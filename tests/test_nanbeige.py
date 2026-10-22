"""CPU numerical checks for Nanbeige against an independent dense oracle."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tests.helpers import tiny_nanbeige_config
from tests.reference import dense_nanbeige_reference
from vllm_rlt.config import CacheConfig, ExecutionConfig, SchedulerConfig
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import AutoModelForCausalLM, NanbeigeConfig, NanbeigeForCausalLM
from vllm_rlt.sampling_params import SamplingParams


def make_cache(config, *, blocks=64):
    return KVCacheManager(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        num_blocks=blocks,
        block_size=2,
        max_loops=config.total_ut_steps,
        device="cpu",
        dtype=torch.float32,
    )


@pytest.fixture
def model():
    torch.manual_seed(42)
    return NanbeigeForCausalLM(tiny_nanbeige_config())


def test_nanbeige_config_validation():
    config = tiny_nanbeige_config()
    assert config.num_loops == 2
    assert config.total_ut_steps == 2
    assert config.hidden_act == "silu"

    with pytest.raises(ValueError, match="head_dim"):
        tiny_nanbeige_config(head_dim=15)

    with pytest.raises(ValueError, match="num_attention_heads"):
        tiny_nanbeige_config(num_attention_heads=5, num_key_value_heads=2)

    with pytest.raises(ValueError, match="hidden_act"):
        tiny_nanbeige_config(hidden_act="gelu")

    with pytest.raises(ValueError, match="model_type"):
        NanbeigeConfig.from_dict({"model_type": "llama"})


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_packed_prefill_matches_dense_at_every_depth(kv_heads):
    torch.manual_seed(123)
    config = tiny_nanbeige_config(num_key_value_heads=kv_heads)
    model = NanbeigeForCausalLM(config)
    tokens = torch.tensor([5, 7, 2, 11, 3])
    expected = dense_nanbeige_reference(model, tokens, config.total_ut_steps)
    cache = make_cache(config)
    assert cache.allocate("prompt", len(tokens))
    hidden = model.prelude(tokens)
    for depth, (expected_hidden, expected_logits) in enumerate(expected):
        hidden, gate = model.recurrent(
            hidden, ["prompt"] * len(tokens), [depth] * len(tokens), list(range(len(tokens))), cache
        )
        torch.testing.assert_close(hidden, expected_hidden, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(model.coda(hidden), expected_logits, atol=2e-6, rtol=3e-5)


def test_incremental_tokens_match_dense_fixed_depth(model):
    tokens = torch.tensor([4, 15, 6, 2, 9, 12, 3])
    expected = dense_nanbeige_reference(model, tokens, model.config.total_ut_steps)
    cache = make_cache(model.config)
    assert cache.allocate("sequence", len(tokens))
    for position, token in enumerate(tokens):
        hidden = model.prelude(token.unsqueeze(0))
        for depth in range(model.config.total_ut_steps):
            hidden, _ = model.recurrent(hidden, ["sequence"], [depth], [position], cache)
            expected_hidden, expected_logits = expected[depth]
            torch.testing.assert_close(hidden[0], expected_hidden[position], atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(
                model.coda(hidden)[0], expected_logits[position], atol=2e-6, rtol=3e-5
            )


def test_mixed_request_depth_batch_matches_separate_dense(model):
    first = torch.tensor([5, 8, 12])
    second = torch.tensor([6, 13, 2, 4])
    expected_first = dense_nanbeige_reference(model, first, 2)[1]
    expected_second = dense_nanbeige_reference(model, second, 1)[0]
    cache = make_cache(model.config)
    assert cache.allocate("first", len(first))
    assert cache.allocate("second", len(second))
    first_hidden = model.prelude(first)
    first_hidden, _ = model.recurrent(first_hidden, ["first"] * 3, [0] * 3, [0, 1, 2], cache)
    hidden, _ = model.recurrent(
        torch.cat([first_hidden, model.prelude(second)]),
        ["first"] * 3 + ["second"] * 4,
        [1] * 3 + [0] * 4,
        [0, 1, 2, 0, 1, 2, 3],
        cache,
    )
    torch.testing.assert_close(
        hidden, torch.cat([expected_first[0], expected_second[0]]), atol=3e-6, rtol=3e-5
    )


@pytest.mark.parametrize("sharded", [False, True])
def test_native_safetensors_roundtrip(tmp_path, model, sharded):
    (tmp_path / "config.json").write_text(json.dumps(model.config.to_dict()))
    weights = model.state_dict()
    if sharded:
        names = list(weights.keys())
        half = len(names) // 2
        first = {name: weights[name] for name in names[:half]}
        second = {name: weights[name] for name in names[half:]}
        save_file(first, tmp_path / "model-00001-of-00002.safetensors")
        save_file(second, tmp_path / "model-00002-of-00002.safetensors")
        index = {
            "weight_map": {
                name: (
                    "model-00001-of-00002.safetensors"
                    if name in first
                    else "model-00002-of-00002.safetensors"
                )
                for name in names
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    else:
        save_file(weights, tmp_path / "model.safetensors")

    loaded = NanbeigeForCausalLM.from_pretrained(tmp_path, dtype=torch.float32)
    for name, parameter in loaded.named_parameters():
        torch.testing.assert_close(parameter, weights[name])

    auto_loaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32)
    assert isinstance(auto_loaded, NanbeigeForCausalLM)


def test_nanbeige_engine_generation(model):
    engine = LLMEngine(model, attention_backend="torch")
    prompt = [1, 5, 8]
    engine.add_request(
        "request-1",
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(max_tokens=5, temperature=0.0),
    )
    all_outputs = []
    while engine.has_unfinished_requests():
        outputs = engine.step()
        all_outputs.extend(outputs)

    assert len(all_outputs) > 0
    final = [o for o in all_outputs if o.request_id == "request-1"][-1]
    assert len(final.token_ids) == 5
    assert final.finish_reason == "length"


def test_nanbeige_caller_revision_is_consistently_used_without_override(
    tmp_path, model, monkeypatch
):
    (tmp_path / "config.json").write_text(json.dumps(model.config.to_dict()))
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    called = {}

    def snapshot_download(**kwargs):
        called.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *a, **k: str(tmp_path / "config.json")
    )
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    NanbeigeForCausalLM.from_pretrained("Nanbeige/Nanbeige4.2-3B", revision=None)
    assert called["revision"] is None
    assert called["repo_id"] == "Nanbeige/Nanbeige4.2-3B"
    assert not any(".py" in pattern for pattern in called["allow_patterns"])

    NanbeigeForCausalLM.from_pretrained("Nanbeige/Nanbeige4.2-3B", revision="v1.0")
    assert called["revision"] == "v1.0"


def test_official_nanbeige_tokenizer_contract():
    try:
        from huggingface_hub import try_to_load_from_cache
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers or huggingface_hub not installed")

    tok_path = try_to_load_from_cache("Nanbeige/Nanbeige4.2-3B", "tokenizer.json")
    if tok_path is None:
        pytest.skip("Official Nanbeige tokenizer is not cached locally")

    cache_dir = Path(tok_path).parent
    tok = AutoTokenizer.from_pretrained(cache_dir, local_files_only=True, trust_remote_code=False)
    assert tok.bos_token_id == 166100
    assert tok.eos_token_id == 166101
    assert tok.pad_token_id == 0

    text = "Hello Nanbeige 世界"
    tokens = tok.encode(text, add_special_tokens=False)
    assert tok.decode(tokens) == text


def test_parity_against_official_hf_transformers():
    try:
        from huggingface_hub import try_to_load_from_cache
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        pytest.skip("transformers or huggingface_hub not installed")

    local_dir = Path("artifacts/models/Nanbeige4.2-3B")
    if (local_dir / "config.json").is_file():
        cache_dir = local_dir
    else:
        config_path = try_to_load_from_cache("Nanbeige/Nanbeige4.2-3B", "config.json")
        if config_path is None:
            pytest.skip("Official Nanbeige remote code is not cached locally")
        cache_dir = Path(config_path).parent

    try:
        cfg = AutoConfig.from_pretrained(cache_dir, local_files_only=True, trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"Failed to load official config: {e}")

    mini_dict = cfg.to_dict()
    mini_dict.update(
        {
            "vocab_size": 1000,
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "kv_channels": 64,
            "num_loops": 2,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        }
    )

    mini_cfg = cfg.__class__.from_dict(mini_dict)
    mini_vllm_cfg = NanbeigeConfig.from_dict(mini_dict)

    torch.manual_seed(42)
    official_model = (
        AutoModelForCausalLM.from_config(mini_cfg, trust_remote_code=True).eval().to(torch.float32)
    )
    vllm_model = NanbeigeForCausalLM(mini_vllm_cfg).eval().to(torch.float32)

    # 1. Parameter names and shapes
    official_params = {k: v.shape for k, v in official_model.named_parameters()}
    vllm_params = {k: v.shape for k, v in vllm_model.named_parameters()}
    assert official_params == vllm_params, (
        f"Param mismatch: {set(official_params) ^ set(vllm_params)}"
    )

    # 2. Transfer weights
    vllm_model.load_state_dict(official_model.state_dict())

    # 3. Compare forward logits against dense_nanbeige_reference
    input_ids = torch.tensor([[12, 45, 88, 120, 250]], dtype=torch.long)
    with torch.no_grad():
        official_logits = official_model(input_ids, use_cache=False).logits.float()

    ref_outputs = dense_nanbeige_reference(vllm_model, input_ids[0], loops=mini_vllm_cfg.num_loops)
    ref_logits = ref_outputs[-1][1].unsqueeze(0).float()
    torch.testing.assert_close(ref_logits, official_logits, atol=1e-5, rtol=1e-4)

    # 4. Compare native engine execution
    cache = make_cache(mini_vllm_cfg)
    tokens = input_ids[0]
    seq_len = len(tokens)
    assert cache.allocate("parity-req", seq_len)
    hidden = vllm_model.prelude(tokens)
    for depth in range(mini_vllm_cfg.num_loops):
        hidden, _ = vllm_model.recurrent(
            hidden,
            ["parity-req"] * seq_len,
            [depth] * seq_len,
            list(range(seq_len)),
            cache,
        )
    engine_logits = vllm_model.coda(hidden).unsqueeze(0).float()
    torch.testing.assert_close(engine_logits, official_logits, atol=1e-5, rtol=1e-4)


def test_parity_against_real_checkpoint_hf_transformers():
    """Verify external reference fidelity against the official HF vendor model on real weights."""
    model_dir = Path("artifacts/models/Nanbeige4.2-3B")
    if (
        not (model_dir / "config.json").is_file()
        or not (model_dir / "model-00001-of-00002.safetensors").is_file()
    ):
        pytest.skip("Real checkpoint weights not downloaded in artifacts/models/Nanbeige4.2-3B")

    import gc

    from transformers import AutoModelForCausalLM as HFAutoModel
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    prompt_set = [
        "你好，请介绍一下你自己。",
        "The capital of France is",
        "def fibonacci(n):",
    ]

    # 1. Official HF model execution (bfloat16)
    hf_model = HFAutoModel.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval()

    # Note: official vendor modeling_nanbeige.py registers inv_freq as persistent=False.
    # When meta-loaded via HuggingFace from_pretrained, the un-serialized buffer must be populated.
    base = hf_model.config.rope_theta
    dim = hf_model.config.head_dim
    correct_inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    for layer in hf_model.model.layers:
        layer.self_attn.rotary_emb.inv_freq = correct_inv_freq.clone()

    hf_results = {}
    with torch.no_grad():
        for prompt in prompt_set:
            input_ids = tok.encode(prompt, return_tensors="pt")
            logits = hf_model(input_ids, use_cache=False).logits[0, -1, :].float()
            top5 = torch.topk(logits, 5).indices.tolist()
            hf_results[prompt] = (logits, top5)

    # Autoregressive multi-step greedy generation from official HF model
    greedy_prompt = "The capital of France is"
    greedy_steps = 5
    cur_ids = tok.encode(greedy_prompt, return_tensors="pt")
    hf_greedy_tokens = []
    with torch.no_grad():
        for _ in range(greedy_steps):
            out = hf_model(cur_ids, use_cache=False)
            next_tok = torch.argmax(out.logits[0, -1, :]).item()
            hf_greedy_tokens.append(next_tok)
            cur_ids = torch.cat([cur_ids, torch.tensor([[next_tok]])], dim=1)

    del hf_model
    gc.collect()

    # 2. vLLM-RLT native model execution (bfloat16)
    vllm_model = NanbeigeForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).eval()

    for prompt in prompt_set:
        input_ids = tok.encode(prompt, return_tensors="pt")
        seq_len = input_ids.shape[1]
        cache = KVCacheManager(
            num_layers=vllm_model.config.num_hidden_layers,
            num_kv_heads=vllm_model.config.num_key_value_heads,
            head_dim=vllm_model.config.head_dim,
            num_blocks=32,
            block_size=16,
            max_loops=vllm_model.config.total_ut_steps,
            device="cpu",
            dtype=torch.bfloat16,
        )
        assert cache.allocate(f"parity-{prompt[:4]}", seq_len)
        tokens = input_ids[0]
        with torch.no_grad():
            hidden = vllm_model.prelude(tokens)
            for depth in range(vllm_model.config.total_ut_steps):
                hidden, _ = vllm_model.recurrent(
                    hidden,
                    [f"parity-{prompt[:4]}"] * seq_len,
                    [depth] * seq_len,
                    list(range(seq_len)),
                    cache,
                )
            vllm_logits = vllm_model.coda(hidden)[-1].float()

        vllm_top5 = torch.topk(vllm_logits, 5).indices.tolist()
        hf_logits, hf_top5 = hf_results[prompt]

        # 3. Assert fidelity: top-1 greedy token matches, top-5 set matches,
        # and cosine similarity >= 0.9998
        assert hf_top5[0] == vllm_top5[0], (
            f"Top-1 mismatch for {prompt!r}: HF {hf_top5[0]} vs vLLM {vllm_top5[0]}"
        )
        assert set(hf_top5) == set(vllm_top5), (
            f"Top-5 set mismatch for {prompt!r}: HF {hf_top5} vs vLLM {vllm_top5}"
        )
        cos_sim = torch.nn.functional.cosine_similarity(
            hf_logits.unsqueeze(0), vllm_logits.unsqueeze(0)
        ).item()
        assert cos_sim >= 0.9998, (
            f"Cosine similarity {cos_sim:.6f} below expected 0.9998 threshold for {prompt!r}"
        )

    # 4. Engine greedy generation comparison
    engine = LLMEngine(vllm_model, attention_backend="torch")
    engine.add_request(
        "parity-greedy",
        prompt_token_ids=tok.encode(greedy_prompt),
        sampling_params=SamplingParams(max_tokens=greedy_steps, temperature=0.0),
    )
    all_outs = []
    while engine.has_unfinished_requests():
        all_outs = engine.step()
    vllm_greedy_tokens = [o for o in all_outs if o.request_id == "parity-greedy"][-1].token_ids
    assert hf_greedy_tokens == vllm_greedy_tokens, (
        f"Greedy tokens mismatch: HF {hf_greedy_tokens} vs vLLM {vllm_greedy_tokens}"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("max_graphs", [1, 16])
def test_nanbeige_graph_replay_matches_eager_with_reuse_and_fallback(max_graphs):
    """Verify Nanbeige recurrent CUDA Graph replay matches eager with reuse and fallback."""
    torch.manual_seed(42)
    model = (
        NanbeigeForCausalLM(
            tiny_nanbeige_config(
                hidden_size=64,
                num_attention_heads=4,
                intermediate_size=128,
            )
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    results = []
    for graph in (False, True):
        engine = LLMEngine(
            model,
            attention_backend="triton",
            cache_config=CacheConfig(num_blocks=256, block_size=4),
            scheduler_config=SchedulerConfig(
                max_num_seqs=4,
                max_num_batched_tokens=16,
                prefill_chunk_size=2,
            ),
            execution_config=ExecutionConfig(cuda_graphs=graph, cuda_graph_max_graphs=max_graphs),
        )
        rounds = []
        for _ in range(2):
            torch.manual_seed(123)
            for i, prompt in enumerate(([4, 7, 3], [9, 2], [3, 8, 7, 6], [4])):
                engine.add_request(
                    str(i),
                    prompt,
                    SamplingParams(
                        max_tokens=4 + i,
                        min_loops=2,
                        max_loops=2,
                        ignore_eos=True,
                    ),
                )
            finished = {}
            while engine.has_unfinished_requests():
                for output in engine.step():
                    if output.finished:
                        finished[output.request_id] = (output.token_ids, output.exit_depths)
            assert len(finished) == 4
            assert engine.cache_manager.num_used_blocks == 0
            rounds.append(finished)
        if graph:
            stats = engine.model_runner.graphs
            assert stats.captures > 0 and stats.replays > stats.captures
            if max_graphs == 1:
                assert stats.fallbacks > 0
        results.append(rounds)
    assert results[0] == results[1]


@pytest.mark.gpu
def test_nanbeige_graph_states_logits_and_kv_match_eager():
    """Verify bitwise match of hidden states, logits, and KV entries between eager and graph."""
    from vllm_rlt.worker.cuda_graph import RecurrentGraphs

    torch.manual_seed(42)
    cfg = tiny_nanbeige_config(hidden_size=64, num_attention_heads=4, intermediate_size=128)
    model = NanbeigeForCausalLM(cfg).to(device="cuda", dtype=torch.bfloat16).eval()

    def make_cuda_cache():
        return KVCacheManager(
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            num_blocks=128,
            block_size=4,
            max_loops=cfg.total_ut_steps,
            device="cuda",
            dtype=torch.bfloat16,
            backend="triton",
        )

    caches = [make_cuda_cache(), make_cuda_cache()]
    graphs = RecurrentGraphs(model, caches[1], ExecutionConfig(cuda_graphs=True), False)
    for c in caches:
        assert c.allocate("request", 5)

    with torch.no_grad():
        for position in range(5):
            token = torch.tensor([4 + position], device="cuda", dtype=torch.long)
            states = [model.prelude(token), model.prelude(token)]
            torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)

            for depth in range(cfg.total_ut_steps):
                batches = [c._prepare_batch(["request"], [depth], [position]) for c in caches]
                states[0], _ = model.recurrent_prepared(
                    states[0], batches[0], caches[0], compute_gate=False
                )
                states[1], _ = graphs.run(states[1], batches[1])
                torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)

                for layer in range(cfg.num_hidden_layers):
                    for eager, replay in zip(
                        caches[0].read(layer, "request", depth),
                        caches[1].read(layer, "request", depth),
                    ):
                        torch.testing.assert_close(eager, replay, rtol=0, atol=0)

            logits = [model.coda(h) for h in states]
            torch.testing.assert_close(logits[0], logits[1], rtol=0, atol=0)
            for c in caches:
                c.finalize_token("request", position, cfg.total_ut_steps - 1)

    assert graphs.captures == 1 and graphs.replays == 10 and graphs.fallbacks == 0
    for c in caches:
        c.free("request")
        assert c.num_used_blocks == 0
