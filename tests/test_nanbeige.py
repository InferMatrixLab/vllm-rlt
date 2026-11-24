"""CPU numerical checks for Nanbeige against an independent dense oracle."""

import gc
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tests.helpers import RecordingConnector, make_cpu_cache, tiny_nanbeige_config
from tests.reference import dense_nanbeige_reference
from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.engine.stage_config import StageConfig
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import AutoModelForCausalLM, NanbeigeConfig, NanbeigeForCausalLM
from vllm_rlt.request import RequestPhase
from vllm_rlt.sampling_params import SamplingParams


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
    cache = make_cpu_cache(config)
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
    cache = make_cpu_cache(model.config)
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
    cache = make_cpu_cache(model.config)
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
    cache = make_cpu_cache(mini_vllm_cfg)
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


def _drain_engine(engine: LLMEngine, *, max_steps: int = 2000):
    finished = {}
    for _ in range(max_steps):
        if not engine.has_unfinished_requests():
            return finished
        for output in engine.step():
            if output.finished:
                finished[output.request_id] = output
    pytest.fail("engine did not drain within step limit")


def _copy_prompt_kv(
    source_engine: StageEngine, source_id: str, target_engine: StageEngine, target_id: str
) -> None:
    for src_table, dst_table in zip(
        source_engine.cache_manager.plane_block_tables(source_id),
        target_engine.cache_manager.plane_block_tables(target_id),
    ):
        for src_block, dst_block in zip(src_table, dst_table):
            target_engine.cache_manager.key_cache[dst_block].copy_(
                source_engine.cache_manager.key_cache[src_block]
            )
            target_engine.cache_manager.value_cache[dst_block].copy_(
                source_engine.cache_manager.value_cache[src_block]
            )


@torch.inference_mode()
def _dense_greedy_tokens(
    model: NanbeigeForCausalLM, prompt: list[int], max_tokens: int
) -> list[int]:
    device = next(model.parameters()).device
    tokens = list(prompt)
    generated = []
    for _ in range(max_tokens):
        logits = dense_nanbeige_reference(
            model, torch.tensor(tokens, device=device), model.config.total_ut_steps
        )[-1][1]
        next_token = int(logits[-1].argmax().item())
        generated.append(next_token)
        tokens.append(next_token)
    return generated


def test_nanbeige_incremental_allocation_growth_matches_dense_and_full_reservation(model):
    prompt = [3, 7, 11, 15, 19]
    params = SamplingParams(max_tokens=9, temperature=0.0, ignore_eos=True)
    oracle = _dense_greedy_tokens(model, prompt, params.max_tokens)

    outputs = {}
    block_traces = {}
    for incremental in (False, True):
        engine = LLMEngine(
            model,
            attention_backend="torch",
            cache_config=CacheConfig(
                num_blocks=64, block_size=2, incremental_allocation=incremental
            ),
            scheduler_config=SchedulerConfig(prefill_chunk_size=2),
        )
        try:
            engine.add_request("req", prompt, params)
            seen_blocks = []
            result = None
            while engine.has_unfinished_requests():
                step_outputs = engine.step()
                if engine.cache_manager.has_allocation("req"):
                    seen_blocks.append(engine.cache_manager.allocated_blocks("req"))
                for item in step_outputs:
                    if item.finished:
                        result = item
            assert result is not None
            assert engine.cache_manager.num_used_blocks == 0
            outputs[incremental] = result
            block_traces[incremental] = seen_blocks
        finally:
            engine.close()

    # Full capacity is 5 + 9 - 1 = 13 tokens -> 7 pages x 2 LAST_EXITED planes = 14 blocks.
    assert block_traces[False][0] == 14
    assert min(block_traces[False]) == 14 and max(block_traces[False]) == 14
    # Incremental allocation starts with the first 2-token chunk (1 page x 2 planes = 2 blocks)
    # and grows monotonically across block_size=2 boundaries via ensure_capacity.
    assert block_traces[True][0] == 2
    assert block_traces[True] == sorted(block_traces[True])
    assert sorted(set(block_traces[True])) == [2, 4, 6, 8, 10, 12, 14]
    assert outputs[True].token_ids == outputs[False].token_ids == oracle
    assert outputs[True].exit_depths == [2] * params.max_tokens


def test_nanbeige_incremental_allocation_budget_watermark_and_abort_reuse(model):
    # Without preemption, incremental allocation still reserves future decode growth logically.
    prompt_a = [3, 7, 11]
    prompt_b = [4, 8, 12]
    prompt_reuse = [10, 14, 18]
    params = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
    oracle_b = _dense_greedy_tokens(model, prompt_b, params.max_tokens)
    oracle_reuse = _dense_greedy_tokens(model, prompt_reuse, params.max_tokens)

    # Full capacity per request: 3 + 8 - 1 = 10 tokens -> 5 pages x 2 planes = 10 blocks.
    # Pool of 16 blocks fits one request's full growth budget, not two concurrently.
    engine = LLMEngine(
        model,
        attention_backend="torch",
        cache_config=CacheConfig(num_blocks=16, block_size=2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2,
            max_num_batched_tokens=4,
            prefill_chunk_size=2,
            enable_preemption=False,
        ),
    )
    try:
        engine.add_request("first", prompt_a, params)
        engine.add_request("second", prompt_b, params)
        engine.step()
        assert engine.cache_manager.allocated_blocks("first") == 2
        assert engine.cache_manager.num_free_blocks == 14
        assert engine.scheduler.requests["second"].phase is RequestPhase.WAITING

        # Advance "first" past its initial page before aborting mid-execution.
        for _ in range(30):
            engine.step()
            if engine.cache_manager.allocated_blocks("first") > 4:
                break
        assert engine.cache_manager.allocated_blocks("first") > 4

        engine.abort_request("first")
        assert not engine.cache_manager.has_allocation("first")
        assert engine.cache_manager.num_used_blocks == 0

        # Reusing the aborted request ID with a distinct prompt must not observe stale KV.
        engine.add_request("first", prompt_reuse, params)
        finished = _drain_engine(engine)
        assert finished["second"].token_ids == oracle_b
        assert finished["first"].token_ids == oracle_reuse
        assert engine.cache_manager.num_used_blocks == 0
    finally:
        engine.close()

    # Watermark headroom limits admission of additional incremental requests while idle admits.
    wm_engine = LLMEngine(
        model,
        attention_backend="torch",
        cache_config=CacheConfig(
            num_blocks=14, block_size=2, incremental_allocation=True, watermark_ratio=0.5
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=8, prefill_chunk_size=4
        ),
    )
    try:
        assert wm_engine.cache_manager.watermark_blocks == 7
        short_params = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
        wm_engine.add_request("a", [1, 2, 3, 4], short_params)
        wm_engine.add_request("b", [5, 6, 7, 8], short_params)
        wm_engine.step()
        # "a" uses 4 blocks (10 free); admitting "b" needs 4 + 7 headroom > 10, so "b" waits.
        assert wm_engine.cache_manager.allocated_blocks("a") == 4
        assert wm_engine.scheduler.requests["b"].phase is RequestPhase.WAITING
        assert wm_engine.cache_manager.num_free_blocks == 10
        finished = _drain_engine(wm_engine)
        assert set(finished) == {"a", "b"}
        assert wm_engine.cache_manager.num_used_blocks == 0
    finally:
        wm_engine.close()


def test_nanbeige_incremental_allocation_preemption_restores_outputs_and_rng(model):
    prompts = [[2, 5, 8], [3, 6, 9], [4, 7, 10]]
    params = SamplingParams(max_tokens=8, temperature=0.7, seed=17, ignore_eos=True)
    results = {}
    for blocks in (128, 12):
        torch.manual_seed(99)
        engine = LLMEngine(
            model,
            attention_backend="torch",
            cache_config=CacheConfig(num_blocks=blocks, block_size=2, incremental_allocation=True),
            scheduler_config=SchedulerConfig(
                max_num_seqs=3,
                max_num_batched_tokens=6,
                prefill_chunk_size=2,
                enable_preemption=True,
            ),
        )
        try:
            for idx, prompt in enumerate(prompts):
                engine.add_request(f"r{idx}", prompt, params)
            finished = _drain_engine(engine)
            assert not engine.preemption.snapshots
            assert engine.cache_manager.num_used_blocks == 0
            if blocks == 12:
                assert engine.preemption.preemptions > 0
                assert engine.preemption.resumptions == engine.preemption.preemptions
            results[blocks] = {rid: out.token_ids for rid, out in finished.items()}
        finally:
            engine.close()

    assert results[12] == results[128]


@pytest.mark.parametrize("wcpb", [False, True])
def test_nanbeige_incremental_allocation_with_prefix_reuse_and_wcpb(model, wcpb):
    prompt = [3, 7, 11, 15, 19]
    continuation = [*prompt, 23]
    params = SamplingParams(max_tokens=7, temperature=0.0, ignore_eos=True)
    oracle_a = _dense_greedy_tokens(model, prompt, params.max_tokens)
    oracle_b = _dense_greedy_tokens(model, continuation, params.max_tokens)

    engine = LLMEngine(
        model,
        attention_backend="torch",
        cache_config=CacheConfig(
            num_blocks=64,
            block_size=2,
            enable_prefix_caching=True,
            incremental_allocation=True,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2,
            max_num_batched_tokens=4,
            prefill_chunk_size=2,
            wavefront_chunked_prefill=wcpb,
        ),
    )
    try:
        engine.add_request("a", prompt, params)
        out_a = _drain_engine(engine)["a"]
        assert out_a.token_ids == oracle_a

        # Two complete 2-token pages of the 5-token prompt must be cached and reusable.
        hit = engine.cache_manager.lookup_prefix(continuation)
        assert len(hit) == 2

        hits_before = engine.cache_manager.prefix_hits
        engine.add_request("b", continuation, params)
        engine.step()
        assert engine.cache_manager.prefix_hits - hits_before == 4
        # Reuses 2 cached pages (4 blocks) + allocates 1 tail prompt page (2 blocks) = 6 blocks,
        # strictly less than full capacity (6 + 7 - 1 = 12 tokens -> 6 pages x 2 = 12 blocks).
        assert engine.cache_manager.allocated_blocks("b") == 6
        out_b = _drain_engine(engine)["b"]
        assert out_b.token_ids == oracle_b
        assert engine.cache_manager.num_used_blocks == 0
    finally:
        engine.close()


def _tiny_gpu_nanbeige() -> NanbeigeForCausalLM:
    torch.manual_seed(42)
    return (
        NanbeigeForCausalLM(
            tiny_nanbeige_config(
                hidden_size=256,
                intermediate_size=512,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=64,
            )
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )


@pytest.mark.gpu
@pytest.mark.parametrize("cuda_graphs", [False, True])
def test_nanbeige_gpu_incremental_allocation_triton_and_cuda_graphs(cuda_graphs):
    model = _tiny_gpu_nanbeige()
    prompts = [[3, 7, 11, 15, 19], [4, 8, 12, 16, 20]]
    params = SamplingParams(max_tokens=9, temperature=0.0, ignore_eos=True)
    outputs = {}
    for incremental in (False, True):
        engine = LLMEngine(
            model,
            attention_backend="triton",
            cache_config=CacheConfig(
                num_blocks=64, block_size=2, incremental_allocation=incremental
            ),
            scheduler_config=SchedulerConfig(
                max_num_seqs=4, max_num_batched_tokens=8, prefill_chunk_size=2
            ),
            execution_config=ExecutionConfig(cuda_graphs=cuda_graphs),
        )
        try:
            for idx, prompt in enumerate(prompts):
                engine.add_request(f"g{idx}", prompt, params)
            engine.step()
            # Two requests: 2 blocks each (4 total) incrementally vs 14 blocks each (28 total) full.
            assert engine.cache_manager.num_used_blocks == (4 if incremental else 28)
            peak_blocks = engine.cache_manager.num_used_blocks
            finished = {}
            while engine.has_unfinished_requests():
                for out in engine.step():
                    if out.finished:
                        finished[out.request_id] = out
                peak_blocks = max(peak_blocks, engine.cache_manager.num_used_blocks)
            assert peak_blocks == 28
            assert engine.cache_manager.num_used_blocks == 0
            if cuda_graphs:
                stats = engine.model_runner.graphs
                assert stats.captures > 0 and stats.replays > stats.captures
            outputs[incremental] = {rid: out.token_ids for rid, out in finished.items()}
        finally:
            engine.close()

    assert outputs[True] == outputs[False]


@pytest.mark.gpu
def test_nanbeige_gpu_incremental_allocation_async_preemption():
    model = _tiny_gpu_nanbeige()
    prompts = [[3, 7, 11], [4, 8, 12], [5, 9, 13]]
    params = SamplingParams(max_tokens=9, temperature=0.0, ignore_eos=True)
    results = {}
    for blocks in (128, 12):
        engine = LLMEngine(
            model,
            attention_backend="triton",
            cache_config=CacheConfig(num_blocks=blocks, block_size=2, incremental_allocation=True),
            scheduler_config=SchedulerConfig(
                max_num_seqs=3,
                max_num_batched_tokens=6,
                prefill_chunk_size=2,
                enable_preemption=True,
            ),
            execution_config=ExecutionConfig(
                async_scheduling=True,
                static_buffers=True,
                pad_to_power_of_two=True,
                cuda_graphs=True,
            ),
            exit_config=ExitConfig("ouro_delayed"),
        )
        try:
            for idx, prompt in enumerate(prompts):
                engine.add_request(f"a{idx}", prompt, params)
            finished = _drain_engine(engine)
            assert not engine.preemption.snapshots
            assert engine.cache_manager.num_used_blocks == 0
            if blocks == 12:
                assert engine.preemption.preemptions > 0
                assert engine.preemption.resumptions == engine.preemption.preemptions
            results[blocks] = {rid: out.token_ids for rid, out in finished.items()}
        finally:
            engine.close()

    assert results[12] == results[128]


def test_nanbeige_incremental_allocation_pd_transfer_budget_and_preemption(model):
    prompt = [3, 7, 11, 15, 19]
    params = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
    oracle = _dense_greedy_tokens(model, prompt, params.max_tokens)

    # Part 1: P->D transfer with incremental allocation, decode-side growth budgeting, and parity.
    prefill_engine = StageEngine(
        model,
        stage_config=StageConfig(role="prefill"),
        cache_config=CacheConfig(num_blocks=32, block_size=2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=2
        ),
    )
    prefill_engine.connector = RecordingConnector()
    # Full decode capacity is 5 + 8 - 1 = 12 tokens -> 6 pages x 2 planes = 12 blocks.
    # Pool of 16 fits one imported request's full growth budget, not two concurrently.
    decode_engine = StageEngine(
        model,
        stage_config=StageConfig(role="decode"),
        cache_config=CacheConfig(num_blocks=16, block_size=2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=2
        ),
    )
    try:
        req_p = prefill_engine.reserve_transfer("pd-1", prompt, params, receiving=False)
        assert req_p is not None
        assert prefill_engine.cache_manager.allocated_blocks("pd-1") == 6
        while req_p.phase != RequestPhase.TRANSFERRING:
            prefill_engine.step()

        req_d = decode_engine.reserve_transfer("pd-1", prompt, params, receiving=True)
        assert req_d is not None
        assert decode_engine.cache_manager.allocated_blocks("pd-1") == 6
        assert decode_engine.reserve_transfer("pd-2", prompt, params, receiving=True) is None
        _copy_prompt_kv(prefill_engine, "pd-1", decode_engine, "pd-1")
        decode_engine.accept_import(req_d, req_p.hidden_state.clone())
        prefill_engine.release_transfer(req_p)
        assert prefill_engine.cache_manager.num_used_blocks == 0

        peak_blocks = decode_engine.cache_manager.num_used_blocks
        finished = {}
        while decode_engine.has_unfinished_requests():
            for out in decode_engine.step():
                if out.finished:
                    finished[out.request_id] = out
            peak_blocks = max(peak_blocks, decode_engine.cache_manager.num_used_blocks)
        assert peak_blocks == 12
        assert decode_engine.cache_manager.num_used_blocks == 0
        assert finished["pd-1"].token_ids == oracle
    finally:
        prefill_engine.close()
        decode_engine.close()

    # Part 2: Decode-stage tight-pool preemption under incremental allocation preserves seeded RNG.
    pre_prompts = [[2, 5, 8], [3, 6, 9], [4, 7, 10]]
    pre_params = SamplingParams(max_tokens=8, temperature=0.7, seed=31, ignore_eos=True)
    p_stage = StageEngine(
        model,
        stage_config=StageConfig(role="prefill"),
        cache_config=CacheConfig(num_blocks=64, block_size=2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=6, prefill_chunk_size=2
        ),
    )
    p_stage.connector = RecordingConnector()
    try:
        p_reqs = [
            p_stage.reserve_transfer(f"p{idx}", pr, pre_params, receiving=False)
            for idx, pr in enumerate(pre_prompts)
        ]
        assert all(req is not None for req in p_reqs)
        while any(req.phase != RequestPhase.TRANSFERRING for req in p_reqs):
            p_stage.step()

        pd_results = {}
        for blocks in (64, 16):
            torch.manual_seed(77)
            d_stage = StageEngine(
                model,
                stage_config=StageConfig(role="decode"),
                cache_config=CacheConfig(
                    num_blocks=blocks, block_size=2, incremental_allocation=True
                ),
                scheduler_config=SchedulerConfig(
                    max_num_seqs=3,
                    max_num_batched_tokens=6,
                    prefill_chunk_size=2,
                    enable_preemption=True,
                ),
            )
            try:
                for idx, r_p in enumerate(p_reqs):
                    r_d = d_stage.reserve_transfer(
                        f"d{idx}", r_p.prompt_token_ids, pre_params, receiving=True
                    )
                    assert r_d is not None
                    _copy_prompt_kv(p_stage, r_p.request_id, d_stage, r_d.request_id)
                    d_stage.accept_import(r_d, r_p.hidden_state.clone())
                done = _drain_engine(d_stage)
                assert not d_stage.preemption.snapshots
                assert d_stage.cache_manager.num_used_blocks == 0
                if blocks == 16:
                    assert d_stage.preemption.preemptions > 0
                    assert d_stage.preemption.resumptions == d_stage.preemption.preemptions
                pd_results[blocks] = {rid: out.token_ids for rid, out in done.items()}
            finally:
                d_stage.close()
        for r_p in p_reqs:
            p_stage.release_transfer(r_p)
        assert p_stage.cache_manager.num_used_blocks == 0
        assert pd_results[16] == pd_results[64]
    finally:
        p_stage.close()


@pytest.mark.parametrize("mode", ["block", "pipeline"])
def test_nanbeige_incremental_allocation_speculative_decoding_and_abort_reuse(model, mode):
    prompt = [3, 7, 11, 15, 19]
    params = SamplingParams(max_tokens=10, temperature=0.0, ignore_eos=True)
    oracle = _dense_greedy_tokens(model, prompt, params.max_tokens)

    engine = LLMEngine(
        model,
        attention_backend="torch",
        cache_config=CacheConfig(num_blocks=64, block_size=2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=16, prefill_chunk_size=2
        ),
        speculative_config=SpeculativeConfig(
            num_speculative_tokens=3, draft_loops=1, target_loops=2, mode=mode
        ),
    )
    try:
        engine.add_request("s0", prompt, params)
        engine.step()
        # Starts with first 2-token chunk (2 blocks) instead of full 14-block reservation.
        assert engine.cache_manager.num_used_blocks == 2
        finished = _drain_engine(engine)
        stats = engine.speculative_runner.stats
        assert stats.rounds > 0
        assert stats.drafted_tokens > stats.accepted_tokens
        assert finished["s0"].token_ids == oracle
        assert engine.cache_manager.num_used_blocks == 0

        if mode == "block":
            # Abort mid-speculation and reuse the same request ID with a distinct prompt.
            reuse_prompt = [10, 14, 18, 22, 26]
            oracle_reuse = _dense_greedy_tokens(model, reuse_prompt, params.max_tokens)
            rounds_before = engine.speculative_runner.stats.rounds
            engine.add_request("s0", prompt, params)
            for _ in range(25):
                engine.step()
                if engine.speculative_runner.stats.rounds > rounds_before:
                    break
            assert engine.speculative_runner.stats.rounds > rounds_before
            engine.abort_request("s0")
            assert engine.cache_manager.num_used_blocks == 0
            engine.add_request("s0", reuse_prompt, params)
            reused = _drain_engine(engine)["s0"]
            assert reused.token_ids == oracle_reuse
            assert engine.cache_manager.num_used_blocks == 0
    finally:
        engine.close()


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["block", "pipeline"])
@pytest.mark.parametrize(
    ("async_scheduling", "cuda_graphs"),
    [(False, False), (False, True), (True, True)],
)
def test_nanbeige_gpu_incremental_allocation_speculative(mode, async_scheduling, cuda_graphs):
    model = _tiny_gpu_nanbeige()
    prompt = [3, 7, 11, 15, 19]
    params = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)

    ref_engine = LLMEngine(
        model,
        attention_backend="triton",
        cache_config=CacheConfig(num_blocks=64, block_size=2, incremental_allocation=False),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=16, prefill_chunk_size=2
        ),
    )
    try:
        ref_engine.add_request("ref", prompt, params)
        expected = _drain_engine(ref_engine)["ref"].token_ids
    finally:
        ref_engine.close()

    spec_engine = LLMEngine(
        model,
        attention_backend="triton",
        cache_config=CacheConfig(num_blocks=64, block_size=2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=16, prefill_chunk_size=2
        ),
        execution_config=ExecutionConfig(
            async_scheduling=async_scheduling, cuda_graphs=cuda_graphs
        ),
        speculative_config=SpeculativeConfig(
            num_speculative_tokens=3, draft_loops=1, target_loops=2, mode=mode
        ),
    )
    try:
        spec_engine.add_request("gs", prompt, params)
        spec_engine.step()
        assert spec_engine.cache_manager.num_used_blocks == 2
        got = _drain_engine(spec_engine)["gs"].token_ids
        assert spec_engine.speculative_runner.stats.rounds > 0
        assert got == expected
        assert spec_engine.cache_manager.num_used_blocks == 0
    finally:
        spec_engine.close()


@pytest.mark.gpu
def test_nanbeige_gpu_real_checkpoint_incremental_allocation():
    """Verify incremental KV growth, Graphs, preemption and prefix reuse on real BF16 weights."""
    model_dir = Path(os.environ.get("NANBEIGE_CHECKPOINT", "artifacts/models/Nanbeige4.2-3B"))
    if (
        not (model_dir / "config.json").is_file()
        or not (model_dir / "model-00001-of-00002.safetensors").is_file()
    ):
        pytest.skip(f"Real checkpoint weights not found in {model_dir}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    model = NanbeigeForCausalLM.from_pretrained(
        model_dir, device="cuda", dtype=torch.bfloat16
    ).eval()
    try:
        prompts = [
            tokenizer.encode("The capital of France is", add_special_tokens=False),
            tokenizer.encode(
                "Explain in clear technical terms how paged KV caching reduces memory "
                "fragmentation in large language model inference:",
                add_special_tokens=False,
            ),
        ]
        params = SamplingParams(max_tokens=32, temperature=0.0, ignore_eos=True)

        # 1. Eager and CUDA Graphs: incremental vs full upfront reservation parity.
        for cuda_graphs in (False, True):
            outputs = {}
            for incremental in (False, True):
                engine = LLMEngine(
                    model,
                    attention_backend="triton",
                    cache_config=CacheConfig(
                        num_blocks=64, block_size=16, incremental_allocation=incremental
                    ),
                    scheduler_config=SchedulerConfig(
                        max_num_seqs=4, max_num_batched_tokens=64, prefill_chunk_size=16
                    ),
                    execution_config=ExecutionConfig(cuda_graphs=cuda_graphs),
                )
                try:
                    for idx, prompt in enumerate(prompts):
                        engine.add_request(f"p{idx}", prompt, params)
                    engine.step()
                    assert engine.cache_manager.num_used_blocks == (4 if incremental else 14)
                    peak = engine.cache_manager.num_used_blocks
                    finished = {}
                    while engine.has_unfinished_requests():
                        for out in engine.step():
                            if out.finished:
                                finished[out.request_id] = out
                        peak = max(peak, engine.cache_manager.num_used_blocks)
                    assert peak == 14
                    assert engine.cache_manager.num_used_blocks == 0
                    if cuda_graphs:
                        stats = engine.model_runner.graphs
                        assert stats.captures > 0 and stats.replays > stats.captures
                    outputs[incremental] = {rid: out.token_ids for rid, out in finished.items()}
                finally:
                    engine.close()
            assert outputs[True] == outputs[False]

        # 2. Tight-pool preemption and lossless restoration under incremental allocation.
        pre_prompts = [
            tokenizer.encode("The capital of France is", add_special_tokens=False),
            tokenizer.encode(
                "Write a short Python function to compute Fibonacci numbers:",
                add_special_tokens=False,
            ),
            tokenizer.encode(
                "What is the difference between synchronous and asynchronous execution?",
                add_special_tokens=False,
            ),
        ]
        pre_params = SamplingParams(max_tokens=24, temperature=0.0, ignore_eos=True)
        pre_outputs = {}
        for blocks in (64, 12):
            engine = LLMEngine(
                model,
                attention_backend="triton",
                cache_config=CacheConfig(
                    num_blocks=blocks, block_size=16, incremental_allocation=True
                ),
                scheduler_config=SchedulerConfig(
                    max_num_seqs=3,
                    max_num_batched_tokens=64,
                    prefill_chunk_size=16,
                    enable_preemption=True,
                ),
            )
            try:
                for idx, prompt in enumerate(pre_prompts):
                    engine.add_request(f"r{idx}", prompt, pre_params)
                finished = _drain_engine(engine, max_steps=5000)
                assert not engine.preemption.snapshots
                assert engine.cache_manager.num_used_blocks == 0
                if blocks == 12:
                    assert engine.preemption.preemptions > 0
                    assert engine.preemption.resumptions == engine.preemption.preemptions
                pre_outputs[blocks] = {rid: out.token_ids for rid, out in finished.items()}
            finally:
                engine.close()
        assert pre_outputs[12] == pre_outputs[64]

        # 3. Prefix caching + incremental allocation parity against cold full-reservation run.
        shared_prefix = tokenizer.encode(
            "You are an expert systems engineer working on a recurrent transformer inference "
            "engine. Answer the following technical question concisely.\nQuestion: ",
            add_special_tokens=False,
        )
        q1 = shared_prefix + tokenizer.encode("What is paged attention?", add_special_tokens=False)
        q2 = shared_prefix + tokenizer.encode(
            "How does incremental KV allocation save memory?", add_special_tokens=False
        )
        pf_params = SamplingParams(max_tokens=20, temperature=0.0, ignore_eos=True)
        ref_engine = LLMEngine(
            model,
            attention_backend="triton",
            cache_config=CacheConfig(
                num_blocks=64,
                block_size=16,
                incremental_allocation=False,
                enable_prefix_caching=False,
            ),
            scheduler_config=SchedulerConfig(
                max_num_seqs=2, max_num_batched_tokens=64, prefill_chunk_size=16
            ),
        )
        try:
            ref_engine.add_request("q2", q2, pf_params)
            ref_q2 = _drain_engine(ref_engine, max_steps=5000)["q2"].token_ids
        finally:
            ref_engine.close()

        pf_engine = LLMEngine(
            model,
            attention_backend="triton",
            cache_config=CacheConfig(
                num_blocks=64,
                block_size=16,
                incremental_allocation=True,
                enable_prefix_caching=True,
            ),
            scheduler_config=SchedulerConfig(
                max_num_seqs=2, max_num_batched_tokens=64, prefill_chunk_size=16
            ),
        )
        try:
            pf_engine.add_request("q1", q1, pf_params)
            _drain_engine(pf_engine, max_steps=5000)
            hits_before = pf_engine.cache_manager.prefix_hits
            pf_engine.add_request("q2", q2, pf_params)
            pf_engine.step()
            assert pf_engine.cache_manager.prefix_hits - hits_before >= 16
            assert pf_engine.cache_manager.allocated_blocks(
                "q2"
            ) < pf_engine.cache_manager.required_blocks(len(q2) + pf_params.max_tokens - 1)
            got_q2 = _drain_engine(pf_engine, max_steps=5000)["q2"].token_ids
            assert got_q2 == ref_q2
            assert pf_engine.cache_manager.num_used_blocks == 0
        finally:
            pf_engine.close()

        # 4. PD handoff + incremental allocation parity on real BF16 weights.
        real_prompt = prompts[0]
        real_params = SamplingParams(max_tokens=24, temperature=0.0, ignore_eos=True)
        local_ref = LLMEngine(
            model,
            attention_backend="triton",
            cache_config=CacheConfig(num_blocks=64, block_size=16, incremental_allocation=False),
            scheduler_config=SchedulerConfig(
                max_num_seqs=2, max_num_batched_tokens=64, prefill_chunk_size=16
            ),
        )
        try:
            local_ref.add_request("ref", real_prompt, real_params)
            real_ref_tokens = _drain_engine(local_ref, max_steps=5000)["ref"].token_ids
        finally:
            local_ref.close()

        rp = StageEngine(
            model,
            stage_config=StageConfig(role="prefill"),
            cache_config=CacheConfig(num_blocks=32, block_size=16, incremental_allocation=True),
            scheduler_config=SchedulerConfig(
                max_num_seqs=2, max_num_batched_tokens=64, prefill_chunk_size=16
            ),
        )
        rp.connector = RecordingConnector()
        rd = StageEngine(
            model,
            stage_config=StageConfig(role="decode"),
            cache_config=CacheConfig(num_blocks=32, block_size=16, incremental_allocation=True),
            scheduler_config=SchedulerConfig(
                max_num_seqs=2, max_num_batched_tokens=64, prefill_chunk_size=16
            ),
        )
        try:
            r_req_p = rp.reserve_transfer("real-pd", real_prompt, real_params, receiving=False)
            assert r_req_p is not None
            while r_req_p.phase != RequestPhase.TRANSFERRING:
                rp.step()
            r_req_d = rd.reserve_transfer("real-pd", real_prompt, real_params, receiving=True)
            assert r_req_d is not None
            assert rd.cache_manager.allocated_blocks("real-pd") == 2
            _copy_prompt_kv(rp, "real-pd", rd, "real-pd")
            rd.accept_import(r_req_d, r_req_p.hidden_state.clone())
            rp.release_transfer(r_req_p)
            assert rp.cache_manager.num_used_blocks == 0
            rd_tokens = _drain_engine(rd, max_steps=5000)["real-pd"].token_ids
            assert rd_tokens == real_ref_tokens
            assert rd.cache_manager.num_used_blocks == 0
        finally:
            rp.close()
            rd.close()

        # 5. Self-speculative decoding (block & pipeline, sync/async, graphs) + incremental KV.
        for mode in ("block", "pipeline"):
            for async_sched, cuda_graphs in ((False, False), (False, True), (True, True)):
                spec_eng = LLMEngine(
                    model,
                    attention_backend="triton",
                    cache_config=CacheConfig(
                        num_blocks=64, block_size=16, incremental_allocation=True
                    ),
                    scheduler_config=SchedulerConfig(
                        max_num_seqs=2, max_num_batched_tokens=64, prefill_chunk_size=16
                    ),
                    execution_config=ExecutionConfig(
                        async_scheduling=async_sched, cuda_graphs=cuda_graphs
                    ),
                    speculative_config=SpeculativeConfig(
                        num_speculative_tokens=3, draft_loops=1, target_loops=2, mode=mode
                    ),
                )
                try:
                    spec_eng.add_request("rs", real_prompt, real_params)
                    spec_eng.step()
                    assert spec_eng.cache_manager.num_used_blocks == 2
                    rs_tokens = _drain_engine(spec_eng, max_steps=5000)["rs"].token_ids
                    assert spec_eng.speculative_runner.stats.rounds > 0
                    assert rs_tokens == real_ref_tokens
                    assert spec_eng.cache_manager.num_used_blocks == 0
                finally:
                    spec_eng.close()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
