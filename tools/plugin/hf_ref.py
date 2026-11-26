"""HF reference for Ouro-1.4B greedy decoding (full depth, no early exit)."""
import json, sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from transformers import modeling_rope_utils as _ru

if "default" not in _ru.ROPE_INIT_FUNCTIONS:
    # transformers 5.x dropped the plain RoPE entry Ouro's remote code asks for.
    def _default_rope(config, device=None, seq_len=None, **_):
        base = getattr(config, "rope_theta", None)
        if base is None:
            base = (getattr(config, "rope_parameters", None) or {}).get("rope_theta", 10000.0)
        head = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head * getattr(config, "partial_rotary_factor", 1.0))
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv, 1.0

    _ru.ROPE_INIT_FUNCTIONS["default"] = _default_rope

MODEL = "/home/zjy/code/david/b_workspace/models/Ouro-1.4B"
n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
dtype = getattr(torch, sys.argv[2]) if len(sys.argv) > 2 else torch.bfloat16
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
# transformers 5.x no longer defaults pad_token_id; Ouro's remote code reads it.
if getattr(cfg, "pad_token_id", None) is None:
    try:
        cfg.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else cfg.eos_token_id
    except Exception:
        object.__setattr__(cfg, "pad_token_id", cfg.eos_token_id)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, config=cfg, trust_remote_code=True, dtype=dtype
).cuda().eval()
# Ouro's remote code calls the mask helpers with the pre-5.x keyword.
import functools as _ft

for _name, _mod in list(sys.modules.items()):
    if "modeling_ouro" in _name:
        for _fn in ("create_causal_mask", "create_sliding_window_causal_mask"):
            _orig = getattr(_mod, _fn, None)
            if _orig is None:
                continue

            import inspect as _in

            _ok = set(_in.signature(_orig).parameters)

            def _wrap(*a, _orig=_orig, _ok=_ok, **k):
                if "input_embeds" in k:
                    k["inputs_embeds"] = k.pop("input_embeds")
                k = {n: v for n, v in k.items() if n in _ok}
                return _orig(*a, **k)

            setattr(_mod, _fn, _ft.wraps(_orig)(_wrap))
prompts = ["The capital of France is", "def fib(n):", "Water boils at"]
for p in prompts:
    ids = tok(p, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=n, do_sample=False, use_cache=True,
            pad_token_id=cfg.pad_token_id,
        )
    gen = out[0, ids.shape[1]:].tolist()
    print(json.dumps({"prompt": p, "ids": gen}))
