"""Minimal streaming /v1/completions benchmark for vLLM and vllm-rlt.

Throughput = successful output tokens / (wave start -> last completion), as in PR #214.
Usage: bench.py PORT PROMPTS.json KEY CONCURRENCY MAX_TOKENS IGNORE_EOS WAVES OUT.json
"""
import asyncio, json, statistics, sys, time
import aiohttp

port, pfile, key, conc, max_tokens, ignore_eos, waves, out = sys.argv[1:9]
conc, max_tokens, waves, ignore_eos = int(conc), int(max_tokens), int(waves), ignore_eos == "1"
prompts = json.load(open(pfile, encoding="utf-8"))
prompts = prompts[key] if isinstance(prompts, dict) else prompts
URL = f"http://127.0.0.1:{port}/v1/completions"


async def one(session, prompt):
    body = {"model": "inkling", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "ignore_eos": ignore_eos, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); stamps = []; usage = None
    async with session.post(URL, json=body) as r:
        if r.status != 200:
            return {"ok": False, "err": (await r.text())[:300]}
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and d["choices"][0].get("text"):
                stamps.append(time.perf_counter())
    n = usage["completion_tokens"] if usage else len(stamps)
    ttft = stamps[0] - t0 if stamps else None
    itl = [b - a for a, b in zip(stamps, stamps[1:])]
    tpot = (stamps[-1] - stamps[0]) / (n - 1) if n > 1 and stamps else None
    return {"ok": True, "n": n, "ttft": ttft, "tpot": tpot, "itl": itl, "end": time.perf_counter()}


async def wave(session, items):
    q = list(items); res = []
    async def lane():
        while q:
            res.append(await one(session, q.pop(0)))
    t0 = time.perf_counter()
    await asyncio.gather(*(lane() for _ in range(conc)))
    ok = [r for r in res if r["ok"]]
    if not ok:
        raise SystemExit(f"all requests failed: {[r.get('err') for r in res][:2]}")
    dur = max(r["end"] for r in ok) - t0
    pct = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else None
    itl = [x for r in ok for x in r["itl"]]
    return {"tok_s": sum(r["n"] for r in ok) / dur, "ok": len(ok), "fail": len(res) - len(ok),
            "ttft_p50_ms": 1e3 * statistics.median([r["ttft"] for r in ok if r["ttft"] is not None]),
            "tpot_p50_ms": 1e3 * statistics.median([r["tpot"] for r in ok if r["tpot"]]),
            "itl_p99_ms": 1e3 * pct(itl, 0.99), "errors": [r["err"] for r in res if not r["ok"]][:3]}


async def main():
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        warm = [prompts[i] for i in range(4, len(prompts), 8)][:conc]
        await asyncio.gather(*(one(s, p) for p in warm))
        results = [await wave(s, prompts) for _ in range(waves)]
    med = statistics.median(r["tok_s"] for r in results)
    json.dump({"key": key, "conc": conc, "waves": results, "median_tok_s": med}, open(out, "w"), indent=1)
    print(key, conc, [round(r["tok_s"], 1) for r in results], "median", round(med, 1))

asyncio.run(main())
