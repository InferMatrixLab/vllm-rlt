"""Choose drafts per async speculative round from measured round cost and acceptance.

Speculation trades extra verification rows for fewer sequential rounds. When the
GPU is latency-bound (few rows) the extra rows are nearly free; when it is
compute- or bandwidth-bound (many rows, long contexts) they are not, and K=0, a
full-depth single-token step with native-equivalent compute, is cheaper per
token. The policy picks K' in [0, K] maximizing expected committed tokens per
predicted round time:

    tokens(K') = n * (1 + sum over j <= K' of prod(p_i for i < j))
    time(K')   = theta . features(n, contexts, K')

theta is fitted online by recursive least squares from compute-stream events
around every submitted round (both events have completed when the round is
collected), with a prior from the startup graph-bucket replay timings. p_i is
the acceptance probability of draft position i given an accepted prefix, with
a Beta prior, updated from collected rounds. A deterministic exploration
round (K' >= 1) every `explore_every` rounds keeps both estimates current when
the policy settles on K'=0.
"""

import math

import torch

# Feature order: core passes, core row-passes, coda rows, context row-passes
# (thousands of tokens), constant. Coda calls are one per draft offset plus one,
# an affine function of passes, so their fixed cost folds into those two terms.
NUM_FEATURES = 5


def round_features(token_starts, counts, draft_loops, target_loops, bucket=None):
    """Work of one round whose request i drafts counts[i]-1 tokens from token_starts[i]."""
    bucket = bucket or (lambda rows: rows)
    longest = max(counts)
    verify_loops = target_loops - draft_loops
    core_rows = coda_rows = ctx = 0.0
    for offset in range(longest):
        active = [i for i, c in enumerate(counts) if offset < c]
        core_rows += draft_loops * bucket(len(active))
        ctx += draft_loops * sum(token_starts[i] + offset + 1 for i in active)
        drafting = sum(1 for i in active if offset + 1 < counts[i])
        if drafting:
            coda_rows += bucket(drafting)
    rows = sum(counts)
    core_rows += verify_loops * bucket(rows)
    ctx += verify_loops * sum(
        token_starts[i] + j + 1 for i, c in enumerate(counts) for j in range(c)
    )
    coda_rows += bucket(rows)
    passes = draft_loops * longest + verify_loops
    return [float(passes), core_rows, coda_rows, ctx / 1000.0, 1.0]


class AdaptiveDraftPolicy:
    def __init__(
        self,
        max_drafts,
        draft_loops,
        target_loops,
        *,
        token_budget,
        bucket=None,
        prior=None,
        alpha_prior=(4.0, 1.0),
        forgetting=0.99,
        explore_every=16,
    ):
        self.max_drafts = max_drafts
        self.draft_loops, self.target_loops = draft_loops, target_loops
        self.token_budget = token_budget
        self.bucket = bucket
        self.forgetting = forgetting
        self.explore_every = explore_every
        self.theta = list(prior) if prior is not None else [0.05, 0.002, 0.002, 0.0, 0.1]
        # Weak prior: the first measured rounds dominate the fit.
        self.covariance = [
            [(100.0 if r == c else 0.0) for c in range(NUM_FEATURES)] for r in range(NUM_FEATURES)
        ]
        # Per draft position: (accepted, stopped) counts with a Beta prior. Later
        # drafts condition on longer accepted prefixes and are accepted less often.
        self.position_counts = [list(alpha_prior) for _ in range(max_drafts)]
        self.rounds = 0
        self.choices = [0] * (max_drafts + 1)
        self.observed = 0
        self.abs_error_ms = 0.0

    def uniform_features(self, n, starts, drafts):
        """round_features for n requests all drafting `drafts`, in closed form."""
        bucket = self.bucket or (lambda rows: rows)
        d, v, width = self.draft_loops, self.target_loops - self.draft_loops, drafts + 1
        core_rows = d * width * bucket(n) + v * bucket(n * width)
        coda_rows = drafts * bucket(n) + bucket(n * width)
        ctx = (d + v) * (width * starts + n * width * (width + 1) / 2)
        return [float(d * width + v), core_rows, coda_rows, ctx / 1000.0, 1.0]

    def acceptance(self, position):
        accepted, stopped = self.position_counts[position]
        return accepted / (accepted + stopped)

    @property
    def alpha(self):
        return self.acceptance(0)

    def expected_tokens(self, drafts):
        """1 + sum over j <= drafts of P(the first j drafts are all accepted)."""
        total, prefix = 1.0, 1.0
        for position in range(drafts):
            prefix *= self.acceptance(position)
            total += prefix
        return total

    def predict_ms(self, features):
        return max(1e-3, sum(t * x for t, x in zip(self.theta, features)))

    def choose(self, token_starts):
        """Drafts for a round over the eligible requests' current token starts."""
        n = len(token_starts)
        if not n:
            # No eligible request: the scheduler builds no speculative round.
            return self.max_drafts
        self.rounds += 1
        cap = min(self.max_drafts, max(0, self.token_budget // n - 1))
        if self.explore_every and cap and self.rounds % self.explore_every == 0:
            drafts = 1 + (self.rounds // self.explore_every) % cap
        else:
            best, drafts = -1.0, 0
            starts = float(sum(token_starts))
            for k in range(cap + 1):
                features = self.uniform_features(n, starts, k)
                expected = self.expected_tokens(k)
                rate = n * expected / self.predict_ms(features)
                if rate > best * (1 + 1e-9):
                    best, drafts = rate, k
        self.choices[drafts] += 1
        return drafts

    def observe_acceptance(self, drafted, accepted):
        """Drafts before the first rejection were accepted; the rejected one stopped."""
        for position in range(min(accepted, len(self.position_counts))):
            self.position_counts[position][0] += 1
        if accepted < drafted and accepted < len(self.position_counts):
            self.position_counts[accepted][1] += 1

    def observe_time(self, features, elapsed_ms):
        """Recursive least squares with exponential forgetting."""
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            return
        predicted = self.predict_ms(features)
        self.observed += 1
        self.abs_error_ms += abs(predicted - elapsed_ms)
        x, p, lam = features, self.covariance, self.forgetting
        px = [sum(p[r][c] * x[c] for c in range(NUM_FEATURES)) for r in range(NUM_FEATURES)]
        denominator = lam + sum(x[r] * px[r] for r in range(NUM_FEATURES))
        gain = [v / denominator for v in px]
        error = elapsed_ms - sum(t * v for t, v in zip(self.theta, x))
        self.theta = [t + g * error for t, g in zip(self.theta, gain)]
        self.covariance = [
            [(p[r][c] - gain[r] * px[c]) / lam for c in range(NUM_FEATURES)]
            for r in range(NUM_FEATURES)
        ]

    def summary(self):
        return dict(
            alpha=self.alpha,
            acceptance=[self.acceptance(j) for j in range(self.max_drafts)],
            theta=list(self.theta),
            choices=list(self.choices),
            observed_rounds=self.observed,
            mean_abs_error_ms=self.abs_error_ms / self.observed if self.observed else None,
        )


def calibrated_prior(core_ms, coda_ms, draft_loops, target_loops):
    """Prior theta from startup bucket replay times {rows: ms} (zero context).

    A round makes (passes - verify_loops) / draft_loops coda calls, so the coda
    call intercept is split between the per-pass and constant terms.
    """

    def fit(points):
        if len(points) < 2:
            return 0.05, 0.002
        xs, ys = zip(*sorted(points.items()))
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        var = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var if var else 0.0
        return max(0.0, my - slope * mx), max(0.0, slope)

    core_pass, core_row = fit(core_ms)
    coda_call, coda_row = fit(coda_ms)
    verify_loops = target_loops - draft_loops
    return [
        core_pass + coda_call / draft_loops,
        core_row,
        coda_row,
        0.0,
        coda_call * (1 - verify_loops / draft_loops),
    ]


@torch.inference_mode()
def time_replays(graphs, repeats=5):
    """Milliseconds per replay of each captured bucket, on the current stream."""
    times = {}
    stream = torch.cuda.current_stream()
    for size, entry in sorted(graphs.entries.items()):
        entry.graph.replay()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(repeats):
            entry.graph.replay()
        end.record(stream)
        end.synchronize()
        times[size] = start.elapsed_time(end) / repeats
    return times
