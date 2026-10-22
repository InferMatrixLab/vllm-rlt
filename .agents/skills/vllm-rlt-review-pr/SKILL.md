---
name: vllm-rlt-review-pr
description: "Review PRs and local changes for InferMatrixLab/vllm-rlt: Ouro engine and KV correctness, serving behavior, and BF16 accuracy/speed evidence. Use for maintainer reviews, repeat reviews, and author self-reviews of this project; not for vLLM-Omni or unrelated repositories."
---

# Review vllm-rlt changes

Review the requested change against its intended base. Return actionable,
evidence-backed defects; zero findings is valid. Distinguish implementation
correctness, reference fidelity, accuracy, speed and memory qualification.

## Human contributor-intent gate

Apply this gate to contributor planning, implementation, and author self-review
before PR submission. It does not block an independent read-only review or
change automated reviewer selection.

Before implementation, have the human contributor explicitly answer:

1. What outcome are you trying to achieve?
2. What is the highest-leverage variable for that outcome?
3. How will this proposed work move that variable, rather than merely create
   activity or busywork?
4. Who is the customer or user, and what do they actually need?

Reuse explicit human answers already present in the request, issue, or
discussion, citing their source. Ask only for missing or ambiguous answers and
wait for the human's response before proceeding with implementation or
submission. Do not invent answers, silently infer intent from a diff, or
complete this gate on the human's behalf.

After those answers are clear, ask the human: **Is a PR needed at all?**
Consider whether an existing capability, a usage/configuration change, an
answer, or an investigation already meets the need. Reuse an explicit,
traceable human decision if one already exists; otherwise wait for it. If the
answer is no or remains unresolved, stop PR-bound work and report the next
step for the human to choose.

For a bugfix, link evidence of an actual observed failure: a reproducible case,
failing test output, logs, or an issue/report describing what happened and what
was expected. Label evidence that has not been reproduced locally. A
hypothetical risk is not observed failure evidence; ask for the missing
evidence rather than manufacture it.

At pre-PR self-review, confirm that the final diff still serves the recorded
outcome, variable, customer need, and PR decision. If intent or scope materially
changed, return only the changed or missing decisions to the human and wait.
Include a concise source-linked record in the existing issue or PR description;
do not create a duplicate planning artifact or copy private discussion into a
public PR. This gate supplements all existing validation and authorization
requirements.

Before submission, inspect the actual PR description against the
[PR template](../../../.github/PULL_REQUEST_TEMPLATE.md). All four required human
answers and the PR-necessity decision must be complete, with observed failure
evidence for a bugfix. Blank fields, placeholders, agent-invented answers, and
unresolved source references are blockers: ask the human only for what is
missing and wait. Reuse explicit, traceable answers instead of asking again.
This is a contributor workflow constraint; the Markdown template does not
provide server-side enforcement.

Contributor entry point: [CONTRIBUTING.md](../../../CONTRIBUTING.md).

## Freeze the review

- Verify the repository/remotes and applicable AGENTS.md. The project is
  `InferMatrixLab/vllm-rlt`; accept its verified forks. Do not inherit vLLM-Omni's
  architecture, reviewers, test requirements or milestone gates.
- For a PR, record base/head SHAs, merge base, state, description, full diff,
  relevant review threads and checks. Read code from the frozen head, not a
  moving local branch. A stacked PR's prerequisite stack is its immediate
  comparison base; distinguish a cumulative comparison against main.
- For local changes, record HEAD, intended base, status and the relevant staged,
  unstaged and untracked diff. Preserve user changes. A review does not authorize
  source fixes, commits, benchmark campaigns or posting a GitHub review.
- Before delivery, check whether the PR head changed. Label the reviewed SHA;
  recheck affected findings if continuing against the newer head. Do not present
  old CI or experiment evidence as validation of an untested newer revision.
- When a prerequisite is reported merged, fetch the verified upstream main
  before selecting a new experiment base. If PR metadata disagrees with Git,
  verify the merge's parents/ancestry or integrated diff rather than relying on
  a cached API state or commit subject alone. Report the discrepancy and the
  verified Git SHA; do not keep treating an integrated prerequisite as open.
  Preserve the requested snapshot for an ongoing review.

## Load relevant references

Read the diff and its callers to select references. Load only those relevant to
the changed behavior or claims, and apply their checks proportionally.

| Reference | Read when |
| --- | --- |
| [Test quality and review execution](references/test-quality-and-execution.md) | Production or test changes, executable validation, or repeat reviews. Covers regression evidence, source identity, simplification and revalidation. |
| [Core and interface review](references/core-and-interface.md) | Core behavior or public API, CLI, configuration, outputs or UI changes. Covers why the change is necessary, Plan B, invariants and compatibility. |
| [A/B evidence](references/ab-evidence.md) | Inference, numerics, memory, measurement logic or accuracy/performance claims are affected. Covers applicable BF16 checks and frozen experiment gates. |

No A/B is required for unaffected runtime/measurement paths. Correctness-only
changes need relevant regression checks, not a speedup.

## Validate and deliver

Use targeted CPU tests or a small reproducer to resolve a concrete uncertainty.
Check test dependencies and skips; a green suite with GPU cases skipped does not
validate kernels or real serving. GPU validation requires the user's applicable
experiment scope and verified scheduler reservation; never use an unreserved
device or silently expand a benchmark's exhausted run budget. If unavailable,
finish the code/evidence review and identify the remaining validation precisely.

For each finding, give priority, a short defect title, the smallest relevant
changed-line location, a reachable trigger, observable impact and supporting
evidence. Trace the failure into unchanged callers if needed, but attribute it
to this change. Exclude speculative risks, unrelated backlog and style-only
feedback. In repeat reviews, verify fixes at the new snapshot and avoid reposting
resolved findings.

Follow the applicable reference's reporting emphasis, then give findings in
priority order and a brief validation/scope note with the reviewed SHA. If there
are no findings, say so without implying unrun gates passed. Use verified PR
diff links or absolute local file links. Draft/post comments only within the
user's explicit authorization; select reviewers from actual ownership evidence.
