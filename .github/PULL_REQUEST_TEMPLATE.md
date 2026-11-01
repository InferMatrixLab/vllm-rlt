<!-- markdownlint-disable -->
<!-- PLEASE FILL IN THE PR DESCRIPTION HERE. -->

## Required human contributor answers

The human contributor must answer all four questions before PR submission.
An explicit answer already recorded elsewhere may be referenced with a precise,
accessible source; a bare link without an identifiable answer is insufficient.
Agents must not invent or fill in the human's answers. Blank, placeholder, or
unresolved answers do not complete this gate. See the
[contributor-intent gate](../.agents/skills/vllm-rlt-review-pr/SKILL.md#human-contributor-intent-gate)
and [contributing guide](../CONTRIBUTING.md).

### 1. What outcome are you trying to achieve?

### 2. What is the highest-leverage variable for that outcome?

### 3. How will this work move that variable rather than create busywork?

### 4. Who is the customer or user, and what do they actually need?

### PR necessity and bugfix evidence

After answering the four questions, explicitly state whether a PR is needed
at all and why. Reuse a traceable human decision if already recorded; otherwise
wait for the human to decide. For a bugfix, link the actual observed failure
(reproducer, failing test output, logs, or issue/report with expected versus
actual behavior); state whether it was reproduced locally.

## Purpose

## Test Plan

**vllm-rlt Version / Commit:**

## Test Result

<!-- BEFORE SUBMITTING: read the contributing guide at https://github.com/InferMatrixLab/vllm-rlt#contributing and run `pre-commit run --all-files` from the repository root (Ruff lint, Ruff format, and the CPU regression test suite). -->
