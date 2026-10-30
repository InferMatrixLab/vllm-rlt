"""CPU checks for the accuracy protocol and comparison failure modes."""

import json
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from benchmarks import gsm8k
from benchmarks.gsm8k import (
    DEFAULT_CASE,
    PROMPT_THINKING,
    SCORER_LM_EVAL,
    SCORER_THINKING,
    baseline_fingerprint,
    baseline_result,
    build_records,
    checkpoint_loops,
    compare,
    depth_summary,
    digest,
    exit_settings,
    file_digest,
    fixed_exit,
    make_task,
    score,
    score_thinking,
    select_doc_ids,
    sequence_differences,
    thinking_close_id,
    validate_prompt_recipe,
)


@pytest.fixture
def task(monkeypatch):
    pytest.importorskip("lm_eval")
    from datasets import Dataset, DatasetDict
    from lm_eval.api.task import ConfigurableTask

    def download(self, *args, **kwargs):
        self.dataset = DatasetDict(
            {
                split: Dataset.from_dict(
                    {
                        "question": [f"What is 6 times 3? Question #{i}" for i in range(1319)],
                        "answer": ["#### 18"] * 1319,
                    }
                )
                for split in ("train", "test")
            }
        )

    monkeypatch.setattr(ConfigurableTask, "download", download)
    return make_task()


@pytest.fixture
def thinking_records():
    # No lm-eval dependency: these new-path checks must execute in default CPU CI.
    task = SimpleNamespace(
        eval_docs=[{"question": f"Question {i}", "answer": "#### 18"} for i in range(10)],
        build_all_requests=lambda *a, **k: pytest.fail("no few-shot requests"),
    )
    options = dict(
        split="test",
        limit=3,
        seed=1,
        max_new_tokens=1024,
        max_length=2048,
        prompt_mode=PROMPT_THINKING,
    )
    return lambda tokenizer, **changes: build_records(task, tokenizer, **(options | changes))


def test_thinking_records_use_the_chat_template_and_eos_only(thinking_records):
    def template(messages, **options):
        assert options == dict(tokenize=False, add_generation_prompt=True, enable_thinking=True)
        assert len(messages) == 1 and messages[0]["role"] == "user"
        return f"chat:{messages[0]['content']}<think>\n"

    tokenizer = SimpleNamespace(
        apply_chat_template=template,
        encode=lambda text, add_special_tokens=True: (
            ([1] if add_special_tokens else []) + list(map(ord, text))
        ),
    )
    records, selection = thinking_records(tokenizer)
    assert [row["id"] for row in records] == selection["ids"]
    for row in records:
        assert row["doc"] == {"question": f"Question {row['id']}", "answer": "#### 18"}
        assert row["prompt"] == f"chat:{row['doc']['question']}<think>\n"
        assert row["prompt_ids"] == [1, *map(ord, row["prompt"])]
        assert row["generation_kwargs"]["until"] == []
    with pytest.raises(ValueError, match="Unknown prompt mode"):
        thinking_records(tokenizer, prompt_mode="bogus")
    with pytest.raises(ValueError, match="truncated"):
        thinking_records(tokenizer, max_length=1)


def test_thinking_records_with_real_checkpoint_tokenizer(thinking_records):
    import os

    source = os.environ.get("OURO_14B_THINKING_TOKENIZER")
    if source is None:
        pytest.skip("set OURO_14B_THINKING_TOKENIZER to prepared local assets; no downloads")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    records, _ = thinking_records(tokenizer)
    for row in records:
        assert row["prompt"].endswith("<think>\n")
        assert row["prompt_ids"][0] == tokenizer.bos_token_id
    assert thinking_close_id(tokenizer) == 4


@pytest.mark.parametrize(
    "edits",
    [
        {"scorer": SCORER_THINKING},
        {"prompt_mode": "thinking"},
        {"apply_chat_template": True},
        {"add_special_tokens": True},
        {"prompt_mode": "unknown"},
    ],
)
def test_prompt_recipe_rejects_inconsistent_fields(edits):
    with pytest.raises(ValueError, match="prompt recipe"):
        validate_prompt_recipe(edits)


@pytest.fixture
def text_tokenizer():
    return SimpleNamespace(decode=lambda ids, **kwargs: "".join(map(chr, ids)))


@pytest.mark.parametrize("markers", [[], [4], [4, *map(ord, r"\boxed{99}"), 4]])
def test_thinking_requires_the_last_real_closing(text_tokenizer, markers):
    final = r"\boxed{18}"
    ids = [*map(ord, "</think>"), *markers, *map(ord, final)]
    result = score_thinking({"doc": {"answer": "#### 18"}}, ids, text_tokenizer, close_id=4)
    assert result["has_closing"] is bool(markers)
    assert result["correct"] is bool(markers)
    assert result["unparseable"] is not bool(markers)
    if markers:
        assert result["final_text"] == final


@pytest.mark.parametrize(
    "final_text,gold,answer,correct",
    [
        (r"The answer is \boxed{18}", "18", "18", True),
        (r"\boxed{18} then \boxed{19}", "19", "19", True),
        ("So the total is #### 18", "18", "18", True),
        (r"\boxed{ $1,234.00. }", "1234", " $1,234.00. ", True),
        (r"\boxed{-3}", "-3", "-3", True),
        ("#### $18.00.", "18", "$18.00.", True),
        (r"\boxed{18} explanatory prose", "18", "18", True),
        (r"\boxed{17}", "18", "17", False),
        (r"\boxed{\frac{36}{2}}", "18", None, False),
        ("The answer is 18.", "18", None, False),
        (r"\boxed{unbalanced", "18", None, False),
        (r"\boxed{NaN}", "18", None, False),
        (r"\boxed{Infinity}", "18", None, False),
        (r"\boxed{1e1}", "10", None, False),
        (r"\boxed{17} then \boxed{oops}", "17", None, False),
        ("", "18", None, False),
    ],
)
def test_score_thinking_extraction(text_tokenizer, final_text, gold, answer, correct):
    row = {"doc": {"answer": f"work.\n#### {gold}"}}
    result = score_thinking(row, [4, *map(ord, final_text)], text_tokenizer, close_id=4)
    assert result["answer"] == answer
    assert result["correct"] is correct
    assert result["unparseable"] is (answer is None)


@pytest.mark.parametrize("candidate", [None, -1, 2, 99, True])
def test_thinking_scorer_validates_marker(candidate):
    tokenizer = SimpleNamespace(
        convert_tokens_to_ids=lambda _: candidate,
        eos_token_id=2,
        all_special_ids=[1, 2, 3, 4],
        added_tokens_decoder={},
        convert_ids_to_tokens=lambda _: "</think>",
        encode=lambda *a, **kw: [4],
    )
    with pytest.raises(ValueError, match="standalone"):
        thinking_close_id(tokenizer)


@pytest.mark.parametrize("metadata_case", ["complete", "legacy", "missing"])
def test_thinking_scorer_combines_special_token_metadata(metadata_case):
    # Transformers 5.14.1 omits special added tokens from all_special_ids.
    added = {}
    if metadata_case == "legacy":
        added = {4: SimpleNamespace(special=True), 9: SimpleNamespace(special=False)}
    tokenizer = SimpleNamespace(
        convert_tokens_to_ids=lambda _: 4,
        eos_token_id=2,
        all_special_ids=[1, 2, 3] if metadata_case != "complete" else [1, 2, 3, 4],
        added_tokens_decoder=added,
        convert_ids_to_tokens=lambda _: "</think>",
        encode=lambda *a, **kw: [4],
    )
    if metadata_case == "missing":
        with pytest.raises(ValueError, match="standalone"):
            thinking_close_id(tokenizer)
    else:
        assert thinking_close_id(tokenizer) == 4


def test_three_shot_prompt_and_strict_scoring(task):
    task.build_all_requests(limit=1, rank=0, world_size=1)
    instance = task.instances[0]
    assert instance.args[0].count("Q:") == 4  # three demonstrations plus the question
    row = {"doc": instance.doc}
    assert score(task, row, "6 * 3 = 18. The answer is 18.")["correct"]
    # A bare number must not silently pass the harness's strict extraction rule.
    assert score(task, row, "18")["unparseable"]
    assert not score(task, row, "The answer is 19.")["correct"]


def test_sample_is_fixed_and_does_not_use_the_first_hundred():
    selected = select_doc_ids(1319, limit=100, seed=0, split="test")
    assert len(selected) == len(set(selected)) == 100
    assert selected == sorted(selected)
    assert selected == select_doc_ids(1319, limit=100, seed=0, split="test")
    assert selected != list(range(100))
    assert selected != select_doc_ids(1319, limit=100, seed=1, split="test")
    assert select_doc_ids(1319, limit=None, seed=0, split="test") == list(range(1319))


@pytest.mark.parametrize("population,limit", [(10, 100), (10, 0), (0, None)])
def test_selection_rejects_missing_coverage(population, limit):
    with pytest.raises(ValueError):
        select_doc_ids(population, limit=limit, seed=0, split="test")


def test_sampled_requests_keep_original_dataset_ids(task):
    tokenizer = SimpleNamespace(encode=lambda prompt, **kwargs: [1, 2, 3])
    records, selection = build_records(
        task, tokenizer, split="test", limit=100, seed=0, max_new_tokens=10, max_length=20
    )
    assert len(records) == 100
    assert [row["id"] for row in records] == selection["ids"]
    for row in records:
        assert row["doc"]["question"] == f"What is 6 times 3? Question #{row['id']}"
        assert row["prompt"].count("Q:") == 4
    with pytest.raises(ValueError, match="truncated"):
        build_records(
            task, tokenizer, split="test", limit=100, seed=0, max_new_tokens=20, max_length=20
        )


def test_default_case_preserves_the_87_observed_questions(task):
    case = json.loads(DEFAULT_CASE.read_text())
    ids = case["source_ids"]
    assert len(ids) == case["baseline"]["examples"] == 87
    assert case["baseline"]["correct"] == 59
    assert ids == select_doc_ids(1319, limit=100, seed=0, split="test")[:87]
    assert ids[-1] == 1136
    assert ids != select_doc_ids(1319, limit=87, seed=0, split="test")
    records, selection = build_records(
        task,
        SimpleNamespace(encode=lambda *a, **kw: [1, 2, 3]),
        split="test",
        limit=None,
        seed=0,
        max_new_tokens=10,
        max_length=20,
        doc_ids=ids,
    )
    assert selection["method"] == "fixed-source-ids-v1"
    assert [row["id"] for row in records] == ids
    for row in records:
        assert row["doc"]["question"] == f"What is 6 times 3? Question #{row['id']}"


@pytest.mark.parametrize("ids", [[], [5, 5], [11, 5], [-1], [1319]])
def test_invalid_fixed_questions_rejected(task, ids):
    with pytest.raises(ValueError, match="Question IDs"):
        build_records(
            task,
            None,
            split="test",
            limit=None,
            seed=0,
            max_new_tokens=10,
            max_length=20,
            doc_ids=ids,
        )


def test_baseline_fingerprint_tracks_recipe_not_checkout():
    protocol = {
        "model_revision": "pinned",
        "model_files": {"weights": "hash"},
        "task_config": {"metric": "strict"},
        "records": [{"id": 5, "prompt_ids": [1, 2, 3]}],
        "max_new_tokens": 1024,
        "max_length": 2048,
        "dtype": "bfloat16",
        "loops": 4,
        "batch_size": 1,
        "add_special_tokens": False,
        "apply_chat_template": False,
        "source": {"sha": "old", "packages": {"transformers": "4.55.0"}},
    }
    expected = baseline_fingerprint(protocol)
    raw = {**protocol, "prompt_mode": "raw", "scorer": SCORER_LM_EVAL}
    assert baseline_fingerprint(raw) == expected
    assert (
        validate_prompt_recipe(protocol) == validate_prompt_recipe(raw) == ("raw", SCORER_LM_EVAL)
    )
    thinking = dict(
        raw,
        prompt_mode=PROMPT_THINKING,
        scorer=SCORER_THINKING,
        apply_chat_template=True,
        add_special_tokens=True,
    )
    assert baseline_fingerprint(thinking) != expected
    assert validate_prompt_recipe(thinking) == (PROMPT_THINKING, SCORER_THINKING)
    protocol["source"]["sha"] = "new"
    protocol["model"] = "/another/checkpoint/path"
    assert baseline_fingerprint(protocol) == expected
    changed_prompt = deepcopy(protocol)
    changed_prompt["records"][0]["prompt_ids"].append(4)
    changed_length = {**protocol, "max_new_tokens": 512}
    changed_packages = deepcopy(protocol)
    changed_packages["source"]["packages"]["transformers"] = "different"
    changed_depth = {**protocol, "loops": 6}
    for changed in (changed_prompt, changed_length, changed_packages, changed_depth):
        assert baseline_fingerprint(changed) != expected


def baseline_fixture(correct=59):
    case = json.loads(DEFAULT_CASE.read_text())
    records = [{"id": i, "prompt_ids": [i]} for i in case["source_ids"]]
    protocol = {
        "baseline": case["baseline"],
        "records": records,
        "max_regression_pp": 1.0,
        "min_reference_accuracy_pct": None,
    }
    rows = [
        {"id": row["id"], "prompt_sha256": digest(row["prompt_ids"]), "correct": n < correct}
        for n, row in enumerate(records)
    ]
    return protocol, rows


@pytest.mark.parametrize("correct,passes", [(59, True), (58, False), (60, True)])
def test_stored_hf_baseline_gate(correct, passes):
    protocol, rows = baseline_fixture(correct)
    result = baseline_result(protocol, rows)
    assert result["reference_accuracy_pct"] == pytest.approx(100 * 59 / 87)
    assert result["delta_pp"] == pytest.approx(100 * (correct - 59) / 87)
    assert result["passes_observed_accuracy_gate"] is passes


@pytest.mark.parametrize("change", ["partial", "wrong_id", "wrong_prompt", "floor"])
def test_stored_baseline_rejects_invalid_run(change):
    protocol, rows = baseline_fixture()
    if change == "floor":
        protocol["min_reference_accuracy_pct"] = 70
        assert not baseline_result(protocol, rows)["passes_observed_accuracy_gate"]
        return
    if change == "partial":
        rows.pop()
    elif change == "wrong_id":
        rows[0]["id"] = 0
    else:
        rows[0]["prompt_sha256"] = "different"
    with pytest.raises(ValueError):
        baseline_result(protocol, rows)


@pytest.mark.parametrize("correct,exit_code", [(59, 0), (58, 1)])
def test_native_cli_enforces_stored_baseline(monkeypatch, correct, exit_code):
    protocol, rows = baseline_fixture(correct)
    monkeypatch.setattr(
        gsm8k, "run", lambda args: {"baseline_comparison": baseline_result(protocol, rows)}
    )
    monkeypatch.setattr(
        sys, "argv", ["gsm8k", "run", "--backend", "native", "--protocol", "p", "--output", "o"]
    )
    if exit_code:
        with pytest.raises(SystemExit) as exc:
            gsm8k.main()
        assert exc.value.code == exit_code
    else:
        gsm8k.main()


def comparison_fixture(tmp_path, native_correct=True, examples=1):
    for backend, correct in (("transformers", True), ("native", native_correct)):
        path = tmp_path / backend
        path.mkdir()
        row = {
            "id": 0,
            "prompt_sha256": "same",
            "correct": correct,
            "answer": "18" if correct else "19",
        }
        (path / "samples.jsonl").write_text(
            "".join(json.dumps(dict(row, id=i)) + "\n" for i in range(examples))
        )
        summary = {
            "backend": backend,
            "protocol_sha256": "same",
            "expected_examples": examples,
            "split": "test",
            "max_regression_pp": 1.0,
            "min_reference_accuracy_pct": 75.92,
            "complete": True,
            "examples": examples,
            "correct": examples * int(correct),
            "accuracy": float(correct),
            "samples_sha256": file_digest(path / "samples.jsonl"),
        }
        (path / "summary.json").write_text(json.dumps(summary))
    return SimpleNamespace(
        transformers=tmp_path / "transformers",
        native=tmp_path / "native",
        output=tmp_path / "comparison.json",
    )


def test_accuracy_regression_fails(tmp_path):
    result = compare(comparison_fixture(tmp_path, native_correct=False))
    assert result["delta_pp"] == -100
    assert result["reference_correct_native_wrong"] == 1
    assert not result["passes_observed_accuracy_gate"]


def test_measured_transformers_score_is_the_default_baseline(tmp_path):
    args = comparison_fixture(tmp_path, native_correct=False)
    for backend in ("transformers", "native"):
        folder = tmp_path / backend
        samples = folder / "samples.jsonl"
        row = json.loads(samples.read_text())
        row.update(correct=False, answer="19")
        samples.write_text(json.dumps(row) + "\n")
        path = folder / "summary.json"
        summary = json.loads(path.read_text())
        summary.update(
            correct=0,
            accuracy=0.0,
            min_reference_accuracy_pct=None,
            samples_sha256=file_digest(samples),
        )
        path.write_text(json.dumps(summary))
    result = compare(args)
    assert result["transformers_correct"] == result["native_correct"] == 0
    assert result["delta_pp"] == 0
    assert result["passes_observed_accuracy_gate"]


@pytest.mark.parametrize("change", ["different_protocol", "partial", "wrong_score", "duplicate"])
def test_invalid_comparisons_rejected(tmp_path, change):
    args = comparison_fixture(tmp_path, examples=2 if change == "duplicate" else 1)
    path = args.native / "summary.json"
    summary = json.loads(path.read_text())
    if change == "different_protocol":
        summary["protocol_sha256"] = "different"
    elif change == "partial":
        summary["complete"] = False
    elif change == "wrong_score":
        summary["accuracy"] = 0.0
    else:
        rows = args.native / "samples.jsonl"
        records = [json.loads(line) for line in rows.read_text().splitlines()]
        records[1]["id"] = records[0]["id"]
        rows.write_text("".join(json.dumps(row) + "\n" for row in records))
        summary.update(samples_sha256=file_digest(rows))
    path.write_text(json.dumps(summary))
    message = "Missing or duplicate examples" if change == "duplicate" else None
    with pytest.raises(ValueError, match=message):
        compare(args)


@pytest.mark.parametrize("loops", [4, 6])
def test_default_exit_is_the_fixed_recipe(loops):
    fixed = {"mode": "ouro", "threshold": 1.0, "min_loops": loops, "max_loops": loops}
    for backend in ("native", "transformers"):
        assert exit_settings(backend, loops) == {**fixed, "async_scheduling": False}
        assert fixed_exit(loops) == exit_settings(backend, loops)


@pytest.mark.parametrize("loops", [4, 6])
@pytest.mark.parametrize(
    "backend,options",
    [
        ("native", dict(mode="ouro", threshold=0.5, min_loops=2)),
        ("native", dict(mode="ouro_delayed", threshold=0.2, min_loops=2, async_scheduling=True)),
        ("transformers", dict(mode="ouro", threshold=0.9, min_loops=1)),
    ],
)
def test_adaptive_exit_settings(backend, options, loops):
    settings = exit_settings(backend, loops, **options)
    assert settings["max_loops"] == loops
    assert settings["threshold"] == options["threshold"]
    assert settings["min_loops"] == options["min_loops"]
    assert settings["async_scheduling"] == options.get("async_scheduling", False)
    assert settings != fixed_exit(loops)


def test_min_loops_is_validated_against_the_checkpoint_depth():
    assert exit_settings("native", 6, threshold=0.5, min_loops=6)["min_loops"] == 6
    with pytest.raises(ValueError, match="between 1 and 6"):
        exit_settings("native", 6, threshold=0.5, min_loops=7)


@pytest.mark.parametrize(
    "config,loops",
    [
        ({"model_type": "ouro", "total_ut_steps": 4}, 4),
        ({"model_type": "ouro", "total_ut_steps": 6}, 6),
        ({"model_type": "llama", "total_ut_steps": 4}, None),
        ({"architectures": ["RavenForCausalLM"], "mean_recurrence": 32}, 32),
        ({"model_type": "ouro"}, None),
        ({"model_type": "ouro", "total_ut_steps": 0}, None),
        ({"model_type": "ouro", "total_ut_steps": 4.0}, None),
    ],
)
def test_checkpoint_loops_come_from_config(config, loops):
    if loops is None:
        with pytest.raises(ValueError, match="total_ut_steps"):
            checkpoint_loops(config)
    else:
        assert checkpoint_loops(config) == loops


def test_run_refuses_the_stored_baseline_at_another_depth(tmp_path, monkeypatch):
    pytest.importorskip("lm_eval")
    case = json.loads(DEFAULT_CASE.read_text())
    protocol = {
        "model_revision": "pinned",
        "model_files": {},
        "task_config": {},
        "records": [],
        "max_new_tokens": 1024,
        "max_length": 2048,
        "dtype": "bfloat16",
        "loops": 4,
        "batch_size": 1,
        "add_special_tokens": False,
        "apply_chat_template": False,
        "source": {"packages": case["baseline"]["packages"]},
    }
    baseline = {**case["baseline"], "protocol_fingerprint": baseline_fingerprint(protocol)}

    class PassedTheGuard(Exception):
        pass

    def stop():
        raise PassedTheGuard

    # The package check follows the baseline guard; stop there instead of loading a model.
    monkeypatch.setattr(gsm8k, "source", stop)
    for loops in (4, 6):
        path = tmp_path / f"protocol-{loops}.json"
        path.write_text(json.dumps({**protocol, "loops": loops, "baseline": baseline}))
        args = SimpleNamespace(
            protocol=str(path),
            backend="native",
            exit_mode="ouro",
            exit_threshold=1.0,
            min_loops=None,
            async_scheduling=False,
        )
        if loops == 4:
            with pytest.raises(PassedTheGuard):
                gsm8k.run(args)
        else:
            # Another depth changes the fingerprint, so the four-loop baseline cannot apply.
            with pytest.raises(ValueError, match="changed after its HF baseline was frozen"):
                gsm8k.run(args)


@pytest.mark.parametrize(
    "backend,options,message",
    [
        ("native", dict(threshold=1.5), "\\[0, 1\\]"),
        ("native", dict(threshold=0.5), "--min-loops"),
        ("native", dict(threshold=0.5, min_loops=5), "--min-loops"),
        # Loop and scheduling options would be silently ignored at fixed depth.
        ("native", dict(min_loops=2), "below 1"),
        ("native", dict(mode="ouro_delayed"), "below 1"),
        ("native", dict(threshold=0.5, min_loops=2, async_scheduling=True), "ouro_delayed"),
        ("transformers", dict(threshold=0.5, min_loops=2), "min-loops 1"),
        ("transformers", dict(mode="ouro_delayed", threshold=0.5, min_loops=1), "min-loops 1"),
    ],
)
def test_invalid_exit_settings_rejected(backend, options, message):
    with pytest.raises(ValueError, match=message):
        exit_settings(backend, 4, **options)


def test_depth_summary_excludes_the_prefill_token():
    rows = [{"exit_depths": [4, 2, 3]}, {"exit_depths": [4, 4]}, {"exit_depths": [4]}]
    summary = depth_summary(rows)
    assert summary["decode_tokens"] == 3
    assert summary["mean_decode_depth"] == pytest.approx(3.0)
    assert summary["decode_depth_histogram"] == {"2": 1, "3": 1, "4": 1}
    assert summary["decode_loops_per_question_mean"] == pytest.approx(3.0)
    assert depth_summary([{"token_ids": [1]}]) is None


def test_compare_reports_exit_policies_between_native_runs(tmp_path):
    args = comparison_fixture(tmp_path)
    adaptive = exit_settings("native", 4, mode="ouro", threshold=0.5, min_loops=2)
    depth = depth_summary([{"exit_depths": [4, 2]}])
    for backend, extra in (
        ("transformers", {"backend": "native"}),
        ("native", {"exit": adaptive, "depth": depth}),
    ):
        path = tmp_path / backend / "summary.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), **extra}))
    result = compare(args)
    # A summary without exit settings predates them and used the fixed four-loop recipe.
    assert result["reference"] == {"backend": "native", "exit": fixed_exit(4), "depth": None}
    assert result["candidate"] == {"backend": "native", "exit": adaptive, "depth": depth}
    assert (result["reference_correct"], result["candidate_correct"]) == (1, 1)
    # Backend-named keys would mislabel a native-vs-native comparison, so they are omitted.
    assert not any(key.startswith(("transformers_", "native_")) for key in result)
    assert "reference_correct_native_wrong" not in result


def test_compare_keeps_backend_named_keys_for_hf_vs_native(tmp_path):
    result = compare(comparison_fixture(tmp_path, native_correct=False))
    assert result["reference_correct_candidate_wrong"] == 1
    assert result["transformers_correct"] == result["reference_correct"] == 1
    assert result["native_correct"] == result["candidate_correct"] == 0
    assert result["native_accuracy_pct"] == result["candidate_accuracy_pct"] == 0


def rows(id_, tokens, depths=None):
    row = {"id": id_, "token_ids": tokens}
    if depths is not None:
        row["exit_depths"] = depths
    return row


def test_sequence_differences_locate_first_divergence():
    pairs = [
        # identical
        (rows(0, [1, 2, 3], [4, 2, 2]), rows(0, [1, 2, 3], [4, 2, 2])),
        # same tokens, exit decision differs at 2 on identical context
        (rows(1, [1, 2, 3], [4, 2, 2]), rows(1, [1, 2, 3], [4, 2, 3])),
        # tokens diverge at 1; the later depth difference follows from it
        (rows(2, [1, 5, 6], [4, 2, 2]), rows(2, [1, 2, 3], [4, 2, 3])),
        # depth differs at 1 while tokens still agree there: identical context
        (rows(3, [1, 2, 9], [4, 3, 2]), rows(3, [1, 2, 3], [4, 2, 2])),
        # one output is a prefix of the other: a length difference, not an exit decision
        (rows(4, [1, 2], [4, 2]), rows(4, [1, 2, 3], [4, 2, 2])),
        # exit decision differs at 2 on identical context and changes token 2 itself
        (rows(5, [1, 2, 3], [4, 2, 2]), rows(5, [1, 2, 9], [4, 2, 3])),
    ]
    result = sequence_differences(pairs)
    tokens, depths = result["token_ids"], result["exit_depths"]
    assert tokens["questions_compared"] == 6
    assert tokens["questions_differing"] == 4
    assert [q["first_divergence"] for q in tokens["questions"]] == [1, 2, 2, 2]
    assert (tokens["earliest_first_divergence"], tokens["median_first_divergence"]) == (1, 2)
    assert depths["questions_differing"] == 5
    assert {q["id"]: q["identical_context"] for q in depths["questions"]} == {
        1: True,
        2: False,
        3: True,
        4: False,
        5: True,
    }
    assert depths["questions_differing_on_identical_context"] == 3


def test_sequence_differences_skip_unrecorded_depths():
    # HF fixed-depth rows record no exit depths.
    result = sequence_differences([(rows(0, [1, 2]), rows(0, [1, 3], [4, 2]))])
    assert result["token_ids"]["questions_differing"] == 1
    assert result["exit_depths"]["questions_compared"] == 0
    assert result["exit_depths"]["earliest_first_divergence"] is None


def test_compare_reports_sequences_without_changing_the_gate(tmp_path):
    args = comparison_fixture(tmp_path, examples=2)
    for backend, depths in (("transformers", [[4, 2], [4, 2]]), ("native", [[4, 2], [4, 3]])):
        path = tmp_path / backend
        samples = [json.loads(line) for line in (path / "samples.jsonl").read_text().splitlines()]
        for sample, sample_depths in zip(samples, depths):
            sample.update(token_ids=[7, 8], exit_depths=sample_depths)
        (path / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in samples))
        summary = json.loads((path / "summary.json").read_text())
        summary.update(backend="native", samples_sha256=file_digest(path / "samples.jsonl"))
        (path / "summary.json").write_text(json.dumps(summary))
    result = compare(args)
    differences = result["sequence_differences"]
    assert differences["token_ids"]["questions_differing"] == 0
    assert differences["exit_depths"]["questions_differing_on_identical_context"] == 1
    assert result["answer_disagreements"] == []
    assert result["passes_observed_accuracy_gate"]
