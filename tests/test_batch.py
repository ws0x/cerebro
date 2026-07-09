from cerebro.batch import BatchItem, run_batch
from cerebro.ir import NodeType
from cerebro.structure import HeuristicStructurer


def test_batch_merges_successful_items_and_survives_failures(tmp_path):
    f1 = tmp_path / "a.txt"
    f1.write_text("Hello world. This is lesson one about topic A. It has detail.")
    f2 = tmp_path / "b.txt"
    f2.write_text("Hello again. This is lesson two about topic B. It has detail too.")

    items = [
        BatchItem("Lesson A", str(f1)),
        BatchItem("Lesson B", str(f2)),
        BatchItem("Missing", str(tmp_path / "missing.txt")),
    ]

    combined, outcomes = run_batch(
        items, lambda: HeuristicStructurer(), level="full", title="Course", max_workers=2
    )

    assert combined.root.title == "Course"
    assert combined.root.type == NodeType.root
    # Only the two successful items become branches; the missing file is skipped, not fatal.
    assert len(combined.root.children) == 2
    assert {c.title for c in combined.root.children} == {"Lesson A", "Lesson B"}
    for child in combined.root.children:
        assert child.type == NodeType.topic  # demoted from root when merged

    errors = [o for o in outcomes if o.error]
    assert len(errors) == 1
    assert errors[0].label == "Missing"


def test_batch_all_fail_yields_placeholder_not_crash(tmp_path):
    items = [BatchItem("Missing", str(tmp_path / "nope.txt"))]
    combined, outcomes = run_batch(items, lambda: HeuristicStructurer(), level="full", title="Course")
    assert combined.root.children  # placeholder node present
    assert all(o.error for o in outcomes)
