from cerebro.batch import BatchItem, forget_batch_snapshot, list_batch_snapshots, run_batch
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

    combined, outcomes, diff = run_batch(
        items, lambda: HeuristicStructurer(), level="full", title="Course", max_workers=2
    )
    assert diff is None  # no batch_source given -> nothing to diff against

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
    combined, outcomes, diff = run_batch(items, lambda: HeuristicStructurer(), level="full", title="Course")
    assert diff is None
    assert combined.root.children  # placeholder node present
    assert all(o.error for o in outcomes)


# --- incremental reruns ----------------------------------------------------


def _lesson_files(tmp_path, names_and_text):
    items = []
    for name, text in names_and_text:
        f = tmp_path / f"{name}.txt"
        f.write_text(text, encoding="utf-8")
        items.append(BatchItem(name, str(f)))
    return items


def test_rerun_with_no_changes_reuses_every_item(tmp_path, tmp_path_factory):
    items = _lesson_files(tmp_path, [("A", "Topic A content here."), ("B", "Topic B content here.")])
    snap_dir = tmp_path_factory.mktemp("snap")

    combined1, outcomes1, diff1 = run_batch(
        items, lambda: HeuristicStructurer(), level="full", title="Course",
        batch_source="course://demo", snapshot_dir=snap_dir,
    )
    assert diff1 is None  # first run

    combined2, outcomes2, diff2 = run_batch(
        items, lambda: HeuristicStructurer(), level="full", title="Course",
        batch_source="course://demo", snapshot_dir=snap_dir,
    )
    assert diff2 is not None
    assert set(diff2.reused) == {"A", "B"}
    assert not diff2.added
    assert not diff2.removed
    assert {c.title for c in combined1.root.children} == {c.title for c in combined2.root.children}


def test_new_item_is_processed_fresh_existing_items_reused(tmp_path, tmp_path_factory):
    items = _lesson_files(tmp_path, [("A", "Topic A content here."), ("B", "Topic B content here.")])
    snap_dir = tmp_path_factory.mktemp("snap")

    run_batch(
        items, lambda: HeuristicStructurer(), level="full", title="Course",
        batch_source="course://demo", snapshot_dir=snap_dir,
    )

    items_with_new = items + _lesson_files(tmp_path, [("C", "Topic C content here.")])
    combined2, outcomes2, diff2 = run_batch(
        items_with_new, lambda: HeuristicStructurer(), level="full", title="Course",
        batch_source="course://demo", snapshot_dir=snap_dir,
    )
    assert set(diff2.reused) == {"A", "B"}
    assert diff2.added == ["C"]
    assert not diff2.removed
    assert {c.title for c in combined2.root.children} == {"A", "B", "C"}


def test_removed_item_is_reported_and_dropped(tmp_path, tmp_path_factory):
    items = _lesson_files(tmp_path, [("A", "Topic A content here."), ("B", "Topic B content here.")])
    snap_dir = tmp_path_factory.mktemp("snap")

    run_batch(
        items, lambda: HeuristicStructurer(), level="full", title="Course",
        batch_source="course://demo", snapshot_dir=snap_dir,
    )

    combined2, outcomes2, diff2 = run_batch(
        items[:1], lambda: HeuristicStructurer(), level="full", title="Course",
        batch_source="course://demo", snapshot_dir=snap_dir,
    )
    assert diff2.removed == ["B"]
    assert {c.title for c in combined2.root.children} == {"A"}


def test_fresh_ignores_history_but_still_saves_a_new_snapshot(tmp_path, tmp_path_factory):
    items = _lesson_files(tmp_path, [("A", "Topic A content here.")])
    snap_dir = tmp_path_factory.mktemp("snap")

    run_batch(
        items, lambda: HeuristicStructurer(), level="full", title="Course",
        batch_source="course://demo", snapshot_dir=snap_dir,
    )
    combined2, outcomes2, diff2 = run_batch(
        items, lambda: HeuristicStructurer(), level="full", title="Course",
        batch_source="course://demo", snapshot_dir=snap_dir, incremental=False,
    )
    assert diff2 is None  # --fresh: no diff reported despite history existing

    combined3, outcomes3, diff3 = run_batch(
        items, lambda: HeuristicStructurer(), level="full", title="Course",
        batch_source="course://demo", snapshot_dir=snap_dir,
    )
    assert diff3 is not None
    assert diff3.reused == ["A"]  # the --fresh run's snapshot is reusable again


def test_different_level_does_not_reuse_an_incompatible_snapshot(tmp_path, tmp_path_factory):
    items = _lesson_files(tmp_path, [("A", "Topic A content here.")])
    snap_dir = tmp_path_factory.mktemp("snap")

    run_batch(
        items, lambda: HeuristicStructurer(), level="brief", title="Course",
        batch_source="course://demo", snapshot_dir=snap_dir,
    )
    combined2, outcomes2, diff2 = run_batch(
        items, lambda: HeuristicStructurer(), level="expert", title="Course",
        batch_source="course://demo", snapshot_dir=snap_dir,
    )
    assert diff2 is None


def test_no_batch_source_never_saves_or_reuses(tmp_path, tmp_path_factory):
    items = _lesson_files(tmp_path, [("A", "Topic A content here.")])
    snap_dir = tmp_path_factory.mktemp("snap")

    run_batch(
        items, lambda: HeuristicStructurer(), level="full", title="Course", snapshot_dir=snap_dir,
    )
    combined2, outcomes2, diff2 = run_batch(
        items, lambda: HeuristicStructurer(), level="full", title="Course", snapshot_dir=snap_dir,
    )
    assert diff2 is None
    assert list(snap_dir.glob("*.json")) == []


# --- forget -----------------------------------------------------------------


def test_forget_deletes_an_existing_snapshot_and_reports_true(tmp_path, tmp_path_factory):
    items = _lesson_files(tmp_path, [("A", "Topic A content here.")])
    snap_dir = tmp_path_factory.mktemp("snap")
    run_batch(items, lambda: HeuristicStructurer(), level="full", title="Course",
              batch_source="course://demo", snapshot_dir=snap_dir)

    assert list(snap_dir.glob("*.json"))  # sanity: a snapshot exists
    assert forget_batch_snapshot("course://demo", snapshot_dir=snap_dir) is True
    assert not list(snap_dir.glob("*.json"))


def test_forget_nonexistent_snapshot_reports_false(tmp_path_factory):
    snap_dir = tmp_path_factory.mktemp("snap")
    assert forget_batch_snapshot("course://never-run", snapshot_dir=snap_dir) is False


def test_forgotten_batch_reprocesses_everything_next_run(tmp_path, tmp_path_factory):
    items = _lesson_files(tmp_path, [("A", "Topic A content here.")])
    snap_dir = tmp_path_factory.mktemp("snap")
    run_batch(items, lambda: HeuristicStructurer(), level="full", title="Course",
              batch_source="course://demo", snapshot_dir=snap_dir)
    forget_batch_snapshot("course://demo", snapshot_dir=snap_dir)

    _combined, _outcomes, diff = run_batch(
        items, lambda: HeuristicStructurer(), level="full", title="Course",
        batch_source="course://demo", snapshot_dir=snap_dir,
    )
    assert diff is None  # no history to diff against, same as a true first run


# --- status / list_batch_snapshots ------------------------------------------


def test_list_batch_snapshots_empty_dir_returns_empty_list(tmp_path_factory):
    snap_dir = tmp_path_factory.mktemp("snap")
    assert list_batch_snapshots(snapshot_dir=snap_dir) == []


def test_list_batch_snapshots_reports_source_and_item_count(tmp_path, tmp_path_factory):
    items = _lesson_files(tmp_path, [("A", "Topic A content here."), ("B", "Topic B content here.")])
    snap_dir = tmp_path_factory.mktemp("snap")
    run_batch(items, lambda: HeuristicStructurer(), level="full", title="Course",
              batch_source="course://demo", snapshot_dir=snap_dir)

    snaps = list_batch_snapshots(snapshot_dir=snap_dir)
    assert len(snaps) == 1
    assert snaps[0]["source"] == "course://demo"
    assert snaps[0]["items"] == 2
    assert snaps[0]["built_at"] != "?"
