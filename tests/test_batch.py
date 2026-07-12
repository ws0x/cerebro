from pathlib import Path

import fitz
import pytest

import cerebro.batch as batch_module
from cerebro.batch import (
    BatchItem,
    _local_file_fingerprint,
    dry_run_batch,
    forget_batch_snapshot,
    list_batch_snapshots,
    run_batch,
)
from cerebro.cache import Cache
from cerebro.ir import NodeType
from cerebro.structure import HeuristicStructurer
from cerebro.structure.document import OutlineAwareStructurer


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


def _build_pdf_with_toc(tmp_path):
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i} body text about chapter content.")
    doc.set_toc([[1, "Chapter One", 1], [1, "Chapter Two", 2]])
    path = tmp_path / "slides.pdf"
    doc.save(str(path))
    doc.close()
    return path


def test_batch_checkpoints_the_snapshot_after_every_item_not_just_at_the_end(tmp_path, monkeypatch):
    # Regression test: a killed/crashed run (Ctrl+C, network drop) partway
    # through a long batch must not lose already-completed items just
    # because the snapshot used to only get saved once, after everything
    # finished. max_workers=1 makes completion order deterministic (item
    # order) so the assertions below aren't racy.
    for i in range(3):
        f = tmp_path / f"lesson{i}.txt"
        f.write_text(f"Lesson {i} covers a distinct topic with enough words to structure.")
    items = [BatchItem(f"Lesson {i}", str(tmp_path / f"lesson{i}.txt")) for i in range(3)]

    snapshot_calls: list[dict] = []
    real_save = batch_module._save_batch_snapshot

    def spy_save(batch_source, params, items_data, snapshot_dir):
        snapshot_calls.append(dict(items_data))  # shallow copy at call time
        return real_save(batch_source, params, items_data, snapshot_dir)

    monkeypatch.setattr("cerebro.batch._save_batch_snapshot", spy_save)

    run_batch(
        items,
        lambda: HeuristicStructurer(),
        level="full",
        title="Course",
        max_workers=1,
        batch_source="test://checkpoint-batch",
        snapshot_dir=tmp_path / "snapshots",
    )

    # One checkpoint per successfully completed item (plus a final redundant
    # save after the loop, for consistency) -- not just one at the very end.
    assert len(snapshot_calls) == 4
    assert len(snapshot_calls[0]) == 1  # after item 1: only item 1 saved so far
    assert len(snapshot_calls[1]) == 2  # after item 2: items 1+2 saved so far
    assert len(snapshot_calls[2]) == 3  # after item 3: all three saved
    assert len(snapshot_calls[3]) == 3  # final save: unchanged, all three


def test_batch_snapshot_write_failure_does_not_corrupt_the_previous_snapshot(tmp_path, tmp_path_factory, monkeypatch):
    items = _lesson_files(tmp_path, [("A", "Topic A content here.")])
    snap_dir = tmp_path_factory.mktemp("snap")

    run_batch(
        items, lambda: HeuristicStructurer(), level="full", title="Course",
        batch_source="course://demo", snapshot_dir=snap_dir,
    )
    snapshot_path = batch_module._batch_snapshot_path("course://demo", snap_dir)
    original_content = snapshot_path.read_text(encoding="utf-8")

    def boom(path, write_fn):
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr("cerebro.batch.atomic_write", boom)

    with pytest.raises(RuntimeError, match="simulated crash"):
        run_batch(
            items, lambda: HeuristicStructurer(), level="full", title="Course",
            batch_source="course://demo", snapshot_dir=snap_dir,
        )

    # The previous good snapshot survives untouched -- proves the snapshot
    # save actually routes through atomic_write, not a plain write_text.
    assert snapshot_path.read_text(encoding="utf-8") == original_content


def test_batch_interrupted_mid_run_still_leaves_completed_items_checkpointed(tmp_path):
    # The realistic version of the checkpoint test above: simulate an actual
    # Ctrl+C (KeyboardInterrupt, which _process()'s `except Exception` does
    # NOT swallow -- correctly, since that's what lets a genuine interrupt
    # still abort the run) partway through a batch, then confirm the items
    # that finished first are already on disk even though run_batch itself
    # never returns normally.
    for i in range(3):
        f = tmp_path / f"lesson{i}.txt"
        content = "CRASH_HERE marker text" if i == 2 else f"Lesson {i} covers a distinct topic with enough words to structure."
        f.write_text(content)
    items = [BatchItem(f"Lesson {i}", str(tmp_path / f"lesson{i}.txt")) for i in range(3)]

    class _CrashOnMarker:
        def structure(self, transcript, level="full"):
            if any("CRASH_HERE" in s.text for s in transcript.segments):
                raise KeyboardInterrupt()
            return HeuristicStructurer().structure(transcript, level=level)

    snapshot_dir = tmp_path / "snapshots"
    raised = False
    try:
        run_batch(
            items, lambda: _CrashOnMarker(), level="full", title="Course",
            max_workers=1, batch_source="test://crash-batch", snapshot_dir=snapshot_dir,
        )
    except KeyboardInterrupt:
        raised = True
    assert raised, "expected the KeyboardInterrupt to propagate out of run_batch, matching real Ctrl+C"

    # Even though run_batch never returned, the two items that finished
    # before the interrupt must already be checkpointed to disk.
    reused, new = dry_run_batch(items, "full", "test://crash-batch", snapshot_dir=snapshot_dir)
    assert set(reused) == {"Lesson 0", "Lesson 1"}


def test_batch_item_with_a_real_pdf_outline_keeps_its_hierarchy_not_flattened(tmp_path):
    # Regression test: course folders can mix PDFs in with videos/subtitles
    # (see ingest/folder.py). A PDF with a real TOC must keep that structure
    # inside its batch branch, not get flattened through the same
    # video-oriented reduce-from-flat-text path used for everything else.
    pdf = _build_pdf_with_toc(tmp_path)
    txt = tmp_path / "notes.txt"
    txt.write_text("Plain lesson notes with no real structure at all here.")

    items = [BatchItem("Slides", str(pdf)), BatchItem("Notes", str(txt))]
    combined, outcomes, _diff = run_batch(
        items,
        lambda: OutlineAwareStructurer(None, Cache(enabled=False)),
        level="full",
        title="Course",
        max_workers=2,
    )

    assert all(o.error is None for o in outcomes)
    slides_branch = next(c for c in combined.root.children if c.title == "Slides")
    assert [c.title for c in slides_branch.children] == ["Chapter One", "Chapter Two"]


def test_batch_pdf_item_with_total_ai_failure_is_reported_as_a_real_failure(tmp_path):
    # Found via a real Groq-vs-Gemini comparison: OutlineAwareStructurer routes
    # a PDF/article item through build_outline_map, which used to silently
    # return a 100%-fallback map when every leaf's LLM call failed -- counted
    # as a genuine success in the batch summary, unlike a video item hitting
    # the same total failure (which run_batch already reports honestly via
    # its existing per-item exception handling). build_outline_map now raises
    # on total failure, so this item must show up as a real, named failure
    # exactly like the video path already does, not silently succeed.
    from cerebro.llm.base import LLMError
    from cerebro.llm.providers import MockProvider

    class AlwaysFailsProvider(MockProvider):
        def complete_json(self, system, user):
            raise LLMError("boom")

    pdf = _build_pdf_with_toc(tmp_path)
    items = [BatchItem("Slides", str(pdf))]
    combined, outcomes, _diff = run_batch(
        items,
        lambda: OutlineAwareStructurer(AlwaysFailsProvider(), Cache(enabled=False)),
        level="full",
        title="Course",
    )

    assert outcomes[0].error is not None
    assert "All section-enrichment calls failed" in outcomes[0].error
    # no fake-success "Slides" branch was merged in -- only the standard
    # all-failed placeholder, same as any other fully-failed batch item
    assert all(c.title != "Slides" for c in combined.root.children)


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


# --- dry_run_batch -----------------------------------------------------------


def test_dry_run_first_ever_run_reports_everything_new(tmp_path, tmp_path_factory):
    items = _lesson_files(tmp_path, [("A", "..."), ("B", "...")])
    snap_dir = tmp_path_factory.mktemp("snap")
    reused, new = dry_run_batch(items, "full", "course://demo", snapshot_dir=snap_dir)
    assert reused == []
    assert set(new) == {"A", "B"}


def test_dry_run_after_a_real_run_reports_reused_and_new_correctly(tmp_path, tmp_path_factory):
    items = _lesson_files(tmp_path, [("A", "..."), ("B", "...")])
    snap_dir = tmp_path_factory.mktemp("snap")
    run_batch(items, lambda: HeuristicStructurer(), level="full", title="Course",
              batch_source="course://demo", snapshot_dir=snap_dir)

    items_with_new = items + _lesson_files(tmp_path, [("C", "...")])
    reused, new = dry_run_batch(items_with_new, "full", "course://demo", snapshot_dir=snap_dir)
    assert set(reused) == {"A", "B"}
    assert new == ["C"]


def test_dry_run_does_not_touch_the_snapshot_file(tmp_path, tmp_path_factory):
    items = _lesson_files(tmp_path, [("A", "...")])
    snap_dir = tmp_path_factory.mktemp("snap")
    dry_run_batch(items, "full", "course://demo", snapshot_dir=snap_dir)
    assert list(snap_dir.glob("*.json")) == []  # no snapshot created just from a dry run


def test_dry_run_with_no_batch_source_reports_everything_new(tmp_path, tmp_path_factory):
    items = _lesson_files(tmp_path, [("A", "...")])
    snap_dir = tmp_path_factory.mktemp("snap")
    reused, new = dry_run_batch(items, "full", None, snapshot_dir=snap_dir)
    assert reused == []
    assert new == ["A"]


def test_local_file_fingerprint_none_for_a_url_or_missing_file(tmp_path):
    assert _local_file_fingerprint("https://www.youtube.com/watch?v=abc123") is None
    assert _local_file_fingerprint(str(tmp_path / "does-not-exist.txt")) is None


def test_local_file_fingerprint_reflects_size_and_mtime(tmp_path):
    f = tmp_path / "lesson.txt"
    f.write_text("original", encoding="utf-8")
    fp1 = _local_file_fingerprint(str(f))
    assert fp1 is not None

    f.write_text("a much longer replacement body", encoding="utf-8")
    fp2 = _local_file_fingerprint(str(f))
    assert fp2 != fp1


def test_edited_local_file_is_reprocessed_not_silently_reused(tmp_path, tmp_path_factory):
    items = _lesson_files(tmp_path, [("A", "Topic A content here.")])
    snap_dir = tmp_path_factory.mktemp("snap")

    run_batch(
        items, lambda: HeuristicStructurer(), level="full", title="Course",
        batch_source="course://demo", snapshot_dir=snap_dir,
    )

    # Same path, genuinely different (and differently-sized) content -- the
    # "lesson file re-recorded/edited in place" scenario a URL-only
    # source-string reuse check would silently miss.
    Path(items[0].source).write_text(
        "A completely rewritten lesson, much longer than the original body.", encoding="utf-8"
    )

    combined2, outcomes2, diff2 = run_batch(
        items, lambda: HeuristicStructurer(), level="full", title="Course",
        batch_source="course://demo", snapshot_dir=snap_dir,
    )
    assert diff2.added == ["A"]
    assert not diff2.reused


def test_dry_run_reports_an_edited_local_file_as_new_not_reused(tmp_path, tmp_path_factory):
    items = _lesson_files(tmp_path, [("A", "Topic A content here.")])
    snap_dir = tmp_path_factory.mktemp("snap")
    run_batch(items, lambda: HeuristicStructurer(), level="full", title="Course",
              batch_source="course://demo", snapshot_dir=snap_dir)

    Path(items[0].source).write_text(
        "A completely rewritten lesson, much longer than the original body.", encoding="utf-8"
    )

    reused, new = dry_run_batch(items, "full", "course://demo", snapshot_dir=snap_dir)
    assert reused == []
    assert new == ["A"]


def test_unchanged_local_file_is_still_reused_across_reruns(tmp_path, tmp_path_factory):
    # Guards against the fingerprint check itself becoming over-eager and
    # invalidating reuse for files that genuinely didn't change.
    items = _lesson_files(tmp_path, [("A", "Topic A content here.")])
    snap_dir = tmp_path_factory.mktemp("snap")
    run_batch(items, lambda: HeuristicStructurer(), level="full", title="Course",
              batch_source="course://demo", snapshot_dir=snap_dir)

    _, _, diff2 = run_batch(items, lambda: HeuristicStructurer(), level="full", title="Course",
                             batch_source="course://demo", snapshot_dir=snap_dir)
    assert diff2.reused == ["A"]
    assert not diff2.added
