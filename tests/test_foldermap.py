import json
import time

from cerebro.cache import Cache
from cerebro.foldermap import (
    build_folder_map,
    finalize_tree_snapshot,
    forget_tree_snapshot,
    label_folders,
    list_tree_snapshots,
)
from cerebro.ir import NodeType
from cerebro.llm.providers import MockProvider


def _make_project(tmp_path):
    (tmp_path / "src" / "auth").mkdir(parents=True)
    (tmp_path / "src" / "auth" / "login.py").write_text("x", encoding="utf-8")
    (tmp_path / "src" / "auth" / "session.py").write_text("x", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_login.py").write_text("x", encoding="utf-8")
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("x", encoding="utf-8")
    # noise that must be filtered out regardless of .gitignore
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.pyc").write_text("x", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.js").write_text("x", encoding="utf-8")
    return tmp_path


def _map(path, tmp_path_factory, **kwargs):
    """build_folder_map + finalize, with an isolated snapshot dir so tests
    never touch the real ~/.cerebro/tree-snapshots/."""
    snap_dir = tmp_path_factory.mktemp("snap")
    return _map_reusing_snapshot(path, snap_dir, **kwargs)


def _map_reusing_snapshot(path, snapshot_dir, label_with=None, **kwargs):
    mm, diff, nodes, pending = build_folder_map(path, snapshot_dir=snapshot_dir, **kwargs)
    if label_with is not None and nodes:
        label_folders(mm, label_with[0], label_with[1], nodes=nodes)
    finalize_tree_snapshot(pending)
    return mm, diff, nodes


def test_builds_hierarchy_matching_real_structure(tmp_path, tmp_path_factory):
    project = _make_project(tmp_path)
    mm, diff, nodes = _map(project, tmp_path_factory)

    titles = {n.title for n in mm.root.walk()}
    assert "src" in titles
    assert "auth" in titles
    assert "login.py" in titles
    assert "tests" in titles
    assert "test_login.py" in titles
    assert diff is None  # first-ever map of this folder, nothing to diff against


def test_default_ignore_list_filters_noise_dirs_always(tmp_path, tmp_path_factory):
    project = _make_project(tmp_path)
    mm, _, _ = _map(project, tmp_path_factory)

    titles = {n.title for n in mm.root.walk()}
    assert "__pycache__" not in titles
    assert "node_modules" not in titles
    assert "junk.pyc" not in titles
    assert "pkg.js" not in titles


def test_notable_files_get_definition_marker(tmp_path, tmp_path_factory):
    project = _make_project(tmp_path)
    mm, _, _ = _map(project, tmp_path_factory)

    by_title = {n.title: n for n in mm.root.walk()}
    assert by_title["README.md"].type == NodeType.definition
    assert by_title["pyproject.toml"].type == NodeType.definition
    assert by_title["login.py"].type == NodeType.detail


def test_gitignore_is_respected_when_present(tmp_path, tmp_path_factory):
    project = _make_project(tmp_path)
    (project / "secrets").mkdir()
    (project / "secrets" / "key.pem").write_text("x", encoding="utf-8")
    (project / ".gitignore").write_text("secrets/\n", encoding="utf-8")

    mm, _, _ = _map(project, tmp_path_factory, respect_gitignore=True)
    titles = {n.title for n in mm.root.walk()}
    assert "secrets" not in titles
    assert "key.pem" not in titles


def test_gitignore_can_be_disabled(tmp_path, tmp_path_factory):
    project = _make_project(tmp_path)
    (project / "secrets").mkdir()
    (project / "secrets" / "key.pem").write_text("x", encoding="utf-8")
    (project / ".gitignore").write_text("secrets/\n", encoding="utf-8")

    mm, _, _ = _map(project, tmp_path_factory, respect_gitignore=False)
    titles = {n.title for n in mm.root.walk()}
    assert "secrets" in titles


def test_max_files_per_folder_truncates_with_a_count(tmp_path, tmp_path_factory):
    project = tmp_path / "many"
    project.mkdir()
    for i in range(30):
        (project / f"file{i}.txt").write_text("x", encoding="utf-8")

    mm, _, _ = _map(project, tmp_path_factory, max_files=10)
    file_titles = [n.title for n in mm.root.children]
    assert len(file_titles) == 11  # 10 shown + 1 "+N more" node
    assert any("more file" in t for t in file_titles)


def test_max_depth_truncates_deep_nesting(tmp_path, tmp_path_factory):
    deep = tmp_path
    for i in range(10):
        deep = deep / f"level{i}"
        deep.mkdir()
    (deep / "buried.txt").write_text("x", encoding="utf-8")

    mm, _, _ = _map(tmp_path, tmp_path_factory, max_depth=3)
    notes = [n.title for n in mm.root.walk()]
    assert any("more item" in t for t in notes)
    assert mm.depth() < 11


def test_nonexistent_or_non_directory_path_raises(tmp_path, tmp_path_factory):
    import pytest

    snap_dir = tmp_path_factory.mktemp("snap")
    with pytest.raises(ValueError):
        build_folder_map(tmp_path / "does_not_exist", snapshot_dir=snap_dir)

    f = tmp_path / "file.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        build_folder_map(f, snapshot_dir=snap_dir)


# --- incremental rebuilds -------------------------------------------------


def test_second_run_with_no_changes_reuses_everything(tmp_path, tmp_path_factory):
    project = _make_project(tmp_path)
    snap_dir = tmp_path_factory.mktemp("snap")

    mm1, diff1, nodes1 = _map_reusing_snapshot(project, snap_dir)
    assert diff1 is None  # first run

    mm2, diff2, nodes2 = _map_reusing_snapshot(project, snap_dir)
    assert diff2 is not None
    assert not diff2.added
    assert not diff2.changed
    assert not diff2.deleted
    assert len(diff2.reused) > 0
    assert nodes2 == []  # nothing needs (re-)labeling
    assert {n.title for n in mm1.root.walk()} == {n.title for n in mm2.root.walk()}


def test_deep_change_propagates_up_to_every_ancestor(tmp_path, tmp_path_factory):
    project = _make_project(tmp_path)
    snap_dir = tmp_path_factory.mktemp("snap")

    _map_reusing_snapshot(project, snap_dir)

    time.sleep(0.01)  # ensure a distinct mtime from anything written above
    (project / "src" / "auth" / "logout.py").write_text("x", encoding="utf-8")

    mm2, diff2, nodes2 = _map_reusing_snapshot(project, snap_dir)
    # A change 2 levels deep must invalidate every signature on the path back
    # to the root (Merkle propagation) — a shallow, direct-children-only
    # signature would miss this entirely.
    assert "src/auth" in diff2.changed
    assert "src" in diff2.changed
    assert "." in diff2.changed
    assert "tests" in diff2.reused  # untouched sibling subtree, unaffected
    titles = {n.title for n in mm2.root.walk()}
    assert "logout.py" in titles


def test_deleted_folder_is_reported(tmp_path, tmp_path_factory):
    project = _make_project(tmp_path)
    snap_dir = tmp_path_factory.mktemp("snap")

    _map_reusing_snapshot(project, snap_dir)

    import shutil

    shutil.rmtree(project / "tests")

    mm2, diff2, _ = _map_reusing_snapshot(project, snap_dir)
    assert "tests" in diff2.deleted
    assert "tests" not in {n.title for n in mm2.root.children}


def test_fresh_flag_ignores_snapshot_but_still_saves_one(tmp_path, tmp_path_factory):
    project = _make_project(tmp_path)
    snap_dir = tmp_path_factory.mktemp("snap")

    _map_reusing_snapshot(project, snap_dir)
    mm2, diff2, nodes2 = _map_reusing_snapshot(project, snap_dir, incremental=False)

    assert diff2 is None  # --fresh: no diff reported even though history exists
    assert len(nodes2) > 0  # everything treated as needing (re-)evaluation

    # But a fresh snapshot was saved, so the *next* incremental run can reuse it.
    mm3, diff3, nodes3 = _map_reusing_snapshot(project, snap_dir)
    assert diff3 is not None
    assert not diff3.added and not diff3.changed


def test_different_params_do_not_reuse_an_incompatible_snapshot(tmp_path, tmp_path_factory):
    project = _make_project(tmp_path)
    snap_dir = tmp_path_factory.mktemp("snap")

    _map_reusing_snapshot(project, snap_dir, max_files=5)
    mm2, diff2, nodes2 = _map_reusing_snapshot(project, snap_dir, max_files=50)
    assert diff2 is None
    assert len(nodes2) > 0


def test_root_folder_is_never_sent_for_ai_labeling(tmp_path, tmp_path_factory):
    project = tmp_path / "flat"
    project.mkdir()
    (project / "a.txt").write_text("x", encoding="utf-8")
    mm, _, nodes = _map(project, tmp_path_factory)

    assert nodes == []  # no subfolders at all -> nothing needs labeling, root excluded
    provider = MockProvider()
    label_folders(mm, provider, Cache(enabled=False), nodes=nodes)
    assert provider.calls == 0


def test_reused_folder_keeps_its_previous_ai_label(tmp_path, tmp_path_factory):
    project = _make_project(tmp_path)
    snap_dir = tmp_path_factory.mktemp("snap")
    cache = (Cache(root=tmp_path_factory.mktemp("cache"), enabled=True),)
    provider = _LabelStubProvider()

    mm1, _, nodes1 = _map_reusing_snapshot(project, snap_dir, label_with=(provider, cache[0]))
    tests_node = next(n for n in mm1.root.walk() if n.title == "tests")
    assert tests_node.note == "Purpose of tests"

    # Rerun with no changes: "tests" is reused wholesale, its label survives,
    # and it shouldn't even appear in nodes_needing_labels this time.
    mm2, diff2, nodes2 = _map_reusing_snapshot(project, snap_dir)
    tests_node2 = next(n for n in mm2.root.walk() if n.title == "tests")
    assert tests_node2.note == "Purpose of tests"
    assert "tests" not in [n.title for n in nodes2]


def test_changed_folder_gets_relabeled_unchanged_folder_does_not(tmp_path, tmp_path_factory):
    project = _make_project(tmp_path)
    snap_dir = tmp_path_factory.mktemp("snap")
    cache = Cache(root=tmp_path_factory.mktemp("cache"), enabled=True)
    provider = _LabelStubProvider()

    _map_reusing_snapshot(project, snap_dir, label_with=(provider, cache))
    calls_after_first = provider.calls
    assert calls_after_first > 0

    time.sleep(0.01)
    (project / "tests" / "test_new.py").write_text("x", encoding="utf-8")

    mm2, diff2, nodes2 = _map_reusing_snapshot(project, snap_dir, label_with=(provider, cache))
    # Only the changed folder ("tests") should have triggered a fresh call —
    # "src" and "src/auth" are untouched and must not be relabeled.
    assert provider.calls == calls_after_first + 1
    assert [n.title for n in nodes2] == ["tests"]


class _LabelStubProvider:
    """Returns a deterministic label derived from the folder name, so tests
    can assert the note actually gets set without needing a real LLM."""

    name = "stub"
    model = "stub-1"

    def __init__(self):
        self.calls = 0

    def complete_json(self, system: str, user: str) -> dict:
        self.calls += 1
        folder = json.loads(user)["folder"]
        return {"label": f"Purpose of {folder}"}


def test_label_folders_sets_note_without_changing_title(tmp_path, tmp_path_factory):
    project = _make_project(tmp_path)
    mm, _, nodes = _map(project, tmp_path_factory)
    provider = _LabelStubProvider()
    cache = Cache(enabled=False)

    label_folders(mm, provider, cache, nodes=nodes)

    src_node = next(n for n in mm.root.walk() if n.title == "src")
    assert src_node.title == "src"  # ground-truth folder name preserved
    assert src_node.note == "Purpose of src"
    assert provider.calls > 0


def test_label_folders_caches_across_calls(tmp_path, tmp_path_factory):
    # Cache must live outside the mapped tree — nesting it inside would make
    # build_folder_map walk the cache directory as if it were project
    # content, and the cache's own contents legitimately change between
    # calls, which would make this test flaky for a subtle, real reason
    # (see the .cerebro default-ignore entry in foldermap.py).
    project = _make_project(tmp_path)
    cache_dir = tmp_path_factory.mktemp("cache")
    provider = _LabelStubProvider()
    cache = Cache(root=cache_dir, enabled=True)

    mm, _, nodes = _map(project, tmp_path_factory)
    label_folders(mm, provider, cache, nodes=nodes)
    calls_after_first = provider.calls
    assert calls_after_first > 0

    mm2, _, nodes2 = _map(project, tmp_path_factory)
    label_folders(mm2, provider, cache, nodes=nodes2)
    assert provider.calls == calls_after_first  # fully served from cache


def test_label_folders_skips_when_no_folders(tmp_path, tmp_path_factory):
    project = tmp_path / "flat"
    project.mkdir()
    (project / "a.txt").write_text("x", encoding="utf-8")
    mm, _, nodes = _map(project, tmp_path_factory)

    provider = MockProvider()
    label_folders(mm, provider, Cache(enabled=False), nodes=nodes)
    assert provider.calls == 0


def test_label_folders_default_labels_whole_tree_when_nodes_not_given(tmp_path, tmp_path_factory):
    # nodes=None (the default) should still work for direct/manual use —
    # labels every folder in the tree, same as before nodes= existed.
    project = _make_project(tmp_path)
    mm, _, _ = _map(project, tmp_path_factory)
    provider = _LabelStubProvider()

    label_folders(mm, provider, Cache(enabled=False))

    src_node = next(n for n in mm.root.walk() if n.title == "src")
    assert src_node.note == "Purpose of src"


# --- forget -----------------------------------------------------------------


def test_forget_deletes_an_existing_snapshot_and_reports_true(tmp_path, tmp_path_factory):
    project = _make_project(tmp_path)
    snap_dir = tmp_path_factory.mktemp("snap")
    _map_reusing_snapshot(project, snap_dir)

    assert list(snap_dir.glob("*.json"))  # sanity: a snapshot exists
    assert forget_tree_snapshot(project, snapshot_dir=snap_dir) is True
    assert not list(snap_dir.glob("*.json"))


def test_forget_nonexistent_snapshot_reports_false(tmp_path, tmp_path_factory):
    snap_dir = tmp_path_factory.mktemp("snap")
    assert forget_tree_snapshot(tmp_path / "never-mapped", snapshot_dir=snap_dir) is False


def test_forgotten_folder_rebuilds_from_scratch_next_run(tmp_path, tmp_path_factory):
    project = _make_project(tmp_path)
    snap_dir = tmp_path_factory.mktemp("snap")
    _map_reusing_snapshot(project, snap_dir)
    forget_tree_snapshot(project, snapshot_dir=snap_dir)

    _mm, diff, _nodes = _map_reusing_snapshot(project, snap_dir)
    assert diff is None  # no history to diff against, same as a true first run


# --- status / list_tree_snapshots -------------------------------------------


def test_list_tree_snapshots_empty_dir_returns_empty_list(tmp_path_factory):
    snap_dir = tmp_path_factory.mktemp("snap")
    assert list_tree_snapshots(snapshot_dir=snap_dir) == []


def test_list_tree_snapshots_reports_source_and_counts(tmp_path, tmp_path_factory):
    project = _make_project(tmp_path)
    snap_dir = tmp_path_factory.mktemp("snap")
    _map_reusing_snapshot(project, snap_dir)

    snaps = list_tree_snapshots(snapshot_dir=snap_dir)
    assert len(snaps) == 1
    assert snaps[0]["source"] == str(project.resolve())
    assert snaps[0]["folders"] > 0
    assert snaps[0]["built_at"] != "?"
