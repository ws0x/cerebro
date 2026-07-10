import json

from cerebro.cache import Cache
from cerebro.foldermap import build_folder_map, label_folders
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


def test_builds_hierarchy_matching_real_structure(tmp_path):
    project = _make_project(tmp_path)
    mm = build_folder_map(project)

    titles = {n.title for n in mm.root.walk()}
    assert "src" in titles
    assert "auth" in titles
    assert "login.py" in titles
    assert "tests" in titles
    assert "test_login.py" in titles


def test_default_ignore_list_filters_noise_dirs_always(tmp_path):
    project = _make_project(tmp_path)
    mm = build_folder_map(project)

    titles = {n.title for n in mm.root.walk()}
    assert "__pycache__" not in titles
    assert "node_modules" not in titles
    assert "junk.pyc" not in titles
    assert "pkg.js" not in titles


def test_notable_files_get_definition_marker(tmp_path):
    project = _make_project(tmp_path)
    mm = build_folder_map(project)

    by_title = {n.title: n for n in mm.root.walk()}
    assert by_title["README.md"].type == NodeType.definition
    assert by_title["pyproject.toml"].type == NodeType.definition
    assert by_title["login.py"].type == NodeType.detail


def test_gitignore_is_respected_when_present(tmp_path):
    project = _make_project(tmp_path)
    (project / "secrets").mkdir()
    (project / "secrets" / "key.pem").write_text("x", encoding="utf-8")
    (project / ".gitignore").write_text("secrets/\n", encoding="utf-8")

    mm = build_folder_map(project, respect_gitignore=True)
    titles = {n.title for n in mm.root.walk()}
    assert "secrets" not in titles
    assert "key.pem" not in titles


def test_gitignore_can_be_disabled(tmp_path):
    project = _make_project(tmp_path)
    (project / "secrets").mkdir()
    (project / "secrets" / "key.pem").write_text("x", encoding="utf-8")
    (project / ".gitignore").write_text("secrets/\n", encoding="utf-8")

    mm = build_folder_map(project, respect_gitignore=False)
    titles = {n.title for n in mm.root.walk()}
    assert "secrets" in titles


def test_max_files_per_folder_truncates_with_a_count(tmp_path):
    project = tmp_path / "many"
    project.mkdir()
    for i in range(30):
        (project / f"file{i}.txt").write_text("x", encoding="utf-8")

    mm = build_folder_map(project, max_files=10)
    file_titles = [n.title for n in mm.root.children]
    assert len(file_titles) == 11  # 10 shown + 1 "+N more" node
    assert any("more file" in t for t in file_titles)


def test_max_depth_truncates_deep_nesting(tmp_path):
    deep = tmp_path
    for i in range(10):
        deep = deep / f"level{i}"
        deep.mkdir()
    (deep / "buried.txt").write_text("x", encoding="utf-8")

    mm = build_folder_map(tmp_path, max_depth=3)
    # 10 real levels of nesting exist on disk; confirm it's meaningfully
    # bounded (nowhere near the full 11 node-levels an unbounded walk would
    # produce) and that a truncation marker is present.
    notes = [n.title for n in mm.root.walk()]
    assert any("more item" in t for t in notes)
    assert mm.depth() < 11


def test_nonexistent_or_non_directory_path_raises(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        build_folder_map(tmp_path / "does_not_exist")

    f = tmp_path / "file.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        build_folder_map(f)


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


def test_label_folders_sets_note_without_changing_title(tmp_path):
    project = _make_project(tmp_path)
    mm = build_folder_map(project)
    provider = _LabelStubProvider()
    cache = Cache(enabled=False)

    label_folders(mm, provider, cache)

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

    mm = build_folder_map(project)
    label_folders(mm, provider, cache)
    calls_after_first = provider.calls
    assert calls_after_first > 0

    mm2 = build_folder_map(project)
    label_folders(mm2, provider, cache)
    assert provider.calls == calls_after_first  # fully served from cache


def test_label_folders_skips_when_no_folders(tmp_path):
    project = tmp_path / "flat"
    project.mkdir()
    (project / "a.txt").write_text("x", encoding="utf-8")
    mm = build_folder_map(project)

    provider = MockProvider()
    label_folders(mm, provider, Cache(enabled=False))
    assert provider.calls == 0
