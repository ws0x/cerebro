"""Integration tests for `cerebro edit` -- it needs a real interactive
terminal (a live tree browser has no sensible piped/scripted equivalent),
so these call cli.edit() directly with questionary mocked at the same
boundary the wizard's own tests already use, rather than driving it via
piped stdin like the wizard's Rich fallback.
"""

from unittest.mock import MagicMock, patch

import pytest
import typer

import cerebro.cli as cli
from cerebro.convert.opml import write_opml
from cerebro.convert.xmind import write_xmind
from cerebro.ir import MindMap, Node, NodeType
from cerebro.merge import read_map


def _sample_file(tmp_path, writer=write_opml, name="test.opml"):
    root = Node(title="Root", type=NodeType.root)
    root.add("Topic A")
    root.add("Topic B")
    mm = MindMap(title="Root", root=root)
    return writer(mm, tmp_path / name)


def _scripted_select(script):
    """``script`` is a list of return values, consumed in call order."""
    calls = iter(script)

    def fake_select(message, choices, style=None):
        mock = MagicMock()
        mock.ask.return_value = next(calls)
        return mock

    return fake_select


def test_edit_requires_a_real_console(tmp_path):
    path = _sample_file(tmp_path)
    with patch("cerebro.cli.has_real_console", return_value=False):
        with pytest.raises(typer.Exit):
            cli.edit(file=path, out=None)


def test_edit_errors_on_missing_file(tmp_path):
    with patch("cerebro.cli.has_real_console", return_value=True):
        with pytest.raises(typer.Exit):
            cli.edit(file=tmp_path / "missing.opml", out=None)


def test_edit_rename_flow_saves_the_new_title(tmp_path):
    path = _sample_file(tmp_path)
    with patch("cerebro.cli.has_real_console", return_value=True):
        with patch("questionary.select", side_effect=_scripted_select([(0,), "rename", "__done__"])):
            with patch("questionary.text") as mock_text:
                mock_text.return_value.ask.return_value = "Renamed Topic A"
                cli.edit(file=path, out=None)

    mm = read_map(path)
    assert [c.title for c in mm.root.children] == ["Renamed Topic A", "Topic B"]


def test_edit_delete_flow_removes_the_node(tmp_path):
    path = _sample_file(tmp_path)
    with patch("cerebro.cli.has_real_console", return_value=True):
        with patch("questionary.select", side_effect=_scripted_select([(1,), "delete", "__done__"])):
            with patch("questionary.confirm") as mock_confirm:
                mock_confirm.return_value.ask.return_value = True
                cli.edit(file=path, out=None)

    mm = read_map(path)
    assert [c.title for c in mm.root.children] == ["Topic A"]


def test_edit_delete_declined_at_confirm_makes_no_change(tmp_path):
    path = _sample_file(tmp_path)
    with patch("cerebro.cli.has_real_console", return_value=True):
        with patch("questionary.select", side_effect=_scripted_select([(1,), "delete", "__done__"])):
            with patch("questionary.confirm") as mock_confirm:
                mock_confirm.return_value.ask.return_value = False
                with pytest.raises(typer.Exit):  # "no changes made" exits early
                    cli.edit(file=path, out=None)

    mm = read_map(path)
    assert [c.title for c in mm.root.children] == ["Topic A", "Topic B"]


def test_edit_cannot_delete_the_root(tmp_path):
    path = _sample_file(tmp_path)
    with patch("cerebro.cli.has_real_console", return_value=True):
        with patch("questionary.select", side_effect=_scripted_select([(), "delete", "__done__"])):
            with pytest.raises(typer.Exit):  # rejected, then Done with nothing changed
                cli.edit(file=path, out=None)

    mm = read_map(path)
    assert mm.root.title == "Root"  # untouched


def test_edit_picking_done_immediately_makes_no_change_and_does_not_write(tmp_path):
    path = _sample_file(tmp_path)
    before = path.read_text(encoding="utf-8")
    with patch("cerebro.cli.has_real_console", return_value=True):
        with patch("questionary.select", side_effect=_scripted_select(["__done__"])):
            with pytest.raises(typer.Exit):
                cli.edit(file=path, out=None)
    assert path.read_text(encoding="utf-8") == before


def test_edit_saves_to_a_different_out_path_when_given(tmp_path):
    path = _sample_file(tmp_path)
    out_path = tmp_path / "edited.opml"
    with patch("cerebro.cli.has_real_console", return_value=True):
        with patch("questionary.select", side_effect=_scripted_select([(0,), "rename", "__done__"])):
            with patch("questionary.text") as mock_text:
                mock_text.return_value.ask.return_value = "New Title"
                cli.edit(file=path, out=out_path)

    assert out_path.exists()
    mm_out = read_map(out_path)
    assert [c.title for c in mm_out.root.children] == ["New Title", "Topic B"]
    mm_original = read_map(path)  # the original file is left untouched when --out points elsewhere
    assert [c.title for c in mm_original.root.children] == ["Topic A", "Topic B"]


def test_edit_works_on_xmind_files_too(tmp_path):
    path = _sample_file(tmp_path, writer=write_xmind, name="test.xmind")
    with patch("cerebro.cli.has_real_console", return_value=True):
        with patch("questionary.select", side_effect=_scripted_select([(0,), "rename", "__done__"])):
            with patch("questionary.text") as mock_text:
                mock_text.return_value.ask.return_value = "Renamed in XMind"
                cli.edit(file=path, out=None)

    mm = read_map(path)
    assert [c.title for c in mm.root.children] == ["Renamed in XMind", "Topic B"]


def test_edit_rejects_an_unsupported_output_extension(tmp_path):
    path = _sample_file(tmp_path)
    with patch("cerebro.cli.has_real_console", return_value=True):
        with patch("questionary.select", side_effect=_scripted_select([(0,), "rename", "__done__"])):
            with patch("questionary.text") as mock_text:
                mock_text.return_value.ask.return_value = "New Title"
                with pytest.raises(typer.Exit):
                    cli.edit(file=path, out=tmp_path / "out.md")
