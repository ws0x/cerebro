from io import StringIO

from rich.console import Console


def test_ui_wizard_and_cli_share_one_console_instance():
    import cerebro.cli as cli
    import cerebro.ui as ui
    import cerebro.wizard as wizard

    assert cli.console is ui.console is wizard.console


def test_mutating_no_color_after_construction_suppresses_future_color():
    buf = StringIO()
    c = Console(file=buf, force_terminal=True, no_color=False)
    c.print("[red]before[/]")
    c.no_color = True
    c.print("[red]after[/]")
    out = buf.getvalue()
    assert "\x1b[" in out.splitlines()[0]  # colored before the flip
    assert "\x1b[" not in out.splitlines()[1]  # plain after it


def test_console_respects_no_color_env_var_at_construction(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    buf = StringIO()
    c = Console(file=buf, force_terminal=True)
    assert c.no_color is True
