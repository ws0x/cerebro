from io import StringIO

from rich.console import Console

from cerebro import console as console_module


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


def test_ascii_mode_toggle(monkeypatch):
    assert console_module.ascii_mode() is False
    console_module.set_ascii(True)
    try:
        assert console_module.ascii_mode() is True
    finally:
        console_module.set_ascii(False)


def test_type_icon_has_an_ascii_counterpart_for_every_node_type():
    from cerebro.ui import _TYPE_ICON, _TYPE_ICON_ASCII

    assert set(_TYPE_ICON) == set(_TYPE_ICON_ASCII)


def test_icon_switches_between_emoji_and_ascii():
    from cerebro.ir import NodeType
    from cerebro.ui import _icon

    console_module.set_ascii(False)
    unicode_icon = _icon(NodeType.root)
    console_module.set_ascii(True)
    try:
        ascii_icon = _icon(NodeType.root)
    finally:
        console_module.set_ascii(False)
    assert unicode_icon != ascii_icon
    assert ascii_icon.isascii()


def test_qprint_is_silent_when_quiet_is_on():
    buf = StringIO()
    c = Console(file=buf, force_terminal=True)
    monkeypatch_console = console_module.console
    console_module.console = c
    try:
        console_module.set_quiet(True)
        console_module.qprint("hidden")
        console_module.set_quiet(False)
        console_module.qprint("shown")
    finally:
        console_module.console = monkeypatch_console
        console_module.set_quiet(False)
    out = buf.getvalue()
    assert "hidden" not in out
    assert "shown" in out


def test_quiet_mode_toggle():
    assert console_module.quiet_mode() is False
    console_module.set_quiet(True)
    try:
        assert console_module.quiet_mode() is True
    finally:
        console_module.set_quiet(False)


def test_json_mode_toggle():
    assert console_module.json_mode() is False
    console_module.set_json(True)
    try:
        assert console_module.json_mode() is True
    finally:
        console_module.set_json(False)
        console_module.set_quiet(False)


def test_json_mode_implies_quiet_mode():
    console_module.set_json(True)
    try:
        assert console_module.quiet_mode() is True
    finally:
        console_module.set_json(False)
        console_module.set_quiet(False)


def test_high_contrast_theme_overrides_dim_to_bold(monkeypatch):
    buf = StringIO()
    c = Console(file=buf, force_terminal=True)
    monkeypatch.setattr(console_module, "console", c)
    c.print("[dim]plain[/]")
    console_module.set_high_contrast(True)
    c.print("[dim]contrasted[/]")
    out = buf.getvalue().splitlines()
    assert "\x1b[2m" in out[0]  # SGR 2 = faint/dim
    assert "\x1b[2m" not in out[1]
    assert "\x1b[1m" in out[1]  # SGR 1 = bold, no dim
