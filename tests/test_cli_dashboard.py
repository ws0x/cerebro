from rich.layout import Layout

from cerebro.cli import _dashboard_layout


def test_dashboard_layout_builds_without_error():
    # Read-only against the real ~/.cerebro (same as `cerebro doctor`/`status`
    # already do) -- just verifies the layout assembles correctly, not its
    # exact content, which depends on the real environment's actual state.
    layout = _dashboard_layout()
    assert isinstance(layout, Layout)


def test_dashboard_layout_has_header_body_footer():
    layout = _dashboard_layout()
    names = {child.name for child in layout.children}
    assert names == {"header", "body", "footer"}


def test_dashboard_body_splits_into_health_and_memory():
    layout = _dashboard_layout()
    body = next(c for c in layout.children if c.name == "body")
    names = {child.name for child in body.children}
    assert names == {"health", "memory"}
