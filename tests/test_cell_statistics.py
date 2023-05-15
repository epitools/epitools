from __future__ import annotations

from typing import TYPE_CHECKING

# TODO: needed for tests
# https://github.com/epitools/epitools/issues/96

if TYPE_CHECKING:
    from typing import Callable

    # TODO: needed for tests
    # https://github.com/epitools/epitools/issues/96


def test_add_cell_statistics_widget(
    make_napari_viewer: Callable,
):
    """Checks that the cell statistics widget can be added inside a dock widget."""

    viewer = make_napari_viewer()
    num_dw = len(viewer.window._dock_widgets)
    viewer.window.add_plugin_dock_widget(
        plugin_name="epitools",
        widget_name="Cell statistics",
    )

    assert len(viewer.window._dock_widgets) == num_dw + 1
