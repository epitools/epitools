from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable


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
