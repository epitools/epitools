from __future__ import annotations

import napari.types
import numpy as np
import numpy.typing as npt
from magicgui import magic_factory, widgets
from magicgui.widgets._bases import Widget
from napari import current_viewer
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error

from napari_epitools.analysis import (
    calculate_projection,
    calculate_segmentation,
)

# Rendering properties of seeds
SEED_SIZE = 3
SEED_EDGE_COLOR = "red"
SEED_FACE_COLOR = "red"

# Defaults
PBAR = {"visible": False, "max": 0, "label": "working..."}

SMOOTHING_RADIUS = {
    "widget_type": "FloatSlider",
    "name": "smoothing_radius",
    "min": 0.0,
    "max": 10.0,
    "step": 0.1,
    "value": 0.2,
}

SURFACE_SMOOTHNESS_1 = {
    "widget_type": "Slider",
    "name": "surface_smoothness_1",
    "min": 0,
    "max": 10,
    "step": 1,
    "value": 5,
}

SURFACE_SMOOTHNESS_2 = {
    "widget_type": "Slider",
    "name": "surface_smoothness_2",
    "min": 0,
    "max": 10,
    "step": 1,
    "value": 5,
}

CUT_OFF_DISTANCE = {
    "widget_type": "Slider",
    "name": "cut_off_distance",
    "min": 0,
    "max": 5,
    "step": 1,
    "value": 2,
}

SPOT_SIGMA = {
    "widget_type": "FloatSlider",
    "name": "spot_sigma",
    "min": 0,
    "max": 20,
    "step": 0.1,
    "value": 3,
}

OUTLINE_SIGMA = {
    "widget_type": "FloatSlider",
    "name": "outline_sigma",
    "min": 0,
    "max": 20,
    "step": 0.1,
    "value": 0,
}

THRESHOLD = {
    "widget_type": "FloatSlider",
    "name": "threshold",
    "min": 0,
    "max": 100,
    "step": 1,
    "value": 20,
}
PROJECTION_LAYER_NAME = "Projection"
SEEDS_LAYER_NAME = "Seeds"
CELLS_LAYER_NAME = "Cells"
WIDGET_NDIM = 3


def _reset_axes(widget: Widget) -> None:
    """Set the dimension sliders to `0`"""
    axes = [0] if widget.viewer.dims.ndim == WIDGET_NDIM else [0, 1]
    for axis in axes:
        widget.viewer.dims.set_current_step(axis, 0)


def _add_layers(widget: Widget, layers: list[napari.types.LayerDataTuple]) -> None:
    add_layer_func = {
        "image": widget.viewer.add_image,
        "labels": widget.viewer.add_labels,
        "points": widget.viewer.add_points,
    }
    for layer in layers:
        data, layer_data, layer_type = layer
        try:
            widget.viewer.layers[layer_data.get("name")].data = data
        except KeyError:
            add_layer_func[layer_type](data, **layer_data)

    _reset_axes(widget)


@magic_factory(
    pbar=PBAR,
    smoothing_radius=SMOOTHING_RADIUS,
    surface_smoothness_1=SURFACE_SMOOTHNESS_1,
    surface_smoothness_2=SURFACE_SMOOTHNESS_2,
    cut_off_distance=CUT_OFF_DISTANCE,
)
def projection_widget(  # noqa: PLR0913
    pbar: widgets.ProgressBar,
    input_image: napari.types.ImageData,
    smoothing_radius: float,
    surface_smoothness_1: int,
    surface_smoothness_2: int,
    cut_off_distance: int,
) -> napari.types.ImageData:
    """Z projection using image interpolation.
    Args:
        pbar:
            Progressbar widget
        input_image:
            Numpy ndarray representation of multi-dimensional image stack.
        smoothing_radius:
            Kernel radius for gaussian blur to apply before estimating the surface.
        surface_smoothness_1:
            Surface smoothness for 1st griddata estimation, larger means smoother.
        surface_smoothness_2:
            Surface smoothness for 3nd griddata estimation, larger means smoother.
        cut_off_distance:
            Cutoff distance in z-planes from the 1st estimated surface.

    Returns:
        Projected image as napari Image layer.
    """

    if input_image is None:
        pbar.hide()
        show_error("Load an image first")
        return None

    def handle_returned(projection: npt.NDArray[np.float64]) -> None:
        """Callback for `run` thread worker."""

        pbar.hide()
        projection_widget.viewer = current_viewer()
        projection_layer = (
            projection,
            {"name": PROJECTION_LAYER_NAME},
            "image",
        )
        _add_layers(projection_widget, [projection_layer])

    @thread_worker(connect={"returned": handle_returned})
    def run() -> npt.NDArray[np.float64]:
        """Handle clicks on the `Run` button. Runs projection in a
        separate thread to avoid blocking GUI."""
        return calculate_projection(
            input_image,
            smoothing_radius,
            surface_smoothness_1,
            surface_smoothness_2,
            cut_off_distance,
        )

    pbar.show()
    return run()


@magic_factory(
    pbar=PBAR,
    spot_sigma=SPOT_SIGMA,
    outline_sigma=OUTLINE_SIGMA,
    threshold=THRESHOLD,
)
def segmentation_widget(
    pbar: widgets.ProgressBar,
    input_image: napari.types.ImageData,
    spot_sigma: float,
    outline_sigma: float,
    threshold: float,
) -> napari.types.LayerDataTuple:
    """Segment cells in a projected image.

    Args:
        pbar:
            Progressbar widget
        input_image:
            Numpy ndarray representation of a projected image stack.
        spot_sigma:
            Controls how close segmented cells can be.
        outline_sigma:
            Controls how precisely segmented cells are outlined.
        threshold:
            Cells with an average intensity below `threshold` are ignored.

    Returns:
        Seed points as a Napari Points layer and segmented cells as a
        Napari Labels layer.
    """
    if input_image is None:
        pbar.hide()
        show_error("Load a projection first")
        return None

    def handle_returned(
        result: tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]
    ) -> None:
        """Callback for `run` thread worker."""

        pbar.hide()
        segmentation_widget.viewer = current_viewer()
        seeds, labels = result
        labels_layer = (labels, {"name": CELLS_LAYER_NAME}, "labels")
        seeds_layer = (
            seeds,
            {
                "name": SEEDS_LAYER_NAME,
                "size": SEED_SIZE,
                "edge_color": SEED_EDGE_COLOR,
                "face_color": SEED_FACE_COLOR,
            },
            "points",
        )
        layers = [labels_layer, seeds_layer]
        _add_layers(segmentation_widget, layers)

    @thread_worker(connect={"returned": handle_returned})
    def run() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Handle clicks on the `Run` button. Runs segmentation in a
        separate thread to avoid blocking GUI.
        """

        return calculate_segmentation(
            input_image,
            spot_sigma,
            outline_sigma,
            threshold,
        )

    pbar.show()
    return run()
