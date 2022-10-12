from typing import Tuple

import numpy as np
import numpy.typing as npt
from magicgui import magic_factory, widgets
from magicgui.widgets._bases import Widget
from napari import current_viewer
from napari.layers import Image
from napari.qt.threading import FunctionWorker, thread_worker
from napari.types import ImageData
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


def _reset_axes(widget: Widget) -> None:
    """Set the dimension sliders to `0`"""
    axes = [0, 1]
    if widget.viewer.dims.ndim == 3:
        axes = [0]

    for axis in axes:
        widget.viewer.dims.set_current_step(axis, 0)


def _add_projection(
    widget: Widget, projection: npt.NDArray[np.float64]
) -> None:
    """Adds a project layer to a widet's viewer."""
    try:
        widget.viewer.layers[PROJECTION_LAYER_NAME].data = projection
    except KeyError:
        widget.viewer.add_image(projection, name=PROJECTION_LAYER_NAME)
    finally:
        _reset_axes(widget)


def _add_segmentation(
    widget: Widget,
    segmentation: Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]],
) -> None:
    """Adds seeds as a points layer and labels as labels layer to a widget's
    viewer.
    """
    seeds, labels = segmentation

    try:
        widget.viewer.layers[SEEDS_LAYER_NAME].data = seeds
        widget.viewer.layers[CELLS_LAYER_NAME].data = labels
    except KeyError:
        widget.viewer.add_points(
            seeds,
            name=SEEDS_LAYER_NAME,
            size=SEED_SIZE,
            edge_color=SEED_EDGE_COLOR,
            face_color=SEED_FACE_COLOR,
        )
        widget.viewer.add_labels(labels, name=CELLS_LAYER_NAME)
    finally:
        _reset_axes(widget)


@magic_factory(
    pbar=PBAR,
    smoothing_radius=SMOOTHING_RADIUS,
    surface_smoothness_1=SURFACE_SMOOTHNESS_1,
    surface_smoothness_2=SURFACE_SMOOTHNESS_2,
    cut_off_distance=CUT_OFF_DISTANCE,
)
def projection_widget(
    pbar: widgets.ProgressBar,
    input_image: ImageData,
    smoothing_radius: float,
    surface_smoothness_1: int,
    surface_smoothness_2: int,
    cut_off_distance: int,
) -> FunctionWorker:
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
        return

    def handle_returned(projection) -> None:
        pbar.hide()
        projection_widget.viewer = current_viewer()
        _add_projection(projection_widget, projection)

    @thread_worker(connect={"returned": handle_returned})
    def run() -> npt.NDArray[np.float64]:
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
    input_image: ImageData,
    spot_sigma: float,
    outline_sigma: float,
    threshold: float,
) -> FunctionWorker:
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
        return

    def handle_returned(result) -> None:
        pbar.hide()
        segmentation_widget.viewer = current_viewer()
        _add_segmentation(segmentation_widget, result)

    @thread_worker(connect={"returned": handle_returned})
    def run() -> npt.NDArray[np.int64]:
        return calculate_segmentation(
            input_image,
            spot_sigma,
            outline_sigma,
            threshold,
        )

    pbar.show()
    return run()


def epitools_widget() -> widgets.Container:
    """
    A composite widget which includes both projection and segmentation widgets.

    Projection is done by implementing the selective plane algorithm from:
    https://epitools.github.io/wiki/Analysis_Modules/00_projection/

    Segmentation is implemented by following the same approach as in:
    https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes
    """
    # Layout for projection
    input_image = widgets.create_widget(annotation=Image, label="Image:")
    smoothing_radius = widgets.FloatSlider(**SMOOTHING_RADIUS)
    surface_smoothness_1 = widgets.Slider(**SURFACE_SMOOTHNESS_1)
    surface_smoothness_2 = widgets.Slider(**SURFACE_SMOOTHNESS_2)
    cut_off_distance = widgets.Slider(**CUT_OFF_DISTANCE)
    run_proj_button = widgets.PushButton(
        name="run_proj_button", label="Run Projection"
    )

    # Layout for segmentation
    seg_image = widgets.create_widget(
        annotation=Image, label="Projected Image:"
    )
    spot_sigma = widgets.FloatSlider(**SPOT_SIGMA)
    outline_sigma = widgets.FloatSlider(**OUTLINE_SIGMA)
    threshold = widgets.Slider(**THRESHOLD)

    run_seg_button = widgets.PushButton(
        name="run_seg_button", label="Run Segmentation"
    )
    widget = widgets.Container(
        widgets=[
            input_image,
            smoothing_radius,
            surface_smoothness_1,
            surface_smoothness_2,
            cut_off_distance,
            run_proj_button,
            seg_image,
            spot_sigma,
            outline_sigma,
            threshold,
            run_seg_button,
        ]
    )
    widget.viewer = current_viewer()

    def handle_projection(projection):
        _add_projection(widget, projection)

    @thread_worker
    def _calculate_projection():
        stack = input_image.value.data.astype(float)
        return calculate_projection(
            stack,
            smoothing_radius.value,
            surface_smoothness_1.value,
            surface_smoothness_2.value,
            cut_off_distance.value,
        )

    @widget.run_proj_button.clicked.connect
    def run_projection() -> None:
        worker = _calculate_projection()
        worker.returned.connect(handle_projection)
        worker.start()

    def handle_segmentation(segmentation):
        _add_segmentation(widget, segmentation)

    @thread_worker
    def _calculate_segmentation():
        proj = seg_image.value.data.astype(float)
        return calculate_segmentation(
            proj,
            spot_sigma.value,
            outline_sigma.value,
            threshold.value,
        )

    @widget.run_seg_button.clicked.connect
    def run_segmentation() -> None:
        worker = _calculate_segmentation()
        worker.returned.connect(handle_segmentation)
        worker.start()

    return widget
