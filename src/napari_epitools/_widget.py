from __future__ import annotations

import os

import magicgui.widgets
import napari.layers
import napari.qt.threading
import napari.types
import numpy as np
import numpy.typing as npt
import pandas as pd
from magicgui import magic_factory
from magicgui.types import FileDialogMode
from magicgui.widgets.bases import Widget
from napari import current_viewer
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error

from napari_epitools.analysis import (
    calculate_cell_statistics,
    calculate_projection,
    calculate_segmentation,
)

# Rendering properties of seeds
SEED_SIZE = 3
SEED_EDGE_COLOR = "red"
SEED_FACE_COLOR = "red"


SMOOTHING_RADIUS = {
    "widget_type": "FloatSlider",
    "name": "smoothing_radius",
    "min": 0.0,
    "max": 10.0,
    "step": 0.1,
    "value": 1,
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
    "value": 3,
}

SPOT_SIGMA = {
    "widget_type": "FloatSlider",
    "name": "spot_sigma",
    "min": 0,
    "max": 20,
    "step": 0.1,
    "value": 10,
}

OUTLINE_SIGMA = {
    "widget_type": "FloatSlider",
    "name": "outline_sigma",
    "min": 0,
    "max": 20,
    "step": 0.1,
    "value": 3,
}

THRESHOLD = {
    "widget_type": "FloatSlider",
    "name": "threshold",
    "min": 0,
    "max": 100,
    "step": 1,
    "value": 3,
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
    smoothing_radius=SMOOTHING_RADIUS,
    surface_smoothness_1=SURFACE_SMOOTHNESS_1,
    surface_smoothness_2=SURFACE_SMOOTHNESS_2,
    cut_off_distance=CUT_OFF_DISTANCE,
)
def projection_widget(
    input_image: napari.types.ImageData,
    smoothing_radius: float,
    surface_smoothness_1: int,
    surface_smoothness_2: int,
    cut_off_distance: int,
) -> napari.qt.threading.FunctionWorker:
    """Z projection using image interpolation.
    Args:
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
        show_error("Load an image first")
        return None

    def handle_returned(projection: npt.NDArray[np.float64]) -> None:
        """Callback for `run` thread worker."""

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

    return run()


def _init_segmentation_widget(
    container: magicgui.widgets._function_gui.FunctionGui,
):
    """Add callbacks for the widget that is used to select the input image"""

    viewer = napari.current_viewer()

    # Automatically select a newly added Image layer
    viewer.layers.events.inserted.connect(
        lambda event: _select_inserted_image(
            new_layer=event.value,
            widget=container.input_image,
        ),
    )


def _select_inserted_image(
    new_layer: napari.layers.Layer,
    widget: magicgui.widgets.ComboBox,
):
    """Update the selected Image when a image layer is added"""

    if not isinstance(new_layer, napari.layers.Image):
        return

    # the new layer is always last in the list
    widget.native.setCurrentIndex(len(widget) - 1)


@magic_factory(
    spot_sigma=SPOT_SIGMA,
    outline_sigma=OUTLINE_SIGMA,
    threshold=THRESHOLD,
    widget_init=_init_segmentation_widget,
)
def segmentation_widget(
    input_image: napari.types.ImageData,
    spot_sigma: float,
    outline_sigma: float,
    threshold: float,
) -> napari.qt.threading.FunctionWorker:
    """Segment cells in a projected image.

    Args:
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
        show_error("Load a projection first")
        return None

    def handle_returned(
        result: tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]
    ) -> None:
        """Callback for `run` thread worker."""

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

    return run()


def cell_statistics_widget() -> magicgui.widgets.Container:
    """Create a widget for calculating cell statistics of labelled segmentations."""

    widgets = _create_cell_statistics_widgets()
    cell_statistics_widget = magicgui.widgets.Container(
        widgets=widgets,
        scrollable=False,
    )

    viewer = napari.current_viewer()
    cell_statistics_widget.viewer = viewer

    # Update cell_statistics when scrolling through frames
    viewer.dims.events.current_step.connect(
        lambda event: _update_cell_statistics(
            layers=viewer.layers, frame=event.value[0]
        ),
    )

    # calculate cell_statistics when pressing 'Run' button
    cell_statistics_widget.run.changed.connect(
        lambda: run_cell_statistics(
            image=cell_statistics_widget.input_image.value,
            labels=cell_statistics_widget.input_labels.value,
        ),
    )

    # write the cell_statistics to CSV
    cell_statistics_widget.export.changed.connect(
        lambda: export_cell_statistics(
            labels=cell_statistics_widget.input_labels.value,
        ),
    )

    return cell_statistics_widget


def _create_cell_statistics_widgets() -> list[Widget]:
    """Create widgets for calculating and exporting cell statistics"""

    image_tooltip = "Select an 'Image' layer to use for calculating cell statistics."
    image = magicgui.widgets.create_widget(
        annotation=napari.layers.Image,
        name="input_image",
        label="image",
        options={"tooltip": image_tooltip},
    )

    labels_tooltip = (
        "Select a 'Labels' layer to use for calculating cell statistics.\n"
        "These should be the corresponding labels for the selected Image."
    )
    labels = magicgui.widgets.create_widget(
        annotation=napari.layers.Labels,
        name="input_labels",
        label="labels",
        options={"tooltip": labels_tooltip},
    )

    run_tooltip = "Calculate cell statistics for the selected Image and Labels"
    run = magicgui.widgets.create_widget(
        name="run",
        label="Calculate statistics",
        widget_type="PushButton",
        options={"tooltip": run_tooltip},
    )

    export_tooltip = (
        "Export the cell statistics for the selected Image and Labels to a CSV file"
    )
    export = magicgui.widgets.create_widget(
        name="export",
        label="Export statistics",
        widget_type="PushButton",
        options={"tooltip": export_tooltip},
    )

    return [image, labels, run, export]


def _update_cell_statistics(
    layers: list[napari.layers.Layer],
    frame: int,
) -> None:
    """Update Labels cell_statistics for current frame"""

    for layer in layers:
        try:
            layer.features = layer.metadata["cell_statistics"][frame]
        except KeyError:
            pass
        except IndexError:
            pass


def run_cell_statistics(
    image: napari.layers.Image,
    labels: napari.layers.Labels,
) -> None:
    """Calculate cell statistics for all frames in the selected Image and Labels"""

    cell_statistics, graphs = calculate_cell_statistics(
        image=image.data,
        labels=labels.data,
    )

    # We will use these to update the cell stats at each frame
    labels.metadata["cell_statistics"] = cell_statistics
    labels.metadata["graphs"] = graphs

    # confirm calculation finished
    message = f"'Finished calculating cell statistics for '{labels.name}'"
    napari.utils.notifications.show_info(message)

    # Set cell stats for the current frame
    viewer = napari.current_viewer()
    current_frame = viewer.dims.current_step[0]
    try:
        labels.features = cell_statistics[current_frame]
    except IndexError:
        pass


def export_cell_statistics(
    labels: napari.layers.Labels,
) -> None:
    """Get filename for exporting Labels cell statitics to CSV"""

    app = magicgui.application.use_app()
    show_file_dialog = app.get_obj("show_file_dialog")
    filename = show_file_dialog(
        mode=FileDialogMode.OPTIONAL_FILE,
        caption="Specify file to save Epitools cell statistics",
        start_path=None,
        filter="*.csv",
    )

    if filename is None:
        return

    _cell_statistics_to_csv(
        filename=filename,
        labels=labels,
    )


def _cell_statistics_to_csv(
    filename: os.PathLike,
    labels: napari.layers.Labels,
) -> None:
    """Write cell statistics for all frames in the selected Labels to CSV"""

    try:
        cell_statistics = labels.metadata["cell_statistics"]
    except KeyError:
        message = f"'{labels.name}' has no cell statistics to export"
        napari.utils.notifications.show_error(message)
        return

    df = pd.concat(
        [pd.DataFrame.from_dict(stats).set_index("index") for stats in cell_statistics],
        keys=list(range(len(cell_statistics))),
    )
    df.to_csv(filename, index_label=["frame", "label"])

    # confirm export
    message = f"'{labels.name}' cell statistics written to {filename}"
    napari.utils.notifications.show_info(message)
