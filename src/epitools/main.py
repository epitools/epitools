from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

import magicgui.widgets
import napari.layers
import napari.qt.threading
from magicgui.types import FileDialogMode

import epitools.analysis
import epitools.widgets

__all__ = [
    "create_projection_widget",
    "create_segmentation_widget",
    "create_cell_statistics_widget",
]

logger = logging.getLogger(__name__)


THREE_DIMENSIONAL = 3  # ZYX
FOUR_DIMENSIONAL = 4  # TZYX


def create_projection_widget() -> magicgui.widgets.Container:
    """Create a widget to project a 4d timeseries (TZYX) along the z dimension"""

    projection_widget = epitools.widgets.create_projection_widget()

    # Project the timeseries when pressing the 'Run' button
    projection_widget.run.changed.connect(
        lambda: run_projection(
            image=projection_widget.input_image.value,
            smoothing_radius=projection_widget.smoothing_radius.value,
            surface_smoothness_1=projection_widget.surface_smoothness_1.value,
            surface_smoothness_2=projection_widget.surface_smoothness_2.value,
            cutoff_distance=projection_widget.cutoff_distance.value,
        ),
    )

    return projection_widget


def run_projection(
    image: napari.layers.Image,
    smoothing_radius,
    surface_smoothness_1,
    surface_smoothness_2,
    cutoff_distance,
) -> None:
    """Project a 4d timeseries along the z dimension"""

    projected_data = epitools.analysis.calculate_projection(
        image.data,
        smoothing_radius,
        surface_smoothness_1,
        surface_smoothness_2,
        cutoff_distance,
    )

    # Remove z dimension from scale and translate arrays
    if image.ndim == THREE_DIMENSIONAL:  # ZYX
        mask = [1, 2]
    elif image.ndim == FOUR_DIMENSIONAL:  # TZYX
        mask = [0, 2, 3]

    two_d_scale = image.metadata["scale"][mask]
    two_d_translate = np.asarray(image.translate)[mask]

    # spacing for regionprops
    two_d_spacing = (
        image.metadata["spacing"]
        if image.ndim == THREE_DIMENSIONAL
        else image.metadata["spacing"][1:]
    )

    two_d_metadata = {
        "scale": two_d_scale,
        "spacing": two_d_spacing,
    }

    viewer = napari.current_viewer()
    viewer.add_image(
        data=projected_data,
        name="Projection",
        translate=two_d_translate,
        metadata=two_d_metadata,
    )


def create_segmentation_widget() -> magicgui.widgets.Container:
    """Create a widget to segment a 3d (TYZ) timeseries"""

    segmentation_widget = epitools.widgets.create_segmentation_widget()
    viewer = napari.current_viewer()

    # Automatically select a newly added Image layer
    viewer.layers.events.inserted.connect(
        lambda event: select_inserted_image(
            new_layer=event.value,
            widget=segmentation_widget.input_image,
        ),
    )

    # Segment the timeseries when pressing the 'Run' button
    segmentation_widget.run.changed.connect(
        lambda: run_segmentation(
            image=segmentation_widget.input_image.value,
            spot_sigma=segmentation_widget.spot_sigma.value,
            outline_sigma=segmentation_widget.outline_sigma.value,
            threshold=segmentation_widget.threshold.value,
        )
    )

    return segmentation_widget


def select_inserted_image(
    new_layer: napari.layers.Layer,
    widget: magicgui.widgets.ComboBox,
):
    """Update the selected Image when a image layer is added"""

    if not isinstance(new_layer, napari.layers.Image):
        message = (
            f"Not selecting new layer {new_layer.name} as input for the "
            f"segmentation widget as {new_layer.name} is {type(new_layer)} "
            "layer not a Labels layer."
        )
        logger.debug(message)
        return

    # the new layer is always last in the list
    widget.native.setCurrentIndex(len(widget) - 1)


def run_segmentation(
    image: napari.layers.Image,
    spot_sigma: float,
    outline_sigma: float,
    threshold: float,
) -> None:
    """Segment a 3d timeserise (TYZ) at each frame"""

    seeds_data, labels_data = epitools.analysis.calculate_segmentation(
        projection=image.data,
        spot_sigma=spot_sigma,
        outline_sigma=outline_sigma,
        threshold=threshold,
    )

    labels_metadata = {
        "scale": image.metadata["scale"],
        "spacing": image.metadata["spacing"],
    }

    viewer = napari.current_viewer()
    viewer.add_labels(
        data=labels_data,
        translate=image.translate,
        metadata=labels_metadata,
        name="Cells",
    )
    viewer.add_points(
        data=seeds_data,
        name="Seeds",
        size=3,
        edge_color="red",
        face_color="red",
    )


def create_cell_statistics_widget() -> magicgui.widgets.Container:
    """Create a widget for calculating cell statistics of labelled segmentations."""

    cell_statistics_widget = epitools.widgets.create_cell_statistics_widget()
    viewer = napari.current_viewer()

    # Update cell_statistics when scrolling through frames
    viewer.dims.events.current_step.connect(
        lambda event: _update_cell_statistics(
            layers=viewer.layers,
            frame=event.value[0],
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

    # create colourmap for selected Labels
    cell_statistics_widget.colourmap_create_button.changed.connect(
        lambda: create_colourmaps(
            labels=cell_statistics_widget.colourmap_input_labels.value,
            colourmap_statistic=cell_statistics_widget.colourmap_statistic.current_choice,
            lower_limit=cell_statistics_widget.colourmap_lower_limit,
            upper_limit=cell_statistics_widget.colourmap_upper_limit,
            autolimits=cell_statistics_widget.colourmap_autolimits.value,
        )
    )

    return cell_statistics_widget


def _update_cell_statistics(
    layers: list[napari.layers.Layer],
    frame: int,
) -> None:
    """Update Labels cell_statistics for current frame.

    Apply any colourmaps for the Labels layers at the current frame.
    """

    for layer in layers:
        try:
            layer.features = layer.metadata["cell_statistics"][frame]
        except KeyError:
            message = f"No cell statistics to load for layer {layer.name}"
            logger.log(level=1, msg=message)
        except IndexError:
            message = (
                f"No cell statistics to load for layer {layer.name} at frame {frame}"
            )
            logger.log(level=9, msg=message)

        try:
            layer.color = layer.metadata["colourmaps"][frame]
        except KeyError:
            message = f"No colourmaps load apply to layer {layer.name}"
            logger.log(level=1, msg=message)
        except IndexError:
            message = f"No colourmap to apply for layer {layer.name} at frame {frame}"
            logger.log(level=9, msg=message)


def run_cell_statistics(
    image: napari.layers.Image,
    labels: napari.layers.Labels,
) -> None:
    """Calculate cell statistics for all frames in the selected Image and Labels"""

    cell_statistics, graphs = epitools.analysis.calculate_cell_statistics(
        image=image.data,
        labels=labels.data,
        pixel_spacing=image.metadata["spacing"],
    )

    # We will use these to update the cell stats at each frame
    labels.metadata["cell_statistics"] = cell_statistics
    labels.metadata["graphs"] = graphs

    # confirm calculation finished
    message = f"Finished calculating cell statistics for '{labels.name}'"
    napari.utils.notifications.show_info(message)

    # Set cell stats for the current frame
    viewer = napari.current_viewer()
    current_frame = viewer.dims.current_step[0]
    try:
        labels.features = cell_statistics[current_frame]
    except IndexError:
        message = (
            f"No cell statistics to load for {labels.name} at frame {current_frame}"
        )
        logger.debug(message)


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
        message = f"Cell statistics not saved for {labels.name} - no filename given"
        napari.utils.notifications.show_info(message)
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
    message = f"'{labels.name}' cell statistics written to '{filename}'"
    napari.utils.notifications.show_info(message)


def create_colourmaps(
    labels: napari.layers.Labels,
    colourmap_statistic: str,
    lower_limit: magicgui.widgets.FloatSpinBox,
    upper_limit: magicgui.widgets.FloatSpinBox,
    *,
    autolimits: bool,
) -> None:
    """Create a colourmap for a cell statistic each frame.

    If the statistic is set to "id", then the colour mode is set to "auto", which will
    colour regions based on their ids.

    If the statistics is set to any other value, a colourmap is created for that
    statistic for each frame.
    """

    if "cell_statistics" not in labels.metadata:
        _msg = (
            "Cannot create colourmaps - "
            "the selected Labels layer has no cell staistics."
        )
        napari.utils.notifications.show_error(_msg)
        return

    if colourmap_statistic == "id":
        labels.metadata.pop("metadata", None)
        labels.color_mode = "auto"
        return

    cell_statistics = [
        frame_statistics[colourmap_statistic]
        for frame_statistics in labels.metadata["cell_statistics"]
    ]
    cell_indices = [
        frame_statistics["index"]
        for frame_statistics in labels.metadata["cell_statistics"]
    ]

    if autolimits:
        lower_limit.value = min(min(statistics) for statistics in cell_statistics)
        upper_limit.value = max(max(statistics) for statistics in cell_statistics)

    colourmaps = [
        _create_colourmap(
            float(lower_limit.value),
            float(upper_limit.value),
            statistics,
            indices,
        )
        for statistics, indices in zip(cell_statistics, cell_indices)
    ]

    labels.metadata["colourmaps"] = colourmaps

    # confirm colourmaps created
    message = f"'{colourmap_statistic}' colourmaps created for '{labels.name}'"
    napari.utils.notifications.show_info(message)

    # Set cell colourmap for the current frame
    viewer = napari.current_viewer()
    current_frame = viewer.dims.current_step[0]
    try:
        labels.color = colourmaps[current_frame]
    except IndexError:
        message = f"No colourmap to apply for {labels.name} at frame {current_frame}"
        logger.debug(message)


def _create_colourmap(
    lower_limit: float,
    upper_limit: float,
    statistics: npt.NDArray,
    indices: npt.NDArray,
) -> dict[int, npt.NDArray]:
    """Create a dictionary of colours - one per unique region"""

    colourmap_data = (statistics - lower_limit) / upper_limit
    colourmap_data = np.clip(colourmap_data, a_min=0, a_max=1)
    colours = plt.cm.turbo(colourmap_data)

    return dict(zip(indices, colours))
