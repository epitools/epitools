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
from napari.utils import progress

import epitools.analysis
import epitools.widgets

__all__ = [
    "create_projection_widget",
    "create_segmentation_widget",
    "create_projection_2ch_widget",
    "create_cell_statistics_widget",
    "create_quality_metrics_widget",
]

logger = logging.getLogger(__name__)


THREE_DIMENSIONAL = 3  # ZYX
FOUR_DIMENSIONAL = 4  # TZYX
DEFAULT_PIXEL_SPACING = (1e-6, 1e-6)  # 1 um


def create_projection_widget() -> magicgui.widgets.Container:
    """Create a widget to project a 4d timeseries (TZYX) along the z dimension"""

    projection_widget = epitools.widgets.create_projection_widget()

    # Project the timeseries when pressing the 'Run' button
    projection_widget.run.changed.connect(
        lambda: run_projection(
            image=projection_widget.input_image.value,
            smoothing_radius=projection_widget.smoothing_radius.value,
            surface_smoothness=[
                projection_widget.surface_smoothness_1.value,
                projection_widget.surface_smoothness_2.value,
            ],
            cutoff_distance=projection_widget.cutoff_distance.value,
        ),
    )

    return projection_widget


def create_projection_2ch_widget() -> magicgui.widgets.Container:
    """Create a widget to project a 2 channel, 4d timeseries (TZYX)
    along the z dimension based on a reference channel"""

    projection_2ch_widget = epitools.widgets.create_projection_2ch_widget()

    # Project the timeseries when pressing the 'Run' button
    projection_2ch_widget.run.changed.connect(
        lambda: run_projection(
            image=projection_2ch_widget.refchannel.value,
            smoothing_radius=projection_2ch_widget.smoothing_radius.value,
            surface_smoothness=[
                projection_2ch_widget.surface_smoothness_1.value,
                projection_2ch_widget.surface_smoothness_2.value,
            ],
            cutoff_distance=projection_2ch_widget.cutoff_distance.value,
            second_image=projection_2ch_widget.channel.value,
        ),
    )

    return projection_2ch_widget


def run_projection(
    image: napari.layers.Image,
    smoothing_radius,
    surface_smoothness,
    cutoff_distance,
    second_image: napari.layers.Image | None = None,
) -> None:
    """Project a 4d timeseries along the z dimension"""

    "If second_image is not empty, project 2 channels based on reference channel"
    if second_image is not None:
        projected_data_1, projected_data_2 = epitools.analysis.calculate_projection(
            image.data,
            smoothing_radius,
            surface_smoothness,
            cutoff_distance,
            second_image.data,
        )

        viewer = napari.current_viewer()
        viewer.add_image(
            data=projected_data_1,
            name="Projection_ch1",
            scale=image.scale,
            translate=image.translate,
            rotate=image.rotate,
            plane=image.plane,
            metadata=image.metadata,
        )

        viewer.add_image(
            data=projected_data_2,
            name="Projection_ch2",
            scale=image.scale,
            translate=image.translate,
            rotate=image.rotate,
            plane=image.plane,
            metadata=image.metadata,
        )

    else:
        projected_data, _ = epitools.analysis.calculate_projection(
            image.data,
            smoothing_radius,
            surface_smoothness,
            cutoff_distance,
        )

        viewer = napari.current_viewer()
        viewer.add_image(
            data=projected_data,
            name="Projection",
            scale=image.scale,
            translate=image.translate,
            rotate=image.rotate,
            plane=image.plane,
            metadata=image.metadata,
        )


def create_quality_metrics_widget() -> magicgui.widgets.Container:
    """
    Create a widget to calculate quality metrics for a 3D (ZYX) or (TYX) timeseries
    """

    quality_metrics_widget = epitools.widgets.create_quality_metrics_widget()
    napari.current_viewer()

    # Calculate quality metrics when pressing the 'Run' button
    quality_metrics_widget.run.changed.connect(
        lambda: run_quality_metrics(
            image=quality_metrics_widget.input_image.value,
            labels=quality_metrics_widget.input_labels.value,
            percentage_of_zslices=quality_metrics_widget.percentage_of_zslices.value,
            run_metrics=quality_metrics_widget.run_metrics.value,
            show_overlay=quality_metrics_widget.show_overlay.value,
        ),
    )

    # write the cell_statistics to CSV
    quality_metrics_widget.export.changed.connect(
        lambda: export_cell_statistics(
            labels=quality_metrics_widget.input_labels.value,
        ),
    )

    return quality_metrics_widget


def run_quality_metrics(
    image, labels, percentage_of_zslices, run_metrics, show_overlay
) -> None:
    """
    Calculate quality metrics for a 3D (ZYX) or (TYX) timeseries
    """

    pbr = progress(total=3)
    pbr.set_description("Quality metrics calculation in progress")
    pbr.update(0)

    quality_metrics = epitools.analysis.calculate_quality_metrics(
        labels=labels.data,
        percentage_of_zslices=percentage_of_zslices,
    )

    pbr.update(1)

    overlay = _show_overlay(
        labels=labels.data,
        correct_cells=quality_metrics["correct_cells"],
    )

    if show_overlay:
        pbr.set_description("Creating overlay in progress")
        viewer = napari.current_viewer()
        viewer.add_labels(
            data=overlay,
            scale=image.scale,
            translate=image.translate,
            rotate=image.rotate,
            plane=image.plane,
            name="Correct_cells",
        )

    pbr.update(2)

    if run_metrics:
        pbr.set_description("Calculating cell statistics in progress")
        [cell_statistics, graphs] = epitools.analysis.calculate_cell_statistics(
            image=image.data,
            labels=overlay,
            pixel_spacing=image.scale,
            id_cells=quality_metrics["correct_cells"],
        )
        labels.metadata["cell_statistics"] = cell_statistics
        labels.metadata["graphs"] = graphs

    pbr.update(3)

    pbr.set_description("Quality metrics calculation complete")

    # must call pbr.close() when using outside for loop
    # or context manager
    pbr.close()

    # Display quality metrics
    message = (
        f"Quality metrics for '{labels.name}':\n"
        f"Correct cells: {len(quality_metrics['correct_cells'])}\n"
        f"Wrong cells: {len(quality_metrics['wrong_cells'])}\n"
    )
    napari.utils.notifications.show_info(message)


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
    """Segment a 3D timeserise (TZYX) at each frame"""

    seeds_data, labels_data = epitools.analysis.calculate_segmentation(
        projection=image.data,
        spot_sigma=spot_sigma,
        outline_sigma=outline_sigma,
        threshold=threshold,
    )

    viewer = napari.current_viewer()
    viewer.add_labels(
        data=labels_data,
        scale=image.scale,
        translate=image.translate,
        rotate=image.rotate,
        plane=image.plane,
        name="Cells",
    )
    viewer.add_points(
        data=seeds_data,
        name="Seeds",
        size=3 * image.scale[-2:].mean(),
        edge_color="red",
        face_color="red",
        scale=image.scale,
        translate=image.translate,
        rotate=image.rotate,
    )


def create_cell_statistics_widget() -> magicgui.widgets.Container:
    """Create a widget for calculating cell statistics of labelled segmentations."""

    cell_statistics_widget = epitools.widgets.create_cell_statistics_widget()
    viewer = napari.current_viewer()

    # Update cell_statistics when scrolling through frames
    viewer.dims.events.current_step.connect(
        lambda event: _update_cell_statistics(
            layers=viewer.layers,
            frame=event.value[0],  # pass in the time frame
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

    pixel_spacing = (
        image.metadata["yx_spacing"]
        if "yx_spacing" in image.metadata
        else DEFAULT_PIXEL_SPACING
    )

    cell_statistics, graphs = epitools.analysis.calculate_cell_statistics(
        image=image.data,
        labels=labels.data,
        pixel_spacing=pixel_spacing,
        id_cells=None,
    )

    # We will use these to update the cell stats at each frame
    labels.metadata["cell_statistics"] = cell_statistics
    labels.metadata["graphs"] = graphs

    # confirm calculation finished
    message = f"Finished calculating cell statistics for '{labels.name}'"
    napari.utils.notifications.show_info(message)

    # TODO: too many regionprops options crashes the client, is there a way to fix this?
    # https://github.com/epitools/epitools/issues/96
    """
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
    """


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

    # Convert np.int64 to regular integers
    for frame_stats in cell_statistics:
        for stat in frame_stats:
            if isinstance(frame_stats[stat], list) and "id_neighbours" in stat:
                frame_stats[stat] = [
                    int(x) if isinstance(x, np.integer) else x
                    for x in frame_stats[stat]
                ]

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
            "the selected Labels layer has no cell statistics."
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


def _show_overlay(
    labels: napari.types.LabelsData,
    correct_cells: list[int],
) -> napari.types.LabelsData:
    """
    Create an overlay of correct cells in a Labels layer

    Args:
        labels : napari.types.LabelsData
            Labelled image
        correct_cells : list[int]
            List of cell ids to highlight in the overlay

    Returns:
        napari.types.LabelsData
            Overlay of correct cells
    """

    # Create overlay
    overlay = np.zeros_like(labels)
    for cell_id in correct_cells:
        overlay[labels == cell_id] = cell_id

    return overlay
