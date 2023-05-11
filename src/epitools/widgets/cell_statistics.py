from __future__ import annotations

import magicgui.widgets
import napari
from magicgui.widgets import Widget

__all__ = [
    "create_cell_statistics_widget",
]


def create_cell_statistics_widget() -> magicgui.widgets.Container:
    """Create a widget for calculating cell statistics of labelled segmentations."""

    cell_statistics_widgets = _create_cell_statistics_widgets()
    colour_labels_widgets = _create_colour_labels_widgets()

    return magicgui.widgets.Container(
        widgets=[*cell_statistics_widgets, *colour_labels_widgets],
        scrollable=False,
    )


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


def _create_colour_labels_widgets() -> list[Widget]:
    """Create widgets for colouring Labels layers by their featuers"""

    create_colour_labels_heading = magicgui.widgets.create_widget(
        label="<b>Colour by statistic</b>",  # bold label
        widget_type="Label",
        gui_only=True,
    )

    labels_tooltip = "Select a 'Labels' layer to set colourmap for"
    labels = magicgui.widgets.create_widget(
        annotation=napari.layers.Labels,
        name="colourmap_input_labels",
        label="labels",
        options={"tooltip": labels_tooltip},
    )

    statisitc_tooltip = "Select which statistic to use for colouring the 'Labels'"
    statistic = magicgui.widgets.create_widget(
        name="colourmap_statistic",
        value="id",
        label="statistic",
        widget_type="ComboBox",
        options={
            "choices": ["id", "area", "perimeter", "orientation", "neighbours"],
            "tooltip": statisitc_tooltip,
        },
    )

    lower_limit_tooltip = "Set the lower limit for the colourmap"
    lower_limit = magicgui.widgets.create_widget(
        value=0.0,
        name="colourmap_lower_limit",
        label="lower limit",
        widget_type="FloatSpinBox",
        options={
            "tooltip": lower_limit_tooltip,
            "min": -3.4 * 10**38,
            "max": 3.4 * 10**38,
        },
    )

    upper_limit_tooltip = "Set the upper limit for the colourmap"
    upper_limit = magicgui.widgets.create_widget(
        value=100.0,
        name="colourmap_upper_limit",
        label="upper limit",
        widget_type="FloatSpinBox",
        options={
            "tooltip": upper_limit_tooltip,
            "min": -3.4 * 10**38,
            "max": 3.4 * 10**38,
        },
    )

    auto_limits_tooltip = "Automatically set colourmap limits based on feature values."
    auto_limits = magicgui.widgets.create_widget(
        value=False,
        name="colourmap_autolimits",
        label="Auto-limits",
        widget_type="CheckBox",
        options={"tooltip": auto_limits_tooltip},
    )

    create_colourmap_tooltip = "Create colourmap for the selected Labels layer"
    create_colourmap = magicgui.widgets.create_widget(
        name="colourmap_create_button",
        label="Create colourmap",
        widget_type="PushButton",
        options={"tooltip": create_colourmap_tooltip},
    )

    return [
        create_colour_labels_heading,
        labels,
        statistic,
        lower_limit,
        upper_limit,
        auto_limits,
        create_colourmap,
    ]
