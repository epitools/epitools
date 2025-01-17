from __future__ import annotations

import magicgui.widgets
import napari

__all__ = [
    "create_quality_metrics_widget",
]


def create_quality_metrics_widget() -> magicgui.widgets.Container:
    """
    Create a widget for quality metrics
    """
    image_tooltip = "Select an 'Image' layer"
    image = magicgui.widgets.create_widget(
        annotation=napari.layers.Image,
        name="input_image",
        label="image",
        options={"tooltip": image_tooltip},
    )

    labels = magicgui.widgets.create_widget(
        annotation=napari.layers.Labels,
        name="labels",
        label="labels",
        options={"tooltip": "Select a 'Labels' layer"},
    )

    percentage_of_zslices = magicgui.widgets.create_widget(
        name="percentage_of_zslices",
        label="Percentage of z-slices",
        widget_type="FloatSpinBox",
        options={"tooltip": "Percentage of z-slices to use", "max": 100.0, "min": 5.0},
        value=80.0,
    )

    run_metrics = magicgui.widgets.create_widget(
        name="run_metrics",
        label="Run metrics",
        widget_type="CheckBox",
        options={"tooltip": "Compute cell statistics"},
    )

    show_overlay = magicgui.widgets.create_widget(
        name="show_overlay",
        label="Show overlay",
        widget_type="CheckBox",
        options={"tooltip": "Show overlay"},
    )

    run_buttom = magicgui.widgets.create_widget(
        name="run",
        label="Run",
        widget_type="PushButton",
        options={"tooltip": "Compute quality metrics"},
    )

    return magicgui.widgets.Container(
        widgets=[
            image,
            labels,
            percentage_of_zslices,
            run_metrics,
            show_overlay,
            run_buttom,
        ],
        scrollable=False,
    )
