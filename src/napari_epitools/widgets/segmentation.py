import magicgui.widgets
import napari

__all__ = [
    "create_segmentation_widget",
]


def create_segmentation_widget() -> magicgui.widgets.Container:
    """Create a widget for segmenting cells in a 3d (TYZ) image"""

    image_tooltip = "Select an 'Image' layer to segment"
    image = magicgui.widgets.create_widget(
        annotation=napari.layers.Image,
        name="input_image",
        label="image",
        options={"tooltip": image_tooltip},
    )

    spot_sigma_tooltip = "Controls how close segmented cells can be."
    spot_sigma = magicgui.widgets.create_widget(
        value=10.0,
        name="spot_sigma",
        label="spot sigma",
        widget_type="FloatSpinBox",
        options={
            "tooltip": spot_sigma_tooltip,
        },
    )

    outline_sigma_tooltip = "Controls how precisely segmented cells are outlined."
    outline_sigma = magicgui.widgets.create_widget(
        value=3.0,
        name="outline_sigma",
        label="outline sigma",
        widget_type="FloatSpinBox",
        options={
            "tooltip": outline_sigma_tooltip,
        },
    )

    threshold_tooltip = "Cells with an average intensity below `threshold` are ignored."
    threshold = magicgui.widgets.create_widget(
        value=3.0,
        name="threshold",
        label="threshold sigma",
        widget_type="FloatSpinBox",
        options={
            "tooltip": threshold_tooltip,
        },
    )

    run_button_tooltip = "Perform the segmentation"
    run_button = magicgui.widgets.create_widget(
        name="run",
        label="Run",
        widget_type="PushButton",
        options={"tooltip": run_button_tooltip},
    )

    return magicgui.widgets.Container(
        widgets=[
            image,
            spot_sigma,
            outline_sigma,
            threshold,
            run_button,
        ],
        scrollable=False,
    )
