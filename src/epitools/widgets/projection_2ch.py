import magicgui.widgets
import napari

__all__ = [
    "create_projection_2ch_widget",
]


def create_projection_2ch_widget() -> magicgui.widgets.Container:
    """Create a widget for projecting a 2 channel, 3d timeseries to a
    2d timeseries based on a reference channel"""

    refchannel_tooltip = (
        "Select a 'Reference' channel to project along the z-dimension."
    )
    refchannel = magicgui.widgets.create_widget(
        annotation=napari.layers.Image,
        name="refchannel",
        label="reference channel",
        options={"tooltip": refchannel_tooltip},
    )

    channel_tooltip = "Select a second channel to project along the \
    z-dimension based on the 'Reference'."
    channel = magicgui.widgets.create_widget(
        annotation=napari.layers.Image,
        name="channel",
        label="second channel",
        options={"tooltip": channel_tooltip},
    )

    smoothing_radius_tooltip = (
        "Kernel radius for gaussian blur to apply before estimating the surface."
    )
    smoothing_radius = magicgui.widgets.create_widget(
        value=1,
        name="smoothing_radius",
        label="smoothing radius",
        widget_type="FloatSpinBox",
        options={
            "tooltip": smoothing_radius_tooltip,
        },
    )

    surface_smoothness_1_tooltip = (
        "Surface smoothness for 1st griddata estimation, larger means smoother."
    )
    surface_smoothness_1 = magicgui.widgets.create_widget(
        value=5,
        name="surface_smoothness_1",
        label="surface smoothness 1",
        widget_type="SpinBox",
        options={
            "tooltip": surface_smoothness_1_tooltip,
        },
    )

    surface_smoothness_2_tooltip = (
        "Surface smoothness for 2nd griddata estimation, larger means smoother."
    )
    surface_smoothness_2 = magicgui.widgets.create_widget(
        value=5,
        name="surface_smoothness_2",
        label="surface smoothness 2",
        widget_type="SpinBox",
        options={
            "tooltip": surface_smoothness_2_tooltip,
        },
    )

    cutoff_distance_tooltip = (
        "Cutoff distance in z-planes from the 1st estimated surface."
    )
    cutoff_distance = magicgui.widgets.create_widget(
        value=3,
        name="cutoff_distance",
        label="z cutoff distance",
        widget_type="SpinBox",
        options={
            "tooltip": cutoff_distance_tooltip,
        },
    )

    run_button_tooltip = "Perform the projection"
    run_button = magicgui.widgets.create_widget(
        name="run",
        label="Run",
        widget_type="PushButton",
        options={"tooltip": run_button_tooltip},
    )

    return magicgui.widgets.Container(
        widgets=[
            refchannel,
            channel,
            smoothing_radius,
            surface_smoothness_1,
            surface_smoothness_2,
            cutoff_distance,
            run_button,
        ],
        scrollable=False,
    )
