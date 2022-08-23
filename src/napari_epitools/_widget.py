from magicgui import widgets
from napari import current_viewer
from napari.layers import Image
from napari.qt.threading import thread_worker

from napari_epitools.analysis import (
    calculate_projection,
    skeletonize,
    thresholded_local_minima_seeded_watershed,
)

# Rendering properties of seeds
SEED_SIZE = 3
SEED_EDGE_COLOR = "red"
SEED_FACE_COLOR = "red"

# Defaults
SMOOTHING_RADIUS = 0.2
SMOOTHING_RADIUS_MIN = 0.0
SMOOTHING_RADIUS_MAX = 2.0
SMOOTHING_RADIUS_STEP = 0.1

SURFACE_SMOOTHNESS_1 = 5
SURFACE_SMOOTHNESS_1_MIN = 0
SURFACE_SMOOTHNESS_1_MAX = 10
SURFACE_SMOOTHNESS_1_STEP = 1

SURFACE_SMOOTHNESS_2 = 5
SURFACE_SMOOTHNESS_2_MIN = 0
SURFACE_SMOOTHNESS_2_MAX = 10
SURFACE_SMOOTHNESS_2_STEP = 1

CUT_OFF_DISTANCE = 2
CUT_OFF_DISTANCE_MIN = 0
CUT_OFF_DISTANCE_MAX = 5
CUT_OFF_DISTANCE_STEP = 1

SPOT_SIGMA = 3
SPOT_SIGMA_MIN = 0
SPOT_SIGMA_MAX = 20
SPOT_SIGMA_STEP = 0.1

OUTLINE_SIGMA = 0
OUTLINE_SIGMA_MIN = 0
OUTLINE_SIGMA_MAX = 20
OUTLINE_SIGMA_STEP = 0.1

THRESHOLD = 20
THRESHOLD_MIN = 0
THRESHOLD_MAX = 255
THRESHOLD_STEP = 1


def epitools_widget() -> widgets.Container:
    """
    Widgets for epitools
    """
    # Layout for projection
    input_image = widgets.create_widget(annotation=Image, label="Image:")
    smoothing_radius = widgets.FloatSlider(
        name="smoothing_radius",
        value=SMOOTHING_RADIUS,
        min=SMOOTHING_RADIUS_MIN,
        max=SMOOTHING_RADIUS_MAX,
        step=SMOOTHING_RADIUS_STEP,
    )
    surface_smoothness_1 = widgets.Slider(
        name="surface_smoothness_1",
        value=SURFACE_SMOOTHNESS_1,
        min=SURFACE_SMOOTHNESS_1_MIN,
        max=SURFACE_SMOOTHNESS_1_MAX,
        step=SURFACE_SMOOTHNESS_1_STEP,
    )
    surface_smoothness_2 = widgets.Slider(
        name="surface_smoothness_2",
        value=SURFACE_SMOOTHNESS_2,
        min=SURFACE_SMOOTHNESS_2_MIN,
        max=SURFACE_SMOOTHNESS_2_MAX,
        step=SURFACE_SMOOTHNESS_2_STEP,
    )
    cut_off_distance = widgets.Slider(
        name="cutoff_distance",
        value=CUT_OFF_DISTANCE,
        min=CUT_OFF_DISTANCE_MIN,
        max=CUT_OFF_DISTANCE_MAX,
        step=CUT_OFF_DISTANCE_STEP,
    )
    run_proj_button = widgets.PushButton(
        name="run_proj_button", label="Run Projection"
    )

    # Layout for segmentation
    seg_image = widgets.create_widget(
        annotation=Image, label="Projected Image:"
    )
    spot_sigma = widgets.FloatSlider(
        name="spot_sigma",
        value=SPOT_SIGMA,
        min=SPOT_SIGMA_MIN,
        max=SPOT_SIGMA_MAX,
        step=SPOT_SIGMA_STEP,
    )
    outline_sigma = widgets.FloatSlider(
        name="outline_sigma",
        value=OUTLINE_SIGMA,
        min=OUTLINE_SIGMA,
        max=OUTLINE_SIGMA,
        step=OUTLINE_SIGMA_STEP,
    )
    threshold = widgets.Slider(
        name="threshold",
        value=THRESHOLD,
        min=THRESHOLD_MIN,
        max=THRESHOLD_MAX,
        step=THRESHOLD_STEP,
    )

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

    def _add_projection(projection):
        widget.viewer.add_image(projection, name="Projection")

    @thread_worker
    def _calculate_projection():
        return calculate_projection(
            input_image.value.data,
            smoothing_radius.value,
            surface_smoothness_1.value,
            surface_smoothness_2.value,
            cut_off_distance.value,
        )

    @widget.run_proj_button.clicked.connect
    def run_projection() -> None:
        worker = _calculate_projection()
        worker.returned.connect(_add_projection)
        worker.start()

    def _add_segmentation(segmentation):
        seeds, labels, lines = segmentation
        widget.viewer.add_points(
            seeds,
            name="Seed Points",
            size=SEED_SIZE,
            edge_color=SEED_EDGE_COLOR,
            face_color=SEED_FACE_COLOR,
        )
        widget.viewer.add_labels(labels, name="Cells")
        widget.viewer.add_labels(lines, name="Cell Outlines")

    @thread_worker
    def _calculate_segmentation():
        seeds, labels = thresholded_local_minima_seeded_watershed(
            seg_image.value.data,
            spot_sigma.value,
            outline_sigma.value,
            threshold.value,
        )
        lines = skeletonize(labels)

        return seeds, labels, lines

    @widget.run_seg_button.clicked.connect
    def run_segmentation() -> None:
        worker = _calculate_segmentation()
        worker.returned.connect(_add_segmentation)
        worker.start()

    return widget
