from typing import List

from magicgui import magic_factory, widgets
from napari.qt.threading import FunctionWorker, thread_worker
from napari.types import ImageData, LayerDataTuple
from napari.utils.notifications import show_info

from napari_epitools.analysis import (
    calculate_projection,
    skeletonize,
    thresholded_local_minima_seeded_watershed,
)

# Rendering properties of seeds
SEED_SIZE = 3
SEED_EDGE_COLOR = "red"
SEED_FACE_COLOR = "red"


@magic_factory(
    pbar={"visible": False, "max": 0, "label": "working..."},
    smoothing_radius={
        "widget_type": "FloatSlider",
        "min": 0.0,
        "max": 2.0,
        "step": 0.1,
    },
    surface_smoothness_1={
        "widget_type": "Slider",
        "min": 0,
        "max": 10,
        "step": 1,
    },
    surface_smoothness_2={
        "widget_type": "Slider",
        "min": 0,
        "max": 10,
        "step": 1,
    },
    cut_off_distance={
        "widget_type": "Slider",
        "min": 0,
        "max": 5,
        "step": 1,
    },
    spot_sigma={
        "widget_type": "FloatSlider",
        "min": 0,
        "max": 20,
        "step": 0.1,
    },
    outline_sigma={
        "widget_type": "FloatSlider",
        "min": 0,
        "max": 20,
        "step": 0.1,
    },
    threshold={
        "widget_type": "FloatSlider",
        "min": 0,
        "max": 100,
        "step": 1,
    },
)
def epitools_widget(
    pbar: widgets.ProgressBar,
    input_image: ImageData,
    smoothing_radius: float = 0.2,
    surface_smoothness_1: int = 5,
    surface_smoothness_2: int = 5,
    cut_off_distance: int = 2,
    spot_sigma: float = 3,
    outline_sigma: float = 0,
    threshold: float = 20,
) -> FunctionWorker[List[LayerDataTuple]]:
    """Z projection using image interpolation.
    Args:
        pbar:
            Progressbar widget
        input_image:
            Numpy ndarray representation of 3D image stack
        smoothing_radius:
            Kernel radius for gaussian blur to apply before estimating the surface.
        surface_smoothness_1:
            Surface smoothness for 1st griddata estimation, larger means smoother.
        surface_smoothness_2:
            Surface smoothness for 3nd griddata estimation, larger means smoother.
        cut_off_distance:
            Cutoff distance in z-planes from the 1st estimated surface.
    Raises:
        ValueError: When no image is loaded.
    Returns:
        Projected image as napari Image layer.
    """
    if input_image is None:
        pbar.hide()
        show_info("Load an image first")
        return

    @thread_worker(connect={"returned": pbar.hide})
    def run() -> List[LayerDataTuple]:
        proj = calculate_projection(
            input_image,
            smoothing_radius,
            surface_smoothness_1,
            surface_smoothness_2,
            cut_off_distance,
        )
        proj_layer = (proj, {"name": "Projection"}, "image")

        seeds, labels = thresholded_local_minima_seeded_watershed(
            proj, spot_sigma, outline_sigma, threshold
        )
        lines = skeletonize(labels)
        seeds_layer = (
            seeds,
            {
                "name": "Seeds",
                "size": SEED_SIZE,
                "edge_color": SEED_EDGE_COLOR,
                "face_color": SEED_FACE_COLOR,
            },
            "points",
        )
        labels_layer = (labels, {"name": "Segmentation"}, "labels")
        lines_layer = (lines, {"name": "Skeletonize"}, "labels")
        return [proj_layer, seeds_layer, labels_layer, lines_layer]

    pbar.show()
    return run()


# @magic_factory
# def segmentation_widget(
#     input_image: ImageData,
#     spot_sigma: Annotated[float, {"min": 0, "max": 20, "step": 0.1}] = 3,
#     outline_sigma: Annotated[float, {"min": 0, "max": 20, "step": 0.1}] = 0,
#     threshold: Annotated[float, {"min": 0, "max": 100, "step": 1}] = 20,
# ) -> LabelsData:

#     return thresholded_local_minima_seeded_watershed(
#         input_image, spot_sigma, outline_sigma, threshold
#     )
