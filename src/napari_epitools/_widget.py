from typing import List

import numpy as np
from magicgui import magic_factory, widgets
from napari import current_viewer
from napari.layers import Image
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

# Defaults
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


@magic_factory(
    pbar={"visible": False, "max": 0, "label": "working..."},
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
) -> FunctionWorker[LayerDataTuple]:
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
    def run() -> LayerDataTuple:
        proj = calculate_projection(
            input_image,
            smoothing_radius,
            surface_smoothness_1,
            surface_smoothness_2,
            cut_off_distance,
        )
        return (proj, {"name": PROJECTION_LAYER_NAME}, "image")

    pbar.show()
    return run()


# this is experimental - looking at how to integrate
# with napari-assistant
# def selective_plane_projection(
#     image: ImageData,
#     smoothing_radius: float = 0.2,
#     surface_smoothness_1: int = 5,
#     surface_smoothness_2: int = 5,
#     cut_off_distance: int = 2,
# ) -> LayerDataTuple:

#     if image is None:
#         show_info("Load an image first")
#         return

#     proj = calculate_projection(
#         image.astype(float),
#         smoothing_radius,
#         surface_smoothness_1,
#         surface_smoothness_2,
#         cut_off_distance,
#     )
#     # @thread_worker
#     # def run() -> LayerDataTuple:
#     #     proj = calculate_projection(
#     #         image,
#     #         smoothing_radius,
#     #         surface_smoothness_1,
#     #         surface_smoothness_2,
#     #         cut_off_distance,
#     #     )
#     #     # return (proj, {"name": PROJECTION_LAYER_NAME}, "image")
#     #     return proj

#     return (proj, {"name": PROJECTION_LAYER_NAME}, "image")


@magic_factory(
    pbar={"visible": False, "max": 0, "label": "working..."},
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
) -> FunctionWorker[List[LayerDataTuple]]:

    if input_image is None or input_image.data.ndim > 2:
        pbar.hide()
        show_info("Load a 2D image first")
        return

    @thread_worker(connect={"returned": pbar.hide})
    def run() -> List[LayerDataTuple]:
        seeds, labels = thresholded_local_minima_seeded_watershed(
            input_image, spot_sigma, outline_sigma, threshold
        )
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
        labels_layer = (labels, {"name": "Segmentation"}, "labels")
        lines = skeletonize(labels)
        lines_layer = (lines, {"name": "Skeletonize"}, "labels")
        return [seeds_layer, labels_layer, lines_layer]

    pbar.show()
    return run()


def epitools_widget() -> widgets.Container:
    """
    Widgets for epitools
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

    def _add_projection(projection):
        print("Finished projection calc")
        try:
            widget.viewer.layers[PROJECTION_LAYER_NAME].data = projection
        except KeyError:
            widget.viewer.add_image(projection, name=PROJECTION_LAYER_NAME)
        finally:
            axes = [0, 1]
            if widget.viewer.dims.ndim == 3:
                axes = [0]

            for axis in axes:
                widget.viewer.dims.set_current_step(axis, 0)

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
        worker.returned.connect(_add_projection)
        worker.start()

    def _add_segmentation(segmentation):
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
            axes = [0, 1]
            if widget.viewer.dims.ndim == 3:
                axes = [0]

            for axis in axes:
                widget.viewer.dims.set_current_step(axis, 0)

    @thread_worker
    def _calculate_segmentation():
        # proj = seg_image.value.data.astype(float)
        # t_size = proj.shape[0]
        # if proj.ndim == 2:
        #     proj = np.expand_dims(proj, axis=0)
        #     t_size = 1

        proj = seg_image.value.data.astype(float)
        print(f"{proj.shape=}")
        t_size = widget.viewer.dims.nsteps[0]
        print(f"{widget.viewer.dims.nsteps=}")
        print(f"{t_size=}")
        seg_seeds = []
        seg_labels = []
        for t in range(t_size):
            print(f"{t=}")
            widget.viewer.dims.set_current_step(0, t)
            seeds, labels = thresholded_local_minima_seeded_watershed(
                proj[t],
                spot_sigma.value,
                outline_sigma.value,
                threshold.value,
            )

            # seeds needs to include time dimension
            for s in seeds:
                s.insert(0, float(t))

            seg_seeds.append(np.array(seeds))
            seg_labels.append(labels)

        return np.vstack(seg_seeds), np.stack(seg_labels)

    @widget.run_seg_button.clicked.connect
    def run_segmentation() -> None:
        worker = _calculate_segmentation()
        worker.returned.connect(_add_segmentation)
        worker.start()

    return widget
