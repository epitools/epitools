"""
This script does the following:
    - Load a 4D dataset
    - Create 2D projections of the images into the stack into 2D
    - Segment the images
    - Calculate regionprops of the segmented images
    - Load the data into a napari viewer

"""

import pathlib

import napari
import skimage.io

from napari_epitools.analysis import (
    calculate_projection,
    calculate_regionprops,
    calculate_segmentation,
)

viewer = napari.Viewer()

# Add widgets to the viewer
_, projection_widget = viewer.window.add_plugin_dock_widget(
    plugin_name="napari-epitools",
    widget_name="Projection (selective plane)",
)

_, segmentation_widget = viewer.window.add_plugin_dock_widget(
    plugin_name="napari-epitools",
    widget_name="Segmentation (local minima seeded watershed)",
)
_, regionprops_widget = viewer.window.add_plugin_dock_widget(
    plugin_name="napari-epitools",
    widget_name="RegionProps (cell statistics)",
)

# This is how the original 4D image was cropped
# image_path = pathlib.Path("sample_data") / "4d" / "181210_DM_CellMOr_subsub_reg_decon-1.tif"  # noqa: E501
# image = skimage.io.imread(image_path)
# image = image[:, :, 300:700, 400:750]

image_path = (
    pathlib.Path("sample_data")
    / "4d"
    / "181210_DM_CellMOr_subsub_reg_decon-1_cropped.tif"
)
image = skimage.io.imread(image_path)


# Project, segment, and calculate region props
projected_image = calculate_projection(
    input_image=image,
    smoothing_radius=1,
    surface_smoothness_1=5,
    surface_smoothness_2=5,
    cut_off_distance=3,
)

seeds, labels = calculate_segmentation(
    projection=projected_image,
    spot_sigma=10,
    outline_sigma=3,
    threshold=3,
)

regionprops = calculate_regionprops(
    image=projected_image,
    labels=labels,
)

# Add results to viewer
viewer.add_image(projected_image, name="Projected")
viewer.add_labels(labels, name="Cells")
viewer.add_points(seeds, name="Seeds", size=3, edge_color="red", face_color="red")

# The napari event loop needs to be run under here to allow the window
# to be spawned from a Python script
napari.run()
