"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/plugins/guides.html?#readers
"""

import pathlib

import numpy as np
import PartSegCore.analysis.load_functions
import PartSegCore.napari_plugins.loader


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    supported_extensions = [
        ".tif",
        ".tiff",
    ]
    extension = pathlib.Path(path).suffix.lower()
    if extension not in supported_extensions:
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of layer.
        Both "meta", and "layer_type" are optional. napari will default to
        layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path

    # load all files using PartSeg
    loader = PartSegCore.analysis.load_functions.LoadStackImage()
    image_stack = loader.load(load_locations=paths)

    image_layers = PartSegCore.napari_plugins.loader.project_to_layers(
        project_info=image_stack,
    )

    for layer in image_layers:
        layer_data, layer_kwargs, layer_type = layer
        layer_kwargs["metadata"] = {
            "scale": np.asarray(
                layer_kwargs.pop("scale")
            ),  # don't scale the image in the viewer
            "spacing": np.asarray(
                image_stack.image.spacing
            ),  # we need this for regionprops
        }

    return image_layers
