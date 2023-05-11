:orphan:

.. _napari_plugin:

Using the ``EpiTools`` plugin for ``napari``
============================================

The following steps are part of the typical ``EpiTools`` workflow.

1. **Projection**

The following video demonstrates how the projection step can be performed.

.. video:: ../_static/epitools-projection.mp4
    :alt: EpiTools Projection Step
    :height: 400

2. **Segmentation**

The following video demonstrates how the segmentation step can be performed.

.. video:: ../_static/epitools-segmentation.mp4
    :alt: EpiTools Segmentation Step
    :height: 400

3. **Correcting Segmentation**

If the segmentation output needs correction we would recommend the
`segment blobs and things with membranes <https://www.napari-hub.org/plugins/napari-segment-blobs-and-things-with-membranes#manual-split-and-merge-labels>`_
plugin. This allows one to manually split and merge labels.

4. **Tracking**

To perform tracking we would recommend the
`brack <https://www.napari-hub.org/plugins/btrack>`_ plugin.

5. **Correcting Tracking**

This is not currently possible with ``EpiTools``/``btrack``.

6. **Cell Statistics**

``EpiTools`` includes methods to extract a variety of statistics from the cells.

.. TODO: restore this video if tooltips are reinstated
.. https://github.com/epitools/epitools/issues/96
..
    This video demonstrates how to view the statistics of a given cell via
    tooltips.

    .. video:: ../_static/epitools-cell-stats.mp4
        :alt: EpiTools Cell Statistics Tooltips
        :height: 400

This video shows how to have a colour map for a given cell property,
i.e. neighbours.

.. video:: ../_static/epitools-colourmaps.mp4
    :alt: EpiTools Cell Statistics Colour Maps
    :height: 400

This video demonstrates how to export and save the cell statistics
output.

.. video:: ../_static/epitools-export.mp4
    :alt: EpiTools Cell Statistics Export
    :height: 400
