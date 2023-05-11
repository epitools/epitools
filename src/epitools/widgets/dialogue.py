from __future__ import annotations

from typing import Any

import magicgui.widgets

__all__ = [
    "select_option",
]


def select_option(
    options,
    *,
    title: str = "",
    prompt: str = "",
    parent: Any | None = None,
) -> str | None:
    """Show a dialog with a set of options and request the user to select one.

    Dialog is modal and immediately blocks execution until user closes it.
    If the dialog is accepted, the selected value is returned, otherwise returns `None`.

    See also the docstring of :func:`magicgui.widgets.create_widget` for more
    information.

    Parameters
    ----------
    options : List[Any]
        List of values to display in a ComboBox
    title : str
        An optional label to use for the window title. Defaults to an empty string.
    prompt : str
        An optional instruction / explanation for the user. This will be displayed below
        the list of options. Defaults to an empty string.
    parent : Widget, optional
        An optional parent widget, by default None.
        The dialog will inherit style from the parent object.
        CAREFUL: if a parent is set, and subsequently deleted, this widget will likely
        be deleted as well (depending on the backend), and will no longer be usable.

    Returns
    -------
    Optional[Any]
        Selected value if accepted, or ``None`` if cancelled.
    """

    options_box = magicgui.widgets.ComboBox(choices=options)
    widgets = (
        [options_box, magicgui.widgets.Label(value=prompt)] if prompt else [options_box]
    )
    dialogue = magicgui.widgets.Dialog(widgets=widgets, parent=parent)
    dialogue.native.setWindowTitle(title)
    selected = dialogue.exec()

    return options_box.value if selected else None
