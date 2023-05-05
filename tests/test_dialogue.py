from unittest.mock import Mock

import magicgui.widgets

from epitools.widgets.dialogue import select_option


def test_select_option(monkeypatch):
    container = magicgui.widgets.Container()

    mock = Mock()

    def _exec(self, **k):
        mock()
        assert self.native.parent() is container.native
        return True

    monkeypatch.setattr(magicgui.widgets.bases.DialogWidget, "exec", _exec)
    selected = select_option(
        options=["one", 2, "C"],
        title="Select option",
        prompt="Please select an option from the list",
        parent=container,
    )

    # The default (first) option is returned
    assert selected == "one"
    mock.assert_called_once()

    mock.reset_mock()
    selected = select_option(
        options=[2, "C", "three"],
        title="Select option",
        prompt="Please select an option from the list",
        parent=container,
    )
    assert selected == 2  # noqa: PLR2004
    mock.assert_called_once()
