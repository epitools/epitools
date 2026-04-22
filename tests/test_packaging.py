import importlib.metadata


def test_vertexmodel_dependency_is_optional():
    requirements = importlib.metadata.requires("epitools") or []

    core_deps = [r for r in requirements if "extra ==" not in r]
    vertexmodel_deps = [r for r in requirements if 'extra == "vertexmodel"' in r]

    assert not any("napari-pyvertexmodel" in r.lower() for r in core_deps)
    assert any("napari-pyvertexmodel" in r.lower() for r in vertexmodel_deps)
