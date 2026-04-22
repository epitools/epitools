from pathlib import Path

import tomllib


def test_vertexmodel_dependency_is_optional():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject_data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    dependencies = pyproject_data["project"]["dependencies"]
    optional_dependencies = pyproject_data["project"]["optional-dependencies"]

    assert "napari-pyVertexModel" not in dependencies
    assert "napari-pyVertexModel" in optional_dependencies["vertexmodel"]
