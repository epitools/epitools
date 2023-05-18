[![Licence](https://img.shields.io/pypi/l/epitools.svg?color=green)](https://raw.githubusercontent.com/epitools/epitools/main/LICENCE.md)
[![PyPI](https://img.shields.io/pypi/v/epitools.svg?color=green)](https://pypi.org/project/epitools)
[![Python Version](https://img.shields.io/pypi/pyversions/epitools.svg?color=green)](https://python.org)
[![tests](https://github.com/epitools/epitools/actions/workflows/test.yml/badge.svg)](https://github.com/epitools/epitools/actions/workflows/test.yml)
[![Documentation](https://readthedocs.org/projects/epitools/badge/?version=latest)](https://epitools.readthedocs.io/en/latest/?badge=latest)
[![coverage](https://coveralls.io/repos/github/epitools/epitools/badge.svg?branch=main)](https://coveralls.io/github/epitools/epitools?branch=main)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/epitools)](https://napari-hub.org/plugins/epitools)

# Welcome to EpiTools!

EpiTools is a Python package and associated [napari](https://napari.org/stable/) plugin to extract the membrane signal from epithelial tissues and analyze it with the aid of computer vision.

The development of EpiTools was inspired by the challenges in analyzing time-lapses of growing Drosophila imaginal discs.

The folded morphology, the very small apical cell surfaces and the long time series required a new automated cell recognition to accurately study growth dynamics.

## Installation

First, install [napari](https://napari.org/index.html#installation).

The recommended way to install `EpiTools` is via
[pip](https://pypi.org/project/pip)

```sh
python -m pip install epitools
```

To install the latest development version of `EpiTools` clone this repository
and run

```sh
python -m pip install -e .
```

If working on Apple Silicon make sure to also install the following package from
[conda-forge](https://conda-forge.org).

```sh
conda install -c conda-forge pyqt
```

### Recommended Companion Napari Plugins

To also install the recommended plugins for the `EpiTools` workflow run

```sh
python -m pip install epitools[wf]
```

and

```sh
python -m pip install -e .[wf]
```

If working on Apple Silicon make sure to also install the following package from
[conda-forge](https://conda-forge.org)

```sh
conda install -c conda-forge cvxopt
```

which is required for [btrack](https://github.com/quantumjot/btrack).

## Issues

If you encounter any problems, please
[file an issue](https://github.com/epitools/epitools/issues) along with a
detailed description.

## Contributing

Contributions are very welcome. Tests can be run with [tox](https://tox.wiki),
please ensure the coverage at least stays the same before you submit a pull request.
