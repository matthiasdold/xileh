# xileh

![CI](https://github.com/matthiasdold/xileh/workflows/CI%20Testing/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/matthiasdold/xileh/branch/main/graph/badge.svg)](https://codecov.io/gh/matthiasdold/xileh)
![Tests](https://img.shields.io/badge/tests-46%20passed-brightgreen)
![Python](https://img.shields.io/badge/python-3.11+-blue)

A lightweight, composable pipeline abstraction built around a hierarchical
data container.

![xileh_gif](./assets/xileh_2022-05-02_16-35_small.gif)

## Overview

`xileh` abstracts complex processing pipelines, motivated by `sklearn.Pipeline`
but with data capacities going beyond a feature matrix `X` and target `y`. It
is built from two components:

- **`xData`** — a hierarchical container holding *data*, a *header*
  (description / control flags), and *meta* information. It can nest other
  `xData` objects to pass multiple data entities through a pipeline.
- **`xPipeline`** — an ordered set of plain Python functions, each taking and
  returning an `xData` object.

![schematic](assets/schematic.png)

### Why build something new?

Placing libraries on two axes — *built-in functionality* and
*specificity / restrictions* — `xileh` deliberately aims for minimal built-in
functionality and maximal freedom of customization.

![func_spec_landscape](./assets/func_spec_landscape.svg)

`sklearn.Pipeline` is great for ML pipelines but specific about data shape and
the `fit`/`transform` contract. Config-driven tools like `hydra` provide a
single source of truth but impose a format that can hinder rapid prototyping.
`xileh` instead:

- imposes as few restrictions on your workflow as possible (arbitrary data
  objects and plain Python functions),
- integrates easily with a function-based workflow during development,
- provides a single source of truth for the processing,
- enables reuse of whole pipelines by strongly motivating composition.

## Installation

```bash
pip install xileh
```

The bare install is **lite** (numpy only). Optional data backends are
available as extras:

```bash
pip install xileh[pandas]   # pandas DataFrame/Series save & load (parquet)
pip install xileh[polars]   # polars DataFrame/Series save & load (parquet)
```

Saving or loading a container that holds a backend-specific type without the
corresponding extra installed raises an error pointing at the extra to install.

Or from source:

```bash
git clone git@github.com:matthiasdold/xileh.git
cd xileh
pip install -e .
```

`xileh` targets Python 3.11+.

## Quick example

```python
from xileh import xData, xPipeline


def add_ones(pdata, name="ones", size=3):
    pdata.add([1] * size, name=name)
    return pdata


pl = xPipeline("demo")
pl.add_steps(("add_ones", add_ones))

root = xData([], name="root")
pl.eval(root)
```

> The container was renamed from `xPData` to `xData`; `xPData` remains
> available as a backwards-compatible alias.

## Documentation

Full documentation, including a quick-start guide, worked examples, and the API
reference, is available at
**[matthiasdold.github.io/xileh](https://matthiasdold.github.io/xileh/)**.

## License

[MIT](LICENSE)
