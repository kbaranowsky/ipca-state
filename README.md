# ipca-state

[![Documentation](https://github.com/kbaranowsky/ipca-state/actions/workflows/docs.yml/badge.svg)](https://github.com/kbaranowsky/ipca-state/actions/workflows/docs.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`ipca-state` is a Python package for preparing asset-level panel data and estimating Instrumented Principal Component Analysis (IPCA) models, including state-conditioned IPCA extensions.

The package provides tools for:

- preparing characteristics and returns data for IPCA estimation;
- constructing cross-sectionally rank-standardized instruments;
- optionally creating state-interacted characteristics for state-conditioned IPCA;
- estimating restricted and unrestricted IPCA models;
- computing total and predictive \(R^2\);
- running bootstrap tests for pricing errors and characteristic relevance;
- comparing forecast performance with Diebold-Mariano style utilities.

The repository name is `ipca-state`, while the Python import package is `ipca`.

## Documentation

Public API documentation is generated with `pdoc` and deployed through GitHub Pages.

Documentation URL:

```text
https://kbaranowsky.github.io/ipca-state/
```

The documentation workflow is located at:

```text
.github/workflows/docs.yml
```

Documentation is rebuilt automatically on every push to the `main` branch.

## Installation

### Install from GitHub

The package is not assumed to be published on PyPI. Install it directly from GitHub:

```bash
python3 -m pip install git+https://github.com/kbaranowsky/ipca-state.git
```

If you use Anaconda, Spyder, or another environment manager, install into the same Python environment that you use to run your code:

```bash
/path/to/your/python -m pip install git+https://github.com/kbaranowsky/ipca-state.git
```

For example:

```bash
/Users/your-name/anaconda3/bin/python -m pip install git+https://github.com/kbaranowsky/ipca-state.git
```

### Upgrade an existing installation

```bash
python3 -m pip install --upgrade --force-reinstall git+https://github.com/kbaranowsky/ipca-state.git
```

### Development installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/kbaranowsky/ipca-state.git
cd ipca-state
python3 -m pip install -e ".[dev]"
```

Editable installation is recommended when modifying the package source code.

## Requirements

The package requires Python 3.10 or newer.

Core dependencies:

- `numpy`
- `pandas`
- `scipy`
- `joblib`
- `statsmodels`

Development and documentation dependencies are available through optional extras:

```bash
python3 -m pip install -e ".[dev]"
python3 -m pip install -e ".[docs]"
```

## Quick start

```python
import pandas as pd
from ipca import Instruments, IPCA

# Raw asset-level panel data.
# The dataframe must contain:
# - a date column;
# - an asset identifier column;
# - a returns column;
# - one or more firm characteristic columns.
data = pd.read_csv("asset_level_panel.csv")

characteristics = ["bm", "mom12m", "size"]

builder = Instruments(
    data=data,
    characteristics=characteristics,
    returns="ret",
    date="date",
    permno="permno",
)

Z, R = builder.prepare_data(
    addConstant=True,
    filterMonths=True,
    make_state_interactions=False,
    printSummary=True,
)

model = IPCA(
    Z=Z,
    R=R,
    K=3,
    alpha=False,
)

model.fit(
    tol=1e-6,
    max_iter=1000,
    printTime=True,
    printInformation=True,
)

model.short_summary()
```

## State-conditioned IPCA example

To construct state-interacted characteristics, provide a two-column state-variable dataframe and set `make_state_interactions=True`.

The first column should contain dates. The second column should contain the state variable.

```python
import pandas as pd
from ipca import Instruments, IPCA

data = pd.read_csv("asset_level_panel.csv")
state_variable = pd.read_csv("state_variable.csv")

characteristics = ["bm", "mom12m", "size"]

builder = Instruments(
    data=data,
    characteristics=characteristics,
    returns="ret",
    date="date",
    permno="permno",
    state_variable=state_variable,
)

Z_state, R_state = builder.prepare_data(
    addConstant=True,
    filterMonths=True,
    make_state_interactions=True,
    printSummary=True,
)

state_model = IPCA(
    Z=Z_state,
    R=R_state,
    K=3,
    alpha=False,
)

state_model.fit()
state_model.short_summary()
```

## Data format

### Asset-level panel data

The main input to `Instruments` is a `pandas.DataFrame` with at least the following columns:

| Column type | Description |
|---|---|
| Date column | Observation date. Dates are normalized to month-end internally. |
| Asset identifier | Asset identifier, such as `permno`. |
| Returns column | Asset return series. |
| Characteristic columns | Firm-level characteristics used as instruments. |

Example:

| date | permno | ret | bm | mom12m | size |
|---|---:|---:|---:|---:|---:|
| 2020-01-31 | 10001 | 0.012 | 0.45 | 0.08 | 10.2 |
| 2020-01-31 | 10002 | -0.004 | 0.37 | -0.02 | 11.4 |

### Risk-free rate

The optional `risk_free` argument should be a two-column dataframe:

1. date column;
2. risk-free rate column.

The risk-free rate should be expressed in decimal form, not percentage points.

Example:

| date | rf |
|---|---:|
| 2020-01-31 | 0.001 |

If provided, the risk-free rate is subtracted from asset returns.

### State variable

The optional `state_variable` argument should be a two-column dataframe:

1. date column;
2. state variable column.

Example:

| date | state |
|---|---:|
| 2020-01-31 | 0.75 |

When `make_state_interactions=True`, characteristics are interacted with the state variable after cross-sectional standardization.

## Public API

The recommended imports are:

```python
from ipca import Instruments, IPCA
```

For compatibility with the original lowercase class name, the package also exposes:

```python
from ipca import ipca
```

Here, `IPCA` is an alias to the original lowercase `ipca` class.

## Main components

### `Instruments`

Prepares raw asset-level data for IPCA estimation.

Main responsibilities:

- date normalization to month-end;
- optional filtering of months with insufficient valid assets;
- optional filtering of assets with insufficient return history;
- cross-sectional rank standardization of firm characteristics;
- optional risk-free rate subtraction;
- construction of dictionaries of characteristics matrices `Z` and return vectors `R`;
- optional construction of state-interacted characteristics.

### `IPCA`

Estimates IPCA models based on prepared `Z` and `R` inputs.

Main functionality:

- restricted IPCA estimation;
- unrestricted IPCA estimation with pricing-error terms;
- latent and observed factor support;
- alternating least squares estimation;
- total and predictive \(R^2\);
- managed-portfolio fit measures;
- bootstrap tests for pricing errors;
- bootstrap tests for characteristic relevance;
- state-level fit diagnostics;
- forecasting and forecast-comparison utilities.

## Building documentation locally

Install the documentation dependencies:

```bash
python3 -m pip install -e ".[docs]"
```

Build the documentation:

```bash
pdoc ipca -o site
```

Open:

```text
site/index.html
```

## Running tests

Install development dependencies:

```bash
python3 -m pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## Repository structure

```text
ipca-state/
├── .github/
│   └── workflows/
│       └── docs.yml
├── src/
│   └── ipca/
│       ├── __init__.py
│       ├── instruments.py
│       └── ipca.py
├── tests/
│   └── test_imports.py
├── .gitignore
├── LICENSE
├── README.md
└── pyproject.toml
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'ipca'`

The package is installed in a different Python environment than the one running your code.

Check your active Python:

```python
import sys
print(sys.executable)
```

Then install using that exact interpreter:

```bash
/path/to/that/python -m pip install git+https://github.com/kbaranowsky/ipca-state.git
```

### Spyder users

In Spyder, first check the Python executable:

```python
import sys
print(sys.executable)
```

Then install from macOS Terminal using that exact executable:

```bash
/path/from/spyder/sys.executable -m pip install git+https://github.com/kbaranowsky/ipca-state.git
```

Restart the Spyder kernel after installation.

### Import name versus repository name

The GitHub repository is named:

```text
ipca-state
```

The Python package is imported as:

```python
import ipca
```

This is the name exposed by `src/ipca/__init__.py`.

## Research background

This package implements tools for Instrumented Principal Component Analysis following Kelly, Pruitt, and Su (2019), with extensions for state-conditioned specifications developed in the context of the author's thesis research.

## References

Baranowski, K. (2026). *State Conditioning in Instrumented Principal Component Analysis: Do Market States Change Characteristics Based Risk Exposures?* Unpublished Bachelor thesis, Erasmus School of Economics, Erasmus University Rotterdam.

Kelly, B. T., Pruitt, S., and Su, Y. (2019). Characteristics are covariances: A unified model of risk and return. *Journal of Financial Economics*, 134(3), 501–524.

## Citation

If you use this package in academic work, cite the underlying methodology:

```bibtex
@article{kelly2019characteristics,
  title={Characteristics are covariances: A unified model of risk and return},
  author={Kelly, Bryan T. and Pruitt, Seth and Su, Yinan},
  journal={Journal of Financial Economics},
  volume={134},
  number={3},
  pages={501--524},
  year={2019}
}
```

You may also cite this repository:

```bibtex
@software{baranowski_ipca_state,
  author = {Baranowski, Kornel},
  title = {ipca-state: Instrumented Principal Component Analysis with State-Conditioned Extensions},
  url = {https://github.com/kbaranowsky/ipca-state},
  year = {2026}
}
```

## License

This project is distributed under the MIT License. See `LICENSE` for details.

## Disclaimer

This package is research software. It is provided without warranty and should be validated independently before use in academic, financial, or production settings.
