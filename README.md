# ipca-state

[![Documentation](https://github.com/kbaranowsky/ipca-state/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/kbaranowsky/ipca-state/actions/workflows/docs.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`ipca-state` is a Python package for preparing asset-level panel data and estimating Instrumented Principal Component Analysis (IPCA) models, including state conditioned IPCA extensions.

The package provides tools for:

- preparing characteristics and returns data for IPCA estimation
- constructing cross-sectionally rank standardized instruments
- optionally creating state interacted characteristics for state conditioned IPCA
- estimating restricted and unrestricted IPCA models
- computing total and predictive \(R^2\)
- running bootstrap tests for pricing errors and characteristic relevance
- comparing forecast performance with Diebold-Mariano test

The repository name is `ipca-state`, while the Python import package is `ipca`.

## Documentation

Public API documentation is generated with `pdoc` and deployed through GitHub Pages.

Documentation URL:

```text
https://kbaranowsky.github.io/ipca-state/
```


## Installation

### Install from GitHub

The package is not published on PyPI. Install it directly from GitHub:

```bash
python3 -m pip install git+https://github.com/kbaranowsky/ipca-state.git
```



## Requirements

The package requires Python 3.10 or newer.

Core dependencies:

- `numpy`
- `pandas`
- `scipy`
- `joblib`
- `statsmodels`


## Example use

```python
import pandas as pd
from ipca import Instruments, IPCA

# Raw asset-level panel data from the authors of Kelly et al. (2019)
data = pd.read_csv("asset_level_panel.csv")

characteristics = ["beta", "mom", "size"]

builder = Instruments(
    data = data,
    characteristics = characteristics,
    returns = "ret",
    date = "date",
    permno =  "permno",
)

Z, R = builder.prepare_data(
    addConstant = True,
    filterMonths = True,
    make_state_interactions = False,
    printSummary = True,
)

model = IPCA(
    Z = Z,
    R = R,
    K = 3,
    alpha = False,
)

model.fit(
    tol = 1e-6,
    max_iter = 1000,
    printTime = True,
    printInformation = True,
)

model.short_summary()
```

## State Conditioned IPCA (SC-IPCA) example use

To construct state interacted characteristics, provide a two column state variable dataframe and set `make_state_interactions = True`.

The first column should contain dates. The second column should contain the state variable.

```python
import pandas as pd
from ipca import Instruments, IPCA

data = pd.read_csv("asset_level_panel.csv")
state_variable = pd.read_csv("state_variable.csv")

characteristics = ["beta", "mom", "size"]

builder = Instruments(
    data = data,
    characteristics =  characteristics,
    returns = "ret",
    date = "date",
    permno = "permno",
    state_variable = state_variable,
)

Z_state, R_state = builder.prepare_data(
    addConstant = True,
    filterMonths = True,
    make_state_interactions = True,
    printSummary = True,
)

state_model = IPCA(
    Z = Z_state,
    R = R_state,
    K = 3,
    alph a = False,
)

state_model.fit(tol = 1e-6, max_iter = 1000, printTime = True, printInformation = True)
state_model.short_summary()
```

## Data format

### Asset-level panel data
The main input should follow the format of Kelly et al. (2019) dataset, with at least the following columns: Date, asset identifier, returns, characteristics columss.

The optional `risk_free` argument should be a two-column dataframe:

1. date column;
2. risk-free rate column.

### State variable

The optional `state_variable` argument should be a two-column dataframe:

1. date column;
2. state variable column.


## Public API

The recommended imports are:

```python
from ipca import Instruments, IPCA
```


## Main components

### `Instruments`

Prepares raw asset level data for IPCA estimation.

Main responsibilities:

- date normalization to month end;
- optional filtering of months with insufficient valid assets;
- optional filtering of assets with insufficient return history;
- cross-sectional rank standardization of firm characteristics;
- optional risk-free rate subtraction;
- construction of dictionaries of characteristics matrices `Z` and return series `R`;
- optional construction of state interacted characteristics.

### `IPCA`

Estimates IPCA models based on prepared `Z` and `R` inputs.

Main functionality:

- restricted IPCA estimation;
- unrestricted IPCA estimation with pricing error terms;
- latent and observed factor support;
- alternating least squares estimation;
- total and predictive \(R^2\);
- managed portfolio fit measures;
- bootstrap tests for pricing errors;
- bootstrap tests for characteristic relevance;
- state level fit diagnostics;
- forecasting and forecast comparison utilities.


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

## References


Kelly, B. T., Pruitt, S., and Su, Y. (2019). Characteristics are covariances: A unified model of risk and return. *Journal of Financial Economics*, 134(3), 501–524.



## Disclaimer

This package is research software. It is provided without warranty and should be validated independently before use in academic, financial, or production settings. See LICENSE for details.

## Development note

This package was developed by me as part of my bachelor thesis. AI tools such as ChatGPT of Claude were used during development for code review, documentation support, debugging support, and packaging guidance according to the requirements and limitations of Thesis Manual 2026. All implementation choices, testing, validation, and responsibility for the code remain with me. For suggestinons on the package improvement please email: kornel.baranowski@gmail.com
