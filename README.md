# ipca

`ipca` is a Python package for preparing asset-level panel data and estimating Instrumented Principal Component Analysis (IPCA) models, including state-conditioned IPCA utilities.

The GitHub repository may be named `ipca-state`, but the installed Python import package is named `ipca`.

## Installation from GitHub

```bash
python3 -m pip install git+https://github.com/kbaranowsky/ipca-state.git
```

To reinstall the newest version from GitHub:

```bash
python3 -m pip install --upgrade --force-reinstall git+https://github.com/kbaranowsky/ipca-state.git
```

If you use Anaconda, Spyder, or another environment manager, install into the exact interpreter that runs your code:

```bash
/path/to/python -m pip install git+https://github.com/kbaranowsky/ipca-state.git
```

You can find that interpreter from Python with:

```python
import sys
print(sys.executable)
```

## Development installation

```bash
git clone https://github.com/kbaranowsky/ipca-state.git
cd ipca-state
python3 -m pip install -e ".[dev]"
```

## Public API

Use:

```python
from ipca import IPCA, Instruments
```

For compatibility with your original lowercase class name, this also works:

```python
from ipca import ipca
```

`IPCA` is an alias to the original `ipca` class.

## Quick import test

After installation, run:

```bash
python3 -c "from ipca import IPCA, Instruments, ipca; print(IPCA, Instruments, ipca)"
```

Or run the package tests:

```bash
pytest
```

## Repository structure

```text
ipca-state/
├── src/
│   └── ipca/
│       ├── __init__.py
│       ├── instruments.py
│       └── ipca.py
├── tests/
│   └── test_imports.py
├── examples/
│   └── basic_import.py
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Basic usage

```python
import pandas as pd
from ipca import Instruments, IPCA

# Raw asset-level panel data.
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

## Build documentation locally

```bash
python3 -m pip install -e ".[docs]"
pdoc ipca -o site
```

Then open `site/index.html`.

## Troubleshooting

### `ModuleNotFoundError: No module named 'ipca'`

Usually this means the package was installed into a different Python environment than the one you are using.

Check your current Python executable:

```python
import sys
print(sys.executable)
```

Then install using that exact executable:

```bash
/path/to/that/python -m pip install git+https://github.com/kbaranowsky/ipca-state.git
```
