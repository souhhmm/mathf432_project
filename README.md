# MATH F432 course project

This project studies how admission records and first-semester results relate to dropout and graduation in the UCI 697 dataset.

## Data source

`data/uci697.csv` contains the 4,424 records from the [official UCI CSV](https://archive.ics.uci.edu/static/public/697/data.csv) in their original order.

Valentim Realinho, Mónica Vieira Martins, Jorge Machado, and Luís Baptista (2021). [Predict Students' Dropout and Academic Success](https://doi.org/10.24432/C5MC89). UCI Machine Learning Repository. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Variable definitions and cohort information are in the [UCI codebook](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) and the [data descriptor](https://doi.org/10.3390/data7110146).

## Reproduce the results

Run from the project directory with Python 3.11 or later.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis.py --results-only
```

The command writes `results.json` using fixed seeds for the data split and simulations.
