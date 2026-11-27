# MATH F432 course project

This project studies how admission records and first-semester results relate to dropout and graduation in the UCI 697 dataset. The report and slides use the four vector PDFs in `figures/`.

## Data source

`data/uci697.csv` contains the 4,424 records and 37 columns from the [official UCI CSV](https://archive.ics.uci.edu/static/public/697/data.csv), with the original header and row order, UTF-8 encoding without a BOM, and LF line endings.

Valentim Realinho, Mónica Vieira Martins, Jorge Machado, and Luís Baptista (2021). [Predict Students' Dropout and Academic Success](https://doi.org/10.24432/C5MC89). UCI Machine Learning Repository. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Variable definitions and cohort information are in the [UCI codebook](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) and the [data descriptor](https://doi.org/10.3390/data7110146).

The [Villar and de Andrade study](https://doi.org/10.1007/s44163-023-00079-z), published in 2024, has an [associated GitHub repository](https://github.com/alicevillar/SML-Comparative-Study). This project uses the UCI release and its own analysis code.

## Build the report and slides

These commands require Tectonic and local copies of `report.tex` and `presentation.tex`. Run them from the project directory. The report uses a single-column layout with 11-point Libertinus text. The widescreen Beamer source includes presenter notes.

```sh
mkdir -p output/pdf
tectonic --outdir output/pdf report.tex
tectonic --outdir output/pdf presentation.tex
```

## Reproduce the analysis and figures

Run from the project directory with Python 3.11 or later. Matplotlib looks for Libertinus Serif among installed fonts and in the Tectonic cache. If you have the report source, building it first lets Tectonic cache the font. You can also set `LIBERTINUS_FONT` to the path to `LibertinusSerif-Regular.otf`.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis.py
```

`analysis.py` writes `results.json` with the data audit, tests, model coefficients, diagnostics and holdout metrics. It calls `plot_results.py` to redraw the figures. The split and simulation seeds are fixed in the script. The report tables use rounded values from `results.json`. Run both build commands again to include the updated figures in the PDFs.
