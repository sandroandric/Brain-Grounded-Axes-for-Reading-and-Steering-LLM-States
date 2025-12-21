# Brain-Grounded Axes Reproducibility Repo

This folder contains the analysis scripts, metadata, and paper sources needed to
reproduce the results in the accompanying arXiv submission.

## Contents

- `analysis/`: all analysis scripts used in the pipeline and evaluations
- `config/`: project configuration
- `mapping/`: run-to-story mappings
- `metadata/`: runs index, lexica, confounds, timebase notes
- `paper/`: arXiv-ready paper (`main.tex`, `references.bib`, `references.bbl`, figures)
- `runbooks/`: step-by-step run instructions and project plan
- `environment/`: dependency lists

## Environment

Create a virtual environment and install dependencies:

```
python -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements.txt
```

For exact package versions, see `environment/requirements.lock.txt`.

## Data

This repo does not include the dataset. Download SMN4Lang (OpenNeuro ds004078 v1.2.1)
with DataLad and fetch the required derivatives. See `data_manifest.md`.

## Reproduce the pipeline

Follow `runbooks/RUNBOOK.md`. The key steps are:

1. Compute MEG PLV states
2. Build word-level atlas
3. Discover axes (ICA)
4. Train LLM adapters
5. Run steering and evaluations
6. Generate figures and compile the paper

## Paper

Paper sources are in `paper/`. To build:

```
cd paper
xelatex main.tex
```

Figures can be regenerated via:

```
python ../analysis/make_paper_figures.py
```

