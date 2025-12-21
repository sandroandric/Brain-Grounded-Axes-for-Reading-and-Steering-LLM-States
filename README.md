# Brain-Grounded Axes Reproducibility Repo

This folder contains the analysis scripts and metadata needed to reproduce the
core results.

## Contents

- `analysis/`: all analysis scripts used in the pipeline and evaluations
- `config/`: project configuration
- `mapping/`: run-to-story mappings
- `metadata/`: runs index, lexica, confounds, timebase notes
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

High-level steps (script entrypoints live in `analysis/`):

1. Compute MEG PLV states
2. Build word-level atlas
3. Discover axes (ICA)
4. Train LLM adapters
5. Run steering and evaluations
