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

### Quick start (core pipeline)

Set a dataset root and run the core steps:

```
export DATASET_ROOT=/path/to/ds004078
```

1) MEG PLV states (Box 1):

```
PYTHONPATH=. python analysis/run_plv_box1.py \
  --dataset-root "$DATASET_ROOT" \
  --run-to-story mapping/id_map.csv \
  --out-dir outputs/meg_plv_states \
  --band theta --metric plv --state-space edge_pca --pca-components 128 \
  --picks grad --window-length 2.0 --window-step 0.5 --decim 1 \
  --subjects 01 02 03 04 05 06 07 08 09 10 11 12 \
  --runs 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 \
  --execute
```

2) Word-level atlas (ridge encoding):

```
PYTHONPATH=. python analysis/encoding_plv_glm.py \
  --dataset-root "$DATASET_ROOT" \
  --plv-root outputs/meg_plv_states \
  --run-to-story mapping/id_map.csv \
  --out-dir outputs/encoding/plv_glm \
  --band theta --metric plv --state-space edge_pca \
  --subjects 01 02 03 04 05 06 07 08 09 10 11 12 \
  --runs 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 \
  --lags 0.0 0.5 1.0 \
  --alpha 200.0 \
  --standardize
```

3) Axis discovery (ICA on the atlas):

```
PYTHONPATH=. python analysis/word_atlas_axes.py \
  --atlas-root outputs/encoding/plv_glm \
  --out-dir outputs/word_axes_ica \
  --subjects 01 02 03 04 05 06 07 08 09 10 11 12 \
  --method ica
```

4) LLM hidden states + adapter:

```
PYTHONPATH=. python analysis/extract_llm_word_hidden.py \
  --time-align-root "$DATASET_ROOT/derivatives/annotations/time_align/word-level" \
  --out-dir outputs/llm_hidden_tinyllama \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --device mps

PYTHONPATH=. python analysis/train_llm_word_axes_adapter.py \
  --word-axes-root outputs/word_axes_ica \
  --hidden-root outputs/llm_hidden_tinyllama \
  --time-align-root "$DATASET_ROOT/derivatives/annotations/time_align/word-level" \
  --axes 13 19 15 2 \
  --out-dir outputs/llm_adapter_word_axes_tinyllama \
  --standardize
```

5) Steering + evaluation:

```
PYTHONPATH=. python analysis/llm_word_axes_steer_batch.py \
  --adapter-dir outputs/llm_adapter_word_axes_tinyllama \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --device mps \
  --axes 13 19 15 2 \
  --prompt-file analysis/prompts_steering_zh.txt \
  --strengths -2 -1 0 1 2 \
  --samples 4 \
  --layer 11 \
  --max-new-tokens 200 \
  --temperature 0.7 \
  --top-p 0.9 \
  --out-dir outputs/llm_steering_tinyllama/layer_11

PYTHONPATH=. python analysis/llm_word_axes_steer_eval.py \
  --adapter-dir outputs/llm_adapter_word_axes_tinyllama \
  --input-glob outputs/llm_steering_tinyllama/layer_11/axis_*.json \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --device mps \
  --out-json outputs/llm_steering_tinyllama/layer_11/steer_eval.json \
  --n-perms 1000
```
