Pipeline runbook (after current PLV run finishes)
=================================================

Use `mapping/id_map.csv` (story_id = run_id). All paths assume dataset root `/Volumes/Neuro_Data_16122025/ds004078`.

0) Environment
--------------
- `python -m venv .venv && source .venv/bin/activate`
- `pip install -r analysis/requirements_step5.txt`

1) Box 1: PLV/wPLI states (edge PCA on gradiometers, theta default)
-------------------------------------------------------------------
```
python analysis/run_plv_box1.py \
  --dataset-root /Volumes/Neuro_Data_16122025/ds004078 \
  --run-to-story mapping/id_map.csv \
  --out-dir outputs/meg_plv_states \
  --band theta \
  --metric plv \
  --state-space edge_pca \
  --pca-components 128 \
  --picks grad \
  --window-length 2.0 \
  --window-step 0.5 \
  --decim 1 \
  --execute
```
Outputs per run: states, window centers (MEG + audio), PCA basis, sidecars/QC.

2) Box 3: Semantic features aligned to windows
----------------------------------------------
```
python analysis/compute_semantic_features.py \
  --dataset-root /Volumes/Neuro_Data_16122025/ds004078 \
  --run-to-story mapping/id_map.csv \
  --out-dir outputs/meg_features \
  --subjects 01 02 03 04 05 06 07 08 09 10 11 12 \
  --runs 1 2 3 ... 60 \
  --window-length 2.0 \
  --window-step 0.5 \
  --features embedding_change logfreq pos_id \
  --embed-model gpt \
  --execute
```

3) Axis discovery (Box 4, MEG-only)
-----------------------------------
```
python analysis/discover_axes.py \
  --plv-root outputs/meg_plv_states \
  --features-root outputs/meg_features \
  --out-dir outputs/meg_axes \
  --band theta \
  --metric plv \
  --state-space edge_pca \
  --feature-names embedding_change logfreq pos_id \
  --train-runs 1 2 ... 40 \
  --test-runs 41 42 ... 60 \
  --subjects 01 02 03 04 05 06 07 08 09 10 11 12 \
  --save-coords
```
Outputs: axes per feature/subject + per-run projections (coords/) + metrics JSON.

4) Box 2: fMRI semantic space
-----------------------------
```
python analysis/fmri_semspace.py \
  --dataset-root /Volumes/Neuro_Data_16122025/ds004078 \
  --out-dir outputs/fmri_semspace \
  --components 100 \
  --space CIFTI \
  --subjects 01 02 03 04 05 06 07 08 09 10 11 12 \
  --runs 1 2 ... 60 \
  --execute
```

4b) Box 2 (recommended): shared PCA per subject
------------------------------------------------
This builds a single PCA basis per subject across runs so components are comparable.
```
python analysis/fmri_semspace_shared.py \
  --dataset-root /Volumes/Neuro_Data_16122025/ds004078 \
  --out-dir outputs/fmri_semspace_shared \
  --components 100 \
  --space CIFTI \
  --subjects 01 02 03 04 05 06 07 08 09 10 11 12 \
  --runs 10 11 ... 60 \
  --execute
```

5) Anchoring MEG axes to fMRI
-----------------------------
```
python analysis/anchor_meg_fmri.py \
  --meg-root outputs/meg_axes \
  --fmri-root outputs/fmri_semspace \
  --out-dir outputs/anchoring \
  --features embedding_change logfreq pos_id \
  --band theta \
  --metric plv \
  --state-space edge_pca \
  --fmri-tag cifti_pca \
  --subjects 01 02 03 04 05 06 07 08 09 10 11 12 \
  --runs 1 2 ... 60 \
  --lags 0.0
```

5b) Anchoring (train/test; shared PCA recommended)
--------------------------------------------------
```
python analysis/anchor_meg_fmri.py \
  --meg-root outputs/meg_axes \
  --fmri-root outputs/fmri_semspace_shared \
  --out-dir outputs/anchoring \
  --features embedding_change logfreq pos_id \
  --band theta \
  --metric plv \
  --state-space edge_pca \
  --fmri-tag cifti_sharedpca \
  --subjects 01 02 03 04 05 06 07 08 09 10 11 12 \
  --train-runs 10 11 ... 40 \
  --test-runs 41 42 ... 60 \
  --lags 0 2 4 6 8
```

5c) Anchoring significance (permutation test)
---------------------------------------------
```
python analysis/anchoring_permtest.py \
  --anchor-json outputs/anchoring/anchoring_metrics_theta_plv_edge_pca_train_test.json \
  --meg-root outputs/meg_axes \
  --fmri-root outputs/fmri_semspace_shared \
  --n-perms 1000 \
  --min-shift-s 30 \
  --seed 0 \
  --out-json outputs/anchoring/anchoring_permtest_train_test.json
```

5d) fMRI SRM (shared response model) on PCA features
----------------------------------------------------
This aligns subjects into a common space using SRM on per-subject PCA features.
```
python analysis/fmri_semspace_srm.py \
  --input-root outputs/fmri_semspace_shared \
  --input-tag cifti_sharedpca \
  --out-dir outputs/fmri_semspace_srm \
  --components 50 \
  --subjects 01 02 03 04 05 06 07 08 09 10 11 12 \
  --train-runs 10 11 ... 40 \
  --test-runs 41 42 ... 60 \
  --n-iter 10 \
  --zscore \
  --execute
```

5e) Anchoring with SRM fMRI space (train/test)
----------------------------------------------
```
python analysis/anchor_meg_fmri.py \
  --meg-root outputs/meg_axes \
  --fmri-root outputs/fmri_semspace_srm \
  --out-dir outputs/anchoring \
  --features embedding_change logfreq pos_id \
  --band theta \
  --metric plv \
  --state-space edge_pca \
  --fmri-tag cifti_srm \
  --subjects 01 02 03 04 05 06 07 08 09 10 11 12 \
  --train-runs 10 11 ... 40 \
  --test-runs 41 42 ... 60 \
  --lags 0 2 4 6 8
```

6) LLM adapter (per subject)
----------------------------
Provide word-level hidden states per story: `${hidden_root}/story_X_hidden.npy`.
Generate with:
```

6b) LLM adapter to MEG word-axes (semantic axes)
-----------------------------------------------
Train a word-level adapter to the ICA word axes (MEG-only semantics).
```

6c) Stability check (multi-split)
---------------------------------
```
python analysis/llm_word_axes_stability.py \
  --word-axes-root outputs/word_axes_ica \
  --hidden-root outputs/llm_hidden \
  --time-align-root derivatives/annotations/time_align/word-level \
  --out-json outputs/llm_adapter_word_axes/adapter_stability.json \
  --axes 13 19 15 2 \
  --train-frac 0.8 \
  --min-count 2 \
  --standardize \
  --n-splits 10 \
  --seed 0
```
python analysis/train_llm_word_axes_adapter.py \
  --word-axes-root outputs/word_axes_ica \
  --hidden-root outputs/llm_hidden \
  --time-align-root derivatives/annotations/time_align/word-level \
  --out-dir outputs/llm_adapter_word_axes \
  --axes 13 19 15 2 \
  --train-frac 0.8 \
  --min-count 2 \
  --standardize \
  --seed 0
```
python analysis/extract_llm_word_hidden.py \
  --time-align-root derivatives/annotations/time_align/word-level \
  --out-dir outputs/llm_hidden \
  --model uer/gpt2-chinese-cluecorpussmall \
  --layer -1 \
  --device mps \
  --max-words 256 \
  --overlap 64
```

```
python analysis/train_llm_adapter.py \
  --dataset-root /Volumes/Neuro_Data_16122025/ds004078 \
  --run-to-story mapping/id_map.csv \
  --meg-root outputs/meg_axes \
  --hidden-root outputs/llm_hidden \
  --out-dir outputs/llm_adapter \
  --features embedding_change logfreq pos_id \
  --band theta \
  --metric plv \
  --state-space edge_pca \
  --alpha 1.0 \
  --train-runs 1 2 ... 40 \
  --test-runs 41 42 ... 60 \
  --subject 01 \
  --standardize \
  --align-mode mean \
  --window-length 2.0
```

7) Steering evaluation (optional)
---------------------------------
```

Word-Atlas Interpretation (MEG-only)
====================================

W1) Lexicon-based semantic axes (concreteness, valence, arousal)
---------------------------------------------------------------
```
python analysis/word_axis_eval_lexica.py \
  --word-axes-root outputs/word_axes_ica \
  --lexica-root metadata/lexica \
  --out-json outputs/word_axes_ica/axis_eval_lexica.json
```

Permutation test (max-stat):
```
python analysis/word_axis_eval_lexica_perm.py \
  --word-axes-root outputs/word_axes_ica \
  --lexica-root metadata/lexica \
  --out-json outputs/word_axes_ica/axis_eval_lexica_perm.json \
  --n-perms 1000 \
  --seed 0
```

W2) HF POS/NER (word-level) labels
----------------------------------
Run in the HF/spaCy venv:
```
PYTHONPATH=. python analysis/word_axis_eval_hf_wordlevel.py \
  --dataset-root . \
  --word-axes-root outputs/word_axes_ica \
  --pos-model ckiplab/bert-base-chinese-pos \
  --ner-model ckiplab/bert-base-chinese-ner \
  --device mps \
  --out-json outputs/word_axes_ica/axis_eval_hf_wordlevel.json \
  --top-k 5
```
python analysis/steering_eval.py \
  --adapter-dir outputs/llm_adapter \
  --subject 01 \
  --hidden-root /path/to/hidden_states \
  --stories story_1 story_2 \
  --axis-index 0 \
  --alpha 1.0 \
  --meg-coords-root outputs/meg_axes \
  --feature-name embedding_change \
  --band theta \
  --metric plv \
  --state-space edge_pca
```
