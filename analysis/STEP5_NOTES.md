Step 5 (PLV + semantic axes) scaffold
======================================

What’s here
-----------
- `step5_plv_axes.py`: minimal pipeline to load preprocessed MEG, compute sliding-window PLV, align to stimulus times, and derive axes from a feature.
- `requirements_step5.txt`: suggested Python deps (install in a venv).

Key dataset timing (from INSTRUCTIONS.md)
-----------------------------------------
- Time alignment files are fMRI-shifted by +10.65 s; subtract 10.65 s for MEG.
- MEG audio delay: add +0.0395 s to stimulus times (or subtract from MEG) for consistency.
- Preprocessed MEG is already bandpass 0.1–40 Hz; stay ≤40 Hz unless you use raw.
- Missing start triggers: sub-08_run-16 and sub-09_run-7 (compute start = end – audio duration).

Story/run mapping
-----------------
MEG story order differs from fMRI. Provide a CSV `run_to_story.csv`:
```
sub,run,story_id
01,10,story_1
...
```
`story_id` should match filenames in `derivatives/annotations/time_align/*/story_XX_*`.

How to run (example)
--------------------
```
python analysis/step5_plv_axes.py \
  --dataset-root /Volumes/Neuro_Data_16122025/ds004078 \
  --subject 01 \
  --runs 10 11 \
  --run-to-story mapping/run_to_story.csv \
  --feature embedding_change \
  --embed-model gpt \
  --decim 5 \
  --out-dir outputs/plv_demo \
  --execute
```
Default window: 2.0 s length, 0.5 s step. Default band: theta (4–7 Hz).
Decimation: use `--decim 5` to resample from 1000 Hz to 200 Hz (safe for ≤40 Hz bands, speeds up).

Next tweaks before full runs
----------------------------
- Features:
  - `embedding_change` (default): L2 norm of successive GPT/word2vec embeddings.
  - `index`: token index (baseline).
- Model choice: `--embed-model gpt` (default) or `word2vec`. BERT loader is not implemented.
- Consider decimation (e.g., to 200 Hz) to reduce memory before PLV.
- Batch per subject to avoid loading all runs at once.
- State space / channels / leakage:
  - Default PLV state space: **edge vectors compressed by PCA to 128 dims** on **gradiometers only** (see config/config.yaml).
  - `--metric wpli` is available for a leakage-robust variant; stays ≤40 Hz because of preprocessing.
  - Band-aware windows: theta 2.0s/0.5s, alpha 1.5s/0.5s, beta 1.0s/0.25s (override if needed).
