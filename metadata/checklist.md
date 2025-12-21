# Checklist (PLAN.md aligned)

## Foundations
- [x] Dataset pinned to ds004078 v1.2.1 (commit 8c4c1355f3); config/config.yaml captures timing/bands/picks/state-space defaults.
- [x] Data downloaded: stimuli, annotations, preprocessed MEG (sensor-level). Mapping: story_id = run_id (mapping/id_map.csv). Duration map kept for QC only.
- [x] Runs index + timebase notes: metadata/runs_index.csv, metadata/timebase_notes.md (10.65 s shift, 39.5 ms delay). Missing-start triggers handled (sub-08_run-16, sub-09_run-7).
- [x] PLAN.md copied to dataset root for reference; AGENTS.md points to checklist.

## Box 1 — MEG PLV/wPLI states
- [x] Script: analysis/compute_plv_states.py (metric plv/wpli; state-space node/edge/PCA; channel policy; sidecars/QC).
- [x] Batch wrapper: analysis/run_plv_box1.py.
- [x] Full run across all subjects/runs with id_map → outputs/meg_plv_states.

## Box 3 — Semantic features
- [x] Script: analysis/compute_semantic_features.py (embedding_change/logfreq/pos_id; aligned windows; sidecars).
- [x] Execute for all subjects/runs to outputs/meg_features.

## Box 4 — Axis discovery (MEG-only)
- [x] Script: analysis/discover_axes.py (train/test split, sign convention, per-run projections, metrics).
- [x] Aggregation helper: analysis/axis_postprocess.py (mean across runs).
- [x] Run axes for embedding_change/logfreq/pos_id (theta, grad, edge_pca, plv) using outputs/meg_plv_states + outputs/meg_features.
- [ ] Re-run discover_axes with fixed per-feature coords (Dec 18 fix) to regenerate coords for all features.

## Box 2 — fMRI semantic space
- [x] Scaffold: analysis/fmri_semspace.py (PCA on CIFTI/MNI; per-run outputs; sidecars).
- [x] Execute for subjects/runs with CIFTI available (runs 10–60) to outputs/fmri_semspace.
- [x] Shared PCA per subject (analysis/fmri_semspace_shared.py) to outputs/fmri_semspace_shared.
- [x] SRM shared space on PCA features (analysis/fmri_semspace_srm.py) to outputs/fmri_semspace_srm.

## Anchoring (MEG ↔ fMRI)
- [x] Script: analysis/anchor_meg_fmri.py (lags, per-run correlations, component IDs).
- [x] Run anchoring once MEG coords + fMRI space are ready.
- [x] Run train/test anchoring with shared fMRI PCA (avoid per-run best-component inflation).
- [x] Run anchoring permutation test (circular-shift null) for statistical significance.
- [x] Run train/test anchoring with SRM fMRI space.

## LLM adapter + steering
- [x] Script: analysis/train_llm_adapter.py (ridge, standardization option).
- [x] Script: analysis/steering_eval.py (axis steering deltas, optional MEG correlation).
- [x] Collect hidden states per story_XX and train adapters per subject.
- [x] Train LLM adapter to MEG word axes (semantic axes) via analysis/train_llm_word_axes_adapter.py.
- [ ] Steering evaluation on held-out stories.

## Legacy step 5 + logging
- [x] step5_plv_axes.py / run_all_plv.py kept for compatibility (axis-per-run outputs).
- [x] Sidecar backfill: analysis/backfill_sidecars.py.
- [ ] Logging: keep updating log.md after each major run (PLV batch, features, axes, fmri, anchoring, adapters).
