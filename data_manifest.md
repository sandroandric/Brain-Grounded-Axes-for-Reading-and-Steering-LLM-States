# Data Manifest (SMN4Lang / ds004078 v1.2.1)

This project uses OpenNeuro ds004078 (SMN4Lang) v1.2.1.

Recommended download (from dataset root):

```
# Base dataset

# Preprocessed MEG (sensor level)
datalad get -J 6 derivatives/preprocessed_data/sub-*/MEG

# Annotations and word-level timing
datalad get -J 6 derivatives/annotations

# fMRI CIFTI (optional; long run)
datalad get -J 6 derivatives/preprocessed_data/sub-{01..12}/CIFTI
```

Notes:
- Two runs have missing start triggers: sub-08_run-16 and sub-09_run-7.
- Word timing lives under `derivatives/annotations/time_align/word-level`.

