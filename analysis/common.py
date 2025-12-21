"""
Shared dataset-specific helpers for ds004078.
"""

from __future__ import annotations

import contextlib
import csv
import pathlib
import subprocess
import wave
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat

# Dataset timing constants
FMRIS_SHIFT_SEC = 10.65
MEG_AUDIO_DELAY_SEC = 0.0395
MISSING_START = {("08", "16"), ("09", "7")}


def git_hash(repo_path: pathlib.Path) -> str:
    """Return short git hash if available."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_path)
        return out.decode().strip()
    except Exception:
        return "unknown"


def normalize_run(run: str) -> str:
    """Normalize run IDs to dataset filenames (strip leading zeros, drop run- prefix)."""
    run = str(run)
    if run.startswith("run-"):
        run = run.split("-")[-1]
    run = run.lstrip("0") or "0"
    return run


def load_run_to_story(mapping_csv: pathlib.Path) -> Dict[Tuple[str, str], str]:
    """mapping_csv columns: sub, run, story_id (story_XX)."""
    lut: Dict[Tuple[str, str], str] = {}
    with mapping_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            sub = row["sub"].zfill(2) if row["sub"].isdigit() else row["sub"]
            run = normalize_run(row["run"])
            lut[(sub, run)] = row["story_id"]
            lut[(sub, run.zfill(2))] = row["story_id"]
    return lut


def load_runs_index(dataset_root: pathlib.Path) -> pd.DataFrame:
    """Canonical runs index (metadata/runs_index.csv)."""
    df = pd.read_csv(dataset_root / "metadata" / "runs_index.csv", dtype=str)
    df["run"] = df["run"].astype(str)
    df["subject"] = df["subject"].astype(str).str.zfill(2)
    return df.set_index(["subject", "run"])


def audio_duration_seconds(dataset_root: pathlib.Path, story_id: str) -> float:
    """Duration of stimuli/audio/story_XX.wav (seconds)."""
    wav = dataset_root / "stimuli" / "audio" / f"{story_id}.wav"
    with contextlib.closing(wave.open(str(wav))) as wf:
        return wf.getnframes() / wf.getframerate()


def load_time_alignment(dataset_root: pathlib.Path, story_id: str, level: str = "word"):
    """Return tokens, starts, ends in audio-onset time (shifted and delay-adjusted)."""
    time_dir = dataset_root / "derivatives" / "annotations" / "time_align" / f"{level}-level"
    mat_path = time_dir / f"{story_id}_{level}_time.mat"
    mat = loadmat(mat_path)
    starts = np.asarray(mat["start"]).squeeze() - FMRIS_SHIFT_SEC + MEG_AUDIO_DELAY_SEC
    ends = np.asarray(mat["end"]).squeeze() - FMRIS_SHIFT_SEC + MEG_AUDIO_DELAY_SEC
    tokens = np.asarray(mat[level]).squeeze()
    return tokens, starts, ends


def iter_windows(n_samples: int, sfreq: float, length_s: float, step_s: float) -> Iterable[Tuple[int, int, float]]:
    """Yield (start_idx, end_idx, center_time_sec) in MEG time."""
    step_samples = int(step_s * sfreq)
    win_samples = int(length_s * sfreq)
    for start in range(0, n_samples - win_samples + 1, step_samples):
        end = start + win_samples
        center = (start + end) / 2 / sfreq
        yield start, end, center


def centers_to_audio_time(
    centers_meg: np.ndarray, subject: str, run: str, runs_idx: pd.DataFrame, story_id: str, dataset_root: pathlib.Path
) -> np.ndarray:
    """
    Convert MEG-time centers to audio-onset time using trig_start and auditory delay.
    For missing-start runs, recompute start = end - audio_duration if trig_start is absent.
    """
    row = runs_idx.loc[(subject, run)]
    aud_delay = float(row.get("auditory_delay_s", MEG_AUDIO_DELAY_SEC))
    trig_start = row.get("trig_start_s")
    if pd.isna(trig_start):
        trig_end = float(row.get("trig_end_s"))
        trig_start = trig_end - audio_duration_seconds(dataset_root, story_id)
    trig_start = float(trig_start)
    centers_audio = centers_meg - (trig_start + aud_delay)
    return centers_audio
