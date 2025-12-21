"""
Step 5 scaffold: compute sensor-level PLV and derive semantic axes.

This script is intentionally conservative: it does not run by default,
and expects you to point it at the version-pinned ds004078 clone
at /Volumes/Neuro_Data_16122025/ds004078 (or another clone).

Main tasks:
1) Load preprocessed MEG (bandpass 0.1–40 Hz already applied upstream).
2) Compute sliding-window PLV per frequency band.
3) Align windows to stimulus time series (word-level by default).
4) Derive axes = mean(PLV | feature high) - mean(PLV | feature low).

Assumptions and dataset-specific timing:
- Preprocessed MEG: derivatives/preprocessed_data/sub-XX/MEG/*.fif
- SamplingFrequency is 1000 Hz (see *_meg.json), but preprocessed has 0.1–40 Hz filter.
- Stimulus time files are in derivatives/annotations/time_align/{word,char}-level/story_XX_*.
  These times are shifted +10.65 s for fMRI. For MEG alignment subtract 10.65 s.
- MEG audio delay: +39.5 ms from trigger to ear. If you align windows to audio,
  add +0.0395 s to stimulus times (or subtract from MEG times) consistently.
- Two runs (sub-08_run-16, sub-09_run-7) have missing start triggers; adjust events
  manually per README if you process those.

Run-to-story mapping:
- MEG story order differs from fMRI. Provide a CSV mapping:
    run_to_story.csv with columns: sub, run, story_id
  Example: sub-01, run-10, story_1
  The script expects this mapping to locate the right story file.

Usage examples (dry run by default):
  python analysis/step5_plv_axes.py \\
    --dataset-root /Volumes/Neuro_Data_16122025/ds004078 \\
    --subject 01 \\
    --runs 10 11 \\
    --run-to-story mapping/run_to_story.csv \\
    --out-dir outputs/plv_demo \\
    --execute

This is a scaffold; it prioritizes clarity over speed. You may want to
optimize PLV computation (e.g., multi-band Hilbert in chunks) before
running across all runs.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
from dataclasses import dataclass
import json
import subprocess
from typing import Dict, Iterable, List, Tuple
import contextlib
import wave

import mne
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat


# Stable dataset defaults
FMRIS_SHIFT_SEC = 10.65  # subtract for MEG
MEG_AUDIO_DELAY_SEC = 0.0395  # add to stimulus times (or subtract from MEG)

# Canonical low-frequency bands (≤40 Hz because preprocessed data are 0.1–40 Hz)
BANDS = {
    "delta": (1.0, 3.0),
    "theta": (4.0, 7.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 40.0),
}


@dataclass
class WindowDef:
    length: float  # seconds
    step: float  # seconds


def load_run_to_story(mapping_csv: pathlib.Path) -> Dict[Tuple[str, str], str]:
    """
    mapping_csv columns: sub, run, story_id (e.g., story_10)
    Returns dict keyed by (sub, run) -> story_id
    """
    lut: Dict[Tuple[str, str], str] = {}
    with mapping_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            sub = row["sub"].zfill(2) if row["sub"].isdigit() else row["sub"]
            run = row["run"]
            if run.startswith("run-"):
                run_id = run.split("-")[-1]
            else:
                run_id = run
            lut[(sub, run_id)] = row["story_id"]
    return lut


def load_time_alignment(dataset_root: pathlib.Path, story_id: str, level: str = "word"):
    """Load time alignment (start/end) for a story. Applies MEG shift by default."""
    time_dir = dataset_root / "derivatives" / "annotations" / "time_align" / f"{level}-level"
    mat_path = time_dir / f"{story_id}_{level}_time.mat"
    mat = loadmat(mat_path)
    starts = np.asarray(mat["start"]).squeeze() - FMRIS_SHIFT_SEC + MEG_AUDIO_DELAY_SEC
    ends = np.asarray(mat["end"]).squeeze() - FMRIS_SHIFT_SEC + MEG_AUDIO_DELAY_SEC
    tokens = np.asarray(mat[level]).squeeze()
    return tokens, starts, ends


def load_embeddings(dataset_root: pathlib.Path, story_id: str, model: str = "gpt", level: str = "word"):
    """
    Load embeddings for a story.
    model: gpt|bert|word2vec
    level: word|char
    Returns (tokens, embeddings ndarray [T x D])
    """
    base = dataset_root / "derivatives" / "annotations" / "embeddings"
    if model == "gpt":
        path = base / "gpt" / f"{level}-level" / f"{story_id}_{level}_gpt_0-24_1024.mat"
        mat = loadmat(path, struct_as_record=False, squeeze_me=True)
        data = np.asarray(mat["data"], dtype=np.float32)
        # data shape: [layers x tokens x dim] (observed 25 x T x 1024). Average layers.
        if data.ndim == 3:
            embeds = data.mean(axis=0)
        elif data.ndim == 2:
            embeds = data
        else:
            raise ValueError(f"Unexpected GPT embed shape {data.shape} in {path}")
        tokens = np.arange(embeds.shape[0])
    elif model == "bert":
        path = base / "bert" / f"{level}-level" / f"{story_id}_{level}_bert_1-12_768.hdf5"
        raise NotImplementedError(f"HDF5 BERT loader not implemented for {path}")
    elif model == "word2vec":
        path = base / "word2vec" / f"{level}-level" / "100d" / f"{story_id}_{level}_word2vec.mat"
        mat = loadmat(path, struct_as_record=False, squeeze_me=True)
        embeds = np.asarray(mat["data"], dtype=np.float32)
        tokens = np.arange(embeds.shape[0])
    else:
        raise ValueError(f"Unknown model {model}")
    return tokens, embeds


def _iter_windows(n_samples: int, sfreq: float, window: WindowDef) -> Iterable[Tuple[int, int, float]]:
    """Yield (start_idx, end_idx, center_time_sec)."""
    step_samples = int(window.step * sfreq)
    win_samples = int(window.length * sfreq)
    for start in range(0, n_samples - win_samples + 1, step_samples):
        end = start + win_samples
        center = (start + end) / 2 / sfreq
        yield start, end, center


def compute_plv(raw: mne.io.BaseRaw, bands: Dict[str, Tuple[float, float]], window: WindowDef, picks: str = "grad"):
    """
    Compute PLV matrices per band and per window.
    Returns dict band -> (n_windows, n_ch, n_ch), window_centers
    """
    raw = raw.copy().pick(picks=picks).load_data()
    sfreq = raw.info["sfreq"]
    data = raw.get_data()
    n_ch, n_samp = data.shape
    window_centers: List[float] = []
    band_plv: Dict[str, List[np.ndarray]] = {b: [] for b in bands}

    for band, (l_freq, h_freq) in bands.items():
        filtered = mne.filter.filter_data(
            data, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, method="fir", verbose=False
        )
        analytic = signal.hilbert(filtered, axis=-1)
        phase = np.angle(analytic)
        band_windows: List[np.ndarray] = []
        for start, end, center in _iter_windows(n_samp, sfreq, window):
            if band == list(bands.keys())[0]:
                window_centers.append(center)
            seg_phase = phase[:, start:end]
            exp_phase = np.exp(1j * seg_phase)
            mean_phase = exp_phase.mean(axis=-1)
            plv_mat = np.abs(np.exp(1j * seg_phase) @ np.exp(-1j * seg_phase).conj().T) / seg_phase.shape[1]
            # Normalize diagonal to 1
            np.fill_diagonal(plv_mat, 1.0)
            band_windows.append(plv_mat.astype(np.float32))
        band_plv[band] = band_windows
    return {b: np.stack(mats, axis=0) for b, mats in band_plv.items()}, np.array(window_centers)


def align_feature_to_windows(window_centers: np.ndarray, starts: np.ndarray, feature_series: np.ndarray):
    """
    Align a feature time series to window centers by nearest onset.
    Assumes starts length matches feature_series length.
    """
    idx = np.searchsorted(starts, window_centers, side="right") - 1
    idx = np.clip(idx, 0, len(feature_series) - 1)
    return feature_series[idx]


def derive_axis(plv_stack: np.ndarray, feature: np.ndarray, quantile: float = 0.2):
    """
    plv_stack: [n_windows, n_ch, n_ch] for one band
    feature: [n_windows] scalar feature
    """
    low_thresh = np.quantile(feature, quantile)
    high_thresh = np.quantile(feature, 1 - quantile)
    low_idx = feature <= low_thresh
    high_idx = feature >= high_thresh
    axis = plv_stack[high_idx].mean(axis=0) - plv_stack[low_idx].mean(axis=0)
    return axis


def embedding_change_feature(embeds: np.ndarray) -> np.ndarray:
    """
    Simple semantic-drift feature: L2 norm of successive embedding differences.
    embeds: [T x D]
    returns: [T] with first element duplicated to keep length
    """
    diff = np.linalg.norm(np.diff(embeds, axis=0), axis=1)
    # Prepend first diff to preserve length alignment
    return np.concatenate(([diff[0]], diff))


def audio_duration_seconds(dataset_root: pathlib.Path, story_id: str) -> float:
    """Duration of a story_XX.wav in seconds (float)."""
    wav = dataset_root / "stimuli" / "audio" / f"{story_id}.wav"
    with contextlib.closing(wave.open(str(wav))) as wf:
        return wf.getnframes() / wf.getframerate()


def adjust_centers_for_missing_start(
    subject: str, run: str, centers: np.ndarray, dataset_root: pathlib.Path, story_id: str
) -> np.ndarray:
    """
    For runs with missing start triggers (sub-08_run-16, sub-09_run-7),
    shift MEG time so that audio end aligns with run end:
        offset = -(event_end - audio_duration)
        centers += offset
    """
    missing = {("08", "16"), ("09", "7")}
    if (subject, run) not in missing:
        return centers
    events_path = (
        dataset_root
        / "derivatives"
        / "preprocessed_data"
        / f"sub-{subject}"
        / "MEG"
        / f"sub-{subject}_task-RDR_run-{run}_events.tsv"
    )
    df = pd.read_csv(events_path, sep="\t")
    if df.empty:
        return centers
    event_end = df["onset"].max()
    audio_dur = audio_duration_seconds(dataset_root, story_id)
    offset = -(event_end - audio_dur)
    print(f"[info] missing start trigger run sub-{subject} run-{run}: shifting centers by {offset:.3f}s")
    return centers + offset


def git_hash(repo_path: pathlib.Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_path)
        return out.decode().strip()
    except Exception:
        return "unknown"


def write_sidecar(
    out_file: pathlib.Path,
    *,
    dataset_root: pathlib.Path,
    subject: str,
    run: str,
    story_id: str,
    band: str,
    window: WindowDef,
    feature: str,
    embed_model: str,
    decim: int,
    picks: str,
    sfreq: float,
    n_channels: int,
    n_windows: int,
    axis_shape: tuple,
    time_align_shift: float,
    auditory_delay: float,
    run_to_story_path: pathlib.Path,
):
    sidecar = {
        "dataset_root": str(dataset_root),
        "subject": subject,
        "run": run,
        "story_id": story_id,
        "band": band,
        "window_length_s": window.length,
        "window_step_s": window.step,
        "feature": feature,
        "embed_model": embed_model,
        "decim": decim,
        "picks": picks,
        "sfreq": sfreq,
        "n_channels": n_channels,
        "n_windows": n_windows,
        "axis_shape": axis_shape,
        "time_align_shift_s": time_align_shift,
        "auditory_delay_s": auditory_delay,
        "run_to_story_mapping": str(run_to_story_path),
        "code_git_hash": git_hash(dataset_root),
    }
    with out_file.with_suffix(out_file.suffix + ".json").open("w") as f:
        json.dump(sidecar, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Step 5: PLV + semantic axis scaffold for ds004078.")
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    parser.add_argument("--subject", required=True, help="e.g., 01")
    parser.add_argument("--runs", nargs="+", required=True, help="e.g., 10 11 12")
    parser.add_argument("--run-to-story", type=pathlib.Path, required=True, help="CSV mapping run -> story_id")
    parser.add_argument("--window-length", type=float, default=2.0)
    parser.add_argument("--window-step", type=float, default=0.5)
    parser.add_argument("--band", choices=list(BANDS.keys()), default="theta")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Actually run PLV (otherwise dry-run).")
    parser.add_argument(
        "--feature",
        choices=["index", "embedding_change"],
        default="embedding_change",
        help="Scalar feature to define axes (index=token index, embedding_change=L2 drift).",
    )
    parser.add_argument(
        "--embed-model",
        choices=["gpt", "word2vec"],
        default="gpt",
        help="Embedding source for feature derivation.",
    )
    parser.add_argument(
        "--decim",
        type=int,
        default=1,
        help="Optional decimation factor applied via raw.resample(sfreq/raw.info['sfreq']/decim) before PLV. Use 5 for ~200 Hz.",
    )
    args = parser.parse_args()

    window = WindowDef(length=args.window_length, step=args.window_step)
    lut = load_run_to_story(args.run_to_story)

    print(f"[info] subject sub-{args.subject}, runs {args.runs}, band {args.band}")
    print(f"[info] dataset root: {args.dataset_root}")
    print(f"[info] window: {window.length}s, step {window.step}s")
    print(f"[info] feature: {args.feature} from {args.embed_model}")
    if args.decim != 1:
        print(f"[info] decimation enabled: factor {args.decim}")

    if not args.execute:
        print("[dry-run] add --execute to run PLV computation.")
        return

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for run in args.runs:
        story_id = lut.get((args.subject, run))
        if story_id is None:
            raise RuntimeError(f"No story mapping for sub-{args.subject} run-{run}")

        meg_path = (
            args.dataset_root
            / "derivatives"
            / "preprocessed_data"
            / f"sub-{args.subject}"
            / "MEG"
            / f"sub-{args.subject}_task-RDR_run-{run}_meg.fif"
        )
        raw = mne.io.read_raw_fif(meg_path, preload=False, verbose="ERROR")
        if args.decim != 1:
            target_sfreq = raw.info["sfreq"] / args.decim
            raw.resample(target_sfreq, npad="auto", verbose="ERROR")
        plv_dict, centers = compute_plv(raw, {args.band: BANDS[args.band]}, window)

        tokens, starts, _ = load_time_alignment(args.dataset_root, story_id, level="word")
        centers = adjust_centers_for_missing_start(args.subject, run, centers, args.dataset_root, story_id)
        # Feature series
        if args.feature == "index":
            feature_series = np.arange(len(tokens), dtype=np.float32)
        elif args.feature == "embedding_change":
            _, embeds = load_embeddings(args.dataset_root, story_id, model=args.embed_model, level="word")
            T = min(len(tokens), embeds.shape[0])
            embeds = embeds[:T]
            feature_series = embedding_change_feature(embeds)
        else:
            raise ValueError(f"Unknown feature {args.feature}")

        feature_aligned = align_feature_to_windows(centers, starts, feature_series)

        axis = derive_axis(plv_dict[args.band], feature_aligned, quantile=0.2)
        out_file = out_dir / f"sub-{args.subject}_run-{run}_{args.band}_axis.npy"
        np.save(out_file, axis)
        write_sidecar(
            out_file,
            dataset_root=args.dataset_root,
            subject=args.subject,
            run=run,
            story_id=story_id,
            band=args.band,
            window=window,
            feature=args.feature,
            embed_model=args.embed_model,
            decim=args.decim,
            picks="grad",
            sfreq=raw.info["sfreq"],
            n_channels=len(raw.ch_names),
            n_windows=plv_dict[args.band].shape[0],
            axis_shape=axis.shape,
            time_align_shift=FMRIS_SHIFT_SEC,
            auditory_delay=MEG_AUDIO_DELAY_SEC,
            run_to_story_path=args.run_to_story,
        )
        print(f"[saved] {out_file}")


if __name__ == "__main__":
    main()
