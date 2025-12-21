"""
Compute sliding-window connectivity (PLV or wPLI) and emit compact state vectors.

Features:
- Band-limited analytic signal (<=40 Hz, consistent with preprocessed MEG)
- Connectivity metric: plv (default) or wpli (leakage-robust)
- State space: node-wise mean, full edges, or PCA-compressed edges
- Channel policy: grad | mag | all
- Audio-time window centers (using runs_index + timing constants)

Outputs per (sub, run, band):
- *_states.npy: shape [n_windows, D] (D depends on state_space or PCA dim)
- *_window_centers_meg_s.npy / *_window_centers_audio_s.npy
- *_pca_components.npy and *_pca_explained_variance.npy (when state_space=edge_pca)
- sidecar JSON capturing parameters and git hash
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Tuple

import mne
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA

from analysis.common import (
    FMRIS_SHIFT_SEC,
    MEG_AUDIO_DELAY_SEC,
    centers_to_audio_time,
    git_hash,
    iter_windows,
    load_run_to_story,
    load_runs_index,
)


# Canonical bands (≤40 Hz because of preprocessing)
BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 3.0),
    "theta": (4.0, 7.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 40.0),
}


def filter_and_hilbert(data: np.ndarray, sfreq: float, band: Tuple[float, float]) -> np.ndarray:
    """Bandpass then analytic signal."""
    filtered = mne.filter.filter_data(data, sfreq=sfreq, l_freq=band[0], h_freq=band[1], method="fir", verbose=False)
    return signal.hilbert(filtered, axis=-1)


def connectivity_window(analytic_win: np.ndarray, metric: str) -> np.ndarray:
    """
    analytic_win: [n_ch, n_time] complex analytic signal
    Returns connectivity matrix [n_ch, n_ch]
    """
    if metric == "plv":
        phase = np.angle(analytic_win)
        exp_phase = np.exp(1j * phase)
        plv_mat = np.abs(exp_phase @ exp_phase.conj().T) / analytic_win.shape[1]
        np.fill_diagonal(plv_mat, 1.0)
        return plv_mat
    elif metric == "wpli":
        cross = analytic_win[:, None, :] * analytic_win[None, :, :].conj()
        im = np.imag(cross)
        num = np.abs(im.mean(axis=-1))
        den = np.mean(np.abs(im), axis=-1) + 1e-12
        wpli = num / den
        np.fill_diagonal(wpli, 0.0)
        return wpli
    else:
        raise ValueError(f"Unknown metric {metric}")


def state_from_mat(mat: np.ndarray, state_space: str, tri_upper: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Convert connectivity matrix to state vector according to state_space."""
    if state_space == "node":
        return mat.mean(axis=1).astype(np.float32)
    elif state_space in {"edge", "edge_pca"}:
        return mat[tri_upper].astype(np.float32)
    else:
        raise ValueError(f"Unknown state_space {state_space}")


def pick_channels(raw: mne.io.BaseRaw, picks: str) -> mne.io.BaseRaw:
    raw = raw.copy()
    if picks == "grad":
        raw.pick_types(meg="grad")
    elif picks == "mag":
        raw.pick_types(meg="mag")
    elif picks == "all":
        raw.pick_types(meg=True)
    else:
        raise ValueError(f"Unknown picks {picks}")
    return raw


def main():
    p = argparse.ArgumentParser(description="Compute windowed PLV/wPLI states.")
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--run-to-story", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--band", choices=list(BANDS.keys()), default="theta")
    p.add_argument("--window-length", type=float, default=2.0)
    p.add_argument("--window-step", type=float, default=0.5)
    p.add_argument("--metric", choices=["plv", "wpli"], default="plv")
    p.add_argument("--state-space", choices=["node", "edge", "edge_pca"], default="edge_pca")
    p.add_argument("--pca-components", type=int, default=128)
    p.add_argument("--picks", choices=["grad", "mag", "all"], default="grad")
    p.add_argument("--decim", type=int, default=1, help="Decimate raw before filtering (integer factor).")
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--save-edges", action="store_true", help="Save edge vectors even when PCA is applied.")
    p.add_argument("--execute", action="store_true", help="Actually compute; otherwise dry run.")
    args = p.parse_args()

    if not args.execute:
        print("[dry-run] add --execute to compute states.")
        return

    runs_idx = load_runs_index(args.dataset_root)
    lut = load_run_to_story(args.run_to_story)
    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    tri_upper = None
    code_hash = git_hash(args.dataset_root)

    for sub in args.subjects:
        for run in args.runs:
            story_id = lut.get((sub, run))
            if story_id is None:
                print(f"[warn] no story mapping for sub-{sub} run-{run}, skipping")
                continue

            meg_path = (
                args.dataset_root
                / "derivatives"
                / "preprocessed_data"
                / f"sub-{sub}"
                / "MEG"
                / f"sub-{sub}_task-RDR_run-{run}_meg.fif"
            )
            raw = mne.io.read_raw_fif(meg_path, preload=True, verbose="ERROR")
            raw = pick_channels(raw, args.picks)
            if args.decim != 1:
                target_sfreq = raw.info["sfreq"] / args.decim
                raw.resample(target_sfreq, npad="auto", verbose="ERROR")
            sfreq = raw.info["sfreq"]
            data = raw.get_data()
            n_ch, n_samp = data.shape
            analytic = filter_and_hilbert(data, sfreq, BANDS[args.band])

            if tri_upper is None:
                tri_upper = np.triu_indices(n_ch, k=1)

            states: List[np.ndarray] = []
            centers_meg: List[float] = []
            for start, end, center in iter_windows(n_samp, sfreq, args.window_length, args.window_step):
                win = analytic[:, start:end]
                conn = connectivity_window(win, args.metric)
                states.append(state_from_mat(conn, args.state_space, tri_upper))
                centers_meg.append(center)

            states_arr = np.stack(states, axis=0) if states else np.zeros((0, 0), dtype=np.float32)
            centers_meg = np.array(centers_meg, dtype=np.float32)
            centers_audio = centers_to_audio_time(centers_meg, sub, run, runs_idx, story_id, args.dataset_root)

            base = out_root / f"sub-{sub}" / f"sub-{sub}_run-{run}_{args.band}_{args.metric}_{args.state_space}"
            base.parent.mkdir(parents=True, exist_ok=True)

            out_payload = states_arr
            pca_components = None
            explained = None
            if args.state_space == "edge_pca" and states_arr.size > 0:
                pca = PCA(n_components=min(args.pca_components, states_arr.shape[1]), svd_solver="randomized")
                out_payload = pca.fit_transform(states_arr)
                pca_components = pca.components_.astype(np.float32)
                explained = pca.explained_variance_ratio_.astype(np.float32)
                np.save(base.parent / f"{base.name}_pca_components.npy", pca_components)
                np.save(base.parent / f"{base.name}_pca_explained_variance.npy", explained)
                if args.save_edges:
                    np.save(base.parent / f"{base.name}_edges.npy", states_arr.astype(np.float32))

            np.save(base.with_suffix(".npy"), out_payload.astype(np.float32))
            np.save(base.parent / f"{base.name}_window_centers_meg_s.npy", centers_meg)
            np.save(base.parent / f"{base.name}_window_centers_audio_s.npy", centers_audio)

            sidecar = {
                "dataset_root": str(args.dataset_root),
                "subject": sub,
                "run": run,
                "story_id": story_id,
                "band": args.band,
                "band_range_hz": BANDS[args.band],
                "window_length_s": args.window_length,
                "window_step_s": args.window_step,
                "metric": args.metric,
                "state_space": args.state_space,
                "pca_components": None if pca_components is None else int(pca_components.shape[0]),
                "picks": args.picks,
                "decim": args.decim,
                "sfreq": sfreq,
                "n_channels": n_ch,
                "n_windows": int(out_payload.shape[0]),
                "state_dim": int(out_payload.shape[1]) if out_payload.ndim == 2 and out_payload.size else 0,
                "time_align_shift_s": FMRIS_SHIFT_SEC,
                "auditory_delay_s": MEG_AUDIO_DELAY_SEC,
                "run_to_story_mapping": str(args.run_to_story),
                "code_git_hash": code_hash,
                "pca_explained_variance": None if explained is None else explained.tolist(),
            }
            with base.with_suffix(".json").open("w") as f:
                json.dump(sidecar, f, indent=2)

            qc = {
                "mean_state": float(np.mean(out_payload)) if out_payload.size else 0.0,
                "std_state": float(np.std(out_payload)) if out_payload.size else 0.0,
                "min_state": float(np.min(out_payload)) if out_payload.size else 0.0,
                "max_state": float(np.max(out_payload)) if out_payload.size else 0.0,
            }
            with (base.parent / f"{base.name}_qc.json").open("w") as f:
                json.dump(qc, f, indent=2)
            print(f"[saved] {base.with_suffix('.npy')} shape={out_payload.shape}, metric={args.metric}, space={args.state_space}")


if __name__ == "__main__":
    main()
