"""
Compute window-level semantic features aligned to MEG windows (audio-onset timebase).

Features implemented:
- embedding_change (GPT or word2vec) at word level (L2 norm of successive embeddings)
- logfreq (word log frequency)
- pos_id (integer-coded POS tags)

Alignment/timebase:
- time_align files are fMRI-shifted; convert to audio by subtracting 10.65 s and adding 0.0395 s delay.
- MEG window centers are converted to audio-onset seconds using runs_index (trig_start, delay) so features align with PLV states.

Outputs (per run):
- features.npy (shape [n_windows, K])
- feature_names.json
- window_centers_s.npy (audio-onset centers) and _window_centers_meg_s.npy
- sidecar.json (window params, feature list, timing constants, code hash)
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat

from analysis.common import (
    MEG_AUDIO_DELAY_SEC,
    centers_to_audio_time,
    git_hash,
    iter_windows,
    load_run_to_story,
    load_runs_index,
    load_time_alignment,
)


@dataclass
class WindowDef:
    length: float
    step: float


def load_embeddings(dataset_root: pathlib.Path, story_id: str, model: str = "gpt", level: str = "word"):
    base = dataset_root / "derivatives" / "annotations" / "embeddings"
    if model == "gpt":
        path = base / "gpt" / f"{level}-level" / f"{story_id}_{level}_gpt_0-24_1024.mat"
        mat = loadmat(path, struct_as_record=False, squeeze_me=True)
        data = np.asarray(mat["data"], dtype=np.float32)
        if data.ndim == 3:
            embeds = data.mean(axis=0)
        elif data.ndim == 2:
            embeds = data
        else:
            raise ValueError(f"Unexpected GPT embed shape {data.shape} in {path}")
    elif model == "word2vec":
        path = base / "word2vec" / f"{level}-level" / "100d" / f"{story_id}_{level}_word2vec.mat"
        mat = loadmat(path, struct_as_record=False, squeeze_me=True)
        embeds = np.asarray(mat["data"], dtype=np.float32)
    else:
        raise ValueError(f"Unknown embed model {model}")
    return embeds


def embedding_change_feature(embeds: np.ndarray) -> np.ndarray:
    diff = np.linalg.norm(np.diff(embeds, axis=0), axis=1)
    return np.concatenate(([diff[0]], diff)).astype(np.float32)


def load_logfreq(dataset_root: pathlib.Path, story_id: str) -> np.ndarray:
    path = dataset_root / "derivatives" / "annotations" / "frequency" / "word-level" / f"{story_id}_word_logfreq.mat"
    mat = loadmat(path)
    wf = np.asarray(mat["wf"]).squeeze().astype(np.float32)
    return wf


def load_pos_ids(dataset_root: pathlib.Path, story_id: str) -> np.ndarray:
    path = dataset_root / "derivatives" / "annotations" / "syntactic_annotations" / "part_of_speech" / f"{story_id}_pos.txt"
    rows = []
    with path.open() as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                rows.append(parts[2])
    tags = pd.Categorical(rows)
    return tags.codes.astype(np.int32), list(tags.categories)


def align_to_windows(window_centers: np.ndarray, starts: np.ndarray, values: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(starts, window_centers, side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return values[idx]


def main():
    p = argparse.ArgumentParser(description="Compute window-level semantic features aligned to MEG windows.")
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--run-to-story", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--window-length", type=float, default=2.0)
    p.add_argument("--window-step", type=float, default=0.5)
    p.add_argument("--features", nargs="+", default=["embedding_change", "logfreq", "pos_id"])
    p.add_argument("--embed-model", choices=["gpt", "word2vec"], default="gpt")
    p.add_argument("--execute", action="store_true", help="Actually compute; otherwise dry run.")
    args = p.parse_args()

    lut = load_run_to_story(args.run_to_story)
    window = WindowDef(length=args.window_length, step=args.window_step)
    code_hash = git_hash(args.dataset_root)
    runs_idx = load_runs_index(args.dataset_root)

    if not args.execute:
        print("[dry-run] add --execute to run.")
        return

    for sub in args.subjects:
        for run in args.runs:
            story_id = lut.get((sub, run))
            if story_id is None:
                print(f"[warn] no story for sub-{sub} run-{run}, skipping")
                continue
            meg_path = (
                args.dataset_root
                / "derivatives"
                / "preprocessed_data"
                / f"sub-{sub}"
                / "MEG"
                / f"sub-{sub}_task-RDR_run-{run}_meg.fif"
            )
            raw = mne.io.read_raw_fif(meg_path, preload=False, verbose="ERROR")
            sfreq = raw.info["sfreq"]
            n_samples = raw.n_times
            centers_meg = np.array([c for _, _, c in iter_windows(n_samples, sfreq, window.length, window.step)], dtype=np.float32)
            centers_audio = centers_to_audio_time(centers_meg, sub, run, runs_idx, story_id, args.dataset_root)

            tokens, starts, _ = load_time_alignment(args.dataset_root, story_id, level="word")

            feat_arrays = []
            feat_names = []
            if "embedding_change" in args.features:
                embeds = load_embeddings(args.dataset_root, story_id, model=args.embed_model, level="word")
                T = min(len(tokens), embeds.shape[0])
                embeds = embeds[:T]
                starts_use = starts[:T]
                emb_change = embedding_change_feature(embeds)
                feat_arrays.append(align_to_windows(centers_audio, starts_use, emb_change))
                feat_names.append("embedding_change")

            if "logfreq" in args.features:
                wf = load_logfreq(args.dataset_root, story_id)
                T = min(len(tokens), len(wf))
                wf = wf[:T]
                starts_use = starts[:T]
                feat_arrays.append(align_to_windows(centers_audio, starts_use, wf))
                feat_names.append("logfreq")

            if "pos_id" in args.features:
                pos_ids, pos_vocab = load_pos_ids(args.dataset_root, story_id)
                T = min(len(tokens), len(pos_ids))
                pos_ids = pos_ids[:T]
                starts_use = starts[:T]
                feat_arrays.append(align_to_windows(centers_audio, starts_use, pos_ids))
                feat_names.append("pos_id")

            if not feat_arrays:
                print(f"[warn] no features computed for sub-{sub} run-{run}")
                continue

            features_mat = np.stack(feat_arrays, axis=1).astype(np.float32)
            out_dir = args.out_dir / f"sub-{sub}"
            out_dir.mkdir(parents=True, exist_ok=True)
            base = out_dir / f"sub-{sub}_run-{run}_features"
            np.save(base.with_suffix(".npy"), features_mat)
            np.save(base.parent / f"{base.name}_window_centers_s.npy", centers_audio)
            np.save(base.parent / f"{base.name}_window_centers_meg_s.npy", centers_meg)
            with (base.parent / f"{base.name}_names.json").open("w") as f:
                json.dump(feat_names, f, indent=2)
            sidecar = {
                "dataset_root": str(args.dataset_root),
                "subject": sub,
                "run": run,
                "story_id": story_id,
                "window_length_s": window.length,
                "window_step_s": window.step,
                "features": feat_names,
                "embed_model": args.embed_model,
                "sfreq": sfreq,
                "n_windows": features_mat.shape[0],
                "timebase": "audio_onset",
                "time_align_shift_s": 10.65,
                "auditory_delay_s": MEG_AUDIO_DELAY_SEC,
                "missing_start_trigger": bool(runs_idx.loc[(sub, run)]["missing_start_trigger"] == "True"),
                "run_to_story_mapping": str(args.run_to_story),
                "code_git_hash": code_hash,
            }
            with base.with_suffix(".json").open("w") as f:
                json.dump(sidecar, f, indent=2)
            print(f"[saved] {base.with_suffix('.npy')} ({features_mat.shape})")


if __name__ == "__main__":
    main()
