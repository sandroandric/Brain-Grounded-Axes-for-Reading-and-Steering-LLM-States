"""
Encoding/GLM scaffold: predict PLV states from word-level features with lags.

Inputs
- PLV states (edge-PCA, theta) from outputs/meg_plv_states
- time_align (word-level), logfreq, embeddings (for embedding_change), POS tags
- run-to-story mapping (mapping/id_map.csv)

Features per word token (concatenated):
- logfreq (scalar)
- embedding_change (scalar; L2 diff between successive embeddings)
- POS one-hot

Design matrix per window:
- For each lag in --lags (sec), pick the latest word onset before (window_center - lag) and use its feature vector.
- Concatenate features across lags.

Model
- Ridge regression (sklearn) predicting PLV PCs (already 128D) from design matrix.
- Trained per subject across all specified runs.

Outputs per subject (under --out-dir/sub-XX):
- weights.npy (feature_dim x plv_dim) and sidecar.json
- preds/sub-XX_run-YY_preds.npy (model predictions per window)
- word_atlas.npy (mean predicted PLV vector per word) + word_vocab.json (word list + counts)
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from analysis.common import (
    MEG_AUDIO_DELAY_SEC,
    centers_to_audio_time,
    git_hash,
    iter_windows,
    load_run_to_story,
    load_runs_index,
    load_time_alignment,
)
from scipy.io import loadmat


def build_global_pos_vocab(dataset_root: pathlib.Path, mapping: Dict[Tuple[str, str], str]) -> List[str]:
    tags_set = set()
    for (_, _), story_id in mapping.items():
        path = (
            dataset_root
            / "derivatives"
            / "annotations"
            / "syntactic_annotations"
            / "part_of_speech"
            / f"{story_id}_pos.txt"
        )
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    tags_set.add(parts[2])
    return sorted(tags_set)


def load_plv_state(plv_root: pathlib.Path, sub: str, run: str, band: str, metric: str, state_space: str):
    base = plv_root / f"sub-{sub}" / f"sub-{sub}_run-{run}_{band}_{metric}_{state_space}"
    states = np.load(base.with_suffix(".npy"))
    centers = np.load(base.parent / f"{base.name}_window_centers_audio_s.npy")
    return states, centers


def load_embeddings(
    dataset_root: pathlib.Path,
    story_id: str,
    level: str = "word",
    model: str = "gpt",
    word2vec_dim: int = 300,
):
    if model == "gpt":
        path = (
            dataset_root
            / "derivatives"
            / "annotations"
            / "embeddings"
            / "gpt"
            / f"{level}-level"
            / f"{story_id}_{level}_gpt_0-24_1024.mat"
        )
    elif model == "word2vec":
        path = (
            dataset_root
            / "derivatives"
            / "annotations"
            / "embeddings"
            / "word2vec"
            / f"{level}-level"
            / f"{word2vec_dim}d"
            / f"{story_id}_{level}_word2vec.mat"
        )
    else:
        raise ValueError(f"Unknown embedding model: {model}")
    mat = loadmat(path, struct_as_record=False, squeeze_me=True)
    data = np.asarray(mat["data"], dtype=np.float32)
    if data.ndim == 3:
        embeds = data.mean(axis=0)
    elif data.ndim == 2:
        embeds = data
    else:
        raise ValueError(f"Unexpected GPT embed shape {data.shape} in {path}")
    return embeds


def embedding_change_feature(embeds: np.ndarray) -> np.ndarray:
    diff = np.linalg.norm(np.diff(embeds, axis=0), axis=1)
    return np.concatenate(([diff[0]], diff)).astype(np.float32)


def load_logfreq(dataset_root: pathlib.Path, story_id: str) -> np.ndarray:
    path = dataset_root / "derivatives" / "annotations" / "frequency" / "word-level" / f"{story_id}_word_logfreq.mat"
    mat = loadmat(path)
    wf = np.asarray(mat["wf"]).squeeze().astype(np.float32)
    return wf


def load_pos_ids(dataset_root: pathlib.Path, story_id: str, pos_vocab: List[str]) -> np.ndarray:
    path = (
        dataset_root
        / "derivatives"
        / "annotations"
        / "syntactic_annotations"
        / "part_of_speech"
        / f"{story_id}_pos.txt"
    )
    rows = []
    with path.open() as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                rows.append(parts[2])
    tags = pd.Categorical(rows, categories=pos_vocab)
    return tags.codes.astype(np.int32)


def build_word_features(
    dataset_root: pathlib.Path,
    story_id: str,
    pos_vocab: List[str],
    features: List[str],
    embed_model: str,
    word2vec_dim: int,
    shuffle_emb: bool,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[str]]:
    tokens, starts, _ = load_time_alignment(dataset_root, story_id, level="word")
    tokens = [str(t) for t in tokens]
    wf = None
    emb_change = None
    pos_ids = None

    lengths = [len(tokens)]
    if "logfreq" in features:
        wf = load_logfreq(dataset_root, story_id)
        lengths.append(len(wf))
    if "embedding_change" in features:
        embeds = load_embeddings(
            dataset_root,
            story_id,
            level="word",
            model=embed_model,
            word2vec_dim=word2vec_dim,
        )
        emb_change = embedding_change_feature(embeds)
        if shuffle_emb:
            emb_change = emb_change[rng.permutation(len(emb_change))]
        lengths.append(len(emb_change))
    if "pos_id" in features:
        pos_ids = load_pos_ids(dataset_root, story_id, pos_vocab)
        lengths.append(len(pos_ids))

    if not features:
        raise ValueError("No features selected.")

    T = min(lengths)
    tokens = tokens[:T]
    starts = starts[:T]
    if wf is not None:
        wf = wf[:T]
    if emb_change is not None:
        emb_change = emb_change[:T]
    if pos_ids is not None:
        pos_ids = pos_ids[:T]

    n_pos = len(pos_vocab) if "pos_id" in features else 0
    feats = []
    for i in range(T):
        parts = []
        if wf is not None:
            parts.append(np.array([wf[i]], dtype=np.float32))
        if emb_change is not None:
            parts.append(np.array([emb_change[i]], dtype=np.float32))
        if pos_ids is not None:
            pos_onehot = np.zeros(n_pos, dtype=np.float32)
            pos_onehot[pos_ids[i]] = 1.0
            parts.append(pos_onehot)
        feat_vec = np.concatenate(parts, axis=0)
        feats.append(feat_vec)
    feats = np.stack(feats, axis=0)
    return feats, tokens


def align_with_lags(window_centers: np.ndarray, word_starts: np.ndarray, word_feats: np.ndarray, lags: List[float]):
    """
    For each window center, and each lag, pick the last word onset before (center - lag).
    Returns X of shape [n_windows, feat_dim * n_lags]
    """
    n_win = len(window_centers)
    feat_dim = word_feats.shape[1]
    X = np.zeros((n_win, feat_dim * len(lags)), dtype=np.float32)
    word_idx = np.searchsorted(word_starts, window_centers, side="right") - 1
    word_idx = np.clip(word_idx, 0, len(word_starts) - 1)
    for j, lag in enumerate(lags):
        shifted = window_centers - lag
        idx = np.searchsorted(word_starts, shifted, side="right") - 1
        idx = np.clip(idx, 0, len(word_starts) - 1)
        X[:, j * feat_dim : (j + 1) * feat_dim] = word_feats[idx]
    return X


def aggregate_word_atlas(tokens_all: List[str], centers_all: List[float], preds_all: List[np.ndarray]):
    word_vecs = {}
    counts = {}
    for tok, pred in zip(tokens_all, preds_all):
        if tok not in word_vecs:
            word_vecs[tok] = pred.copy()
            counts[tok] = 1
        else:
            word_vecs[tok] += pred
            counts[tok] += 1
    vocab = sorted(word_vecs.keys())
    mat = np.stack([word_vecs[w] / counts[w] for w in vocab], axis=0)
    return vocab, mat, counts


def main():
    p = argparse.ArgumentParser(description="Encoding model: predict PLV PCs from word features with lags.")
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--plv-root", type=pathlib.Path, required=True)
    p.add_argument("--run-to-story", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--band", default="theta")
    p.add_argument("--metric", default="plv")
    p.add_argument("--state-space", default="edge_pca")
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--lags", nargs="+", type=float, default=[0.0, 0.5, 1.0], help="Lags in seconds for design matrix.")
    p.add_argument("--features", nargs="+", default=["logfreq", "embedding_change", "pos_id"])
    p.add_argument("--embed-model", choices=["gpt", "word2vec"], default="gpt")
    p.add_argument("--word2vec-dim", type=int, choices=[100, 300], default=300)
    p.add_argument("--shuffle-embedding-change", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization.")
    p.add_argument("--standardize", action="store_true", help="Standardize design matrix (recommended).")
    args = p.parse_args()

    lut = load_run_to_story(args.run_to_story)
    code_hash = git_hash(args.dataset_root)
    pos_vocab_global = build_global_pos_vocab(args.dataset_root, lut) if "pos_id" in args.features else []
    rng = np.random.default_rng(args.seed)

    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    for sub in args.subjects:
        X_all = []
        Y_all = []
        tokens_all = []
        preds_tokens = []
        for run in args.runs:
            story_id = lut.get((sub, run))
            if story_id is None:
                continue
            try:
                plv, centers_audio = load_plv_state(args.plv_root, sub, run, args.band, args.metric, args.state_space)
            except FileNotFoundError:
                continue
            feats, tokens = build_word_features(
                args.dataset_root,
                story_id,
                pos_vocab_global,
                args.features,
                args.embed_model,
                args.word2vec_dim,
                args.shuffle_embedding_change,
                rng,
            )
            # word starts in audio time
            _, starts, _ = load_time_alignment(args.dataset_root, story_id, level="word")
            starts = starts[: len(feats)]

            X = align_with_lags(centers_audio, starts, feats, args.lags)
            n = min(len(X), len(plv))
            X = X[:n]
            Y = plv[:n]
            X_all.append(X)
            Y_all.append(Y)
            # store tokens aligned to window centers for atlas from preds later
            idx = np.searchsorted(starts, centers_audio[:n], side="right") - 1
            idx = np.clip(idx, 0, len(tokens) - 1)
            tokens_all.extend([tokens[i] for i in idx])
        if not X_all:
            print(f"[warn] no data for sub-{sub}")
            continue
        X_all = np.concatenate(X_all, axis=0)
        Y_all = np.concatenate(Y_all, axis=0)

        scaler = None
        if args.standardize:
            scaler = StandardScaler()
            X_all = scaler.fit_transform(X_all)

        model = Ridge(alpha=args.alpha, fit_intercept=True)
        model.fit(X_all, Y_all)

        sub_dir = out_root / f"sub-{sub}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        np.save(sub_dir / "weights.npy", model.coef_.astype(np.float32))
        sidecar = {
            "subject": sub,
            "band": args.band,
            "metric": args.metric,
            "state_space": args.state_space,
            "lags": args.lags,
            "alpha": args.alpha,
            "standardize": bool(args.standardize),
            "plv_dim": int(Y_all.shape[1]),
            "feature_dim_per_lag": int(X_all.shape[1] / len(args.lags)),
            "n_samples": int(X_all.shape[0]),
            "feature_names": args.features,
            "embed_model": args.embed_model if "embedding_change" in args.features else None,
            "word2vec_dim": args.word2vec_dim if args.embed_model == "word2vec" else None,
            "shuffle_embedding_change": bool(args.shuffle_embedding_change),
            "pos_vocab_size": int(len(pos_vocab_global)),
            "code_git_hash": code_hash,
        }
        with (sub_dir / "weights_sidecar.json").open("w") as f:
            json.dump(sidecar, f, indent=2)
        if scaler is not None:
            np.save(sub_dir / "scaler_mean.npy", scaler.mean_.astype(np.float32))
            np.save(sub_dir / "scaler_scale.npy", scaler.scale_.astype(np.float32))

        # predictions per run + word atlas from predictions
        preds_dir = sub_dir / "preds"
        preds_dir.mkdir(exist_ok=True)
        tokens_all = []
        preds_tokens = []
        for run in args.runs:
            story_id = lut.get((sub, run))
            if story_id is None:
                continue
            try:
                plv, centers_audio = load_plv_state(args.plv_root, sub, run, args.band, args.metric, args.state_space)
            except FileNotFoundError:
                continue
            feats, tokens = build_word_features(
                args.dataset_root,
                story_id,
                pos_vocab_global,
                args.features,
                args.embed_model,
                args.word2vec_dim,
                args.shuffle_embedding_change,
                rng,
            )
            _, starts, _ = load_time_alignment(args.dataset_root, story_id, level="word")
            starts = starts[: len(feats)]
            X = align_with_lags(centers_audio, starts, feats, args.lags)
            n = min(len(X), len(plv))
            X = X[:n]
            if scaler is not None:
                X = scaler.transform(X)
            pred = model.predict(X)
            np.save(preds_dir / f"sub-{sub}_run-{run}_preds.npy", pred.astype(np.float32))
            idx = np.searchsorted(starts, centers_audio[:n], side="right") - 1
            idx = np.clip(idx, 0, len(tokens) - 1)
            tokens_all.extend([tokens[i] for i in idx])
            preds_tokens.extend([p for p in pred])

        vocab, word_mat, counts = aggregate_word_atlas(tokens_all, [], preds_tokens)
        np.save(sub_dir / "word_atlas.npy", word_mat.astype(np.float32))
        with (sub_dir / "word_vocab.json").open("w") as f:
            json.dump({"vocab": vocab, "counts": counts}, f, indent=2)
        print(f"[saved] sub-{sub} weights + word atlas ({len(vocab)} words)")


if __name__ == "__main__":
    main()
