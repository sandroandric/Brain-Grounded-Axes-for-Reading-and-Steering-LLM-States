"""
Box 6: Brain-Derived Steering Evaluation

Uses ACTUAL brain axis directions extracted from the LLM adapter weights.
The adapter weight vector for each axis IS the direction in LLM embedding
space that maximally predicts that brain axis.

Key difference from steering_eval.py:
- OLD: Steering vectors from word contrast lists (proxy)
- NEW: Steering vectors from trained adapter weights (actual brain data)

Run:
python analysis/steering_eval_brain.py \
  --atlas-root outputs/encoding/plv_glm \
  --tinyllama-root derivatives/annotations/embeddings/tinyllama/word-level \
  --time-align-root derivatives/annotations/time_align/word-level \
  --out-dir outputs/steering_brain \
  --device mps
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import scipy.io
import torch
from scipy import stats
from sklearn.decomposition import FastICA
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModelForCausalLM


# Genuine axes from confound analysis
GENUINE_AXES = {
    2: "concreteness",
    10: "concreteness_neg",
    12: "concreteness",
    13: "arousal_neg",
    15: "affect",
    16: "arousal",
    19: "conc_arousal",
}

# Semantic interpretations for evaluation
AXIS_SEMANTICS = {
    2: ("concreteness", +1),   # Higher = more concrete
    12: ("concreteness", +1),  # Higher = more concrete
    13: ("arousal", -1),       # Higher = LESS arousing (function words)
    15: ("affect", +1),        # Higher = positive affect
    16: ("arousal", +1),       # Higher = more arousing
    19: ("mixed", +1),         # Concreteness + arousal
}


def clean_word(w: str) -> str:
    return re.sub(r'\s+', '', str(w).strip())


def load_sub_atlas(sub_dir: pathlib.Path):
    atlas = np.load(sub_dir / "word_atlas.npy")
    meta = json.load((sub_dir / "word_vocab.json").open())
    vocab = meta["vocab"]
    return vocab, atlas


def build_split_atlas(atlas_root: pathlib.Path, subjects: List[str]) -> Tuple[List[str], np.ndarray]:
    union = set()
    per_sub = {}

    for sub in subjects:
        sub_dir = atlas_root / f"sub-{sub}"
        if not sub_dir.exists():
            continue
        vocab, atlas = load_sub_atlas(sub_dir)
        per_sub[sub] = (vocab, atlas)
        union.update(vocab)

    vocab_union = sorted(list(union))
    word_to_idx = {w: i for i, w in enumerate(vocab_union)}

    sum_mat = None
    counts = np.zeros(len(vocab_union), dtype=np.int32)

    for sub, (vocab, atlas) in per_sub.items():
        mat = np.zeros((len(vocab_union), atlas.shape[1]), dtype=np.float32)
        for i, w in enumerate(vocab):
            j = word_to_idx[w]
            mat[j] = atlas[i]
        if sum_mat is None:
            sum_mat = mat
        else:
            sum_mat += mat
        counts += np.array([w in vocab for w in vocab_union], dtype=np.int32)

    counts = np.maximum(counts, 1)
    avg_mat = sum_mat / counts[:, None]

    return vocab_union, avg_mat


def load_story_tinyllama_embeddings(npy_path: pathlib.Path, layer: int = -1) -> np.ndarray:
    data = np.load(str(npy_path))
    return data[layer]


def load_story_words(time_align_path: pathlib.Path) -> List[str]:
    mat = scipy.io.loadmat(str(time_align_path))
    words = mat['word'].flatten()
    return [clean_word(str(w)) for w in words]


def build_word_type_embeddings(
    tinyllama_root: pathlib.Path,
    time_align_root: pathlib.Path,
    hidden_dim: int = 2048,
) -> Dict[str, np.ndarray]:
    word_emb_sums = defaultdict(lambda: np.zeros(hidden_dim))
    word_counts = defaultdict(int)

    for story_id in range(1, 61):
        tinyllama_files = list(tinyllama_root.glob(f"story_{story_id}_word_tinyllama*.npy"))
        if not tinyllama_files:
            continue
        tinyllama_path = tinyllama_files[0]

        time_path = time_align_root / f"story_{story_id}_word_time.mat"
        if not time_path.exists():
            continue

        embeddings = load_story_tinyllama_embeddings(tinyllama_path, layer=-1)
        words = load_story_words(time_path)

        min_len = min(len(words), len(embeddings))
        for word, emb in zip(words[:min_len], embeddings[:min_len]):
            if word:
                word_emb_sums[word] += emb
                word_counts[word] += 1

    result = {}
    for word in word_emb_sums:
        result[word] = word_emb_sums[word] / word_counts[word]

    return result


def train_adapter_and_extract_weights(
    tinyllama_embeddings: Dict[str, np.ndarray],
    vocab: List[str],
    ica_scores: np.ndarray,
    axes_to_train: List[int],
) -> Dict[int, np.ndarray]:
    """Train Ridge adapter and extract weight vectors as brain-derived steering directions."""

    # Build matched data
    X_list = []
    Y_list = []
    vocab_clean = {clean_word(w): i for i, w in enumerate(vocab)}

    for word, emb in tinyllama_embeddings.items():
        word_clean = clean_word(word)
        if word_clean in vocab_clean:
            idx = vocab_clean[word_clean]
            X_list.append(emb)
            Y_list.append(ica_scores[idx])

    X = np.array(X_list)
    Y = np.array(Y_list)

    print(f"  Training data: {X.shape[0]} words, {X.shape[1]} dims", flush=True)

    # Train adapter for each axis and extract weights
    steering_vectors = {}
    alphas = np.logspace(-1, 5, 20)

    for ax in axes_to_train:
        model = RidgeCV(alphas=alphas, cv=3)
        model.fit(X, Y[:, ax])

        # The coefficient vector IS the brain axis direction in embedding space
        weight_vector = model.coef_

        # Normalize for steering
        weight_vector = weight_vector / (np.linalg.norm(weight_vector) + 1e-8)

        steering_vectors[ax] = weight_vector

        # Report training performance
        y_pred = model.predict(X)
        r, p = pearsonr(y_pred, Y[:, ax])
        print(f"  Axis {ax}: r={r:.4f}, p={p:.2e}, alpha={model.alpha_:.1f}", flush=True)

    return steering_vectors


class SteeringHook:
    """Hook that applies steering vector to model activations."""

    def __init__(self, steering_vector: torch.Tensor, strength: float = 1.0):
        self.steering_vector = steering_vector
        self.strength = strength

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            steered = hidden + self.strength * self.steering_vector.unsqueeze(0).unsqueeze(0)
            return (steered,) + output[1:]
        else:
            return output + self.strength * self.steering_vector.unsqueeze(0).unsqueeze(0)


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    steering_vector: Optional[np.ndarray] = None,
    strength: float = 1.0,
    layer_idx: int = 11,
    max_new_tokens: int = 50,
    device: str = "mps",
    temperature: float = 0.7,
) -> Tuple[str, float]:
    """Generate text with optional steering vector applied."""
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    hook_handle = None

    if steering_vector is not None:
        steering_tensor = torch.tensor(steering_vector, dtype=model.dtype).to(device)

        if hasattr(model, 'model'):
            layers = model.model.layers
        elif hasattr(model, 'transformer'):
            layers = model.transformer.h
        else:
            raise ValueError("Unknown model architecture")

        target_layer = layers[layer_idx]
        hook = SteeringHook(steering_tensor, strength)
        hook_handle = target_layer.register_forward_hook(hook)

    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        if outputs.scores:
            log_probs = []
            for i, score in enumerate(outputs.scores):
                probs = torch.softmax(score, dim=-1)
                token_id = outputs.sequences[0, inputs.input_ids.shape[1] + i]
                log_prob = torch.log(probs[0, token_id] + 1e-10)
                log_probs.append(log_prob.item())

            avg_log_prob = np.mean(log_probs) if log_probs else 0
            perplexity = np.exp(-avg_log_prob)
        else:
            perplexity = float('nan')

    finally:
        if hook_handle is not None:
            hook_handle.remove()

    return generated_text, perplexity


def compute_semantic_projection(
    text: str,
    steering_vectors: Dict[int, np.ndarray],
    model,
    tokenizer,
    device: str,
) -> Dict[int, float]:
    """Project generated text onto brain axis directions."""
    model.eval()

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0].cpu().numpy()

    mean_hidden = np.mean(hidden, axis=0)

    projections = {}
    for ax, sv in steering_vectors.items():
        proj = float(np.dot(mean_hidden, sv))
        projections[ax] = proj

    return projections


def cohens_d(group1: List[float], group2: List[float]) -> float:
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)


def bootstrap_ci(data: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> tuple:
    data = np.array(data)
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    alpha = (1 - ci) / 2
    return np.percentile(means, alpha * 100), np.percentile(means, (1 - alpha) * 100)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas-root", type=pathlib.Path, required=True)
    p.add_argument("--tinyllama-root", type=pathlib.Path, required=True)
    p.add_argument("--time-align-root", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--device", default="mps")
    p.add_argument("--layer-idx", type=int, default=11)
    p.add_argument("--n-prompts", type=int, default=20, help="Number of test prompts")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("BOX 6: BRAIN-DERIVED STEERING EVALUATION", flush=True)
    print("=" * 70, flush=True)
    print("\nKey difference from proxy approach:", flush=True)
    print("  OLD: Steering vectors from English word contrasts", flush=True)
    print("  NEW: Steering vectors from trained adapter weights (ACTUAL BRAIN DATA)", flush=True)

    # ===== PART 1: EXTRACT BRAIN-DERIVED STEERING VECTORS =====
    print("\n" + "=" * 70, flush=True)
    print("1. EXTRACTING BRAIN-DERIVED STEERING VECTORS", flush=True)
    print("=" * 70, flush=True)

    # Build atlas from all subjects (we need steering vectors, not testing adapter)
    all_subjects = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    print(f"\n[build] Building word atlas from all {len(all_subjects)} subjects...", flush=True)
    vocab, atlas = build_split_atlas(args.atlas_root, all_subjects)
    print(f"  {len(vocab)} words, {atlas.shape[1]} features", flush=True)

    # Run ICA
    print("\n[ica] Running ICA with 20 components...", flush=True)
    ica_model = FastICA(n_components=20, random_state=42, max_iter=500)
    ica_scores = ica_model.fit_transform(atlas)

    # Load TinyLlama embeddings
    print("\n[load] Loading TinyLlama embeddings...", flush=True)
    tinyllama_embeddings = build_word_type_embeddings(args.tinyllama_root, args.time_align_root)
    print(f"  {len(tinyllama_embeddings)} word types", flush=True)

    # Train adapters and extract brain-derived steering vectors
    axes_to_test = list(GENUINE_AXES.keys())
    print(f"\n[train] Training adapters for genuine axes: {axes_to_test}", flush=True)
    brain_steering_vectors = train_adapter_and_extract_weights(
        tinyllama_embeddings, vocab, ica_scores, axes_to_test
    )

    # Save steering vectors
    for ax, sv in brain_steering_vectors.items():
        np.save(args.out_dir / f"brain_steering_axis_{ax}.npy", sv)
    print(f"\n✓ Saved {len(brain_steering_vectors)} brain-derived steering vectors", flush=True)

    # ===== PART 2: LOAD LLM FOR STEERING =====
    print("\n" + "=" * 70, flush=True)
    print("2. LOADING LLM FOR STEERING", flush=True)
    print("=" * 70, flush=True)

    print(f"\n[load] Loading {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device != "cpu" else torch.float32,
    )
    model = model.to(args.device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"  Loaded on {args.device}, {n_layers} layers", flush=True)
    print(f"  Steering layer: {args.layer_idx}", flush=True)

    # ===== PART 3: STEERING EVALUATION =====
    print("\n" + "=" * 70, flush=True)
    print("3. STEERING EVALUATION", flush=True)
    print("=" * 70, flush=True)

    # Extended test prompts for adequate statistical power
    test_prompts = [
        "The old man walked slowly through the",
        "She felt a sudden wave of",
        "The scientist discovered that the",
        "In the quiet village, people often",
        "The story begins with a",
        "After years of hard work, she finally",
        "The ancient temple stood silent in the",
        "He remembered the day when his",
        "The children played happily in the",
        "Deep in the forest, there lived a",
        "The news spread quickly through the",
        "She opened the door and saw",
        "The machine started making strange",
        "In the distance, they could hear",
        "The letter contained shocking",
        "Every morning, he would sit by the",
        "The meeting was interrupted by a",
        "She couldn't believe what she had just",
        "The city was known for its",
        "As the sun set, the sky turned",
        "The experiment yielded unexpected",
        "He spent hours thinking about",
        "The book described a world where",
        "They traveled for days until they reached",
        "The music reminded her of",
    ][:args.n_prompts]

    print(f"\n[eval] Testing with {len(test_prompts)} prompts", flush=True)

    strengths = [-1.5, -0.75, 0.0, 0.75, 1.5]

    all_results = {
        "method": "brain_derived",
        "model": args.model,
        "layer_idx": args.layer_idx,
        "n_prompts": len(test_prompts),
        "axes": {},
    }

    # Test each brain axis
    for ax in axes_to_test:
        axis_name = GENUINE_AXES[ax]
        print(f"\n[axis {ax}] Testing {axis_name}...", flush=True)

        sv = brain_steering_vectors[ax]
        axis_results = {"generations": {}, "prompts": test_prompts}

        for strength in strengths:
            key = f"strength_{strength:.2f}"
            axis_results["generations"][key] = []

            for prompt in test_prompts:
                if strength == 0.0:
                    text, ppl = generate_with_steering(
                        model, tokenizer, prompt,
                        steering_vector=None,
                        device=args.device,
                    )
                else:
                    text, ppl = generate_with_steering(
                        model, tokenizer, prompt,
                        steering_vector=sv,
                        strength=strength,
                        layer_idx=args.layer_idx,
                        device=args.device,
                    )

                # Project back to brain axes
                projections = compute_semantic_projection(
                    text, brain_steering_vectors, model, tokenizer, args.device
                )

                axis_results["generations"][key].append({
                    "text": text,
                    "perplexity": float(ppl) if not np.isnan(ppl) else None,
                    "projections": {str(k): float(v) for k, v in projections.items()},
                })

        all_results["axes"][str(ax)] = axis_results

        # Quick preview
        neg_proj = np.mean([g["projections"][str(ax)] for g in axis_results["generations"]["strength_-1.50"]])
        pos_proj = np.mean([g["projections"][str(ax)] for g in axis_results["generations"]["strength_1.50"]])
        print(f"  Mean projection: neg={neg_proj:+.2f}, pos={pos_proj:+.2f}, Δ={pos_proj-neg_proj:+.2f}", flush=True)

    # ===== PART 4: STATISTICAL ANALYSIS =====
    print("\n" + "=" * 70, flush=True)
    print("4. STATISTICAL ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    validation = {"per_axis": {}, "summary": {}}

    print(f"\n{'Axis':<6} {'Name':<15} {'Δ Proj':<10} {'t':<10} {'p':<12} {'Cohen d':<10} {'Sig':<6}", flush=True)
    print("-" * 79, flush=True)

    n_sig = 0
    all_d = []

    for ax in axes_to_test:
        axis_name = GENUINE_AXES[ax]
        axis_data = all_results["axes"][str(ax)]

        neg_projs = [g["projections"][str(ax)] for g in axis_data["generations"]["strength_-1.50"]]
        pos_projs = [g["projections"][str(ax)] for g in axis_data["generations"]["strength_1.50"]]
        base_projs = [g["projections"][str(ax)] for g in axis_data["generations"]["strength_0.00"]]

        # Statistical tests
        t_stat, p_value = stats.ttest_ind(pos_projs, neg_projs)
        d = cohens_d(pos_projs, neg_projs)
        delta = np.mean(pos_projs) - np.mean(neg_projs)

        sig = "✓" if p_value < 0.05 else ""
        if p_value < 0.05:
            n_sig += 1
        all_d.append(abs(d))

        print(f"{ax:<6} {axis_name:<15} {delta:+<10.2f} {t_stat:<10.2f} {p_value:<12.4f} {d:<10.2f} {sig:<6}", flush=True)

        # Bootstrap CIs
        neg_ci = bootstrap_ci(neg_projs)
        pos_ci = bootstrap_ci(pos_projs)

        # Perplexity analysis
        neg_ppl = [g["perplexity"] for g in axis_data["generations"]["strength_-1.50"] if g["perplexity"]]
        pos_ppl = [g["perplexity"] for g in axis_data["generations"]["strength_1.50"] if g["perplexity"]]
        base_ppl = [g["perplexity"] for g in axis_data["generations"]["strength_0.00"] if g["perplexity"]]

        validation["per_axis"][str(ax)] = {
            "name": axis_name,
            "neg_mean": float(np.mean(neg_projs)),
            "neg_ci": [float(neg_ci[0]), float(neg_ci[1])],
            "pos_mean": float(np.mean(pos_projs)),
            "pos_ci": [float(pos_ci[0]), float(pos_ci[1])],
            "delta": float(delta),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(d),
            "significant": bool(p_value < 0.05),
            "ppl_baseline": float(np.mean(base_ppl)) if base_ppl else None,
            "ppl_steered": float(np.mean(neg_ppl + pos_ppl)) if neg_ppl and pos_ppl else None,
        }

    # Summary
    print(f"\n{'='*79}")
    print(f"SUMMARY: {n_sig}/{len(axes_to_test)} axes significant (p<0.05)")
    print(f"Mean |Cohen's d|: {np.mean(all_d):.2f}")

    d_interp = "LARGE" if np.mean(all_d) > 0.8 else ("MEDIUM" if np.mean(all_d) > 0.5 else "SMALL")
    print(f"Effect size interpretation: {d_interp}")

    validation["summary"] = {
        "n_axes_tested": len(axes_to_test),
        "n_significant": n_sig,
        "mean_cohens_d": float(np.mean(all_d)),
        "effect_size_interpretation": d_interp,
        "n_prompts": len(test_prompts),
    }

    all_results["validation"] = validation

    # ===== SAVE RESULTS =====
    with open(args.out_dir / "brain_steering_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(args.out_dir / "brain_steering_validation.json", "w") as f:
        json.dump(validation, f, indent=2)

    print(f"\n✓ Results saved to {args.out_dir}/", flush=True)

    # ===== PART 5: COMPARISON WITH PROXY METHOD =====
    print("\n" + "=" * 70, flush=True)
    print("5. METHODOLOGY COMPARISON", flush=True)
    print("=" * 70, flush=True)

    print("\n| Method | Steering Vectors | Brain Data? |")
    print("|--------|------------------|-------------|")
    print("| Proxy (OLD) | English word contrasts | No |")
    print("| Brain-derived (NEW) | Adapter weight matrix | YES |")

    print("\n✓ Box 6 brain-derived steering complete!", flush=True)


if __name__ == "__main__":
    main()
