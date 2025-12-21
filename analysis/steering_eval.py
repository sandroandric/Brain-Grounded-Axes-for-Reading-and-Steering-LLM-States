"""
Box 6: Reading + Steering Evaluation

Part 1 - Reading: Project LLM hidden states to brain axis coordinates
Part 2 - Steering: Apply steering vectors to manipulate LLM generation

This script uses activation steering (adding vectors to hidden states during
generation) to test whether brain-derived semantic directions can control
LLM behavior.

Run:
python analysis/steering_eval.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --out-dir outputs/steering_eval \
  --device mps
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Genuine axes from our analysis (axes that survived confound control)
GENUINE_AXES = {
    2: "concreteness",
    10: "concreteness_neg",
    12: "concreteness",
    13: "arousal_neg",
    15: "affect",
    16: "arousal",
    19: "conc_arousal",
}


def compute_steering_vectors(
    model,
    tokenizer,
    positive_words: List[str],
    negative_words: List[str],
    layer: int = -1,
    device: str = "mps",
) -> np.ndarray:
    """Compute steering vector as difference between positive and negative word embeddings."""
    model.eval()

    def get_word_embedding(word: str) -> np.ndarray:
        inputs = tokenizer(word, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(inputs.input_ids, output_hidden_states=True)
            # Get last layer hidden state for the last token
            hidden = outputs.hidden_states[layer][0, -1].cpu().numpy()
        return hidden

    pos_embs = [get_word_embedding(w) for w in positive_words if w.strip()]
    neg_embs = [get_word_embedding(w) for w in negative_words if w.strip()]

    if not pos_embs or not neg_embs:
        return None

    pos_mean = np.mean(pos_embs, axis=0)
    neg_mean = np.mean(neg_embs, axis=0)

    steering_vector = pos_mean - neg_mean
    # Normalize
    steering_vector = steering_vector / (np.linalg.norm(steering_vector) + 1e-8)

    return steering_vector


class SteeringHook:
    """Hook that applies steering vector to model activations."""

    def __init__(self, steering_vector: torch.Tensor, strength: float = 1.0):
        self.steering_vector = steering_vector
        self.strength = strength

    def __call__(self, module, input, output):
        # output is typically (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden = output[0]
            # Add steering vector to all positions
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
    layer_idx: int = 10,  # Middle layer
    max_new_tokens: int = 50,
    device: str = "mps",
    temperature: float = 0.7,
) -> Tuple[str, float]:
    """Generate text with optional steering vector applied."""
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    hook_handle = None

    if steering_vector is not None:
        # Register steering hook
        steering_tensor = torch.tensor(steering_vector, dtype=model.dtype).to(device)

        # Get the target layer
        if hasattr(model, 'model'):
            # LLaMA-style architecture
            layers = model.model.layers
        elif hasattr(model, 'transformer'):
            # GPT-2 style
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

        # Compute perplexity from scores
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


def evaluate_steering_effect(
    model,
    tokenizer,
    steering_vector: np.ndarray,
    test_prompts: List[str],
    strengths: List[float] = [-2.0, -1.0, 0.0, 1.0, 2.0],
    layer_idx: int = 10,
    device: str = "mps",
) -> Dict:
    """Evaluate steering effect across different strengths."""
    results = {"prompts": [], "generations": {}}

    for prompt in test_prompts:
        results["prompts"].append(prompt)

        for strength in strengths:
            key = f"strength_{strength:.1f}"
            if key not in results["generations"]:
                results["generations"][key] = []

            if strength == 0.0:
                # No steering
                text, ppl = generate_with_steering(
                    model, tokenizer, prompt,
                    steering_vector=None,
                    device=device,
                )
            else:
                text, ppl = generate_with_steering(
                    model, tokenizer, prompt,
                    steering_vector=steering_vector,
                    strength=strength,
                    layer_idx=layer_idx,
                    device=device,
                )

            results["generations"][key].append({
                "text": text,
                "perplexity": float(ppl) if not np.isnan(ppl) else None,
            })

    return results


def compute_semantic_shift(
    text: str,
    steering_vectors: Dict[str, np.ndarray],
    model,
    tokenizer,
    device: str,
) -> Dict[str, float]:
    """Compute how far generated text falls along each semantic dimension."""
    model.eval()

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0].cpu().numpy()  # [seq_len, hidden_dim]

    # Mean hidden state
    mean_hidden = np.mean(hidden, axis=0)

    # Project onto each steering vector
    projections = {}
    for name, sv in steering_vectors.items():
        proj = float(np.dot(mean_hidden, sv))
        projections[name] = proj

    return projections


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--layer-idx", type=int, default=10, help="Layer to apply steering (0-21 for TinyLlama)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("BOX 6: READING + STEERING EVALUATION", flush=True)
    print("=" * 70, flush=True)

    # Load model
    print(f"\n[load] Loading {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device != "cpu" else torch.float32,
    )
    model = model.to(args.device)
    model.eval()

    # Get model info
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Model loaded on {args.device}", flush=True)
    print(f"  Layers: {n_layers}, Hidden dim: {hidden_dim}", flush=True)
    print(f"  Steering layer: {args.layer_idx}", flush=True)

    # Define semantic contrasts for steering
    # These are based on our brain axis interpretations:
    # - Axis 2: Concreteness (proper nouns vs common words)
    # - Axis 13: Arousal/grammatical class (content vs function words)
    # - Axis 15: Valence/affect (positive vs negative)
    semantic_contrasts = {
        "concreteness": {
            "positive": ["stone", "table", "house", "mountain", "river", "tree", "book", "chair", "dog", "car"],
            "negative": ["freedom", "love", "democracy", "justice", "hope", "idea", "concept", "theory", "soul", "mind"],
            "description": "Concrete vs Abstract",
        },
        "arousal": {
            "positive": ["exciting", "thrilling", "intense", "passionate", "furious", "ecstatic", "terrifying", "shocking"],
            "negative": ["calm", "peaceful", "quiet", "serene", "relaxed", "still", "gentle", "soft"],
            "description": "High vs Low Arousal",
        },
        "valence": {
            "positive": ["happy", "wonderful", "beautiful", "love", "joy", "success", "peace", "delight", "kind", "good"],
            "negative": ["sad", "terrible", "ugly", "hate", "fear", "failure", "war", "angry", "cruel", "bad"],
            "description": "Positive vs Negative Valence",
        },
    }

    # Compute steering vectors for each contrast
    print("\n[steer] Computing steering vectors...", flush=True)
    steering_vectors = {}

    for contrast_name, contrast_data in semantic_contrasts.items():
        print(f"  Computing {contrast_name} vector...", flush=True)
        sv = compute_steering_vectors(
            model, tokenizer,
            contrast_data["positive"],
            contrast_data["negative"],
            layer=-1,
            device=args.device,
        )
        if sv is not None:
            steering_vectors[contrast_name] = sv
            print(f"    Vector norm: {np.linalg.norm(sv):.4f}", flush=True)

    # Test prompts for steering evaluation
    test_prompts = [
        "The old man walked slowly through the",
        "She felt a sudden wave of",
        "The scientist discovered that the",
        "In the quiet village, people often",
        "The story begins with a",
    ]

    # Steering strengths to test
    strengths = [-1.5, -0.75, 0.0, 0.75, 1.5]

    # Evaluate steering for each contrast
    print("\n" + "=" * 70, flush=True)
    print("STEERING EVALUATION", flush=True)
    print("=" * 70, flush=True)

    all_results = {
        "model": args.model,
        "layer_idx": args.layer_idx,
        "contrasts": {},
    }

    for contrast_name, sv in steering_vectors.items():
        print(f"\n[eval] Testing {contrast_name} steering...", flush=True)
        print(f"  Description: {semantic_contrasts[contrast_name]['description']}", flush=True)

        results = evaluate_steering_effect(
            model, tokenizer, sv,
            test_prompts,
            strengths=strengths,
            layer_idx=args.layer_idx,
            device=args.device,
        )

        # Compute semantic shift for each generation
        for strength_key, gens in results["generations"].items():
            for gen in gens:
                shifts = compute_semantic_shift(
                    gen["text"], steering_vectors, model, tokenizer, args.device
                )
                gen["semantic_shifts"] = shifts

        all_results["contrasts"][contrast_name] = results

        # Show examples
        print(f"\n  Example generations (prompt: '{test_prompts[0][:30]}...'):", flush=True)
        for strength in [-1.5, 0.0, 1.5]:
            key = f"strength_{strength:.1f}"
            gen = results["generations"][key][0]
            direction = "neg" if strength < 0 else ("baseline" if strength == 0 else "pos")
            text_preview = gen["text"][len(test_prompts[0]):len(test_prompts[0])+50]
            print(f"    [{direction:8s}] ...{text_preview}...", flush=True)

    # Part 2: Reading - Project sample texts to semantic space
    print("\n" + "=" * 70, flush=True)
    print("READING EVALUATION (Semantic Space Projection)", flush=True)
    print("=" * 70, flush=True)

    sample_texts = [
        "The concrete bridge over the river was built with stone and steel.",
        "Freedom and democracy are abstract concepts that inspire hope.",
        "The exciting chase through the city streets was thrilling and intense.",
        "The peaceful garden offered a calm and serene escape from stress.",
        "She felt happy and grateful for the wonderful surprise.",
        "He was angry and frustrated by the terrible news.",
    ]

    reading_results = {"texts": [], "projections": []}

    print("\n[read] Projecting sample texts to semantic space...", flush=True)
    for text in sample_texts:
        print(f"\n  Text: '{text[:50]}...'", flush=True)

        projections = compute_semantic_shift(text, steering_vectors, model, tokenizer, args.device)

        for contrast_name, proj in projections.items():
            print(f"    {contrast_name}: {proj:+.4f}", flush=True)

        reading_results["texts"].append(text)
        reading_results["projections"].append(projections)

    all_results["reading"] = reading_results

    # Summary statistics
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY: STEERING EFFECTS", flush=True)
    print("=" * 70, flush=True)

    print(f"\n{'Contrast':<15} {'PPL(neg)':<12} {'PPL(base)':<12} {'PPL(pos)':<12} {'Δ Shift':<12}", flush=True)
    print("-" * 63, flush=True)

    for contrast_name, results in all_results["contrasts"].items():
        neg_ppls = [g["perplexity"] for g in results["generations"]["strength_-1.5"] if g["perplexity"]]
        base_ppls = [g["perplexity"] for g in results["generations"]["strength_0.0"] if g["perplexity"]]
        pos_ppls = [g["perplexity"] for g in results["generations"]["strength_1.5"] if g["perplexity"]]

        neg_mean = np.mean(neg_ppls) if neg_ppls else float('nan')
        base_mean = np.mean(base_ppls) if base_ppls else float('nan')
        pos_mean = np.mean(pos_ppls) if pos_ppls else float('nan')

        # Compute semantic shift difference
        neg_shifts = [g["semantic_shifts"][contrast_name] for g in results["generations"]["strength_-1.5"]]
        pos_shifts = [g["semantic_shifts"][contrast_name] for g in results["generations"]["strength_1.5"]]
        shift_delta = np.mean(pos_shifts) - np.mean(neg_shifts) if neg_shifts and pos_shifts else float('nan')

        print(f"{contrast_name:<15} {neg_mean:<12.2f} {base_mean:<12.2f} {pos_mean:<12.2f} {shift_delta:+<12.3f}", flush=True)

    # Save results
    output_path = args.out_dir / "steering_results.json"

    # Convert numpy to list for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(convert_for_json(all_results), f, indent=2)

    # Save steering vectors
    for contrast_name, sv in steering_vectors.items():
        np.save(args.out_dir / f"steering_vector_{contrast_name}.npy", sv)

    print(f"\n✓ Results saved to {args.out_dir}/", flush=True)
    print("\nBox 6 complete!", flush=True)


if __name__ == "__main__":
    main()
