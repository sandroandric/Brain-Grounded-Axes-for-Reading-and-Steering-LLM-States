"""
Steer LLM generation along MEG-derived word axes using a linear adapter.

Uses adapter_W/adapter_b from outputs/llm_adapter_word_axes and applies
a steering vector to hidden states during generation.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_adapter(adapter_dir: pathlib.Path):
    W = np.load(adapter_dir / "adapter_W.npy")
    b = np.load(adapter_dir / "adapter_b.npy")
    sidecar_path = adapter_dir / "adapter_sidecar.json"
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    axes = sidecar.get("axes")
    mean_path = adapter_dir / "adapter_scaler_mean.npy"
    scale_path = adapter_dir / "adapter_scaler_scale.npy"
    mean = np.load(mean_path) if mean_path.exists() else None
    scale = np.load(scale_path) if scale_path.exists() else None
    return W, b, axes, mean, scale


def ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def resolve_axis_index(axis_id: int, axes: List[int] | None, n_axes: int) -> int:
    if axes and axis_id in axes:
        return axes.index(axis_id)
    if 0 <= axis_id < n_axes:
        return axis_id
    raise ValueError(f"Axis {axis_id} not found. Available axes: {axes}")


def axis_scores(W: np.ndarray, b: np.ndarray, h: np.ndarray, mean: np.ndarray | None, scale: np.ndarray | None):
    if mean is not None and scale is not None:
        h = (h - mean) / scale
    return W @ h + b


def get_prompt_hidden(model, tokenizer, prompt: str, layer_idx: int, device: str) -> np.ndarray:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hs = outputs.hidden_states[layer_idx][0]  # [seq, hidden]
    return hs.mean(dim=0).cpu().numpy()


class SteeringHook:
    def __init__(self, vector: torch.Tensor, strength: float):
        self.vector = vector
        self.strength = strength

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            steered = hidden + self.strength * self.vector
            return (steered,) + output[1:]
        return output + self.strength * self.vector


def generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int, temperature: float, top_p: float):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser(description="Steer LLM along MEG word axes.")
    p.add_argument("--adapter-dir", type=pathlib.Path, required=True)
    p.add_argument("--model", default="uer/gpt2-chinese-cluecorpussmall")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    p.add_argument("--axis", type=int, required=True, help="Axis id (e.g., 13) or index (0..).")
    p.add_argument("--strengths", nargs="+", type=float, default=[-5, -2, -1, 0, 1, 2, 5])
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    args = p.parse_args()

    W, b, axes, mean, scale = load_adapter(args.adapter_dir)
    axis_idx = resolve_axis_index(args.axis, axes, W.shape[0])

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    ensure_pad_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device != "cpu" else torch.float32,
    )
    if args.device == "mps":
        model = model.to("mps")
    elif args.device == "cuda":
        model = model.to("cuda")
    model.eval()

    # Choose layer index
    n_layers = model.config.num_hidden_layers
    layer_idx = args.layer if args.layer >= 0 else n_layers + args.layer + 1
    # GPT-2 hidden_states includes embeddings at index 0; last layer = n_layers
    if layer_idx < 0 or layer_idx > n_layers:
        layer_idx = n_layers

    # Steering vector in raw hidden space
    w = W[axis_idx]
    if scale is not None:
        w = w * scale
    norm = np.linalg.norm(w) + 1e-8
    w = w / norm
    steer_vec = torch.tensor(w, dtype=model.dtype, device=model.device).view(1, 1, -1)

    # Baseline scores on prompt
    h_prompt = get_prompt_hidden(model, tokenizer, args.prompt, layer_idx, args.device)
    z_prompt = axis_scores(W, b, h_prompt, mean, scale).tolist()

    # Attach hook to target layer
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        target_layer = model.model.layers[layer_idx - 1] if layer_idx > 0 else model.model.embed_tokens
    elif hasattr(model, "transformer"):
        target_layer = model.transformer.h[layer_idx - 1] if layer_idx > 0 else model.transformer.wte
    else:
        raise ValueError("Unsupported model architecture for steering.")

    results = {
        "axis_id": args.axis,
        "axis_index": axis_idx,
        "axes": axes,
        "strengths": args.strengths,
        "prompt_axis_scores": z_prompt,
        "generations": [],
    }

    for strength in args.strengths:
        hook = None
        if strength != 0.0:
            hook = target_layer.register_forward_hook(SteeringHook(steer_vec, strength))
        try:
            text = generate(model, tokenizer, args.prompt, args.device, args.max_new_tokens, args.temperature, args.top_p)
        finally:
            if hook is not None:
                hook.remove()
        results["generations"].append({"strength": strength, "text": text})

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
