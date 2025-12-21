"""
Random-direction steering baseline for LLMs.

Generates outputs using random steering vectors matched in dimensionality to the adapter
and saves JSONs compatible with llm_word_axes_steer_eval.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_adapter_meta(adapter_dir: pathlib.Path):
    W = np.load(adapter_dir / "adapter_W.npy")
    sidecar_path = adapter_dir / "adapter_sidecar.json"
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    mean_path = adapter_dir / "adapter_scaler_mean.npy"
    scale_path = adapter_dir / "adapter_scaler_scale.npy"
    scale = np.load(scale_path) if scale_path.exists() else None
    return W, sidecar.get("axes"), scale


def ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def model_layer_index(model, layer: int) -> int:
    n_layers = model.config.num_hidden_layers
    layer_idx = layer if layer >= 0 else n_layers + layer + 1
    if layer_idx < 0 or layer_idx > n_layers:
        return n_layers
    return layer_idx


def load_prompts(path: pathlib.Path) -> List[str]:
    prompts = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            prompts.append(line)
    return prompts


def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


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


def main():
    p = argparse.ArgumentParser(description="Random-direction steering baseline.")
    p.add_argument("--adapter-dir", type=pathlib.Path, required=True)
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    p.add_argument("--axis-id", type=int, default=15)
    p.add_argument("--prompt-file", type=pathlib.Path, required=True)
    p.add_argument("--strengths", nargs="+", type=float, default=[-5, -2, -1, 0, 1, 2, 5])
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--layer", type=int, default=11)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--n-random", type=int, default=5)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = p.parse_args()

    prompts = load_prompts(args.prompt_file)
    if not prompts:
        raise ValueError("No prompts loaded from prompt file.")

    W, axes, scale = load_adapter_meta(args.adapter_dir)
    dim = W.shape[1]

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

    layer_idx = model_layer_index(model, args.layer)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        target_layer = model.model.layers[layer_idx - 1] if layer_idx > 0 else model.model.embed_tokens
    elif hasattr(model, "transformer"):
        target_layer = model.transformer.h[layer_idx - 1] if layer_idx > 0 else model.transformer.wte
    else:
        raise ValueError("Unsupported model architecture for steering.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for i in range(args.n_random):
        w = rng.normal(size=dim).astype(np.float32)
        if scale is not None:
            w = w * scale
        w = w / (np.linalg.norm(w) + 1e-8)
        steer_vec = torch.tensor(w, dtype=model.dtype, device=model.device).view(1, 1, -1)

        results = {
            "axis_id": args.axis_id,
            "axis_index": -1,
            "axes": axes,
            "vector_type": "random",
            "random_seed": int(args.seed + i),
            "strengths": args.strengths,
            "samples": args.samples,
            "prompts": prompts,
            "generations": [],
        }

        counter = 0
        for prompt_idx, prompt in enumerate(prompts):
            for strength in args.strengths:
                for sample in range(args.samples):
                    torch.manual_seed(args.seed + i * 1000 + counter)
                    if args.device == "cuda":
                        torch.cuda.manual_seed_all(args.seed + i * 1000 + counter)
                    counter += 1

                    hook = None
                    if strength != 0.0:
                        hook = target_layer.register_forward_hook(SteeringHook(steer_vec, strength))
                    try:
                        text = generate(
                            model,
                            tokenizer,
                            prompt,
                            args.max_new_tokens,
                            args.temperature,
                            args.top_p,
                        )
                    finally:
                        if hook is not None:
                            hook.remove()
                    results["generations"].append(
                        {
                            "prompt": prompt,
                            "prompt_index": prompt_idx,
                            "strength": strength,
                            "sample": sample,
                            "seed": args.seed + i * 1000 + counter - 1,
                            "text": text,
                        }
                    )

        out_path = args.out_dir / f"axis_{args.axis_id}_random_{i}.json"
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
