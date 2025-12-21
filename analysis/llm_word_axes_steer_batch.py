"""
Batch steering runs for multiple prompts and samples per strength.
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


def ensure_pad_token(tokenizer, padding_side: str | None = None):
    if padding_side in {"left", "right"}:
        tokenizer.padding_side = padding_side
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


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
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
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in out]


def prompt_hidden_rms(model, tokenizer, prompt: str, layer_idx: int) -> float:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hs = outputs.hidden_states[layer_idx][0]
    return float(torch.sqrt((hs ** 2).mean()).item())


def main():
    p = argparse.ArgumentParser(description="Batch LLM steering for word axes.")
    p.add_argument("--adapter-dir", type=pathlib.Path, required=True)
    p.add_argument("--model", default="uer/gpt2-chinese-cluecorpussmall")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    p.add_argument("--axes", nargs="+", type=int, required=True)
    p.add_argument("--prompt-file", type=pathlib.Path, required=True)
    p.add_argument("--strengths", nargs="+", type=float, default=[-5, -2, -1, 0, 1, 2, 5])
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strength-mode", choices=["raw", "hidden_rms"], default="raw")
    p.add_argument("--strength-factor", type=float, default=1.0)
    p.add_argument("--strength-clip", type=float, default=None, help="Optional absolute clip after scaling.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--padding-side", choices=["left", "right"], default=None)
    p.add_argument("--max-batch-prompts", type=int, default=None)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = p.parse_args()
    args.batch_size = max(1, args.batch_size)

    prompts = load_prompts(args.prompt_file)
    if not prompts:
        raise ValueError("No prompts loaded from prompt file.")

    W, b, axes, mean, scale = load_adapter(args.adapter_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    ensure_pad_token(tokenizer, args.padding_side)
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
        # LLaMA-style
        target_layer = model.model.layers[layer_idx - 1] if layer_idx > 0 else model.model.embed_tokens
    elif hasattr(model, "transformer"):
        # GPT-2 style
        target_layer = model.transformer.h[layer_idx - 1] if layer_idx > 0 else model.transformer.wte
    else:
        raise ValueError("Unsupported model architecture for steering.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.strength_mode == "hidden_rms":
        prompt_scales = [prompt_hidden_rms(model, tokenizer, prompt, layer_idx) for prompt in prompts]
    else:
        prompt_scales = [1.0 for _ in prompts]

    max_batch = args.max_batch_prompts if args.max_batch_prompts else len(prompts)

    for axis_id in args.axes:
        axis_idx = resolve_axis_index(axis_id, axes, W.shape[0])
        w = W[axis_idx]
        if scale is not None:
            w = w * scale
        w = w / (np.linalg.norm(w) + 1e-8)
        steer_vec = torch.tensor(w, dtype=model.dtype, device=model.device).view(1, 1, -1)

        results = {
            "axis_id": axis_id,
            "axis_index": axis_idx,
            "axes": axes,
            "strengths": args.strengths,
            "samples": args.samples,
            "strength_mode": args.strength_mode,
            "strength_factor": args.strength_factor,
            "strength_clip": args.strength_clip,
            "batch_size": args.batch_size,
            "prompts": prompts,
            "generations": [],
        }

        counter = 0
        for strength in args.strengths:
            for sample in range(args.samples):
                for start in range(0, len(prompts), args.batch_size):
                    if start >= max_batch:
                        break
                    batch_prompts = prompts[start:min(start + args.batch_size, max_batch)]
                    batch_scales = prompt_scales[start:min(start + args.batch_size, max_batch)]
                    scaled_strengths = []
                    for scale in batch_scales:
                        scaled = strength * scale * args.strength_factor
                        if args.strength_clip is not None:
                            clip = abs(args.strength_clip)
                            if clip > 0:
                                scaled = max(-clip, min(clip, scaled))
                        scaled_strengths.append(scaled)

                    torch.manual_seed(args.seed + counter)
                    if args.device == "cuda":
                        torch.cuda.manual_seed_all(args.seed + counter)
                    counter += 1

                    hook = None
                    if any(s != 0.0 for s in scaled_strengths):
                        strength_tensor = torch.tensor(
                            scaled_strengths,
                            dtype=model.dtype,
                            device=model.device,
                        ).view(-1, 1, 1)
                        hook = target_layer.register_forward_hook(SteeringHook(steer_vec, strength_tensor))
                    try:
                        texts = generate_batch(
                            model,
                            tokenizer,
                            batch_prompts,
                            args.max_new_tokens,
                            args.temperature,
                            args.top_p,
                        )
                    finally:
                        if hook is not None:
                            hook.remove()

                    for offset, text in enumerate(texts):
                        prompt_idx = start + offset
                        results["generations"].append(
                            {
                                "prompt": batch_prompts[offset],
                                "prompt_index": prompt_idx,
                                "strength": strength,
                                "strength_scaled": scaled_strengths[offset],
                                "strength_scale": batch_scales[offset],
                                "sample": sample,
                                "seed": args.seed + counter - 1,
                                "text": text,
                            }
                        )

        out_path = args.out_dir / f"axis_{axis_id}.json"
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
