import argparse
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen Image Edit Inference")

    parser.add_argument("--source_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--instruction", type=str, required=True, help="Editing instruction")
    parser.add_argument("--output", type=str, default="output.jpg", help="Output image path")

    parser.add_argument("--model_root", type=str, default="Qwen/Qwen-Image-Edit-2509")
    parser.add_argument("--transformer_path", type=str, default="sparkling621/EditHF-Reward/transformer")

    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--device", type=str, default="cuda:0")

    return parser.parse_args()


def load_pipeline(model_root, transformer_path, device):
    print("Loading transformer...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch.bfloat16
    ).to(device)

    print("Loading pipeline...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_root,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to(device)

    pipe.set_progress_bar_config(disable=None)
    return pipe


def run_inference(pipe, image_path, prompt, args):
    image = Image.open(image_path).convert("RGB")

    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    inputs = {
        "image": [image],
        "prompt": prompt,
        "generator": generator,
        "true_cfg_scale": args.cfg,
        "negative_prompt": " ",
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "num_images_per_prompt": 1,
    }

    with torch.inference_mode():
        output = pipe(**inputs)
        return output.images[0]


def main():
    args = parse_args()

    pipe = load_pipeline(
        model_root=args.model_root,
        transformer_path=args.transformer_path,
        device=args.device
    )

    result = run_inference(
        pipe,
        args.source_image,
        args.instruction,
        args
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
    result.save(args.output)

    print(f"✅ Saved to {args.output}")


if __name__ == "__main__":
    main()