import os
import re
import json
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from transformers import AutoProcessor, AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from scipy.stats import spearmanr, kendalltau
from scipy.stats import pearsonr

# ===================== 常量 =====================
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ===================== Transform =====================
def build_transform(input_size=448):
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def build_query(instruction, mode="preservation"):
    if mode == "visual":
        query = (
            f"With the image editing prompt [{instruction}], "
            "the source image <image> is edited to generate the target image <image>. "
            "Please rate the target image from visual quality aspect "
            "on a scale of 0-100. "
            "Just reply a sentence in the following format: "
            "[The quality score is XX.XX (a number with two decimal)]"
        )

    elif mode == "alignment":
        query = (
            f"With the image editing prompt [{instruction}], "
            "the source image <image> is edited to generate the target image <image>. "
            "Please rate the target image from editing instruction alignment aspect "
            "on a scale of 0-100. "
            "Just reply a sentence in the following format: "
            "[The quality score is XX.XX (a number with two decimal)]"
        )

    elif mode == "preservation":
        query = (
            f"With the image editing prompt [{instruction}], "
            "the source image <image> is edited to generate the target image <image>. "
            "Please rate the target image from attribute preservation aspect "
            "on a scale of 0-100. "
            "Just reply a sentence in the following format: "
            "[The quality score is XX.XX (a number with two decimal)]"
        )

    else:
        raise ValueError("mode must be one of: visual, alignment, preservation")

    return query
# ===================== 数据集 =====================
class ImageQualityDataset(Dataset):
    """
    假设每行 JSONL 的结构类似：
    {"images": ["path1.png", "path2.png"], "query": "some text", "response": "The quality score is 3.45"}
    """
    def __init__(self, json_file, transform=None):
        self.samples = []
        self.transform = transform
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                img_paths = entry.get("images")
                query = entry.get("query", "")
                response = entry.get("response", "")
                if not img_paths or not response:
                    continue
                # 提取 score
                match = re.search(r"The quality score is ([0-9]+\.?[0-9]*)", response)
                if match:
                    label = float(match.group(1))
                    self.samples.append((img_paths, query, label, response))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_paths, query, label, response = self.samples[idx]
        # 返回列表形式的 img_paths（例如 [path1, path2]）
        return img_paths, torch.tensor(label, dtype=torch.float32), query, response

# ===================== collate_fn =====================
def collate_fn(batch):
    img_paths, labels, queries, response = zip(*batch)
    labels = torch.stack(labels)
    return list(img_paths), labels, list(queries), list(response)

# ===================== MLP =====================
# ===================== MLP 回归器 =====================
class MLPRegressor(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=1024, hidden_dim2=256, output_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim)
        )
    def forward(self, x):
        return self.mlp(x)
# ===================== Infer 主函数 =====================
def infer_single(
    source_image,
    edited_image,
    instruction,
    peft_dir,
    mode,
    model_path="OpenGVLab/InternVL3_5-8B",
    input_size=448,
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    device = torch.device(device)

    transform = build_transform(input_size)

    # -------- tokenizer --------
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    image_size = config.force_image_size or config.vision_config.image_size
    patch_size = config.vision_config.patch_size
    num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))

    # -------- base model --------
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    model.img_context_token_id = img_context_token_id

    vision_lora = PeftModel.from_pretrained(model.vision_model, f"{peft_dir}/vision_lora", torch_dtype=dtype)
    llm_lora = PeftModel.from_pretrained(model.language_model, f"{peft_dir}/llm_lora", torch_dtype=dtype)
    mlp_ckpt = os.path.join(peft_dir, "mlp_head.pth")

    model.vision_model = vision_lora
    model.language_model = llm_lora

    model.to(device).eval()

    # -------- MLP --------
    hidden_size = getattr(model.config, "hidden_size", 4096)

    mlp = MLPRegressor(
        input_dim=hidden_size,
        hidden_dim=hidden_size // 2,
        hidden_dim2=128,
        output_dim=1
    ).to(device)

    mlp.load_state_dict(torch.load(mlp_ckpt, map_location=device))
    mlp.eval()

    # -------- 读取图像 --------
    img1 = Image.open(source_image).convert("RGB")
    img2 = Image.open(edited_image).convert("RGB")

    pixel_values1 = transform(img1).unsqueeze(0).to(dtype).to(device)
    pixel_values2 = transform(img2).unsqueeze(0).to(dtype).to(device)

    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

    num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

    # -------- query --------
    query = build_query(instruction, mode) + " The quality score is 00.00"

    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)

    inputs = tokenizer(query, return_tensors="pt").to(device)

    image_flags = torch.ones(
        (len(num_patches_list), num_patches),
        dtype=torch.long,
        device=device
    )

    with torch.inference_mode():

        outputs = model(
            pixel_values=pixel_values,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_flags=image_flags,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states
        feats = hidden_states[-1][:, -6, :].detach().to(torch.float32)

        pred = abs(mlp(feats)).squeeze(1) * 1e2

    print("Quality Score:", pred.item())

    return pred.item()

# ===================== main =====================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Infer image quality with LoRA + MLP")
    parser.add_argument("--source_image", type=str, required=True, help="Path to source image")
    parser.add_argument("--edited_image", type=str, required=True, help="Path to edited image")
    parser.add_argument("--instruction", type=str, required=True, help="Editing instruction")
    parser.add_argument("--peft_dir", type=str, required=True, help="Directory of LoRA checkpoints")
    parser.add_argument("--mode", type=str, default="preservation", choices=["visual", "alignment", "preservation"],
                        help="Evaluation mode")
    parser.add_argument("--input_size", type=int, default=448, help="Image input size")
    parser.add_argument("--device", type=str, default=None, help="Device to run on, e.g., cuda or cpu")

    args = parser.parse_args()

    score = infer_single(
        source_image=args.source_image,
        edited_image=args.edited_image,
        instruction=args.instruction,
        peft_dir=args.peft_dir,
        mode=args.mode,
        input_size=args.input_size,
        device=args.device
    )

    print("Predicted Quality Score:", score)