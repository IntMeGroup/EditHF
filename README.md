# EditHF-1M: A Million-Scale Rich Human Preference Feedback for Image Editing

## EditHF-1M
**EditHF-1M** is a **million-scale** image editing dataset containing over **29M human preference pairs** and **148K human mean opinion scores (MOS)**, evaluated across three dimensions: **visual quality**, **editing alignment**, and **attribute preservation**.

## EditHF
**EditHF** is an MLLM-based evaluation model trained on EditHF-1M to provide fine-grained, human-aligned scores for image editing across dimensions: **visual quality**, **editing alignment**, and **attribute preservation**.
 
📥 Model Weights

You can download the pre-trained LoRA checkpoints from the following link:
[EditHF](https://huggingface.co/sparkling621/EditHF/tree/main)

📦 Installation

```bash
git clone https://github.com/IntMeGroup/EditHF.git
cd EditHF
pip install requirements.txt
```
⚡ Quick Start

```bash
python inference.py \
    --source_image "/path/to/source.jpg" \  # Path to the original/source image
    --edited_image "/path/to/edited.jpg" \  # Path to the edited/target image
    --instruction "Editing instruction" \  # Editing instruction describing desired modifications
    --peft_dir "lora_checkpoints_visual" \  # Directory containing LoRA checkpoints and MLP head. 
    --mode visual  # Evaluation dimension: 'visual' for visual quality, 'alignment' for editing instruction alignment, 'preservation' for attribute preservation
```
## EditHF-Reward
**EditHF-Reward** is a reward modeling approach that utilizes EditHF signals to improve text-guided image editing models through reinforcement learning.

📥 Model Weights

You can download the advanced image editing model **Qwen-Image-Edit refined with our EditHF-Reward** from the following link:
[Qwen-Image-Edit(EditHF-Reward)](https://huggingface.co/sparkling621/EditHF-Reward/tree/main)
