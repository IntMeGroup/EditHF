<div align="center">
   <h1>EditHF-1M: A Million-Scale Rich Human Preference Feedback for Image Editing</h1>
   <div>
      <!-- <a href="https://arxiv.org/abs/2603.14916"><img src="https://arxiv.org/abs/2603.14916"/></a> -->
      <a href="https://arxiv.org/abs/2603.14916"><img src="https://img.shields.io/badge/Arxiv-2603.14916-red"/></a>
   </div>
</div>

## EditHF-1M
**EditHF-1M** is a **million-scale** image editing dataset containing over **29M human preference pairs** and **148K human mean opinion scores (MOS)**, evaluated across three dimensions: **visual quality**, **editing alignment**, and **attribute preservation**.
![](benchmark.png)

**IEQA**: A subset of the EditHF-1M dataset is adopted as the IEQA dataset for the New Trends in Image Restoration and Enhancement (NTIRE) Workshop and Challenge @ CVPR 2026, under the [X-AIGC Quality Assessment – Track 2: Image Editing](https://www.codabench.org/competitions/13031/).

You can download the IEQA dataset from the following link: [IEQA](https://huggingface.co/datasets/sparkling621/IEQA/tree/main)

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

⚡ Quick Start

```bash
pip install diffusers==0.36.0
python Qweninfer.py \
  --source_image "/path/to/source.jpg" \  # Path to the original/source image
  --instruction "apply a warm cinematic tone" \  # Editing instruction describing desired modifications
  --output "/path/to/output.jpg" \ # Output image path
```

🎨 Editing Examples

![](visualization.png)

## 🎓 Citations
If you find our work useful, please cite our paper as:
```bash
@article{xu2026edithf1mmillionscalerichhuman,
      title={EditHF-1M: A Million-Scale Rich Human Preference Feedback for Image Editing}, 
      author={Zitong Xu and Huiyu Duan and Zhongpeng Ji and Xinyun Zhang and Yutao Liu and Xiongkuo Min and others},
      year={2026},
      journal={arXiv preprint arXiv:2603.14916}, 
}
```
