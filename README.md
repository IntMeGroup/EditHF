# EditHF-1M: A Million-Scale Rich Human Preference Feedback for Image Editing

## EditHF
📦 Installation

```bash
git clone https://github.com/YourUsername/EditHF.git
cd EditHF
pip install requirements.txt
```
⚡ Quick Start

```bash
python inference.py \
    --source_image "/path/to/source.jpg" \  # Path to the original/source image
    --edited_image "/path/to/edited.jpg" \  # Path to the edited/target image
    --instruction "Editing instruction" \  # Editing instruction describing desired modifications
    --peft_dir "lora_checkpoints_visual" \  # Directory containing LoRA checkpoints. Options: lora_checkpoints_visual / lora_checkpoints_editing / lora_checkpoints_preservation
    --mode visual  # Evaluation dimension: 'visual' for visual quality, 'alignment' for instruction adherence, 'preservation' for attribute preservation
```
