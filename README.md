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
    --source_image "/path/to/source.jpg" \
    --edited_image "/path/to/edited.jpg" \
    --instruction "Add a cinematic vintage film effect by subtly desaturating colors and adding warm tones." \
    --peft_dir "lora_checkpoints_visual" \ #lora checkpoints path, lora_checkpoints_visual/lora_checkpoints_editing/lora_checkpoints_preservation
    --mode visual #evaluation dimension, visual/editing/preservation 
```
