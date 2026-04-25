# Challenge 3: Multilingual Toxic Comment Classification

## Results
|         Metric         |        Score         |
|------------------------|----------------------|
| **Validation ROC-AUC** | **0.9870**           |
| Best Epoch             | 7 / 10               |
| Early Stopping         | Triggered at epoch 9 |
| Final Training Loss    | 0.0088               |

## Approach
Fine-tuned Google's MuRIL (Multilingual Representations for Indian Languages)
model for binary toxic comment classification on English and Hindi text.

## Model
- Model: google/muril-base-cased
- Task: Binary Classification (0 = non-toxic, 1 = toxic)
- Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
- Max sequence length: 128 tokens
- Batch size: 16
- Train/Val split: 90/10 stratified

## Why MuRIL
MuRIL is specifically pre-trained on 17 Indian languages including Hindi and
English code-mixed text, making it ideal for this multilingual dataset compared
to general models like XLM-RoBERTa.

## How to Reproduce
1. Clone this repository
2. Place with_label.csv and no_label.csv in the same folder
3. Install requirements:
   pip install -r requirements.txt
4. Run training:
   python train.py
5. Output: no_label.csv with predicted labels filled in

## Requirements
- Python 3.10
- PyTorch with CUDA support
- Windows 11
- Install all dependencies: pip install -r requirements.txt

## Files
- train.py          → Training and inference code
- requirements.txt  → Dependencies
- no_label.csv      → Final predictions
- README.md         → This file
