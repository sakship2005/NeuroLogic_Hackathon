# Challenge 3: Multilingual Toxic Comment Classification

## Approach
Fine-tuned Google's MuRIL (Multilingual Representations for Indian Languages) 
model for binary toxic comment classification on English and Hindi text.

## Model
- Model: google/muril-base-cased
- Task: Binary Classification (0 = non-toxic, 1 = toxic)
- Best Validation ROC-AUC: 0.9870

## Why MuRIL
MuRIL is specifically pre-trained on Indian languages including Hindi and 
English code-mixed text, making it ideal for this dataset.

## Results
- Best Validation ROC-AUC: 0.9870
- Early stopping triggered at epoch 9
- Training loss reduced from 0.53 to 0.008

## Requirements
Install dependencies:
    pip install -r requirements.txt

## How to Reproduce
1. Clone this repository
2. Place toxic_labeled.xlsx and toxic_no_label_evaluation.xlsx in the same folder
3. Install requirements:
       pip install -r requirements.txt
4. Run training:
       python train.py
5. Output: no_label.csv with predicted labels

## Environment
- Python 3.10
- PyTorch with CUDA support
- Windows 11

## Files
- train.py         → Training and inference code
- requirements.txt → Dependencies
- no_label.csv     → Final predictions
- README.txt       → This file
