print("Step 1: Importing libraries...")
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
print("Step 1: Done ✓")

class ToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), max_length=self.max_len,
                             padding="max_length", truncation=True, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts, self.tokenizer, self.max_len = texts, tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), max_length=self.max_len,
                             padding="max_length", truncation=True, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}

if __name__ == '__main__':
    print("Step 2: Setting config...")
    MODEL_NAME  = "google/muril-base-cased"
    TRAIN_FILE  = "toxic_labeled.xlsx"
    EVAL_FILE   = "toxic_no_label_evaluation.xlsx"
    OUTPUT_FILE = "submission.csv"
    MAX_LEN     = 128
    BATCH_SIZE  = 16
    # add these to your config at the top
    EPOCHS        = 10       # max epochs, early stopping will cut it short
    PATIENCE      = 2        # stop if no improvement for 2 epochs
    LR          = 2e-5
    SEED        = 42
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Step 2: Done ✓ | Device: {DEVICE}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Step 3: Loading data...")
    train_df = pd.read_excel(TRAIN_FILE)
    eval_df  = pd.read_excel(EVAL_FILE)
    text_col      = [c for c in train_df.columns if "text"  in c.lower()][0]
    label_col     = [c for c in train_df.columns if "label" in c.lower() or "toxic" in c.lower()][0]
    eval_text_col = [c for c in eval_df.columns  if "text"  in c.lower()][0]
    print(f"Step 3: Done ✓ | Text col: '{text_col}' | Label col: '{label_col}'")
    print(train_df[label_col].value_counts())

    texts  = train_df[text_col].tolist()
    labels = train_df[label_col].astype(int).tolist()
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.1, random_state=SEED, stratify=labels)

    print("Step 4: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Step 4: Done ✓")

    print("Step 5: Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    print(f"Step 5: Done ✓ | On GPU: {next(model.parameters()).is_cuda}")

    # ── num_workers=0 fixes the Windows crash ──
    train_loader = DataLoader(ToxicDataset(X_train, y_train, tokenizer, MAX_LEN),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(ToxicDataset(X_val,   y_val,   tokenizer, MAX_LEN),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    num_steps = EPOCHS * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer,
                              num_warmup_steps=int(0.1 * num_steps),
                              num_training_steps=num_steps)

    print("Step 6: Starting training...\n")
    best_auc, best_state = 0.0, None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            loss   = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item()
            if (step + 1) % 20 == 0:
                print(f"  Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                labels_b = batch.pop("labels")
                probs = torch.softmax(model(**{k: v.to(DEVICE) for k, v in batch.items()}).logits, dim=-1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels_b.numpy())

        auc = roc_auc_score(all_labels, all_probs)
        print(f"\nEpoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc     = auc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  ✓ Best model saved (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            print(f"  ✗ No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping triggered at epoch {epoch+1}")
                break

    print(f"\nBest Val AUC: {best_auc:.4f}")

    print("\nStep 7: Running inference...")
    model.load_state_dict(best_state)
    model.to(DEVICE).eval()

    eval_loader = DataLoader(InferenceDataset(eval_df[eval_text_col].tolist(), tokenizer, MAX_LEN),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    all_probs = []
    with torch.no_grad():
        for batch in eval_loader:
            probs = torch.softmax(model(**{k: v.to(DEVICE) for k, v in batch.items()}).logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy())

    preds = (np.array(all_probs) >= 0.5).astype(int)
    submission = pd.DataFrame({"label": preds})
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\nStep 7: Done ✓ | Saved {OUTPUT_FILE} with {len(submission)} predictions")
    print(submission["label"].value_counts())