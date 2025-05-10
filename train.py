import os
import json
import torch
import random
import pickle
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.model import BERT4Rec
from utils.evaluate import evaluate_model
from utils.config import (
    MASK_PROB, EPOCHS, PATIENCE, LR, MODEL_SAVE_PATH,
    SEQ_LEN, PROCESSED_DIR, BATCH_SIZE
)

random.seed(42)

def pad_sequence(seq, max_len):
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq

def mask_items(seqs, num_items, mask_prob):
    MASK_ID = num_items + 1
    masked_seqs, labels = [], []

    for seq in seqs:
        padded = pad_sequence(seq, SEQ_LEN)
        masked, label = [], []
        for item in padded:
            if item != 0 and random.random() < mask_prob:
                masked.append(MASK_ID)
                label.append(item)
            else:
                masked.append(item)
                label.append(0)
        masked_seqs.append(masked)
        labels.append(label)

    return torch.LongTensor(masked_seqs), torch.LongTensor(labels)

def evaluate_val_loss(model, val_data, criterion, num_items, device, mask_prob):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        loop = tqdm(range(0, len(val_data), BATCH_SIZE), desc="Validating", leave=False)
        for i in loop:
            batch = val_data[i:i+BATCH_SIZE]
            masked_inputs, labels = mask_items(batch, num_items, mask_prob)
            masked_inputs, labels = masked_inputs.to(device), labels.to(device)
            mask = (masked_inputs == 0)
            logits = model(masked_inputs, mask)

            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            loss = criterion(logits, labels)
            total_loss += loss.item()

    return total_loss / len(val_data)

def train_model(model, train_data, val_data, num_items, device, mask_prob=MASK_PROB):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max'
    )
    model.to(device)

    history = []
    best_ndcg = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        random.shuffle(train_data)

        loop = tqdm(range(0, len(train_data), 128), desc=f"Epoch {epoch}")
        for i in loop:
            batch = train_data[i:i+128]
            masked_inputs, labels = mask_items(batch, num_items, mask_prob)
            masked_inputs, labels = masked_inputs.to(device), labels.to(device)
            mask = (masked_inputs == 0)
            logits = model(masked_inputs, mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            loop.set_postfix(loss=loss.item(), lr=current_lr)

        val_loss = evaluate_val_loss(model, val_data, criterion, num_items, device, mask_prob)
        val_metrics = evaluate_model(model, val_data, num_items, device, k_values=[10])
        val_ndcg = val_metrics['ndcg'][10]
        val_recall = val_metrics['recall'][10]

        scheduler.step(val_ndcg)

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_data),
            "val_loss": val_loss,
            "val_ndcg": val_ndcg,
            "val_recall": val_recall,
            "lr": current_lr
        }
        history.append(log_entry)

        print(f"\nEpoch {epoch} Summary | Train Loss: {log_entry['train_loss']:.4f} | "
              f"Val Loss: {val_loss:.4f} | NDCG@10: {val_ndcg:.4f} | Recall@10: {val_recall:.4f}\n")

        if val_ndcg > best_ndcg + 1e-4:
            best_ndcg = val_ndcg
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered by NDCG@10.")
                break

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open('results/model_performance.json', 'w') as f:
        json.dump(history, f, indent=2)

def load_processed_data(processed_dir):
    with open(os.path.join(processed_dir, 'train_seqs.pkl'), 'rb') as f:
        train_seqs = pickle.load(f)
    with open(os.path.join(processed_dir, 'val_seqs.pkl'), 'rb') as f:
        val_seqs = pickle.load(f)
    with open(os.path.join(processed_dir, 'test_seqs.pkl'), 'rb') as f:
        test_seqs = pickle.load(f)

    all_items = set(item for seq in train_seqs + val_seqs + test_seqs for item in seq if item != 0)
    num_items = max(all_items)

    return train_seqs, val_seqs, test_seqs, num_items

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_seqs, val_seqs, test_seqs, num_items = load_processed_data(PROCESSED_DIR)
    model = BERT4Rec(num_items=num_items)

    train_model(model, train_seqs, val_seqs, num_items, device)
