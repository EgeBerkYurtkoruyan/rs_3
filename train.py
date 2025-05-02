import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from model.model import BERT4Rec
from utils.dataloader import BERT4RecDataset
from utils.preprocess import (
    load_data,
    preprocess_data,
    remap_movie_ids,
    generate_user_sequences,
    split_data_by_user,
    process_sequences_to_fixed_length
)


def recall_ndcg_at_k(logits, labels, k=10):
    """
    Compute Recall@k and NDCG@k for masked positions.
    """
    logits = logits.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    recall_list = []
    ndcg_list = []

    for logit_seq, label_seq in zip(logits, labels):
        for idx, true_item in enumerate(label_seq):
            if true_item == -100:
                continue
            top_k = np.argsort(logit_seq[idx])[::-1][:k]
            if true_item in top_k:
                recall_list.append(1)
                rank = np.where(top_k == true_item)[0][0]
                ndcg_list.append(1 / np.log2(rank + 2))
            else:
                recall_list.append(0)
                ndcg_list.append(0)

    recall = np.mean(recall_list)
    ndcg = np.mean(ndcg_list)
    return recall, ndcg


def train_bert4rec(train_data, val_data, num_items, max_seq_len=20, batch_size=64, epochs=20, lr=1e-4, device='cuda'):
    print("Initializing BERT4Rec model...")
    model = BERT4Rec(num_items=num_items, max_seq_len=max_seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2 )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    train_dataset = BERT4RecDataset(train_data, max_seq_len=max_seq_len, mask_token_id=model.mask_token_id)
    val_dataset = BERT4RecDataset(val_data, max_seq_len=max_seq_len, mask_token_id=model.mask_token_id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_ndcg = 0
    wait = 0
    patience = 3

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for input_ids, attention_mask, labels in progress_bar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_recalls, val_ndcgs = [], []
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                logits = model(input_ids, attention_mask)
                recall, ndcg = recall_ndcg_at_k(logits, labels, k=10)
                val_recalls.append(recall)
                val_ndcgs.append(ndcg)

        val_recall = np.mean(val_recalls)
        val_ndcg = np.mean(val_ndcgs)
        scheduler.step(val_ndcg)

        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Recall@10 = {val_recall:.4f}, NDCG@10 = {val_ndcg:.4f}")

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            wait = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    return model


if __name__ == "__main__":
    print("=== BERT4Rec Training Pipeline ===")
    movies, ratings, users = load_data()
    ratings = preprocess_data(ratings)
    ratings, movie_id_map = remap_movie_ids(ratings)
    user_sequences = generate_user_sequences(ratings)
    train_seqs, val_seqs, test_seqs = split_data_by_user(user_sequences)
    train_fixed = process_sequences_to_fixed_length(train_seqs, max_length=20)
    val_fixed = process_sequences_to_fixed_length(val_seqs, max_length=20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_bert4rec(
        train_data=train_fixed,
        val_data=val_fixed,
        num_items=len(movie_id_map),
        max_seq_len=20,
        batch_size=128,
        epochs=20,
        lr=1e-4,
        device=device
    )
    print("✅ Training complete.")
