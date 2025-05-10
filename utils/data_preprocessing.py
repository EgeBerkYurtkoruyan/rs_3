import os
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from config import SEQ_LEN, PROCESSED_DIR, DATA_DIR

def load_data():
    """Load the raw MovieLens data and filter by rating >= 4."""
    df = pd.read_csv(DATA_DIR, sep='::', engine='python', names=['user', 'item', 'rating', 'timestamp'])
    df = df[df['rating'] >= 4].sort_values(by=['user', 'timestamp'])
    return df

def remap_items(df):
    """Map raw item IDs to contiguous indices starting from 1."""
    unique_items = df['item'].unique()
    item2id = {item: idx + 1 for idx, item in enumerate(unique_items)}  # +1 to reserve 0 for padding
    df['item'] = df['item'].map(item2id)
    return df, item2id

def build_sequences(df):
    """Build a dictionary of user -> list of item interactions."""
    user_seqs = defaultdict(list)
    for _, row in df.iterrows():
        user_seqs[row['user']].append(row['item'])
    return user_seqs

def split_sequence(seq):
    """Split a single user sequence into train/val/test splits."""
    n_total = len(seq)
    train_end = int(n_total * 0.7)
    val_end = int(n_total * 0.85)
    return seq[:train_end], seq[train_end:val_end], seq[val_end:]

def pad_sequence(seq, max_len):
    """Truncate or pad a sequence to fixed length."""
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq

def process_and_save():
    df = load_data()
    df, item2id = remap_items(df)
    user_seqs = build_sequences(df)

    train_data, val_data, test_data = [], [], []

    for user, seq in user_seqs.items():
        if len(seq) < 5:
            continue

        train, val, test = split_sequence(seq)
        train_data.append(pad_sequence(train, SEQ_LEN))
        val_data.append(pad_sequence(val, SEQ_LEN))
        test_data.append(pad_sequence(test, SEQ_LEN))

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(os.path.join(PROCESSED_DIR, 'train_seqs.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(PROCESSED_DIR, 'val_seqs.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
    with open(os.path.join(PROCESSED_DIR, 'test_seqs.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    with open(os.path.join(PROCESSED_DIR, 'item2id.pkl'), 'wb') as f:
        pickle.dump(item2id, f)

    print(f"✅ Saved {len(train_data)} train, {len(val_data)} val, {len(test_data)} test sequences.")
    print(f"✅ Number of unique items: {len(item2id)}")

if __name__ == "__main__":
    process_and_save()
