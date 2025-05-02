import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Set paths to data files
DATA_DIR = "data"
MOVIES_FILE = os.path.join(DATA_DIR, "movies.dat")
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.dat")
USERS_FILE = os.path.join(DATA_DIR, "users.dat")

# Function to load MovieLens data files
def load_data():
    movies_columns = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(MOVIES_FILE, sep='::', engine='python',
                         encoding='ISO-8859-1', header=None, names=movies_columns)

    ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(RATINGS_FILE, sep='::', engine='python',
                          header=None, names=ratings_columns)

    users_columns = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
    users = pd.read_csv(USERS_FILE, sep='::', engine='python',
                        header=None, names=users_columns)

    return movies, ratings, users

# Preprocess ratings into binary implicit feedback and chronological sequences
def preprocess_data(ratings, min_interactions=5):
    positive_ratings = ratings[ratings['rating'] >= 4].copy()
    positive_ratings.sort_values(by=['user_id', 'timestamp'], inplace=True)

    user_interaction_count = positive_ratings.groupby('user_id').size()
    valid_users = user_interaction_count[user_interaction_count >= min_interactions].index
    filtered_ratings = positive_ratings[positive_ratings['user_id'].isin(valid_users)]

    return filtered_ratings

# Remap movie IDs to a contiguous range starting at 1
def remap_movie_ids(filtered_ratings):
    unique_movie_ids = filtered_ratings['movie_id'].unique()
    movie_id_map = {old: new_id for new_id, old in enumerate(unique_movie_ids, start=1)}
    filtered_ratings['movie_id'] = filtered_ratings['movie_id'].map(movie_id_map)
    return filtered_ratings, movie_id_map

# Generate user -> sequence dict
def generate_user_sequences(filtered_ratings):
    user_sequences = {}
    for user_id, group in filtered_ratings.groupby('user_id'):
        sequence = group['movie_id'].tolist()
        user_sequences[user_id] = sequence
    return user_sequences

# Split user sequences into train/val/test
def split_data_by_user(user_sequences, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10

    users = list(user_sequences.keys())
    train_users, temp_users = train_test_split(users, test_size=(val_ratio + test_ratio), random_state=42)
    val_users, test_users = train_test_split(temp_users, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

    train_sequences = {u: user_sequences[u] for u in train_users}
    val_sequences = {u: user_sequences[u] for u in val_users}
    test_sequences = {u: user_sequences[u] for u in test_users}

    print(f"Split dataset by users:")
    print(f"  Train: {len(train_sequences)}")
    print(f"  Val:   {len(val_sequences)}")
    print(f"  Test:  {len(test_sequences)}")
    return train_sequences, val_sequences, test_sequences

# Pad or truncate sequences
def process_sequences_to_fixed_length(user_sequences, max_length=20):
    processed = {}
    for user_id, seq in user_sequences.items():
        if len(seq) > max_length:
            processed[user_id] = seq[-max_length:]
        elif len(seq) < max_length:
            processed[user_id] = [0] * (max_length - len(seq)) + seq
        else:
            processed[user_id] = seq
    return processed

def main():
    print("Loading data...")
    movies, ratings, users = load_data()

    print("Preprocessing ratings...")
    filtered_ratings = preprocess_data(ratings)

    print("Remapping movie IDs...")
    filtered_ratings, movie_id_map = remap_movie_ids(filtered_ratings)

    print("Generating user sequences...")
    user_sequences = generate_user_sequences(filtered_ratings)

    print("Splitting dataset...")
    train_sequences, val_sequences, test_sequences = split_data_by_user(user_sequences)

    print("Processing fixed-length sequences...")
    train_fixed = process_sequences_to_fixed_length(train_sequences)
    val_fixed = process_sequences_to_fixed_length(val_sequences)
    test_fixed = process_sequences_to_fixed_length(test_sequences)

    print("Done.")
    return (filtered_ratings, user_sequences, movie_id_map,
            train_sequences, val_sequences, test_sequences,
            train_fixed, val_fixed, test_fixed)

if __name__ == "__main__":
    (filtered_ratings, user_sequences, movie_id_map,
     train_sequences, val_sequences, test_sequences,
     train_fixed, val_fixed, test_fixed) = main()
