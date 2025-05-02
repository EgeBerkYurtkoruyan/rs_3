from torch.utils.data import Dataset
import torch


def mask_sequences(sequences, mask_token_id, mask_ratio=0.3):
    """
    Randomly masks items in the input sequences.
    Returns:
        input_ids: Masked input sequences
        labels: Labels for computing loss (-100 for non-masked positions)
    """
    input_ids = sequences.clone()
    labels = sequences.clone()

    # Create a mask for positions to be masked
    mask = (torch.rand(sequences.shape) < mask_ratio) & (sequences != 0)
    input_ids[mask] = mask_token_id
    labels[~mask] = -100  # Ignore non-masked positions in loss computation

    return input_ids, labels


class BERT4RecDataset(Dataset):
    def __init__(self, user_sequences_dict, max_seq_len, mask_token_id, mask_ratio=0.15):
        self.user_sequences = list(user_sequences_dict.values())
        self.max_seq_len = max_seq_len
        self.mask_token_id = mask_token_id
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.user_sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.user_sequences[idx], dtype=torch.long)
        input_ids, labels = mask_sequences(seq, self.mask_token_id, self.mask_ratio)
        attention_mask = (input_ids != 0).long()
        return input_ids, attention_mask, labels
