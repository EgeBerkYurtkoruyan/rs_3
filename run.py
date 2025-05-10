import torch
from model.model import BERT4Rec
from train import train_model, load_processed_data
from utils.config import PROCESSED_DIR

def run_experiment(exp_name,
                   embed_dim=256,
                   num_layers=4,
                   num_heads=4,
                   dropout=0.2,
                   mask_prob=0.15,
                   epochs=10,
                   model_name=None,
                   seq_len=50):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running experiment '{exp_name}' on device: {device}")

    # Load preprocessed sequences
    train_seqs, val_seqs, test_seqs, num_items = load_processed_data(PROCESSED_DIR)

    # Initialize model
    model = BERT4Rec(
        num_items=num_items,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
    )

    # Default file names for saving if not provided
    json_name = f"{exp_name}.json"
    model_file = model_name if model_name else f"{exp_name}.pt"

    # Train and return training history
    history = train_model(
        model=model,
        train_data=train_seqs,
        val_data=val_seqs,
        num_items=num_items,
        device=device,
        mask_prob=mask_prob,
        result_name=json_name,
        model_name=model_file,
        epochs=epochs,
        seq_len=seq_len
    )
    return history

if __name__ == "__main__":
    history = run_experiment(
        exp_name="test",
        embed_dim=256,
        num_layers=4,
        num_heads=4,
        dropout=0.2,
        mask_prob=0.15,
        epochs=1,
        model_name="test_model.pt",
        seq_len=20
    )
