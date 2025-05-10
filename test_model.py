import torch
import pickle
from model.model import BERT4Rec
from utils.evaluate import evaluate_model
from utils.config import PROCESSED_DIR, SEQ_LEN
import torch
import pickle
from model.model import BERT4Rec
from utils.evaluate import evaluate_model
from utils.config import PROCESSED_DIR, SEQ_LEN

from train import load_processed_data  # reuse logic
from utils.config import PROCESSED_DIR

def test_model(model_path, seq_len=SEQ_LEN, k_values=[5, 10, 20]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Load all processed sequences to determine correct num_items
    from train import load_processed_data
    train_seqs, val_seqs, test_data, num_items = load_processed_data(PROCESSED_DIR)

    # Load model with correct num_items
    model = BERT4Rec(num_items=num_items)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Run evaluation
    metrics = evaluate_model(model, test_data, num_items, device, k_values=k_values)

    print("\n Evaluation Results:")
    print("| Metric   | " + " | ".join([f"@{k}" for k in k_values]) + " |")
    print("|----------|" + "|".join(["--------" for _ in k_values]) + "|")
    for metric in ["recall", "ndcg"]:
        values = [f"{metrics[metric][k]:.4f}" for k in k_values]
        print(f"| {metric.capitalize():<8}| " + " | ".join(values) + " |")

    return metrics


if __name__ == "__main__":
    model_path = "results/best_model.pt"  # Path to the saved model
    test_model(model_path)