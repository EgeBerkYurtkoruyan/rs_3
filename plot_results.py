import json
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics(json_path=None):
    # Load training history
    with open(json_path, "r") as f:
        history = json.load(f)

    # Extract values
    epochs = [entry["epoch"] for entry in history]
    val_ndcg = [entry["val_ndcg"] for entry in history]
    val_recall = [entry["val_recall"] for entry in history]

    # Set Seaborn theme
    sns.set(style="whitegrid", font_scale=1.1)

    # Create the plot
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=epochs, y=val_ndcg, label="NDCG@10", marker='o', markersize=4, linewidth=2)
    sns.lineplot(x=epochs, y=val_recall, label="Recall@10", marker='s', markersize=4, linewidth=2)

    # Labels and styling
    plt.xlabel("Epoch")
    plt.ylabel("Validation Score")
    plt.title("Validation NDCG@10 and Recall@10 Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_losses(json_path=None):
    # Load training history
    with open(json_path, "r") as f:
        history = json.load(f)

    # Extract data
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]

    # Set Seaborn theme
    sns.set(style="whitegrid", font_scale=1.1)

    # Create the plot
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=epochs, y=train_loss, label="Train Loss", marker='o', markersize=4, linewidth=2)
    sns.lineplot(x=epochs, y=val_loss, label="Validation Loss", marker='s', markersize=4, linewidth=2)

    # Labels and formatting
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    json_path = "results/model_performance.json" 
    plot_metrics(json_path)
    plot_losses(json_path)
