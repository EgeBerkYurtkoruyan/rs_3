import json
import os
from run import run_experiment

# Default baseline configuration
BASE_CONFIG = {
    "embed_dim": 256,
    "num_layers": 4,
    "num_heads": 4,
    "dropout": 0.2,
    "mask_prob": 0.2,
    "epochs": 40,
    "seq_len": 20
}

# Define ablation ranges
ABLATION_SPACE = {
    "embed_dim": [256, 512,1024], # running
    "num_layers": [4,5,6],
    "num_heads": [2, 4, 8],
    "dropout": [0.1, 0.2, 0.3],
    "mask_prob": [0.10, 0.15, 0.20] # kerem
}

def run_ablation(target_param, output_path="results/ablation"):
    assert target_param in ABLATION_SPACE, f"Invalid param: {target_param}"

    values_to_try = ABLATION_SPACE[target_param]
    print(f"\nRunning ablation study on: `{target_param}`")
    results = {}

    for val in values_to_try:
        config = BASE_CONFIG.copy()
        config[target_param] = val

        exp_name = f"{target_param}_{str(val).replace('.', '')}"
        model_name = f"{exp_name}.pt"

        print(f"\nRunning experiment `{exp_name}` with {target_param} = {val}")
        result_temp = run_experiment(
            exp_name=exp_name,
            embed_dim=config["embed_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            dropout=config["dropout"],
            mask_prob=config["mask_prob"],
            epochs=config["epochs"],
            seq_len=config["seq_len"],
            model_name=model_name
        )
        results[str(val)] = result_temp  # keys as strings for JSON compatibility

    # Save results
    output_path = os.path.join(output_path, target_param)
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"ablation_{target_param}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results saved to {output_file}")

    return results

if __name__ == "__main__":
    results = run_ablation("embed_dim")
