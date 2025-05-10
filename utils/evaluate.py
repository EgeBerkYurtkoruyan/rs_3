import torch
import numpy as np
from tqdm import tqdm

def evaluate_model(model, test_data, num_items, device, k_values=[10]):
    model.eval()
    recalls = {k: [] for k in k_values}
    ndcgs   = {k: [] for k in k_values}

    MASK_ID = num_items + 1
    batch_size = 512

    with torch.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
            batch = test_data[i:i+batch_size]
            batch = torch.LongTensor(batch).to(device)
            inputs = batch.clone()
            labels = torch.zeros_like(inputs)

            # Mask exactly one random non-pad item in each sequence
            for b in range(inputs.size(0)):
                non_pad = (inputs[b] != 0).nonzero(as_tuple=True)[0]
                if len(non_pad) == 0:
                    continue
                pos = np.random.choice(non_pad.cpu())
                labels[b, pos] = inputs[b, pos]
                inputs[b, pos] = MASK_ID

            pad_mask = (inputs == 0)
            logits = model(inputs, pad_mask)

            for b in range(inputs.size(0)):
                for pos in range(inputs.size(1)):
                    target = labels[b, pos].item()
                    if target == 0:
                        continue  # not a masked position

                    pred_logits = logits[b, pos]
                    topk = torch.topk(pred_logits, max(k_values)).indices.cpu().tolist()

                    for k in k_values:
                        topk_k = topk[:k]
                        if i == 0 and b == 0:
                            print("Target:", target)
                            print("Top-10:", topk_k)
                        hit = int(target in topk_k)
                        recalls[k].append(hit)

                        rel = [1 if item == target else 0 for item in topk_k]
                        dcg = rel[0] + sum([rel[j]/np.log2(j+2) for j in range(1, len(rel))])
                        ndcg = dcg / 1.0  # perfect DCG = 1
                        ndcgs[k].append(ndcg)

    return {
        "recall": {k: np.mean(recalls[k]) for k in k_values},
        "ndcg":   {k: np.mean(ndcgs[k])   for k in k_values},
    }
