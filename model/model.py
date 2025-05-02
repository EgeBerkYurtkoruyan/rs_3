import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BERT4Rec(nn.Module):
    def __init__(self, num_items, max_seq_len=20, hidden_size=256, num_layers=2, num_heads=2, dropout=0.1):
        super(BERT4Rec, self).__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size

        # Define a special token ID for masking
        self.mask_token_id = num_items  # Assuming item IDs range from 1 to num_items
        vocab_size = num_items + 1  # Including the mask token

        # Embedding layers
        self.item_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        # BERT configuration
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_seq_len,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )

        # BERT model
        self.bert = BertModel(config)

        # Output layer to project hidden states to item vocabulary
        self.output_layer = nn.Linear(hidden_size, num_items)

    def forward(self, input_ids, attention_mask):
        """
        input_ids: Tensor of shape [batch_size, seq_len]
        attention_mask: Tensor of shape [batch_size, seq_len]
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Compute embeddings
        item_embeds = self.item_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        embeddings = item_embeds + position_embeds

        # Pass through BERT
        outputs = self.bert(inputs_embeds=embeddings, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Compute logits over the item vocabulary
        logits = self.output_layer(sequence_output)  # [batch_size, seq_len, num_items]
        return logits
