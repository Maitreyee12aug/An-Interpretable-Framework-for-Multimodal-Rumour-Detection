# =============================================================
# text_encoder.py
# Implements Section 4.1: Text Branch
# Paper: "Gated Consistency and Uncertainty-Aware Fusion:
#         An Interpretable Framework for Multimodal Rumour Detection"
# =============================================================

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer


class TextEncoder(nn.Module):
    """
    Section 4.1 — Text Branch

    Pipeline:
        Raw Caption
            → Text Cleaning & Tokenization (DistilBERT tokenizer)
            → Fine-tuned DistilBERT (last 4 layers unfrozen)
            → [CLS] token embedding → f_text ∈ R^768
            → Linear(768→512) + ReLU + Dropout(0.3) → z_text ∈ R^512
            → Sigmoid head → u_text ∈ R^1  (uncertainty proxy)

    Outputs:
        z_text  : projected text feature [B, 512]
        u_text  : modality confidence proxy [B, 1]
    """

    def __init__(
        self,
        pretrained_model: str = "distilbert-base-uncased",
        proj_dim: int = 512,
        dropout_rate: float = 0.3,
        max_length: int = 128,
        unfreeze_last_n_layers: int = 4,
    ):
        super().__init__()

        self.max_length = max_length

        # ── DistilBERT Backbone ──────────────────────────────────
        # DistilBERT reduces BERT params by 40% and is 60% faster
        # while retaining 97% of BERT-Base performance (Sanh et al.)
        self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model)
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model)

        # Freeze all layers first
        for param in self.distilbert.parameters():
            param.requires_grad = False

        # Unfreeze last N transformer layers for domain adaptation
        # Eq. in paper: f_text = DistilBERT_finetuned(T_raw)[CLS] ∈ R^768
        transformer_layers = self.distilbert.transformer.layer
        num_layers = len(transformer_layers)  # 6 layers in DistilBERT
        for i in range(num_layers - unfreeze_last_n_layers, num_layers):
            for param in transformer_layers[i].parameters():
                param.requires_grad = True

        # ── Linear Projection ────────────────────────────────────
        # Eq (1): h_t = ReLU(W_t * f_text + b_t)
        # W_t ∈ R^{512×768}, b_t ∈ R^512
        self.projection = nn.Linear(768, proj_dim)

        # ── ReLU Activation ──────────────────────────────────────
        self.relu = nn.ReLU()

        # ── Dropout ──────────────────────────────────────────────
        # Eq (2): z_text = Dropout(h_t), p=0.3
        # Kept active during inference for MC Dropout
        self.dropout = nn.Dropout(p=dropout_rate)

        # ── Uncertainty Proxy Head ───────────────────────────────
        # Eq (3): u_text = Sigmoid(W_ut * z_text + b_ut) ∈ R^1
        # Learned confidence signal for fusion weighting
        # Distinct from MC Dropout predictive uncertainty
        self.uncertainty_head = nn.Linear(proj_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def tokenize(self, texts, device):
        """
        Tokenize a list of raw caption strings.
        Applies subword tokenization, padding, truncation
        to max_length=128 tokens.
        Returns input_ids and attention_masks on device.
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return (
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device),
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the Text Branch.

        Args:
            input_ids     : [B, seq_len] tokenized input
            attention_mask: [B, seq_len] attention mask

        Returns:
            z_text : [B, 512] projected text feature
            u_text : [B, 1]   learned uncertainty proxy
        """

        # Extract contextual embedding via DistilBERT
        # Use [CLS] token (index 0) as sentence representation
        # f_text ∈ R^768
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # last_hidden_state: [B, seq_len, 768]
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [B, 768]

        # Linear projection + ReLU
        # h_t = ReLU(W_t * f_text + b_t) ∈ R^512
        h_t = self.relu(self.projection(cls_embedding))  # [B, 512]

        # Dropout for regularization and MC Dropout
        # z_text = Dropout(h_t) ∈ R^512
        z_text = self.dropout(h_t)  # [B, 512]

        # Uncertainty proxy
        # u_text = Sigmoid(W_ut * z_text + b_ut) ∈ R^1
        u_text = self.sigmoid(self.uncertainty_head(z_text))  # [B, 1]

        return z_text, u_text


# =============================================================
# Quick test
# =============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoder = TextEncoder(
        proj_dim=512,
        dropout_rate=0.3,
        max_length=128,
        unfreeze_last_n_layers=4,
    ).to(device)

    # Count trainable parameters
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    # Test forward pass
    sample_texts = [
        "American troops landed at the National Palace in Haiti.",
        "Defense Secretary orders soldiers to leave Ramadi immediately.",
    ]
    input_ids, attention_mask = encoder.tokenize(sample_texts, device)
    z_text, u_text = encoder(input_ids, attention_mask)

    print(f"z_text shape : {z_text.shape}")   # Expected: [2, 512]
    print(f"u_text shape : {u_text.shape}")   # Expected: [2, 1]
    print(f"u_text values: {u_text.detach().cpu().numpy()}")
