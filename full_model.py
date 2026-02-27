# =============================================================
# full_model.py
# Complete Integrated Framework — Sections 4.1 through 4.6
# Paper: "Gated Consistency and Uncertainty-Aware Fusion:
#         An Interpretable Framework for Multimodal Rumour Detection"
#
# Architecture Flow (Algorithm 1):
#   Step 1 : Text Branch        → z_text, u_text    (Section 4.1)
#   Step 2 : Image Branch       → z_img,  u_img     (Section 4.2)
#   Step 3 : Consistency GNN    → s_cons            (Section 4.3)
#   Step 4 : Dynamic Gate       → z'_text, z'_img   (Section 4.4)
#   Step 5 : Adaptive Fusion    → w_text, w_img      (Section 4.5)
#   Step 6 : Final Classifier   → logits            (Section 4.6)
#   Step 7 : MC Dropout         → sigma^2           (Section 4.6)
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from transformers import DistilBertModel, DistilBertTokenizer

from models.text_encoder import TextEncoder
from models.dynamic_gate import DynamicGate
from models.consistency_gnn import CrossModalAttentionGNN


# =============================================================
# Image Encoder — Section 4.2
# =============================================================

class ImageEncoder(nn.Module):
    """
    Section 4.2 — Image Branch

    Pipeline:
        Raw Image
            → ResNet-50 (blocks 3 & 4 unfrozen)
            → Global Average Pooling → f_img ∈ R^2048
            → Linear(2048→512) + ReLU + Dropout(0.3) → z_img ∈ R^512
            → Sigmoid head → u_img ∈ R^1 (uncertainty proxy)

    Outputs:
        z_img  : projected image feature [B, 512]
        u_img  : modality confidence proxy [B, 1]
    """

    def __init__(
        self,
        proj_dim: int = 512,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        # ── ResNet-50 Backbone ────────────────────────────────────
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Freeze all layers first
        for param in backbone.parameters():
            param.requires_grad = False

        # Unfreeze Blocks 3 & 4 (layer3, layer4) for fine-tuning
        # This captures high-level semantic features while preserving
        # low-level representations learned on ImageNet
        for param in backbone.layer3.parameters():
            param.requires_grad = True
        for param in backbone.layer4.parameters():
            param.requires_grad = True

        # Remove final FC layer, keep GAP
        # f_img ∈ R^2048 after GAP
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # ── Linear Projection ─────────────────────────────────────
        # Eq (7): h_i = ReLU(W_i * f_img + b_i)
        self.projection = nn.Linear(2048, proj_dim)
        self.relu = nn.ReLU()

        # ── Dropout ───────────────────────────────────────────────
        # Eq (8): z_img = Dropout(h_i), p=0.3
        self.dropout = nn.Dropout(p=dropout_rate)

        # ── Uncertainty Proxy Head ────────────────────────────────
        # Eq (9): u_img = Sigmoid(W_ui * z_img + b_ui) ∈ R^1
        self.uncertainty_head = nn.Linear(proj_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        """
        Args:
            images : [B, 3, 224, 224] normalized image tensors

        Returns:
            z_img  : [B, 512] projected image feature
            u_img  : [B, 1]   learned uncertainty proxy
        """

        # Extract ResNet features → GAP → f_img ∈ R^2048
        B = images.size(0)
        f_img = self.backbone(images).view(B, -1)  # [B, 2048]

        # Linear projection + ReLU
        h_i = self.relu(self.projection(f_img))    # [B, 512]

        # Dropout
        z_img = self.dropout(h_i)                  # [B, 512]

        # Uncertainty proxy
        u_img = self.sigmoid(
            self.uncertainty_head(z_img)
        )                                          # [B, 1]

        return z_img, u_img


# =============================================================
# Uncertainty-Aware Adaptive Fusion — Section 4.5
# =============================================================

class UncertaintyAwareFusion(nn.Module):
    """
    Section 4.5 — Uncertainty-Aware Adaptive Modality Fusion

    Computes per-sample adaptive weights w_text, w_img
    conditioned on both gated features AND uncertainty proxies.

    Input  : [z'_text || z'_img || u_text || u_img] ∈ R^{1026}
    Output : [w_text, w_img] ∈ R^2  (Softmax normalized)

    Weights reflect modality reliability:
        High u_text (high confidence) → w_text ↑
        Low  u_img  (low confidence)  → w_img  ↓

    Outputs:
        w_text : [B, 1] text modality weight
        w_img  : [B, 1] image modality weight
    """

    def __init__(
        self,
        proj_dim: int = 512,
        fusion_hidden: int = 256,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        # Input: z'_text + z'_img + u_text + u_img
        # = 512 + 512 + 1 + 1 = 1026
        fusion_input_dim = proj_dim + proj_dim + 1 + 1

        # ── Fusion MLP ────────────────────────────────────────────
        # Eq (10): [w_text, w_img] = Softmax(MLP([z'_text || z'_img
        #                                          || u_text || u_img]))
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden, 2),
            nn.Softmax(dim=-1),  # Ensures w_text + w_img = 1
        )

    def forward(self, z_prime_text, z_prime_img, u_text, u_img):
        """
        Args:
            z_prime_text : [B, 512] gated text feature
            z_prime_img  : [B, 512] gated image feature
            u_text       : [B, 1]   text uncertainty proxy
            u_img        : [B, 1]   image uncertainty proxy

        Returns:
            w_text : [B, 1] text modality weight
            w_img  : [B, 1] image modality weight
        """

        # Concatenate all inputs
        fusion_in = torch.cat(
            [z_prime_text, z_prime_img, u_text, u_img], dim=-1
        )  # [B, 1026]

        # Compute adaptive modality weights
        weights = self.fusion_mlp(fusion_in)  # [B, 2]

        w_text = weights[:, 0:1]  # [B, 1]
        w_img  = weights[:, 1:2]  # [B, 1]

        return w_text, w_img


# =============================================================
# Full Integrated Model — Sections 4.1–4.6
# =============================================================

class GatedConsistencyRumorDetector(nn.Module):
    """
    Complete Framework: Gated Consistency and Uncertainty-Aware
    Fusion for Multimodal Rumour Detection.

    Integrates all 6 components from Algorithm 1:
        1. Text Branch        (Section 4.1)
        2. Image Branch       (Section 4.2)
        3. Consistency GNN    (Section 4.3)
        4. Dynamic Gate       (Section 4.4)
        5. Adaptive Fusion    (Section 4.5)
        6. Final Classifier   (Section 4.6)

    MC Dropout inference handled separately in mc_dropout.py.
    """

    def __init__(
        self,
        proj_dim: int = 512,
        gnn_hidden_dim: int = 256,
        gnn_num_heads: int = 4,
        gate_hidden_dim: int = 512,
        fusion_hidden_dim: int = 256,
        classifier_hidden: int = 128,
        dropout_rate: float = 0.3,
        distilbert_model: str = "distilbert-base-uncased",
        max_text_length: int = 128,
        unfreeze_last_n_layers: int = 4,
    ):
        super().__init__()

        self.proj_dim = proj_dim

        # ── Step 1: Text Branch (Section 4.1) ────────────────────
        self.text_encoder = TextEncoder(
            pretrained_model=distilbert_model,
            proj_dim=proj_dim,
            dropout_rate=dropout_rate,
            max_length=max_text_length,
            unfreeze_last_n_layers=unfreeze_last_n_layers,
        )

        # ── Step 2: Image Branch (Section 4.2) ───────────────────
        self.image_encoder = ImageEncoder(
            proj_dim=proj_dim,
            dropout_rate=dropout_rate,
        )

        # ── Step 3: Consistency GNN (Section 4.3) ────────────────
        # Used offline to pre-compute s_cons for each sample
        # Can also be trained end-to-end if CLIP features extracted first
        self.consistency_gnn = CrossModalAttentionGNN(
            clip_dim=512,
            hidden_dim=gnn_hidden_dim,
            num_heads=gnn_num_heads,
            dropout_rate=0.1,
        )

        # ── Step 4: Dynamic Gate (Section 4.4) ───────────────────
        self.dynamic_gate = DynamicGate(
            proj_dim=proj_dim,
            gate_hidden_dim=gate_hidden_dim,
            dropout_rate=dropout_rate,
        )

        # ── Step 5: Uncertainty-Aware Fusion (Section 4.5) ───────
        self.adaptive_fusion = UncertaintyAwareFusion(
            proj_dim=proj_dim,
            fusion_hidden=fusion_hidden_dim,
            dropout_rate=dropout_rate,
        )

        # ── Step 6: Final Classifier (Section 4.6) ───────────────
        # Input: [z'_text || z'_img || s_cons]
        # = 512 + 512 + 1 = 1025
        classifier_input_dim = proj_dim + proj_dim + 1

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden, 2),  # Binary: Real / Fake
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        images,
        s_cons,
        image_clip_feats=None,
        text_clip_feats=None,
    ):
        """
        Full forward pass through all 6 components.

        Args:
            input_ids        : [B, seq_len]       DistilBERT token IDs
            attention_mask   : [B, seq_len]       DistilBERT attention mask
            images           : [B, 3, 224, 224]   ResNet-50 input
            s_cons           : [B, 1]             pre-computed consistency score
            image_clip_feats : [B, num_reg, 512]  optional CLIP image features
            text_clip_feats  : [B, num_seg, 512]  optional CLIP text features

        Returns:
            logits     : [B, 2]   classification logits
            w_text     : [B, 1]   text modality weight  (interpretability)
            w_img      : [B, 1]   image modality weight (interpretability)
            g_text     : [B, 512] text gate values      (interpretability)
            g_img      : [B, 512] image gate values     (interpretability)
            s_cons     : [B, 1]   consistency score     (interpretability)
            u_text     : [B, 1]   text uncertainty proxy
            u_img      : [B, 1]   image uncertainty proxy
        """

        # ── Step 1: Text Branch ───────────────────────────────────
        # z_text ∈ R^512, u_text ∈ R^1
        z_text, u_text = self.text_encoder(input_ids, attention_mask)

        # ── Step 2: Image Branch ──────────────────────────────────
        # z_img ∈ R^512, u_img ∈ R^1
        z_img, u_img = self.image_encoder(images)

        # ── Step 3: Consistency Score ─────────────────────────────
        # s_cons is pre-computed offline (see consistency_gnn.py)
        # and passed as input for efficient end-to-end training.
        # If CLIP features are provided, compute dynamically:
        if image_clip_feats is not None and text_clip_feats is not None:
            s_cons = self.consistency_gnn(
                image_clip_feats, text_clip_feats
            )  # [B, 1]

        # ── Step 4: Dynamic Gate ──────────────────────────────────
        # z'_text = z_text ⊙ g_text
        # z'_img  = z_img  ⊙ g_img
        z_prime_text, z_prime_img, g_text, g_img = self.dynamic_gate(
            s_cons, z_text, z_img
        )

        # ── Step 5: Uncertainty-Aware Adaptive Fusion ─────────────
        # w_text, w_img ∈ [0,1], sum = 1
        w_text, w_img = self.adaptive_fusion(
            z_prime_text, z_prime_img, u_text, u_img
        )

        # ── Step 6: Final Classifier ──────────────────────────────
        # Input: [z'_text || z'_img || s_cons] ∈ R^1025
        clf_input = torch.cat(
            [z_prime_text, z_prime_img, s_cons], dim=-1
        )  # [B, 1025]
        logits = self.classifier(clf_input)  # [B, 2]

        return (
            logits,
            w_text,
            w_img,
            g_text,
            g_img,
            s_cons,
            u_text,
            u_img,
        )


# =============================================================
# MC Dropout Inference — Section 4.6
# =============================================================

def enable_dropout(model):
    """
    Enable dropout layers during inference for MC Dropout.
    Call before mc_inference() to activate stochastic passes.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_inference(
    model,
    input_ids,
    attention_mask,
    images,
    s_cons,
    T: int = 20,
    device: str = "cpu",
):
    """
    Monte Carlo Dropout Inference — Algorithm 1, Step 7

    Performs T=20 stochastic forward passes with dropout enabled
    to estimate predictive uncertainty sigma^2.

    Args:
        model          : GatedConsistencyRumorDetector
        input_ids      : [B, seq_len]
        attention_mask : [B, seq_len]
        images         : [B, 3, 224, 224]
        s_cons         : [B, 1]
        T              : number of MC samples (default=20)
        device         : compute device

    Returns:
        p_hat          : [B, 2]  mean predictive probability
        sigma_sq       : [B]     predictive uncertainty (variance)
        mean_w_text    : [B, 1]  mean text weight across T passes
        mean_w_img     : [B, 1]  mean image weight across T passes
    """

    model.eval()
    enable_dropout(model)  # Keep dropout active for stochastic passes

    all_probs   = []
    all_w_text  = []
    all_w_img   = []

    with torch.no_grad():
        for t in range(T):
            logits, w_text, w_img, _, _, _, _, _ = model(
                input_ids, attention_mask, images, s_cons
            )
            probs = torch.softmax(logits, dim=-1)  # [B, 2]
            all_probs.append(probs.unsqueeze(0))   # [1, B, 2]
            all_w_text.append(w_text)
            all_w_img.append(w_img)

    # Stack T passes: [T, B, 2]
    all_probs = torch.cat(all_probs, dim=0)

    # Mean prediction p_hat = (1/T) Σ p_t
    p_hat = all_probs.mean(dim=0)  # [B, 2]

    # Predictive variance σ² = (1/T) Σ (p_t - p_hat)²
    sigma_sq = all_probs.var(dim=0).mean(dim=-1)  # [B]

    # Mean modality weights across T passes
    mean_w_text = torch.stack(all_w_text, dim=0).mean(dim=0)  # [B, 1]
    mean_w_img  = torch.stack(all_w_img,  dim=0).mean(dim=0)  # [B, 1]

    return p_hat, sigma_sq, mean_w_text, mean_w_img


# =============================================================
# Quick test
# =============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Instantiate full model
    model = GatedConsistencyRumorDetector(
        proj_dim=512,
        gnn_hidden_dim=256,
        gnn_num_heads=4,
        gate_hidden_dim=512,
        fusion_hidden_dim=256,
        classifier_hidden=128,
        dropout_rate=0.3,
    ).to(device)

    # Parameter count
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable : {trainable:,}")
    print(f"Total     : {total:,}")

    # ── Dummy inputs ─────────────────────────────────────────────
    B = 2

    # Text inputs (DistilBERT tokenized)
    input_ids      = torch.randint(0, 30522, (B, 64)).to(device)
    attention_mask = torch.ones(B, 64, dtype=torch.long).to(device)

    # Image inputs (ResNet-50 normalized)
    images = torch.randn(B, 3, 224, 224).to(device)

    # Pre-computed consistency scores
    s_cons = torch.rand(B, 1).to(device)

    # ── Standard forward pass ─────────────────────────────────────
    model.train()
    (
        logits,
        w_text,
        w_img,
        g_text,
        g_img,
        s_cons_out,
        u_text,
        u_img,
    ) = model(input_ids, attention_mask, images, s_cons)

    print(f"\n--- Forward Pass Output Shapes ---")
    print(f"logits    : {logits.shape}")      # [2, 2]
    print(f"w_text    : {w_text.shape}")      # [2, 1]
    print(f"w_img     : {w_img.shape}")       # [2, 1]
    print(f"g_text    : {g_text.shape}")      # [2, 512]
    print(f"g_img     : {g_img.shape}")       # [2, 512]
    print(f"s_cons    : {s_cons_out.shape}")  # [2, 1]
    print(f"u_text    : {u_text.shape}")      # [2, 1]
    print(f"u_img     : {u_img.shape}")       # [2, 1]

    # ── Interpretability check ────────────────────────────────────
    print(f"\n--- Interpretability Values ---")
    print(f"w_text (text weights)   : {w_text.detach().cpu().numpy().flatten()}")
    print(f"w_img  (image weights)  : {w_img.detach().cpu().numpy().flatten()}")
    print(f"w_text + w_img = 1.0   : {torch.allclose(w_text + w_img, torch.ones_like(w_text))}")

    # ── MC Dropout inference ──────────────────────────────────────
    print(f"\n--- MC Dropout Inference (T=20) ---")
    p_hat, sigma_sq, mean_w_text, mean_w_img = mc_inference(
        model, input_ids, attention_mask, images, s_cons, T=20, device=device
    )
    print(f"p_hat       : {p_hat.shape}")      # [2, 2]
    print(f"sigma_sq    : {sigma_sq.shape}")   # [2]
    print(f"mean_w_text : {mean_w_text.shape}")# [2, 1]
    print(f"mean_w_img  : {mean_w_img.shape}") # [2, 1]
    print(f"Uncertainty (sigma^2): {sigma_sq.detach().cpu().numpy()}")
