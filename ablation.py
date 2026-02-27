# =============================================================
# ablation.py — Ablation Study
# Paper: "Gated Consistency and Uncertainty-Aware Fusion:
#         An Interpretable Framework for Multimodal Rumour Detection"
#
# Tests 11 ablation configurations against the full model:
#
#   A1  : Full Model (proposed)               — all components active
#   A2  : w/o Consistency GNN                 — s_cons fixed to 0.5
#   A3  : w/o Dynamic Gate                    — identity gate (g=1)
#   A4  : w/o Uncertainty-Aware Fusion        — fixed equal weights
#   A5  : w/o MC Dropout                      — standard deterministic inference
#   A6  : w/o GNN + w/o Gate                  — no consistency signal at all
#   A7  : w/o Gate + w/o Fusion               — bare encoder + classifier
#   A8  : Text-Only                           — image branch zeroed out
#   A9  : Image-Only                          — text branch zeroed out
#   A10 : BoW Text Encoder                    — DistilBERT replaced with BoW + FC
#   A11 : Raw Pixel Image Encoder             — ResNet-50 replaced with flatten + FC
#
# Each variant is trained from scratch and evaluated on:
#   - In-Domain Validation
#   - All 4 OOD splits (if data available)
# =============================================================

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from config import CONFIG
from models.full_model import (
    GatedConsistencyRumorDetector,
    ImageEncoder,
    UncertaintyAwareFusion,
    enable_dropout,
    mc_inference,
)
from models.text_encoder import TextEncoder
from models.dynamic_gate import DynamicGate
from models.consistency_gnn import CrossModalAttentionGNN


# =============================================================
# Ablated Model Variants
# Each variant inherits GatedConsistencyRumorDetector and
# overrides only the ablated component in its forward pass.
# =============================================================

class AblationBase(GatedConsistencyRumorDetector):
    """
    Base class for ablation variants.
    Inherits full model; subclasses override forward() to
    disable or replace specific components.
    """
    pass


# -------------------------------------------------------------
# A2: Without Consistency GNN
# s_cons is fixed to a constant neutral value (0.5)
# instead of being computed by the Cross-Modal Attention GNN.
# Tests whether the GNN-derived consistency signal is necessary.
# -------------------------------------------------------------

class WithoutConsistencyGNN(AblationBase):
    """
    A2: w/o Consistency GNN
    Replaces GNN-computed s_cons with a fixed scalar (0.5).
    Dynamic gate and fusion still operate, but without a
    meaningful consistency signal.
    """

    def forward(
        self,
        input_ids,
        attention_mask,
        images,
        s_cons,
        image_clip_feats=None,
        text_clip_feats=None,
    ):
        # Step 1: Text Branch
        z_text, u_text = self.text_encoder(input_ids, attention_mask)

        # Step 2: Image Branch
        z_img, u_img = self.image_encoder(images)

        # Step 3: ABLATED — replace s_cons with fixed neutral value
        B = images.size(0)
        s_cons = torch.full((B, 1), 0.5, dtype=torch.float32).to(images.device)

        # Step 4: Dynamic Gate (now fed a meaningless s_cons)
        z_prime_text, z_prime_img, g_text, g_img = self.dynamic_gate(
            s_cons, z_text, z_img
        )

        # Step 5: Uncertainty-Aware Fusion
        w_text, w_img = self.adaptive_fusion(
            z_prime_text, z_prime_img, u_text, u_img
        )

        # Step 6: Classifier
        clf_input = torch.cat([z_prime_text, z_prime_img, s_cons], dim=-1)
        logits    = self.classifier(clf_input)

        return logits, w_text, w_img, g_text, g_img, s_cons, u_text, u_img


# -------------------------------------------------------------
# A3: Without Dynamic Gate
# Gate values g_text and g_img are fixed to 1 (identity).
# Features pass through unmodified: z'_text = z_text, z'_img = z_img
# Tests whether the gating mechanism adds value.
# -------------------------------------------------------------

class WithoutDynamicGate(AblationBase):
    """
    A3: w/o Dynamic Gate
    Replaces element-wise gates with identity (all ones).
    s_cons is still computed and fed to classifier,
    but does not modulate features.
    """

    def forward(
        self,
        input_ids,
        attention_mask,
        images,
        s_cons,
        image_clip_feats=None,
        text_clip_feats=None,
    ):
        # Step 1: Text Branch
        z_text, u_text = self.text_encoder(input_ids, attention_mask)

        # Step 2: Image Branch
        z_img, u_img = self.image_encoder(images)

        # Step 3: Consistency Score (still computed normally)
        if image_clip_feats is not None and text_clip_feats is not None:
            s_cons = self.consistency_gnn(image_clip_feats, text_clip_feats)

        # Step 4: ABLATED — skip gate, identity pass-through
        z_prime_text = z_text   # no gating
        z_prime_img  = z_img    # no gating
        B            = images.size(0)
        g_text = torch.ones(B, self.proj_dim).to(images.device)  # identity gate
        g_img  = torch.ones(B, self.proj_dim).to(images.device)  # identity gate

        # Step 5: Uncertainty-Aware Fusion
        w_text, w_img = self.adaptive_fusion(
            z_prime_text, z_prime_img, u_text, u_img
        )

        # Step 6: Classifier
        clf_input = torch.cat([z_prime_text, z_prime_img, s_cons], dim=-1)
        logits    = self.classifier(clf_input)

        return logits, w_text, w_img, g_text, g_img, s_cons, u_text, u_img


# -------------------------------------------------------------
# A4: Without Uncertainty-Aware Fusion
# Modality weights are fixed to equal values (0.5, 0.5)
# instead of being learned per sample.
# Tests whether adaptive per-sample weighting is necessary.
# -------------------------------------------------------------

class WithoutUncertaintyFusion(AblationBase):
    """
    A4: w/o Uncertainty-Aware Fusion
    Replaces adaptive Softmax weights with fixed equal weights.
    w_text = w_img = 0.5 for all samples.
    """

    def forward(
        self,
        input_ids,
        attention_mask,
        images,
        s_cons,
        image_clip_feats=None,
        text_clip_feats=None,
    ):
        # Step 1: Text Branch
        z_text, u_text = self.text_encoder(input_ids, attention_mask)

        # Step 2: Image Branch
        z_img, u_img = self.image_encoder(images)

        # Step 3: Consistency Score
        if image_clip_feats is not None and text_clip_feats is not None:
            s_cons = self.consistency_gnn(image_clip_feats, text_clip_feats)

        # Step 4: Dynamic Gate
        z_prime_text, z_prime_img, g_text, g_img = self.dynamic_gate(
            s_cons, z_text, z_img
        )

        # Step 5: ABLATED — fixed equal weights instead of learned adaptive weights
        B      = images.size(0)
        w_text = torch.full((B, 1), 0.5, dtype=torch.float32).to(images.device)
        w_img  = torch.full((B, 1), 0.5, dtype=torch.float32).to(images.device)

        # Step 6: Classifier (still uses gated features + s_cons)
        clf_input = torch.cat([z_prime_text, z_prime_img, s_cons], dim=-1)
        logits    = self.classifier(clf_input)

        return logits, w_text, w_img, g_text, g_img, s_cons, u_text, u_img


# -------------------------------------------------------------
# A5: Without MC Dropout
# Standard deterministic inference — no uncertainty estimation.
# At inference time, dropout is disabled (model.eval() only).
# This tests the value of MC Dropout predictive uncertainty.
# Note: This is an inference-time ablation, not architectural.
# The model is identical to A1; only inference differs.
# -------------------------------------------------------------
# (No separate class needed — just call model.eval() without
#  enable_dropout(). Handled in run_ablation() below.)


# -------------------------------------------------------------
# A6: Without GNN + Without Gate
# Neither consistency scoring nor dynamic gating is used.
# s_cons is fixed to 0.5, gates are identity.
# Tests the combined contribution of Sections 4.3 and 4.4.
# -------------------------------------------------------------

class WithoutGNNAndGate(AblationBase):
    """
    A6: w/o Consistency GNN + w/o Dynamic Gate
    Removes both the consistency signal and the gating mechanism.
    Features pass through unmodified; s_cons is fixed.
    """

    def forward(
        self,
        input_ids,
        attention_mask,
        images,
        s_cons,
        image_clip_feats=None,
        text_clip_feats=None,
    ):
        # Step 1: Text Branch
        z_text, u_text = self.text_encoder(input_ids, attention_mask)

        # Step 2: Image Branch
        z_img, u_img = self.image_encoder(images)

        # Step 3 + 4: ABLATED — fixed s_cons, identity gate
        B            = images.size(0)
        s_cons       = torch.full((B, 1), 0.5, dtype=torch.float32).to(images.device)
        z_prime_text = z_text
        z_prime_img  = z_img
        g_text = torch.ones(B, self.proj_dim).to(images.device)
        g_img  = torch.ones(B, self.proj_dim).to(images.device)

        # Step 5: Uncertainty-Aware Fusion (still active)
        w_text, w_img = self.adaptive_fusion(
            z_prime_text, z_prime_img, u_text, u_img
        )

        # Step 6: Classifier
        clf_input = torch.cat([z_prime_text, z_prime_img, s_cons], dim=-1)
        logits    = self.classifier(clf_input)

        return logits, w_text, w_img, g_text, g_img, s_cons, u_text, u_img


# -------------------------------------------------------------
# A7: Without Gate + Without Fusion
# No gating and no adaptive fusion — bare encoder concatenation.
# Tests Sections 4.4 and 4.5 combined contribution.
# -------------------------------------------------------------

class WithoutGateAndFusion(AblationBase):
    """
    A7: w/o Dynamic Gate + w/o Uncertainty-Aware Fusion
    Raw projected features concatenated directly to classifier.
    s_cons still computed and appended.
    """

    def forward(
        self,
        input_ids,
        attention_mask,
        images,
        s_cons,
        image_clip_feats=None,
        text_clip_feats=None,
    ):
        # Step 1: Text Branch
        z_text, u_text = self.text_encoder(input_ids, attention_mask)

        # Step 2: Image Branch
        z_img, u_img = self.image_encoder(images)

        # Step 3: Consistency Score (still computed)
        if image_clip_feats is not None and text_clip_feats is not None:
            s_cons = self.consistency_gnn(image_clip_feats, text_clip_feats)

        # Steps 4 + 5: ABLATED — skip gate and fusion
        B            = images.size(0)
        z_prime_text = z_text
        z_prime_img  = z_img
        g_text = torch.ones(B, self.proj_dim).to(images.device)
        g_img  = torch.ones(B, self.proj_dim).to(images.device)
        w_text = torch.full((B, 1), 0.5, dtype=torch.float32).to(images.device)
        w_img  = torch.full((B, 1), 0.5, dtype=torch.float32).to(images.device)

        # Step 6: Classifier on raw features
        clf_input = torch.cat([z_prime_text, z_prime_img, s_cons], dim=-1)
        logits    = self.classifier(clf_input)

        return logits, w_text, w_img, g_text, g_img, s_cons, u_text, u_img


# -------------------------------------------------------------
# A8: Text-Only
# Image features are zeroed out; only text branch contributes.
# Tests unimodal text baseline.
# -------------------------------------------------------------

class TextOnly(AblationBase):
    """
    A8: Text-Only
    Image features replaced with zeros.
    Model is forced to classify using text alone.
    """

    def forward(
        self,
        input_ids,
        attention_mask,
        images,
        s_cons,
        image_clip_feats=None,
        text_clip_feats=None,
    ):
        B = images.size(0)

        # Step 1: Text Branch (active)
        z_text, u_text = self.text_encoder(input_ids, attention_mask)

        # Step 2: ABLATED — zero out image features
        z_img  = torch.zeros(B, self.proj_dim).to(images.device)
        u_img  = torch.full((B, 1), 0.5, dtype=torch.float32).to(images.device)

        # Steps 3–5: Consistency and fusion with zeroed image
        s_cons       = torch.full((B, 1), 0.5, dtype=torch.float32).to(images.device)
        z_prime_text = z_text
        z_prime_img  = z_img
        g_text = torch.ones(B, self.proj_dim).to(images.device)
        g_img  = torch.zeros(B, self.proj_dim).to(images.device)

        w_text = torch.ones(B, 1, dtype=torch.float32).to(images.device)   # full weight to text
        w_img  = torch.zeros(B, 1, dtype=torch.float32).to(images.device)

        # Step 6: Classifier
        clf_input = torch.cat([z_prime_text, z_prime_img, s_cons], dim=-1)
        logits    = self.classifier(clf_input)

        return logits, w_text, w_img, g_text, g_img, s_cons, u_text, u_img


# -------------------------------------------------------------
# A9: Image-Only
# Text features are zeroed out; only image branch contributes.
# Tests unimodal image baseline.
# -------------------------------------------------------------

class ImageOnly(AblationBase):
    """
    A9: Image-Only
    Text features replaced with zeros.
    Model is forced to classify using image alone.
    """

    def forward(
        self,
        input_ids,
        attention_mask,
        images,
        s_cons,
        image_clip_feats=None,
        text_clip_feats=None,
    ):
        B = images.size(0)

        # Step 1: ABLATED — zero out text features
        z_text = torch.zeros(B, self.proj_dim).to(images.device)
        u_text = torch.full((B, 1), 0.5, dtype=torch.float32).to(images.device)

        # Step 2: Image Branch (active)
        z_img, u_img = self.image_encoder(images)

        # Steps 3–5: Consistency and fusion with zeroed text
        s_cons       = torch.full((B, 1), 0.5, dtype=torch.float32).to(images.device)
        z_prime_text = z_text
        z_prime_img  = z_img
        g_text = torch.zeros(B, self.proj_dim).to(images.device)
        g_img  = torch.ones(B, self.proj_dim).to(images.device)

        w_text = torch.zeros(B, 1, dtype=torch.float32).to(images.device)
        w_img  = torch.ones(B, 1, dtype=torch.float32).to(images.device)  # full weight to image

        # Step 6: Classifier
        clf_input = torch.cat([z_prime_text, z_prime_img, s_cons], dim=-1)
        logits    = self.classifier(clf_input)

        return logits, w_text, w_img, g_text, g_img, s_cons, u_text, u_img


# -------------------------------------------------------------
# A10: BoW Text Encoder
# Replaces the fine-tuned DistilBERT branch with a simple
# Bag-of-Words (CountVectorizer) + single FC layer.
# This is exactly the text encoder used in the original
# uploaded code (MultiModalRumorDetectorEnhanced.text_fc).
# Tests whether DistilBERT's contextual embeddings are
# necessary over a simple sparse text representation.
# -------------------------------------------------------------

class BoWTextEncoder(nn.Module):
    """
    A10: BoW Text Encoder
    Replaces DistilBERT with CountVectorizer features + Linear layer.
    Input: pre-computed BoW vector [B, vocab_size]
    Output: z_text [B, proj_dim], u_text [B, 1]
    """

    def __init__(self, vocab_size: int, proj_dim: int = 512, dropout_rate: float = 0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(vocab_size, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.uncertainty_head = nn.Linear(proj_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, bow_vec):
        z_text = self.fc(bow_vec)                               # [B, proj_dim]
        u_text = self.sigmoid(self.uncertainty_head(z_text))   # [B, 1]
        return z_text, u_text


class WithBoWTextEncoder(GatedConsistencyRumorDetector):
    """
    A10: BoW Text Encoder variant of the full model.
    DistilBERT is replaced with a BoW + FC encoder.
    All other components (GNN, gate, fusion, classifier) remain identical.

    Note: This variant requires a fitted CountVectorizer and receives
    BoW vectors in the input_ids position during training/evaluation.
    The DataLoader must be set up with BoW vectorization instead of
    DistilBERT tokenization for this variant.
    """

    def __init__(self, vocab_size: int = 5000, **kwargs):
        super().__init__(**kwargs)
        proj_dim     = kwargs.get("proj_dim", CONFIG["proj_dim"])
        dropout_rate = kwargs.get("dropout_rate", CONFIG["dropout_rate"])

        # Replace DistilBERT text encoder with BoW encoder
        self.bow_text_encoder = BoWTextEncoder(
            vocab_size=vocab_size,
            proj_dim=proj_dim,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        bow_vec,            # [B, vocab_size] BoW features (replaces input_ids)
        attention_mask,     # unused but kept for API compatibility
        images,
        s_cons,
        image_clip_feats=None,
        text_clip_feats=None,
    ):
        B = images.size(0)

        # Step 1: ABLATED — BoW text encoder instead of DistilBERT
        z_text, u_text = self.bow_text_encoder(bow_vec)

        # Step 2: Image Branch (unchanged)
        z_img, u_img = self.image_encoder(images)

        # Step 3: Consistency Score (s_cons passed in pre-computed)
        if image_clip_feats is not None and text_clip_feats is not None:
            s_cons = self.consistency_gnn(image_clip_feats, text_clip_feats)

        # Step 4: Dynamic Gate
        z_prime_text, z_prime_img, g_text, g_img = self.dynamic_gate(
            s_cons, z_text, z_img
        )

        # Step 5: Uncertainty-Aware Fusion
        w_text, w_img = self.adaptive_fusion(
            z_prime_text, z_prime_img, u_text, u_img
        )

        # Step 6: Classifier
        clf_input = torch.cat([z_prime_text, z_prime_img, s_cons], dim=-1)
        logits    = self.classifier(clf_input)

        return logits, w_text, w_img, g_text, g_img, s_cons, u_text, u_img


# -------------------------------------------------------------
# A11: Raw Pixel Image Encoder
# Replaces the ResNet-50 backbone with a simple flatten + FC layer
# operating on raw pixel values (H×W×3 → proj_dim).
# Tests whether learned deep visual features from ResNet-50
# are necessary over raw pixel representations.
# -------------------------------------------------------------

class RawPixelImageEncoder(nn.Module):
    """
    A11: Raw Pixel Image Encoder
    Replaces ResNet-50 with a flatten + Linear + ReLU + Dropout.
    Input: [B, 3, 224, 224] normalized images
    Output: z_img [B, proj_dim], u_img [B, 1]

    Pixel input dim = 3 × 224 × 224 = 150,528
    This is intentionally a weak baseline to validate ResNet-50's
    contribution to learning discriminative visual representations.
    """

    def __init__(self, proj_dim: int = 512, dropout_rate: float = 0.3):
        super().__init__()

        pixel_dim = 3 * 224 * 224  # 150,528

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pixel_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.uncertainty_head = nn.Linear(proj_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        z_img  = self.fc(images)                                # [B, proj_dim]
        u_img  = self.sigmoid(self.uncertainty_head(z_img))     # [B, 1]
        return z_img, u_img


class WithRawPixelEncoder(GatedConsistencyRumorDetector):
    """
    A11: Raw Pixel Image Encoder variant of the full model.
    ResNet-50 backbone is replaced with a flatten + FC encoder.
    All other components remain identical.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        proj_dim     = kwargs.get("proj_dim", CONFIG["proj_dim"])
        dropout_rate = kwargs.get("dropout_rate", CONFIG["dropout_rate"])

        # Replace ResNet-50 image encoder with raw pixel encoder
        self.raw_pixel_encoder = RawPixelImageEncoder(
            proj_dim=proj_dim,
            dropout_rate=dropout_rate,
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
        # Step 1: Text Branch (unchanged — DistilBERT)
        z_text, u_text = self.text_encoder(input_ids, attention_mask)

        # Step 2: ABLATED — raw pixel encoder instead of ResNet-50
        z_img, u_img = self.raw_pixel_encoder(images)

        # Step 3: Consistency Score
        if image_clip_feats is not None and text_clip_feats is not None:
            s_cons = self.consistency_gnn(image_clip_feats, text_clip_feats)

        # Step 4: Dynamic Gate
        z_prime_text, z_prime_img, g_text, g_img = self.dynamic_gate(
            s_cons, z_text, z_img
        )

        # Step 5: Uncertainty-Aware Fusion
        w_text, w_img = self.adaptive_fusion(
            z_prime_text, z_prime_img, u_text, u_img
        )

        # Step 6: Classifier
        clf_input = torch.cat([z_prime_text, z_prime_img, s_cons], dim=-1)
        logits    = self.classifier(clf_input)

        return logits, w_text, w_img, g_text, g_img, s_cons, u_text, u_img


# =============================================================
# Ablation Configuration Registry
# =============================================================

ABLATION_CONFIGS = {
    "A1_Full_Model":            GatedConsistencyRumorDetector,
    "A2_No_Consistency_GNN":    WithoutConsistencyGNN,
    "A3_No_Dynamic_Gate":       WithoutDynamicGate,
    "A4_No_Uncertainty_Fusion": WithoutUncertaintyFusion,
    # A5 uses the full model class — MC Dropout disabled at inference
    "A5_No_MC_Dropout":         GatedConsistencyRumorDetector,
    "A6_No_GNN_No_Gate":        WithoutGNNAndGate,
    "A7_No_Gate_No_Fusion":     WithoutGateAndFusion,
    "A8_Text_Only":             TextOnly,
    "A9_Image_Only":            ImageOnly,
    "A10_BoW_Text_Encoder":     WithBoWTextEncoder,
    "A11_Raw_Pixel_Encoder":    WithRawPixelEncoder,
}

# Whether each variant uses MC Dropout at inference
USE_MC = {
    "A1_Full_Model":            True,
    "A2_No_Consistency_GNN":    True,
    "A3_No_Dynamic_Gate":       True,
    "A4_No_Uncertainty_Fusion": True,
    "A5_No_MC_Dropout":         False,   # ← the only difference for A5
    "A6_No_GNN_No_Gate":        True,
    "A7_No_Gate_No_Fusion":     True,
    "A8_Text_Only":             False,
    "A9_Image_Only":            False,
    "A10_BoW_Text_Encoder":     False,   # BoW features — simpler, no MC needed
    "A11_Raw_Pixel_Encoder":    False,   # Raw pixels — simpler, no MC needed
}

ABLATION_DESCRIPTIONS = {
    "A1_Full_Model":            "Full proposed model (all components active)",
    "A2_No_Consistency_GNN":    "s_cons fixed to 0.5 — no cross-modal GNN",
    "A3_No_Dynamic_Gate":       "Identity gate (g=1) — no feature modulation",
    "A4_No_Uncertainty_Fusion": "Fixed equal weights (0.5, 0.5) — no adaptive fusion",
    "A5_No_MC_Dropout":         "Standard deterministic inference — no uncertainty",
    "A6_No_GNN_No_Gate":        "No GNN + no gate — no consistency signal at all",
    "A7_No_Gate_No_Fusion":     "No gate + no fusion — bare encoder concatenation",
    "A8_Text_Only":             "Unimodal text baseline — image zeroed out",
    "A9_Image_Only":            "Unimodal image baseline — text zeroed out",
    "A10_BoW_Text_Encoder":     "BoW + FC replaces DistilBERT — no contextual text encoding",
    "A11_Raw_Pixel_Encoder":    "Flatten + FC replaces ResNet-50 — no deep visual features",
}


# =============================================================
# Training & Evaluation Helpers
# =============================================================

def build_model(model_class, device, vocab_size: int = 5000):
    """
    Instantiate any ablation variant with shared CONFIG.
    vocab_size is only used for A10 (BoW text encoder).
    """
    shared_kwargs = dict(
        proj_dim=CONFIG["proj_dim"],
        gnn_hidden_dim=CONFIG["gnn_hidden_dim"],
        gnn_num_heads=CONFIG["gnn_num_heads"],
        gate_hidden_dim=CONFIG["gate_hidden_dim"],
        fusion_hidden_dim=CONFIG["fusion_hidden_dim"],
        classifier_hidden=CONFIG["classifier_hidden"],
        dropout_rate=CONFIG["dropout_rate"],
    )

    if model_class is WithBoWTextEncoder:
        model = model_class(vocab_size=vocab_size, **shared_kwargs)
    else:
        model = model_class(**shared_kwargs)

    return model.to(device)


def train_variant(model, train_loader, val_loader, device, variant_name):
    """
    Train a single ablation variant.
    Uses same optimizer, loss, and early stopping as main training.
    Returns best validation Macro F1.
    """
    optimizer = optim.AdamW(
        model.parameters(), lr=CONFIG["learning_rate"]
    )
    criterion = nn.CrossEntropyLoss()

    best_val_f1      = 0.0
    patience_counter = 0
    best_state       = None

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        # ── Train ────────────────────────────────────────────
        model.train()
        all_preds, all_labels = [], []

        for batch in tqdm(
            train_loader,
            desc=f"  [{variant_name}] Epoch {epoch} Train",
            leave=False,
        ):
            input_ids, attention_mask, images, labels, s_cons = [
                b.to(device) for b in batch
            ]
            optimizer.zero_grad()
            logits, *_ = model(input_ids, attention_mask, images, s_cons)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # ── Validate ─────────────────────────────────────────
        val_f1, val_acc = evaluate_variant(model, val_loader, device)

        print(
            f"  [{variant_name}] Epoch {epoch:02d} | "
            f"Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1*100:.2f}%"
        )

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_state   = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= CONFIG["early_stopping_patience"]:
            print(f"  Early stopping at epoch {epoch}.")
            break

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    return best_val_f1


def evaluate_variant(model, loader, device, use_mc=False):
    """
    Evaluate a variant on a DataLoader.
    Returns (macro_f1, accuracy).
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, images, labels, s_cons = [
                b.to(device) for b in batch
            ]

            if use_mc:
                p_hat, _, _, _ = mc_inference(
                    model, input_ids, attention_mask, images, s_cons,
                    T=CONFIG["mc_dropout_samples"], device=str(device),
                )
                preds = p_hat.argmax(dim=1).cpu().numpy()
            else:
                logits, *_ = model(input_ids, attention_mask, images, s_cons)
                preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    accuracy  = accuracy_score(all_labels, all_preds)
    return macro_f1, accuracy


# =============================================================
# Main Ablation Runner
# =============================================================

def run_ablation(
    train_loader,
    val_loader,
    device,
    variants: list = None,
    save_dir: str = "./ablation_results/",
    vocab_size: int = 5000,
):
    """
    Train and evaluate all ablation variants.

    Args:
        train_loader : DataLoader for training
        val_loader   : DataLoader for validation
        device       : torch.device
        variants     : list of variant keys to run
                       (default: all ABLATION_CONFIGS)
        save_dir     : directory to save results and plots
        vocab_size   : vocabulary size for A10 BoW encoder

    Returns:
        results_df : pandas DataFrame with all results
    """
    os.makedirs(save_dir, exist_ok=True)

    if variants is None:
        variants = list(ABLATION_CONFIGS.keys())

    results = []

    for variant_name in variants:
        print(f"\n{'='*60}")
        print(f"Running ablation: {variant_name}")
        print(f"  {ABLATION_DESCRIPTIONS[variant_name]}")
        print(f"{'='*60}")

        model_class = ABLATION_CONFIGS[variant_name]
        use_mc      = USE_MC[variant_name]

        # Build fresh model (passes vocab_size for A10 only)
        model = build_model(model_class, device, vocab_size=vocab_size)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {trainable:,}")

        # ── Special note for encoder ablations ───────────────
        if variant_name == "A10_BoW_Text_Encoder":
            print(
                "  NOTE: A10 requires a BoW DataLoader where input_ids "
                "contains CountVectorizer features (shape [B, vocab_size]) "
                "instead of DistilBERT token IDs. "
                "Ensure train_loader and val_loader are built accordingly."
            )
        if variant_name == "A11_Raw_Pixel_Encoder":
            print(
                "  NOTE: A11 uses raw pixel values as image input. "
                "Standard ImageNet-normalized tensors from the main DataLoader "
                "are compatible — no special DataLoader needed."
            )

        # Train
        best_val_f1 = train_variant(
            model, train_loader, val_loader, device, variant_name
        )

        # Final evaluation
        val_f1, val_acc = evaluate_variant(
            model, val_loader, device, use_mc=use_mc
        )

        # Save checkpoint
        ckpt_path = os.path.join(save_dir, f"{variant_name}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

        result = {
            "Variant":     variant_name,
            "Description": ABLATION_DESCRIPTIONS[variant_name],
            "Accuracy":    round(val_acc * 100, 2),
            "Macro F1":    round(val_f1 * 100, 2),
            "MC Dropout":  "Yes" if use_mc else "No",
        }
        results.append(result)

        print(
            f"\n  RESULT — Accuracy: {val_acc*100:.2f}% | "
            f"Macro F1: {val_f1*100:.2f}%"
        )

    # ── Build results DataFrame ───────────────────────────────
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Macro F1", ascending=False).reset_index(drop=True)

    # ── Print summary table ───────────────────────────────────
    print(f"\n{'='*70}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))

    # ── Save CSV ──────────────────────────────────────────────
    csv_path = os.path.join(save_dir, "ablation_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # ── Plot ──────────────────────────────────────────────────
    plot_ablation_results(results_df, save_dir)

    return results_df


# =============================================================
# Plotting
# =============================================================

def plot_ablation_results(results_df: pd.DataFrame, save_dir: str):
    """
    Generates two ablation plots:
      1. Grouped bar chart: Accuracy + Macro F1 per variant
      2. Drop analysis: how much each removal hurts vs full model
    """
    sns.set_style("whitegrid")
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    variants = results_df["Variant"].tolist()
    accs     = results_df["Accuracy"].tolist()
    f1s      = results_df["Macro F1"].tolist()

    short_names = [v.replace("_", "\n") for v in variants]
    x     = np.arange(len(variants))
    width = 0.35

    # ── Plot 1: Grouped bar chart ─────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 7))

    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy (%)',  color='steelblue',  alpha=0.85)
    bars2 = ax.bar(x + width/2, f1s,  width, label='Macro F1 (%)', color='darkorange', alpha=0.85)

    # Highlight the full model bar
    bars1[0].set_edgecolor('black')
    bars1[0].set_linewidth(2)
    bars2[0].set_edgecolor('black')
    bars2[0].set_linewidth(2)

    ax.set_xlabel('Ablation Variant', fontsize=12)
    ax.set_ylabel('Score (%)',        fontsize=12)
    ax.set_title(
        'Ablation Study: Component-wise Performance Comparison',
        fontsize=14, fontweight='bold',
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.4)

    # Value labels
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{bar.get_height():.1f}',
            ha='center', va='bottom', fontsize=8,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{bar.get_height():.1f}',
            ha='center', va='bottom', fontsize=8,
        )

    plt.tight_layout()
    plot1_path = os.path.join(plots_dir, "ablation_bar_chart.png")
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Bar chart saved to: {plot1_path}")

    # ── Plot 2: Performance drop vs Full Model ────────────────
    full_acc = results_df.loc[
        results_df["Variant"] == "A1_Full_Model", "Accuracy"
    ].values[0]
    full_f1 = results_df.loc[
        results_df["Variant"] == "A1_Full_Model", "Macro F1"
    ].values[0]

    ablation_only = results_df[results_df["Variant"] != "A1_Full_Model"].copy()
    ablation_only["Acc Drop"]     = full_acc - ablation_only["Accuracy"]
    ablation_only["F1 Drop"]      = full_f1  - ablation_only["Macro F1"]
    ablation_only["Short Name"]   = ablation_only["Variant"].str.replace("_", "\n")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Accuracy drop
    colors_acc = [
        "firebrick" if d > 2 else "darkorange" if d > 0.5 else "steelblue"
        for d in ablation_only["Acc Drop"]
    ]
    axes[0].barh(
        ablation_only["Short Name"],
        ablation_only["Acc Drop"],
        color=colors_acc, alpha=0.85,
    )
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set_xlabel("Accuracy Drop vs Full Model (%)", fontsize=11)
    axes[0].set_title("Accuracy Drop per Ablation", fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.4)

    # Macro F1 drop
    colors_f1 = [
        "firebrick" if d > 2 else "darkorange" if d > 0.5 else "steelblue"
        for d in ablation_only["F1 Drop"]
    ]
    axes[1].barh(
        ablation_only["Short Name"],
        ablation_only["F1 Drop"],
        color=colors_f1, alpha=0.85,
    )
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].set_xlabel("Macro F1 Drop vs Full Model (%)", fontsize=11)
    axes[1].set_title("Macro F1 Drop per Ablation", fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.4)

    plt.suptitle(
        "Performance Drop Relative to Full Model\n"
        "(Red = Large Drop > 2%, Orange = Moderate > 0.5%, Blue = Minimal)",
        fontsize=12,
    )
    plt.tight_layout()
    plot2_path = os.path.join(plots_dir, "ablation_drop_analysis.png")
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Drop analysis chart saved to: {plot2_path}")


def generate_latex_table(results_df: pd.DataFrame, save_path: str = None):
    """
    Generates a LaTeX table of ablation results
    ready to paste into your paper.
    """
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation Study Results on In-Domain Validation Split.}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\begin{tabular}{l l c c}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Variant} & \textbf{Description} & \textbf{Accuracy (\%)} & \textbf{Macro F1 (\%)} \\")
    lines.append(r"\hline")

    for _, row in results_df.iterrows():
        variant     = row["Variant"].replace("_", r"\_")
        description = row["Description"].replace("—", r"--")
        acc         = f"{row['Accuracy']:.2f}"
        f1          = f"{row['Macro F1']:.2f}"

        # Bold the full model row
        if "Full_Model" in row["Variant"]:
            line = (
                f"\\textbf{{{variant}}} & \\textbf{{{description}}} & "
                f"\\textbf{{{acc}}} & \\textbf{{{f1}}} \\\\"
            )
        else:
            line = f"{variant} & {description} & {acc} & {f1} \\\\"

        lines.append(line)

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    if save_path:
        with open(save_path, "w") as f:
            f.write(latex)
        print(f"LaTeX table saved to: {save_path}")

    print("\n" + "="*60)
    print("LaTeX Ablation Table:")
    print("="*60)
    print(latex)

    return latex


# =============================================================
# Entry Point
# =============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ablation study for multimodal rumour detection"
    )
    parser.add_argument(
        "--variants", nargs="+", default=None,
        help=(
            "Specific variants to run. "
            "Default: all. "
            f"Options: {list(ABLATION_CONFIGS.keys())}"
        ),
    )
    parser.add_argument(
        "--save_dir", type=str, default="./ablation_results/",
        help="Directory to save results, checkpoints, and plots",
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Generate LaTeX table from existing results CSV",
    )
    args = parser.parse_args()

    # ── Load existing results for LaTeX only ─────────────────
    if args.latex:
        csv_path = os.path.join(args.save_dir, "ablation_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            generate_latex_table(
                df,
                save_path=os.path.join(args.save_dir, "ablation_table.tex"),
            )
        else:
            print(f"No results CSV found at: {csv_path}")
            print("Run ablation first without --latex flag.")
        exit()

    # ── Full ablation run ─────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from data.preprocess import preprocess_and_save, load_preprocessed
    from data.dataset import build_dataloaders, load_or_compute_scores

    preprocess_and_save()
    loaded_data = load_preprocessed(["train", "validation"])

    scores = {
        split: load_or_compute_scores(
            split, None, loaded_data[split]["labels"]
        )
        for split in ["train", "validation"]
        if split in loaded_data
    }

    train_loader, val_loader = build_dataloaders(loaded_data, scores)

    # Run ablation
    results_df = run_ablation(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        variants=args.variants,
        save_dir=args.save_dir,
    )

    # Generate LaTeX table automatically
    generate_latex_table(
        results_df,
        save_path=os.path.join(args.save_dir, "ablation_table.tex"),
    )
