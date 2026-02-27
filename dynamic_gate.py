# =============================================================
# dynamic_gate.py
# Implements Section 4.4: Dynamic Gating Mechanism
# Paper: "Gated Consistency and Uncertainty-Aware Fusion:
#         An Interpretable Framework for Multimodal Rumour Detection"
# =============================================================

import torch
import torch.nn as nn


class DynamicGate(nn.Module):
    """
    Section 4.4 — Dynamic Reliability Gate

    Computes per-sample element-wise gates g_text and g_img
    that modulate feature contributions based on cross-modal
    consistency score s_cons.

    Architecture:
        Input  : [s_cons || z_text || z_img] ∈ R^{1 + 512 + 512}
        MLP    : Linear(1025→512) → ReLU → Dropout
                 Linear(512→1024) → Sigmoid
        Output : [g_text || g_img] ∈ R^{512 + 512}

    Gated Features:
        z'_text = z_text ⊙ g_text   (element-wise Hadamard product)
        z'_img  = z_img  ⊙ g_img

    Behaviour:
        s_cons → 1.0  :  both gates open  → both modalities trusted
        s_cons → 0.0  :  gates selectively suppress unreliable modality
                         (model learns which to suppress per sample)

    Outputs:
        z'_text : gated text feature   [B, 512]
        z'_img  : gated image feature  [B, 512]
        g_text  : text gate values     [B, 512]  (for interpretability)
        g_img   : image gate values    [B, 512]  (for interpretability)
    """

    def __init__(
        self,
        proj_dim: int = 512,
        gate_hidden_dim: int = 512,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        # Gate input dimension:
        # 1 (s_cons) + proj_dim (z_text) + proj_dim (z_img)
        gate_input_dim = 1 + proj_dim + proj_dim  # = 1025

        # Gate output dimension:
        # proj_dim (g_text) + proj_dim (g_img)
        gate_output_dim = proj_dim + proj_dim      # = 1024

        # ── Gating MLP ───────────────────────────────────────────
        # Eq (4): g = Sigmoid(MLP([s_cons || z_text || z_img]))
        # Sigmoid constrains gates to [0,1] for interpretable weighting
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(gate_hidden_dim, gate_output_dim),
            nn.Sigmoid(),
        )

        self.proj_dim = proj_dim

    def forward(self, s_cons, z_text, z_img):
        """
        Forward pass of the Dynamic Gating Mechanism.

        Args:
            s_cons : [B, 1]   cross-modal consistency score
            z_text : [B, 512] projected text feature
            z_img  : [B, 512] projected image feature

        Returns:
            z_prime_text : [B, 512] gated text feature
            z_prime_img  : [B, 512] gated image feature
            g_text       : [B, 512] text gate (for interpretability)
            g_img        : [B, 512] image gate (for interpretability)
        """

        # ── Concatenate gate inputs ───────────────────────────────
        # gate_in = [s_cons || z_text || z_img] ∈ R^{B × 1025}
        gate_in = torch.cat([s_cons, z_text, z_img], dim=-1)  # [B, 1025]

        # ── Compute gates ─────────────────────────────────────────
        # gates ∈ R^{B × 1024}, values in [0, 1]
        gates = self.gate_mlp(gate_in)  # [B, 1024]

        # ── Split into modality-specific gates ────────────────────
        g_text = gates[:, : self.proj_dim]   # [B, 512]
        g_img  = gates[:, self.proj_dim :]   # [B, 512]

        # ── Apply gates via Hadamard product ─────────────────────
        # Eq (5): z'_text = z_text ⊙ g_text
        # Eq (6): z'_img  = z_img  ⊙ g_img
        z_prime_text = z_text * g_text  # [B, 512]
        z_prime_img  = z_img  * g_img   # [B, 512]

        return z_prime_text, z_prime_img, g_text, g_img


# =============================================================
# Quick test
# =============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    gate = DynamicGate(
        proj_dim=512,
        gate_hidden_dim=512,
        dropout_rate=0.3,
    ).to(device)

    trainable = sum(p.numel() for p in gate.parameters() if p.requires_grad)
    print(f"DynamicGate trainable parameters: {trainable:,}")

    B = 4
    s_cons = torch.rand(B, 1).to(device)
    z_text = torch.randn(B, 512).to(device)
    z_img  = torch.randn(B, 512).to(device)

    z_prime_text, z_prime_img, g_text, g_img = gate(s_cons, z_text, z_img)

    print(f"z_prime_text shape : {z_prime_text.shape}")  # [4, 512]
    print(f"z_prime_img  shape : {z_prime_img.shape}")   # [4, 512]
    print(f"g_text shape       : {g_text.shape}")        # [4, 512]
    print(f"g_img  shape       : {g_img.shape}")         # [4, 512]

    # Gate values should be in [0, 1]
    print(f"g_text in [0,1]    : {(g_text >= 0).all() and (g_text <= 1).all()}")
    print(f"g_img  in [0,1]    : {(g_img  >= 0).all() and (g_img  <= 1).all()}")

    # Test interpretability: high s_cons should open both gates
    print("\n--- Interpretability Check ---")
    high_cons = torch.ones(1, 1).to(device)
    low_cons  = torch.zeros(1, 1).to(device)
    z_t = torch.randn(1, 512).to(device)
    z_i = torch.randn(1, 512).to(device)

    _, _, g_t_high, g_i_high = gate(high_cons, z_t, z_i)
    _, _, g_t_low,  g_i_low  = gate(low_cons,  z_t, z_i)

    print(f"High s_cons → mean g_text: {g_t_high.mean().item():.4f}, "
          f"mean g_img: {g_i_high.mean().item():.4f}")
    print(f"Low  s_cons → mean g_text: {g_t_low.mean().item():.4f},  "
          f"mean g_img: {g_i_low.mean().item():.4f}")
