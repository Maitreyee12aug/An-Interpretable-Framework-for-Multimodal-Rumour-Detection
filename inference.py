# =============================================================
# inference.py — Single Sample Inference
# Paper: "Gated Consistency and Uncertainty-Aware Fusion:
#         An Interpretable Framework for Multimodal Rumour Detection"
#
# Usage:
#   python inference.py \
#       --image ./sample/image.jpg \
#       --caption "Your caption text here" \
#       --checkpoint ./checkpoints/best_model.pt
# =============================================================

import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import DistilBertTokenizer

from config import CONFIG
from models.full_model import GatedConsistencyRumorDetector, mc_inference
from models.consistency_gnn import CrossModalAttentionGNN, CrossModalConsistencyScorer


# ── Image transform for ResNet-50 ────────────────────────────
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model(checkpoint_path: str, device):
    model = GatedConsistencyRumorDetector(
        proj_dim=CONFIG["proj_dim"],
        gnn_hidden_dim=CONFIG["gnn_hidden_dim"],
        gnn_num_heads=CONFIG["gnn_num_heads"],
        gate_hidden_dim=CONFIG["gate_hidden_dim"],
        fusion_hidden_dim=CONFIG["fusion_hidden_dim"],
        classifier_hidden=CONFIG["classifier_hidden"],
        dropout_rate=CONFIG["dropout_rate"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"Model loaded from: {checkpoint_path}")
    return model


def predict(
    image_path: str,
    caption: str,
    checkpoint_path: str,
    device: str = None,
    use_mc: bool = True,
):
    """
    Run inference on a single image-caption pair.

    Args:
        image_path      : path to input image
        caption         : image caption string
        checkpoint_path : path to model checkpoint
        device          : 'cuda' or 'cpu' (auto-detected if None)
        use_mc          : use MC Dropout for uncertainty (T=20)

    Returns:
        dict with prediction, confidence, modality weights, uncertainty
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # ── Load model ────────────────────────────────────────────
    model = load_model(checkpoint_path, device)

    # ── Preprocess image ──────────────────────────────────────
    pil_image  = Image.open(image_path).convert("RGB")
    img_tensor = IMAGE_TRANSFORM(pil_image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # ── Tokenize caption ──────────────────────────────────────
    tokenizer = DistilBertTokenizer.from_pretrained(CONFIG["distilbert_model"])
    encoded   = tokenizer(
        caption,
        padding='max_length',
        truncation=True,
        max_length=CONFIG["max_text_length"],
        return_tensors='pt',
    )
    input_ids      = encoded['input_ids'].to(device)       # [1, seq_len]
    attention_mask = encoded['attention_mask'].to(device)  # [1, seq_len]

    # ── Compute consistency score ─────────────────────────────
    print("Computing cross-modal consistency score...")
    gnn    = CrossModalAttentionGNN(clip_dim=512, hidden_dim=256).to(device)
    scorer = CrossModalConsistencyScorer(
        gnn_model=gnn,
        device=str(device),
    )
    s_cons_val = scorer.compute_score(pil_image, caption)
    s_cons     = torch.tensor([[s_cons_val]], dtype=torch.float32).to(device)
    print(f"Consistency score: {s_cons_val:.4f}")

    # ── Inference ─────────────────────────────────────────────
    if use_mc:
        p_hat, sigma_sq, mean_w_text, mean_w_img = mc_inference(
            model, input_ids, attention_mask, img_tensor, s_cons,
            T=CONFIG["mc_dropout_samples"], device=str(device),
        )
        pred        = p_hat.argmax(dim=1).item()
        confidence  = p_hat.max(dim=1).values.item()
        uncertainty = sigma_sq.item()
        w_text      = mean_w_text.item()
        w_img       = mean_w_img.item()
    else:
        with torch.no_grad():
            logits, w_text_t, w_img_t, g_text, g_img, _, u_text, u_img = model(
                input_ids, attention_mask, img_tensor, s_cons
            )
        probs       = torch.softmax(logits, dim=-1)
        pred        = probs.argmax(dim=1).item()
        confidence  = probs.max(dim=1).values.item()
        uncertainty = None
        w_text      = w_text_t.item()
        w_img       = w_img_t.item()

    # ── Results ───────────────────────────────────────────────
    label_name = "REAL" if pred == 0 else "FAKE"

    result = {
        'prediction':        label_name,
        'label':             pred,
        'confidence':        round(confidence, 4),
        'consistency_score': round(s_cons_val, 4),
        'text_weight':       round(w_text, 4),
        'image_weight':      round(w_img, 4),
        'uncertainty':       round(uncertainty, 6) if uncertainty else None,
    }

    print("\n" + "="*50)
    print(f"  PREDICTION    : {result['prediction']}")
    print(f"  Confidence    : {result['confidence']:.4f}")
    print(f"  s_cons        : {result['consistency_score']:.4f}")
    print(f"  w_text        : {result['text_weight']:.4f}")
    print(f"  w_img         : {result['image_weight']:.4f}")
    if uncertainty:
        print(f"  Uncertainty σ²: {result['uncertainty']:.6f}")
    print("="*50)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multimodal Rumour Detection — Single Sample Inference"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--caption", type=str, required=True,
        help="Image caption text",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="./checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="'cuda' or 'cpu' (auto-detected if not specified)",
    )
    parser.add_argument(
        "--no_mc", action="store_true",
        help="Disable MC Dropout uncertainty estimation",
    )
    args = parser.parse_args()

    predict(
        image_path=args.image,
        caption=args.caption,
        checkpoint_path=args.checkpoint,
        device=args.device,
        use_mc=not args.no_mc,
    )
