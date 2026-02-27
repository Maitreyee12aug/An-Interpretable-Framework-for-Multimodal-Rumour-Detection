# =============================================================
# config.py — All hyperparameters from the paper
# Paper: "Gated Consistency and Uncertainty-Aware Fusion:
#         An Interpretable Framework for Multimodal Rumour Detection"
# =============================================================

CONFIG = {
    # ── Model Architecture (Section 4) ───────────────────────
    "proj_dim": 512,               # Projection dimension for both branches
    "gnn_hidden_dim": 256,         # GNN hidden layer dimension
    "gnn_num_heads": 4,            # Multi-head attention heads in GNN
    "gate_hidden_dim": 512,        # Dynamic gate MLP hidden dim
    "fusion_hidden_dim": 256,      # Adaptive fusion MLP hidden dim
    "classifier_hidden": 128,      # Final classifier hidden dim

    # ── Text Branch (Section 4.1) ─────────────────────────────
    "distilbert_model": "distilbert-base-uncased",
    "max_text_length": 128,        # Max token length
    "unfreeze_last_n_layers": 4,   # Last N DistilBERT layers unfrozen

    # ── Image Branch (Section 4.2) ────────────────────────────
    "img_size": 224,               # ResNet-50 input size
    "img_feat_dim": 2048,          # ResNet-50 GAP output dim

    # ── Consistency GNN (Section 4.3) ────────────────────────
    "clip_model": "ViT-B/32",      # CLIP backbone
    "clip_dim": 512,               # CLIP embedding dimension
    "yolo_model": "yolov8n.pt",    # YOLOv8 nano for speed/accuracy balance
    "max_regions": 5,              # Max YOLOv8 regions per image
    "max_segments": 10,            # Max spaCy segments per caption

    # ── Training (Section 5.1) ────────────────────────────────
    "optimizer": "AdamW",
    "learning_rate": 2e-4,
    "batch_size": 32,
    "num_epochs": 50,
    "early_stopping_patience": 5,
    "dropout_rate": 0.3,

    # ── MC Dropout (Section 4.6) ──────────────────────────────
    "mc_dropout_samples": 20,      # T=20 stochastic forward passes

    # ── Paths ─────────────────────────────────────────────────
    "output_dir": "./preprocessed_miragenews_hf/",
    "scores_dir": "./consistency_scores/",
    "plots_dir": "./plots/",
    "checkpoint_dir": "./checkpoints/",

    # ── Dataset ───────────────────────────────────────────────
    "dataset_name": "anson-huang/mirage-news",
    "image_col": "image",
    "caption_col": "text",
    "label_col": "label",
}
