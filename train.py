# =============================================================
# train.py — Section 5.1: Training Dynamics
# Paper: "Gated Consistency and Uncertainty-Aware Fusion:
#         An Interpretable Framework for Multimodal Rumour Detection"
#
# Training config (from paper):
#   Optimizer : AdamW, lr=2e-4
#   Epochs    : max 50, early stopping patience=5
#   Dropout   : 0.3 throughout
#   Loss      : CrossEntropyLoss
# =============================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from config import CONFIG
from models.full_model import GatedConsistencyRumorDetector
from data.preprocess import preprocess_and_save, load_preprocessed
from data.dataset import build_dataloaders, load_or_compute_scores


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Single training epoch."""
    model.train()
    total_loss    = 0.0
    all_preds     = []
    all_labels    = []
    num_samples   = 0

    pbar = tqdm(loader, desc="  Train")
    for batch in pbar:
        input_ids, attention_mask, images, labels, s_cons = [
            b.to(device) for b in batch
        ]

        optimizer.zero_grad()

        logits, w_text, w_img, g_text, g_img, _, u_text, u_img = model(
            input_ids, attention_mask, images, s_cons
        )

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss  += loss.item() * labels.size(0)
        num_samples += labels.size(0)

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_samples
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, acc, f1


def validate(model, loader, criterion, device):
    """Validation epoch."""
    model.eval()
    total_loss   = 0.0
    all_preds    = []
    all_labels   = []
    all_w_text   = []
    all_w_img    = []
    num_samples  = 0

    pbar = tqdm(loader, desc="  Val")
    with torch.no_grad():
        for batch in pbar:
            input_ids, attention_mask, images, labels, s_cons = [
                b.to(device) for b in batch
            ]

            logits, w_text, w_img, g_text, g_img, _, u_text, u_img = model(
                input_ids, attention_mask, images, s_cons
            )

            loss         = criterion(logits, labels)
            total_loss  += loss.item() * labels.size(0)
            num_samples += labels.size(0)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_w_text.extend(w_text.cpu().numpy())
            all_w_img.extend(w_img.cpu().numpy())

    avg_loss = total_loss / num_samples
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, acc, f1, all_preds, all_labels, all_w_text, all_w_img


def save_checkpoint(model, optimizer, epoch, val_f1, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':      epoch,
        'model':      model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'val_f1':     val_f1,
    }, path)
    print(f"  Checkpoint saved: {path}")


def plot_training_curves(
    train_losses, train_accs, val_accs, val_f1s, save_path
):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_accs, label='Train Accuracy', marker='o')
    axes[0].plot(val_accs,   label='Val Accuracy',   marker='o')
    axes[0].set_title('Accuracy per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_ylim(0, 1)

    axes[1].plot(train_losses, label='Train Loss', color='red', marker='o')
    axes[1].set_title('Training Loss per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Training curves saved to: {save_path}")


def main():
    # ── Device ───────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────
    print("\nLoading preprocessed data...")
    preprocess_and_save()
    loaded_data = load_preprocessed(['train', 'validation'])

    if not loaded_data:
        print("No data loaded. Run preprocess.py first.")
        return

    # Load consistency scores
    consistency_scores = {}
    for split in ['train', 'validation']:
        if split in loaded_data:
            consistency_scores[split] = load_or_compute_scores(
                split_name=split,
                raw_dataset=None,
                preprocessed_labels=loaded_data[split]['labels'],
                scorer=None,  # Use pre-computed scores if available
            )

    train_loader, val_loader = build_dataloaders(
        loaded_data, consistency_scores
    )

    # ── Model ─────────────────────────────────────────────────
    print("\nInitializing model...")
    model = GatedConsistencyRumorDetector(
        proj_dim=CONFIG["proj_dim"],
        gnn_hidden_dim=CONFIG["gnn_hidden_dim"],
        gnn_num_heads=CONFIG["gnn_num_heads"],
        gate_hidden_dim=CONFIG["gate_hidden_dim"],
        fusion_hidden_dim=CONFIG["fusion_hidden_dim"],
        classifier_hidden=CONFIG["classifier_hidden"],
        dropout_rate=CONFIG["dropout_rate"],
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # ── Optimizer & Loss ──────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training Loop ─────────────────────────────────────────
    best_val_f1       = 0.0
    patience_counter  = 0
    checkpoint_dir    = CONFIG["checkpoint_dir"]
    plots_dir         = CONFIG["plots_dir"]
    os.makedirs(plots_dir, exist_ok=True)

    train_losses, train_accs = [], []
    val_accs, val_f1s        = [], []

    print(f"\nStarting training for {CONFIG['num_epochs']} epochs...")
    print(f"Early stopping patience: {CONFIG['early_stopping_patience']}")
    print("-" * 60)

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}")

        # Train
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels, w_text, w_img = validate(
            model, val_loader, criterion, device
        )

        # Log
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        print(
            f"  Train — Loss: {train_loss:.4f}, "
            f"Acc: {train_acc:.4f}, F1: {train_f1:.4f}"
        )
        print(
            f"  Val   — Loss: {val_loss:.4f}, "
            f"Acc: {val_acc:.4f}, F1: {val_f1:.4f}"
        )

        # Save best checkpoint
        if val_f1 > best_val_f1:
            best_val_f1     = val_f1
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_f1,
                os.path.join(checkpoint_dir, 'best_model.pt'),
            )
            print(f"  New best Val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(
                f"  No improvement. "
                f"Patience: {patience_counter}/"
                f"{CONFIG['early_stopping_patience']}"
            )

        # Early stopping
        if patience_counter >= CONFIG["early_stopping_patience"]:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    print(f"\nTraining complete. Best Val F1: {best_val_f1:.4f}")

    # Plot training curves
    plot_training_curves(
        train_losses, train_accs, val_accs, val_f1s,
        os.path.join(plots_dir, 'training_curves.png'),
    )


if __name__ == "__main__":
    main()
