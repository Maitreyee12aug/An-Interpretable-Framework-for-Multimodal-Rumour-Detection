# =============================================================
# evaluate.py — Section 5.2: Performance Analysis
# Paper: "Gated Consistency and Uncertainty-Aware Fusion:
#         An Interpretable Framework for Multimodal Rumour Detection"
#
# Evaluates model on:
#   - In-Domain validation (NYT + MJ)
#   - OOD splits: BBC+DALL-E3, CNN+DALL-E3, BBC+SDXL, CNN+SDXL
# =============================================================

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)

from config import CONFIG
from models.full_model import GatedConsistencyRumorDetector, mc_inference
from data.dataset import MirageNewsDataset
from torch.utils.data import DataLoader


def load_model(checkpoint_path: str, device):
    """
    Loads model from checkpoint.
    """
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
    print(
        f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} "
        f"with Val F1: {checkpoint.get('val_f1', '?'):.4f}"
    )
    return model


def evaluate_split(model, loader, device, split_name="Split", use_mc=False):
    """
    Evaluate model on a single data split.

    Args:
        model       : GatedConsistencyRumorDetector
        loader      : DataLoader
        device      : torch device
        split_name  : name for logging
        use_mc      : use MC Dropout inference (T=20)

    Returns:
        dict with accuracy, macro_f1, report, predictions, labels,
             modality_weights, uncertainties
    """
    model.eval()
    all_preds        = []
    all_labels       = []
    all_w_text       = []
    all_w_img        = []
    all_uncertainties = []

    pbar = tqdm(loader, desc=f"  Evaluating {split_name}")

    with torch.no_grad():
        for batch in pbar:
            input_ids, attention_mask, images, labels, s_cons = [
                b.to(device) for b in batch
            ]

            if use_mc:
                # MC Dropout inference for uncertainty estimation
                p_hat, sigma_sq, mean_w_text, mean_w_img = mc_inference(
                    model, input_ids, attention_mask, images, s_cons,
                    T=CONFIG["mc_dropout_samples"], device=str(device),
                )
                preds = p_hat.argmax(dim=1).cpu().numpy()
                all_uncertainties.extend(sigma_sq.cpu().numpy())
                all_w_text.extend(mean_w_text.cpu().numpy())
                all_w_img.extend(mean_w_img.cpu().numpy())
            else:
                logits, w_text, w_img, _, _, _, _, _ = model(
                    input_ids, attention_mask, images, s_cons
                )
                preds = logits.argmax(dim=1).cpu().numpy()
                all_w_text.extend(w_text.cpu().numpy())
                all_w_img.extend(w_img.cpu().numpy())

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    report   = classification_report(
        all_labels, all_preds,
        target_names=["Real (0)", "Fake (1)"],
        zero_division=0,
    )

    print(f"\n{'='*50}")
    print(f"  {split_name}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Macro F1  : {macro_f1*100:.2f}%")
    print(f"{'='*50}")
    print(report)

    return {
        'split':          split_name,
        'accuracy':       acc,
        'macro_f1':       macro_f1,
        'report':         report,
        'predictions':    np.array(all_preds),
        'labels':         np.array(all_labels),
        'w_text':         np.array(all_w_text),
        'w_img':          np.array(all_w_img),
        'uncertainties':  np.array(all_uncertainties) if all_uncertainties else None,
    }


def plot_confusion_matrix(labels, preds, split_name, save_path):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(labels.astype(int), preds.astype(int))
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=["Real (0)", "Fake (1)"],
        yticklabels=["Real (0)", "Fake (1)"],
    )
    plt.title(f'Confusion Matrix — {split_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to: {save_path}")


def plot_ood_comparison(results: list, save_path: str):
    """
    Bar chart comparing Accuracy and Macro F1
    across all evaluated splits.
    """
    splits   = [r['split'] for r in results]
    accs     = [r['accuracy'] * 100 for r in results]
    f1s      = [r['macro_f1'] * 100 for r in results]

    x     = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy (%)', color='steelblue')
    bars2 = ax.bar(x + width/2, f1s,  width, label='Macro F1 (%)', color='coral')

    ax.set_xlabel('Dataset Split')
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Performance Across In-Domain and OOD Splits')
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=20, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    # Value labels on bars
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{bar.get_height():.1f}',
            ha='center', va='bottom', fontsize=9,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{bar.get_height():.1f}',
            ha='center', va='bottom', fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"OOD comparison chart saved to: {save_path}")


def main():
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plots_dir  = CONFIG["plots_dir"]
    os.makedirs(plots_dir, exist_ok=True)

    checkpoint = os.path.join(CONFIG["checkpoint_dir"], 'best_model.pt')
    if not os.path.exists(checkpoint):
        print(f"Checkpoint not found at: {checkpoint}")
        print("Please run train.py first.")
        return

    model = load_model(checkpoint, device)

    from data.preprocess import load_preprocessed
    from data.dataset import build_dataloaders, load_or_compute_scores

    # Load all available splits
    all_splits   = ['train', 'validation']
    loaded_data  = load_preprocessed(all_splits)
    scores       = {
        split: load_or_compute_scores(
            split, None, loaded_data[split]['labels']
        )
        for split in all_splits if split in loaded_data
    }

    train_loader, val_loader = build_dataloaders(loaded_data, scores)

    # Evaluate
    all_results = []

    # In-Domain Validation
    val_result = evaluate_split(
        model, val_loader, device,
        split_name="In-Domain Validation",
        use_mc=True,
    )
    all_results.append(val_result)

    # Plot confusion matrix for validation
    plot_confusion_matrix(
        val_result['labels'],
        val_result['predictions'],
        "In-Domain Validation",
        os.path.join(plots_dir, 'confusion_matrix_val.png'),
    )

    # Summary table
    summary = pd.DataFrame([
        {
            'Split':    r['split'],
            'Accuracy': f"{r['accuracy']*100:.2f}%",
            'Macro F1': f"{r['macro_f1']*100:.2f}%",
        }
        for r in all_results
    ])
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(summary.to_string(index=False))

    # OOD comparison plot
    if len(all_results) > 1:
        plot_ood_comparison(
            all_results,
            os.path.join(plots_dir, 'ood_comparison.png'),
        )

    return all_results


if __name__ == "__main__":
    main()
