# =============================================================
# visualize.py — Section 5.3: Modality Attribution Analysis
# Paper: "Gated Consistency and Uncertainty-Aware Fusion:
#         An Interpretable Framework for Multimodal Rumour Detection"
#
# Reproduces Figures 7-10 from the paper:
#   Fig 7: Modality weight distributions by class
#   Fig 8: Consistency score vs gate values
#   Fig 9: Consistency score vs modality weights scatter
#   Fig 10: Qualitative sample visualizations
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image, ImageDraw
from datasets import load_dataset
from sklearn.manifold import TSNE
import torch

from config import CONFIG

sns.set_style("whitegrid")
os.makedirs(CONFIG["plots_dir"], exist_ok=True)


# =============================================================
# Figure 7: Modality Weight Distribution by Class
# =============================================================

def plot_modality_weight_distributions(
    w_text: np.ndarray,
    w_img: np.ndarray,
    labels: np.ndarray,
    save_path: str = None,
):
    """
    Plots text and image weight distributions
    separated by true class (Real vs Fake).

    Reproduces Figure 7 from the paper:
    - Real samples → high image weights
    - Fake samples → high text weights
    """
    w_text  = np.array(w_text).flatten()
    w_img   = np.array(w_img).flatten()
    labels  = np.array(labels).astype(int)

    df = pd.DataFrame({
        'Text Weight':  w_text,
        'Image Weight': w_img,
        'True Label':   labels,
        'Class':        ['Real (0)' if l == 0 else 'Fake (1)' for l in labels],
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Text Weight
    sns.histplot(
        data=df, x='Text Weight', hue='Class',
        kde=True, bins=50, palette='viridis', ax=axes[0],
    )
    axes[0].set_title('Distribution of Text Modality Weight by Class')
    axes[0].set_xlabel('Text Weight (w_text)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xlim(0, 1)
    axes[0].grid(True)

    # Image Weight
    sns.histplot(
        data=df, x='Image Weight', hue='Class',
        kde=True, bins=50, palette='viridis', ax=axes[1],
    )
    axes[1].set_title('Distribution of Image Modality Weight by Class')
    axes[1].set_xlabel('Image Weight (w_img)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlim(0, 1)
    axes[1].grid(True)

    plt.suptitle(
        'Modality Weight Distribution: Real vs Fake Samples',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure 7 saved to: {save_path}")
    plt.show()

    # Print summary statistics
    print("\nAverage modality weights by class:")
    print(df.groupby('Class')[['Text Weight', 'Image Weight']].mean().round(4))


# =============================================================
# Figure 8: Consistency Score vs Gate Values
# =============================================================

def plot_consistency_vs_gates(
    s_cons: np.ndarray,
    g_text_mean: np.ndarray,
    g_img_mean: np.ndarray,
    labels: np.ndarray,
    save_path: str = None,
):
    """
    Scatter plot: consistency score vs mean gate values.

    Reproduces Figure 8 from the paper:
    - High s_cons → both gates high
    - Low s_cons  → one gate suppressed
    """
    s_cons      = np.array(s_cons).flatten()
    g_text_mean = np.array(g_text_mean).flatten()
    g_img_mean  = np.array(g_img_mean).flatten()
    labels      = np.array(labels).astype(int)
    classes     = ['Real' if l == 0 else 'Fake' for l in labels]

    df = pd.DataFrame({
        'Consistency Score': s_cons,
        'Mean g_text':       g_text_mean,
        'Mean g_img':        g_img_mean,
        'Class':             classes,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # s_cons vs g_text
    for cls, color in [('Real', 'steelblue'), ('Fake', 'coral')]:
        sub = df[df['Class'] == cls]
        axes[0].scatter(
            sub['Consistency Score'], sub['Mean g_text'],
            alpha=0.4, s=10, color=color, label=cls,
        )
    axes[0].set_xlabel('Consistency Score (s_cons)')
    axes[0].set_ylabel('Mean Text Gate (g_text)')
    axes[0].set_title('Consistency Score vs Text Gate Value')
    axes[0].legend()
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    # s_cons vs g_img
    for cls, color in [('Real', 'steelblue'), ('Fake', 'coral')]:
        sub = df[df['Class'] == cls]
        axes[1].scatter(
            sub['Consistency Score'], sub['Mean g_img'],
            alpha=0.4, s=10, color=color, label=cls,
        )
    axes[1].set_xlabel('Consistency Score (s_cons)')
    axes[1].set_ylabel('Mean Image Gate (g_img)')
    axes[1].set_title('Consistency Score vs Image Gate Value')
    axes[1].legend()
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    plt.suptitle(
        'Cross-Modal Consistency Score vs Dynamic Gate Values',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure 8 saved to: {save_path}")
    plt.show()


# =============================================================
# Figure 9: Consistency Score vs Modality Weights Scatter
# =============================================================

def plot_consistency_vs_weights(
    s_cons: np.ndarray,
    w_text: np.ndarray,
    w_img: np.ndarray,
    labels: np.ndarray,
    save_path: str = None,
):
    """
    Scatter plot: consistency score vs adaptive modality weights.
    Reproduces Figure 9 from the paper.
    """
    s_cons  = np.array(s_cons).flatten()
    w_text  = np.array(w_text).flatten()
    w_img   = np.array(w_img).flatten()
    labels  = np.array(labels).astype(int)
    classes = ['Real' if l == 0 else 'Fake' for l in labels]

    df = pd.DataFrame({
        'Consistency Score': s_cons,
        'Text Weight':       w_text,
        'Image Weight':      w_img,
        'Class':             classes,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for cls, color in [('Real', 'steelblue'), ('Fake', 'coral')]:
        sub = df[df['Class'] == cls]
        axes[0].scatter(
            sub['Consistency Score'], sub['Text Weight'],
            alpha=0.4, s=10, color=color, label=cls,
        )
    axes[0].set_xlabel('Consistency Score (s_cons)')
    axes[0].set_ylabel('Text Modality Weight (w_text)')
    axes[0].set_title('Consistency Score vs Text Modality Weight')
    axes[0].legend()

    for cls, color in [('Real', 'steelblue'), ('Fake', 'coral')]:
        sub = df[df['Class'] == cls]
        axes[1].scatter(
            sub['Consistency Score'], sub['Image Weight'],
            alpha=0.4, s=10, color=color, label=cls,
        )
    axes[1].set_xlabel('Consistency Score (s_cons)')
    axes[1].set_ylabel('Image Modality Weight (w_img)')
    axes[1].set_title('Consistency Score vs Image Modality Weight')
    axes[1].legend()

    plt.suptitle(
        'Cross-Modal Consistency Score vs Adaptive Modality Weights',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure 9 saved to: {save_path}")
    plt.show()


# =============================================================
# Figure 10: Qualitative Sample Visualization
# =============================================================

def visualize_sample(
    pil_image: Image.Image,
    caption: str,
    true_label: int,
    pred_label: int,
    s_cons: float,
    w_text: float,
    w_img: float,
    g_text_mean: float,
    g_img_mean: float,
    uncertainty: float = None,
    yolo_boxes: list = None,
    save_path: str = None,
):
    """
    Visualizes a single sample with YOLO detections,
    caption, and all interpretability values.
    Reproduces Figure 10 style from the paper.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: image with YOLO boxes
    ax_img = axes[0]
    img_draw = pil_image.copy()
    draw = ImageDraw.Draw(img_draw)
    if yolo_boxes:
        for box in yolo_boxes:
            draw.rectangle(box, outline="red", width=3)
    ax_img.imshow(img_draw)
    ax_img.axis("off")
    ax_img.set_title(
        f"True: {'Real' if true_label==0 else 'Fake'}  |  "
        f"Pred: {'Real' if pred_label==0 else 'Fake'}",
        fontsize=13,
        color='green' if true_label == pred_label else 'red',
    )

    # Right: interpretability values
    ax_info = axes[1]
    ax_info.axis("off")

    info_text = (
        f"Caption:\n"
        f"\"{caption[:120]}{'...' if len(caption)>120 else ''}\"\n\n"
        f"Consistency Score (s_cons) : {s_cons:.3f}\n\n"
        f"Dynamic Gate Values:\n"
        f"  g_text (mean) : {g_text_mean:.3f}\n"
        f"  g_img  (mean) : {g_img_mean:.3f}\n\n"
        f"Adaptive Modality Weights:\n"
        f"  w_text : {w_text:.3f}\n"
        f"  w_img  : {w_img:.3f}\n"
    )
    if uncertainty is not None:
        info_text += f"\nPredictive Uncertainty (σ²): {uncertainty:.6f}\n"

    # Interpretation note
    if true_label == pred_label:
        if w_img > w_text:
            note = "✓ Correct: Model relied on visual evidence"
        else:
            note = "✓ Correct: Model relied on textual evidence"
    else:
        note = "✗ Misclassified: Low consistency → misleading modality"

    info_text += f"\n{note}"

    ax_info.text(
        0.05, 0.95, info_text,
        transform=ax_info.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round', facecolor='lightyellow',
            edgecolor='gray', alpha=0.8,
        ),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sample visualization saved to: {save_path}")
    plt.show()


# =============================================================
# Consistency Score Distribution
# =============================================================

def plot_consistency_distribution(
    consistency_scores: dict,
    save_path: str = None,
):
    """Plots consistency score distributions per split."""
    num_splits = len(consistency_scores)
    fig, axes  = plt.subplots(1, num_splits, figsize=(7 * num_splits, 5))

    if num_splits == 1:
        axes = [axes]

    colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum']

    for ax, (split, scores), color in zip(
        axes, consistency_scores.items(), colors
    ):
        sns.histplot(scores, bins=50, kde=True, color=color, ax=ax)
        ax.set_title(f'Consistency Score Distribution\n({split})')
        ax.set_xlabel('s_cons')
        ax.set_ylabel('Frequency')
        ax.set_xlim(0, 1)
        mean_val = np.mean(scores)
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Consistency distribution saved to: {save_path}")
    plt.show()


# =============================================================
# Dataset Statistics Plots
# =============================================================

def plot_dataset_statistics(processed_data_paths: dict, save_path: str = None):
    """Plots class balance across dataset splits."""
    import numpy as np

    splits      = list(processed_data_paths.keys())
    real_counts = []
    fake_counts = []

    for split in splits:
        labels = np.load(processed_data_paths[split])['labels']
        real_counts.append((labels == 0).sum())
        fake_counts.append((labels == 1).sum())

    x     = np.arange(len(splits))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, real_counts, width, label='Real', color='steelblue')
    bars2 = ax.bar(x + width/2, fake_counts, width, label='Fake', color='coral')

    ax.set_xlabel('Split')
    ax.set_ylabel('Number of Samples')
    ax.set_title('MiRAGeNews Class Balance Across Splits')
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Dataset statistics saved to: {save_path}")
    plt.show()


# =============================================================
# CLIP Similarity Heatmap (GNN Input Visualization)
# =============================================================

def plot_clip_similarity_heatmap(
    pil_image: Image.Image,
    caption: str,
    clip_model,
    clip_processor,
    yolo_model,
    save_path: str = None,
):
    """
    Visualizes the CLIP similarity matrix between
    image regions (YOLOv8) and text segments (sentence split).
    Shows the raw input to the Cross-Modal Attention GNN.
    """
    import torch
    import torch.nn.functional as F
    from matplotlib.gridspec import GridSpec

    # Detect regions
    results  = yolo_model(pil_image, verbose=False)
    boxes    = results[0].boxes.xyxy.cpu().numpy().astype(int)[:5]

    if len(boxes) == 0:
        print("No YOLO regions detected.")
        return

    # Draw boxes
    img_boxed = pil_image.copy()
    draw      = ImageDraw.Draw(img_boxed)
    for box in boxes:
        draw.rectangle(box.tolist(), outline="red", width=2)
    region_crops = [pil_image.crop(box) for box in boxes]

    # Text segments
    segments = [s.strip() for s in caption.split('.') if s.strip()][:5]
    if not segments:
        segments = [caption]

    # CLIP similarity
    sim_matrix = np.zeros((len(segments), len(region_crops)))
    for i, seg in enumerate(segments):
        inputs = clip_processor(
            text=[seg], images=region_crops,
            return_tensors="pt", padding=True, truncation=True,
        )
        with torch.no_grad():
            outputs = clip_model(**inputs)
            sim = F.cosine_similarity(
                outputs.image_embeds,
                outputs.text_embeds.unsqueeze(1),
                dim=-1,
            )
        sim_matrix[i, :] = sim.squeeze().numpy()

    # Plot
    fig = plt.figure(figsize=(16, 6))
    gs  = GridSpec(1, 3, width_ratios=[1.2, 1.5, 1])

    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(img_boxed)
    ax_img.axis("off")
    ax_img.set_title("YOLOv8-Detected Regions")

    ax_heat = fig.add_subplot(gs[1])
    sns.heatmap(
        sim_matrix, annot=True, fmt=".2f", cmap="viridis",
        xticklabels=[f"R{i+1}" for i in range(len(region_crops))],
        yticklabels=[f"T{i+1}" for i in range(len(segments))],
        ax=ax_heat,
    )
    ax_heat.set_title("CLIP Similarity Matrix (GNN Input)")
    ax_heat.set_xlabel("Image Regions")
    ax_heat.set_ylabel("Text Segments")

    ax_text = fig.add_subplot(gs[2])
    ax_text.axis("off")
    ax_text.set_title("Text Segments", fontsize=12)
    full_text = "\n\n".join(
        [f"T{i+1}: {seg}" for i, seg in enumerate(segments)]
    )
    ax_text.text(0, 1, full_text, fontsize=10, va='top', wrap=True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"CLIP heatmap saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    import numpy as np

    plots_dir = CONFIG["plots_dir"]
    os.makedirs(plots_dir, exist_ok=True)

    # Demo with dummy data
    N = 500
    np.random.seed(42)

    dummy_labels = np.random.randint(0, 2, N)
    dummy_s_cons = np.random.beta(2, 5, N)
    dummy_w_text = np.random.dirichlet([1, 1], N)[:, 0]
    dummy_w_img  = 1 - dummy_w_text
    dummy_g_text = dummy_s_cons * 0.8 + np.random.normal(0, 0.05, N)
    dummy_g_img  = dummy_s_cons * 0.7 + np.random.normal(0, 0.05, N)

    print("Generating Figure 7: Modality Weight Distributions...")
    plot_modality_weight_distributions(
        dummy_w_text, dummy_w_img, dummy_labels,
        save_path=os.path.join(plots_dir, 'fig7_modality_weights.png'),
    )

    print("\nGenerating Figure 8: Consistency vs Gate Values...")
    plot_consistency_vs_gates(
        dummy_s_cons, dummy_g_text, dummy_g_img, dummy_labels,
        save_path=os.path.join(plots_dir, 'fig8_consistency_vs_gates.png'),
    )

    print("\nGenerating Figure 9: Consistency vs Modality Weights...")
    plot_consistency_vs_weights(
        dummy_s_cons, dummy_w_text, dummy_w_img, dummy_labels,
        save_path=os.path.join(plots_dir, 'fig9_consistency_vs_weights.png'),
    )

    print("\nAll visualization scripts ready.")
