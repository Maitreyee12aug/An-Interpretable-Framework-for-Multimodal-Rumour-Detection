# =============================================================
# data/dataset.py — Section 3: PyTorch Dataset & DataLoader
# Paper: "Gated Consistency and Uncertainty-Aware Fusion:
#         An Interpretable Framework for Multimodal Rumour Detection"
# =============================================================

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import DistilBertTokenizer
from config import CONFIG


# =============================================================
# Image Transform for ResNet-50
# =============================================================

IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet mean
        std=[0.229, 0.224, 0.225],    # ImageNet std
    ),
])


# =============================================================
# MiRAGeNews Dataset with Consistency Scores
# =============================================================

class MirageNewsDataset(Dataset):
    """
    PyTorch Dataset for MiRAGeNews multimodal rumour detection.

    Each sample returns:
        input_ids        : DistilBERT tokenized caption [seq_len]
        attention_mask   : DistilBERT attention mask    [seq_len]
        image            : ResNet-50 normalized image   [3, 224, 224]
        label            : ground truth label           scalar
        consistency_score: pre-computed s_cons          [1]
    """

    def __init__(
        self,
        texts: np.ndarray,
        images: np.ndarray,
        labels: np.ndarray,
        consistency_scores: list,
        max_length: int = CONFIG["max_text_length"],
    ):
        assert len(texts) == len(images) == len(labels) == len(consistency_scores), \
            "All data components must have the same length."

        self.texts               = texts
        self.images              = images
        self.labels              = labels
        self.consistency_scores  = consistency_scores
        self.max_length          = max_length

        # DistilBERT tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            CONFIG["distilbert_model"]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # ── Text Tokenization ─────────────────────────────────
        encoded = self.tokenizer(
            str(self.texts[idx]),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        input_ids      = encoded['input_ids'].squeeze(0)       # [seq_len]
        attention_mask = encoded['attention_mask'].squeeze(0)  # [seq_len]

        # ── Image Transform ───────────────────────────────────
        img_np = self.images[idx]  # (H, W, 3) float32
        if isinstance(img_np, np.ndarray) and img_np.ndim == 3:
            # Convert to uint8 for ToPILImage
            img_uint8  = (img_np * 255).clip(0, 255).astype(np.uint8)
            img_tensor = IMAGE_TRANSFORM(img_uint8)  # [3, 224, 224]
        else:
            img_tensor = torch.zeros(
                3, CONFIG["img_size"], CONFIG["img_size"],
                dtype=torch.float32,
            )

        # ── Label ─────────────────────────────────────────────
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        # ── Consistency Score ─────────────────────────────────
        score = self.consistency_scores[idx]
        if not isinstance(score, (float, int, np.floating)):
            score = 0.0
        s_cons = torch.tensor([float(score)], dtype=torch.float32)  # [1]

        return input_ids, attention_mask, img_tensor, label, s_cons


# =============================================================
# Consistency Score Loading / Saving
# =============================================================

def load_or_compute_scores(
    split_name: str,
    raw_dataset,
    preprocessed_labels: np.ndarray,
    scorer=None,
) -> list:
    """
    Loads pre-computed consistency scores if available,
    otherwise computes them using the CrossModalConsistencyScorer.

    Args:
        split_name           : 'train', 'validation', etc.
        raw_dataset          : HuggingFace dataset split
        preprocessed_labels  : labels array (for length check)
        scorer               : CrossModalConsistencyScorer instance

    Returns:
        scores : list of float consistency scores
    """
    scores_dir = CONFIG["scores_dir"]
    os.makedirs(scores_dir, exist_ok=True)
    score_file = os.path.join(scores_dir, f'{split_name}_consistency_scores.pkl')

    # Try to load existing scores
    if os.path.exists(score_file):
        print(f"Loading existing consistency scores for '{split_name}'...")
        try:
            with open(score_file, "rb") as f:
                scores = pickle.load(f)
            if len(scores) == len(preprocessed_labels):
                print(f"  Loaded {len(scores)} scores.")
                return scores
            else:
                print(f"  Length mismatch. Recomputing...")
        except Exception as e:
            print(f"  Error loading scores: {e}. Recomputing...")

    # Compute scores
    if scorer is None:
        print(f"No scorer provided. Using default score 0.5 for '{split_name}'.")
        return [0.5] * len(preprocessed_labels)

    print(f"Computing consistency scores for '{split_name}'...")
    scores = []
    from tqdm import tqdm

    for i in tqdm(range(len(raw_dataset)), desc=f"Scoring {split_name}"):
        sample = raw_dataset[i]
        score  = scorer.compute_score(
            sample[CONFIG["image_col"]],
            sample[CONFIG["caption_col"]],
        )
        scores.append(score if score is not None else 0.0)

    # Save scores
    try:
        with open(score_file, "wb") as f:
            pickle.dump(scores, f)
        print(f"  Saved {len(scores)} scores to {score_file}")
    except Exception as e:
        print(f"  Error saving scores: {e}")

    return scores


# =============================================================
# DataLoader Factory
# =============================================================

def build_dataloaders(loaded_data: dict, consistency_scores: dict):
    """
    Builds PyTorch DataLoaders for train and validation splits.

    Args:
        loaded_data         : dict from preprocess.load_preprocessed()
        consistency_scores  : dict of {split: list of scores}

    Returns:
        train_loader, val_loader
    """
    batch_size  = CONFIG["batch_size"]
    num_workers = 0 if os.name == 'nt' else 2

    loaders = {}
    for split in ['train', 'validation']:
        if split not in loaded_data:
            print(f"Split '{split}' not found in loaded_data.")
            continue
        d      = loaded_data[split]
        scores = consistency_scores.get(split, [0.5] * len(d['labels']))

        dataset = MirageNewsDataset(
            texts=d['texts'],
            images=d['images'],
            labels=d['labels'],
            consistency_scores=scores,
        )

        shuffle = (split == 'train')
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        print(
            f"{split} loader: {len(dataset)} samples, "
            f"{len(loaders[split])} batches"
        )

    return loaders.get('train'), loaders.get('validation')


if __name__ == "__main__":
    from data.preprocess import load_preprocessed

    loaded_data = load_preprocessed(['train', 'validation'])

    # Use dummy scores for testing
    dummy_scores = {
        split: [0.5] * len(d['labels'])
        for split, d in loaded_data.items()
    }

    train_loader, val_loader = build_dataloaders(loaded_data, dummy_scores)

    # Test one batch
    batch = next(iter(train_loader))
    input_ids, attention_mask, images, labels, s_cons = batch
    print(f"input_ids      : {input_ids.shape}")        # [32, 128]
    print(f"attention_mask : {attention_mask.shape}")   # [32, 128]
    print(f"images         : {images.shape}")           # [32, 3, 224, 224]
    print(f"labels         : {labels.shape}")           # [32]
    print(f"s_cons         : {s_cons.shape}")           # [32, 1]
