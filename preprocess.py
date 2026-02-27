# =============================================================
# data/preprocess.py — Section 3: Dataset Preprocessing
# Paper: "Gated Consistency and Uncertainty-Aware Fusion:
#         An Interpretable Framework for Multimodal Rumour Detection"
# =============================================================

import os
import re
import warnings
import numpy as np
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from config import CONFIG

STOPWORDS = set(stopwords.words('english'))

IMG_WIDTH    = CONFIG["img_size"]
IMG_HEIGHT   = CONFIG["img_size"]
IMG_CHANNELS = 3
OUTPUT_DIR   = CONFIG["output_dir"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')


# =============================================================
# Text Cleaning
# =============================================================

def clean_text(text: str) -> str:
    """
    Removes HTML tags, special characters, normalizes whitespace,
    removes stopwords, and lowercases the caption.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join(
        [w for w in text.split() if w not in STOPWORDS]
    )
    return text


# =============================================================
# Image Preprocessing
# =============================================================

def resize_normalize_image(
    img: Image.Image,
    target_size: tuple = (IMG_WIDTH, IMG_HEIGHT),
) -> np.ndarray:
    """
    Resizes image with aspect-ratio-preserving padding to target_size.
    Normalizes pixel values to [0, 1].
    Returns float32 numpy array of shape (H, W, 3).
    """
    if img is None:
        return None
    try:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        w, h = img.size
        if w == 0 or h == 0:
            return None

        ratio    = min(target_size[0] / w, target_size[1] / h)
        new_size = (int(w * ratio), int(h * ratio))
        if new_size[0] == 0 or new_size[1] == 0:
            return None

        img      = img.resize(new_size, Image.Resampling.LANCZOS)
        new_img  = Image.new("RGB", target_size, (128, 128, 128))
        paste_x  = (target_size[0] - new_size[0]) // 2
        paste_y  = (target_size[1] - new_size[1]) // 2
        new_img.paste(img, (paste_x, paste_y))

        img_array = np.array(new_img) / 255.0
        return img_array.astype(np.float32)
    except Exception as e:
        print(f"Image processing error: {e}")
        return None


# =============================================================
# Main Preprocessing Pipeline
# =============================================================

def preprocess_and_save():
    """
    Loads the MiRAGeNews dataset from HuggingFace,
    preprocesses all splits, and saves .npz files.
    """
    print("Loading MiRAGeNews dataset...")
    try:
        raw_dataset = load_dataset(CONFIG["dataset_name"])
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

    print(raw_dataset)

    processed_data_paths = {}

    for split_name in raw_dataset.keys():
        output_path = os.path.join(
            OUTPUT_DIR, f'miragenews_processed_{split_name}.npz'
        )
        processed_data_paths[split_name] = output_path

        if os.path.exists(output_path):
            print(f"Preprocessed file exists for '{split_name}'. Skipping.")
            continue

        print(f"\nProcessing split: {split_name}")
        current_split = raw_dataset[split_name]
        required_cols = [
            CONFIG["image_col"],
            CONFIG["caption_col"],
            CONFIG["label_col"],
        ]
        if not all(c in current_split.column_names for c in required_cols):
            print(f"Warning: Split '{split_name}' missing columns. Skipping.")
            continue

        processed_texts, processed_images, processed_labels = [], [], []
        skipped = 0

        for example in tqdm(current_split, desc=f"Preprocessing {split_name}"):
            img = example[CONFIG["image_col"]]
            if img is None or not isinstance(img, Image.Image):
                skipped += 1
                continue

            cleaned_caption  = clean_text(example[CONFIG["caption_col"]])
            processed_image  = resize_normalize_image(img)
            label            = example[CONFIG["label_col"]]

            if processed_image is not None and cleaned_caption:
                processed_texts.append(cleaned_caption)
                processed_images.append(processed_image)
                processed_labels.append(label)
            else:
                skipped += 1

        if skipped > 0:
            print(f"Skipped {skipped} samples in '{split_name}'.")

        if processed_texts:
            np.savez_compressed(
                output_path,
                texts=np.array(processed_texts, dtype=object),
                images=np.array(processed_images),
                labels=np.array(processed_labels),
            )
            print(
                f"Saved {split_name} ({len(processed_texts)} samples) "
                f"to {output_path}"
            )
        else:
            print(f"No samples processed for split '{split_name}'.")

    return processed_data_paths


def load_preprocessed(splits=('train', 'validation')):
    """
    Loads preprocessed .npz files for the given splits.
    Returns a dict: {split_name: {texts, images, labels}}
    """
    loaded = {}
    for split in splits:
        path = os.path.join(
            OUTPUT_DIR, f'miragenews_processed_{split}.npz'
        )
        if not os.path.exists(path):
            print(f"File not found for split '{split}': {path}")
            continue
        data = np.load(path, allow_pickle=True)
        loaded[split] = {
            'texts':  data['texts'],
            'images': data['images'],
            'labels': data['labels'],
        }
        print(
            f"Loaded {split}: {len(data['labels'])} samples | "
            f"Image shape: {data['images'][0].shape}"
        )
    return loaded


if __name__ == "__main__":
    paths = preprocess_and_save()
    data  = load_preprocessed()
    for split, d in data.items():
        print(f"{split}: {len(d['labels'])} samples")
