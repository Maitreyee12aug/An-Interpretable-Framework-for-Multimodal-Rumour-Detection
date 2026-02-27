# =============================================================
# consistency_gnn.py
# Implements Section 4.3: Cross-Modal Consistency Scoring
# Paper: "Gated Consistency and Uncertainty-Aware Fusion:
#         An Interpretable Framework for Multimodal Rumour Detection"
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from PIL import Image

try:
    import clip
except ImportError:
    raise ImportError(
        "CLIP not found. Install with: "
        "pip install git+https://github.com/openai/CLIP.git"
    )

try:
    import spacy
except ImportError:
    raise ImportError("spaCy not found. Install with: pip install spacy")

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "YOLOv8 not found. Install with: pip install ultralytics"
    )


# =============================================================
# Cross-Modal Attention GNN
# =============================================================

class CrossModalAttentionGNN(nn.Module):
    """
    Section 4.3 — Cross-Modal Attention Graph Neural Network

    Processes l2-normalized CLIP embeddings of:
        - Image regions  {v_j} detected by YOLOv8
        - Text segments  {t_i} parsed by spaCy

    Architecture:
        1. Self-attention within each modality
           (intra-modal relationship modeling)
        2. Cross-attention between modalities
           (bidirectional image ↔ text interaction)
        3. MLP aggregation → scalar consistency score s_cons ∈ [0,1]

    Output:
        s_cons : scalar cross-modal consistency score [B, 1]
    """

    def __init__(
        self,
        clip_dim: int = 512,    # CLIP ViT-B/32 embedding dim
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.clip_dim = clip_dim
        self.hidden_dim = hidden_dim

        # ── Intra-Modal Self-Attention ────────────────────────────
        # Captures co-occurrence between image regions
        # and coreference across text segments
        self.image_self_attn = nn.MultiheadAttention(
            embed_dim=clip_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.text_self_attn = nn.MultiheadAttention(
            embed_dim=clip_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        # ── Cross-Modal Attention ─────────────────────────────────
        # Bidirectional: image attends to text AND text attends to image
        # Enables reasoning about contextually consistent/inconsistent
        # subgraphs (object clusters + phrase clusters)
        self.image_to_text_attn = nn.MultiheadAttention(
            embed_dim=clip_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.text_to_image_attn = nn.MultiheadAttention(
            embed_dim=clip_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        # ── Layer Normalization ───────────────────────────────────
        self.img_norm1 = nn.LayerNorm(clip_dim)
        self.img_norm2 = nn.LayerNorm(clip_dim)
        self.txt_norm1 = nn.LayerNorm(clip_dim)
        self.txt_norm2 = nn.LayerNorm(clip_dim)

        # ── MLP Aggregator ────────────────────────────────────────
        # Produces scalar s_cons ∈ [0,1]
        # Input: pooled image features + pooled text features
        self.aggregator = nn.Sequential(
            nn.Linear(clip_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Constrain to [0,1]
        )

    def forward(self, image_feats, text_feats):
        """
        Args:
            image_feats : [B, num_regions, clip_dim]  l2-normalized
            text_feats  : [B, num_segments, clip_dim] l2-normalized

        Returns:
            s_cons : [B, 1] scalar consistency score
        """

        # ── Step 1: Intra-Modal Self-Attention ────────────────────
        # Image regions attend to each other
        img_self, _ = self.image_self_attn(
            image_feats, image_feats, image_feats
        )
        img_self = self.img_norm1(image_feats + img_self)  # residual

        # Text segments attend to each other
        txt_self, _ = self.text_self_attn(
            text_feats, text_feats, text_feats
        )
        txt_self = self.txt_norm1(text_feats + txt_self)  # residual

        # ── Step 2: Bidirectional Cross-Modal Attention ───────────
        # Image features attend to text features
        img_cross, _ = self.image_to_text_attn(
            img_self, txt_self, txt_self
        )
        img_cross = self.img_norm2(img_self + img_cross)  # residual

        # Text features attend to image features
        txt_cross, _ = self.text_to_image_attn(
            txt_self, img_self, img_self
        )
        txt_cross = self.txt_norm2(txt_self + txt_cross)  # residual

        # ── Step 3: Global Pooling ────────────────────────────────
        # Mean pool across regions/segments → single vector per modality
        img_pooled = img_cross.mean(dim=1)  # [B, clip_dim]
        txt_pooled = txt_cross.mean(dim=1)  # [B, clip_dim]

        # ── Step 4: MLP Aggregation → s_cons ─────────────────────
        # Concatenate and pass through MLP
        combined = torch.cat([img_pooled, txt_pooled], dim=-1)  # [B, 2*clip_dim]
        s_cons = self.aggregator(combined)  # [B, 1]

        return s_cons


# =============================================================
# Cross-Modal Consistency Scorer
# Wraps YOLOv8 + spaCy + CLIP + GNN
# =============================================================

class CrossModalConsistencyScorer:
    """
    Full pipeline for computing cross-modal consistency scores.

    Pipeline (Algorithm 1, Step 3):
        Raw Image  → YOLOv8 → object regions {r_j}
        Raw Text   → spaCy  → text segments  {s_i}
        {r_j}, {s_i} → CLIP encoders → normalized embeddings {v_j}, {t_i}
        {v_j}, {t_i} → CrossModalAttentionGNN → s_cons ∈ [0,1]
    """

    def __init__(
        self,
        gnn_model: CrossModalAttentionGNN,
        clip_model_name: str = "ViT-B/32",
        yolo_model_path: str = "yolov8n.pt",
        max_regions: int = 5,
        max_segments: int = 10,
        device: str = "cpu",
    ):
        self.device = device
        self.max_regions = max_regions
        self.max_segments = max_segments
        self.gnn = gnn_model.to(device)

        # ── Load CLIP ─────────────────────────────────────────────
        print("Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load(
            clip_model_name, device=device
        )
        self.clip_model.eval()
        self.clip_dim = 512  # ViT-B/32 embedding dim

        # ── Load YOLOv8 ───────────────────────────────────────────
        # Pragmatic choice: state-of-the-art balance between
        # inference speed and detection accuracy
        print("Loading YOLOv8 model...")
        self.yolo = YOLO(yolo_model_path)

        # ── Load spaCy ────────────────────────────────────────────
        # Segments captions into: Noun Chunks, Named Entities,
        # Verb Phrases using syntactic dependency parser
        print("Loading spaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(
                ["python", "-m", "spacy", "download", "en_core_web_sm"],
                check=True,
            )
            self.nlp = spacy.load("en_core_web_sm")

    # ── Region Detection ─────────────────────────────────────────
    def detect_regions(self, pil_image: Image.Image):
        """
        Apply YOLOv8 to detect object-level regions.
        Returns list of cropped PIL images (bounding box crops).
        """
        results = self.yolo(pil_image, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        boxes = boxes[: self.max_regions]

        crops = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for (x1, y1, x2, y2) in boxes:
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(pil_image.width, x2)
                y2 = min(pil_image.height, y2)
                if x1 >= x2 or y1 >= y2:
                    continue
                crop = pil_image.crop((x1, y1, x2, y2))
                if crop.size[0] > 0 and crop.size[1] > 0:
                    crops.append(crop)

        return crops  # List of PIL Images

    # ── Text Segmentation ─────────────────────────────────────────
    def segment_text(self, caption: str):
        """
        Use spaCy syntactic dependency parser to split caption into
        phrase-level segments: Noun Chunks, Named Entities, Verb Phrases.

        This enables direct mapping of textual entities onto
        corresponding visual regions found by YOLOv8.
        """
        doc = self.nlp(caption)
        segments = []

        # Noun chunks (e.g., "the red car")
        for chunk in doc.noun_chunks:
            if chunk.text.strip():
                segments.append(chunk.text.strip())

        # Named entities (e.g., "New York", "Tuesday")
        for ent in doc.ents:
            if ent.text.strip() not in segments:
                segments.append(ent.text.strip())

        # Verb phrases (root verb + its subtree)
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                verb_phrase = " ".join(
                    [t.text for t in token.subtree]
                ).strip()
                if verb_phrase and verb_phrase not in segments:
                    segments.append(verb_phrase)

        # Fallback: use full caption if no segments extracted
        if not segments:
            segments = [caption]

        return segments[: self.max_segments]

    # ── CLIP Encoding ─────────────────────────────────────────────
    def encode_regions(self, crops):
        """
        Encode image region crops using CLIP image encoder.
        Returns l2-normalized embeddings [num_regions, clip_dim].
        """
        if not crops:
            return None

        tensors = []
        for crop in crops:
            try:
                t = self.clip_preprocess(crop).unsqueeze(0).to(self.device)
                tensors.append(t)
            except Exception:
                continue

        if not tensors:
            return None

        img_batch = torch.cat(tensors, dim=0)  # [num_regions, 3, 224, 224]

        with torch.no_grad():
            feats = self.clip_model.encode_image(img_batch)

        # l2 normalization
        feats = F.normalize(feats, p=2, dim=-1)  # [num_regions, clip_dim]
        return feats

    def encode_segments(self, segments):
        """
        Encode text segments using CLIP text encoder.
        Returns l2-normalized embeddings [num_segments, clip_dim].
        """
        if not segments:
            return None

        try:
            tokens = clip.tokenize(segments, truncate=True).to(self.device)
        except Exception:
            return None

        with torch.no_grad():
            feats = self.clip_model.encode_text(tokens)

        # l2 normalization
        feats = F.normalize(feats, p=2, dim=-1)  # [num_segments, clip_dim]
        return feats

    # ── Full Pipeline ─────────────────────────────────────────────
    def compute_score(
        self,
        pil_image: Image.Image,
        caption: str,
    ) -> float:
        """
        Compute scalar consistency score for one image-caption pair.

        Returns:
            s_cons : float in [0, 1]
                     Higher = stronger alignment between modalities
                     Lower  = cross-modal inconsistency detected
        """

        if pil_image is None or not caption or not caption.strip():
            return 0.0

        try:
            # Step 1: Detect regions and segment text
            crops = self.detect_regions(pil_image)
            segments = self.segment_text(caption)

            if not crops or not segments:
                return 0.0

            # Step 2: Encode with CLIP
            image_feats = self.encode_regions(crops)
            text_feats = self.encode_segments(segments)

            if image_feats is None or text_feats is None:
                return 0.0

            # Step 3: Add batch dimension
            # image_feats: [num_regions, 512] → [1, num_regions, 512]
            # text_feats:  [num_segments, 512] → [1, num_segments, 512]
            image_feats = image_feats.unsqueeze(0)
            text_feats = text_feats.unsqueeze(0)

            # Step 4: Pass through Cross-Modal Attention GNN
            self.gnn.eval()
            with torch.no_grad():
                s_cons = self.gnn(image_feats, text_feats)  # [1, 1]

            score = s_cons.squeeze().item()
            score = max(0.0, min(1.0, score))  # Clamp to [0,1]
            return score

        except Exception as e:
            print(f"Error computing consistency score: {e}")
            return 0.0

    def compute_batch_scores(
        self,
        pil_images: list,
        captions: list,
    ) -> list:
        """
        Compute consistency scores for a batch of image-caption pairs.

        Args:
            pil_images : list of PIL Images
            captions   : list of caption strings

        Returns:
            scores : list of float scores in [0, 1]
        """
        scores = []
        for img, cap in zip(pil_images, captions):
            score = self.compute_score(img, cap)
            scores.append(score)
        return scores


# =============================================================
# Quick test
# =============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Instantiate GNN
    gnn = CrossModalAttentionGNN(
        clip_dim=512,
        hidden_dim=256,
        num_heads=4,
        dropout_rate=0.1,
    ).to(device)

    trainable = sum(p.numel() for p in gnn.parameters() if p.requires_grad)
    print(f"GNN trainable parameters: {trainable:,}")

    # Test GNN forward pass with dummy data
    B = 2
    num_regions = 4
    num_segments = 3
    clip_dim = 512

    dummy_img_feats = F.normalize(
        torch.randn(B, num_regions, clip_dim), p=2, dim=-1
    ).to(device)
    dummy_txt_feats = F.normalize(
        torch.randn(B, num_segments, clip_dim), p=2, dim=-1
    ).to(device)

    s_cons = gnn(dummy_img_feats, dummy_txt_feats)
    print(f"s_cons shape  : {s_cons.shape}")    # Expected: [2, 1]
    print(f"s_cons values : {s_cons.detach().cpu().numpy()}")
    print(f"All in [0,1]  : {(s_cons >= 0).all() and (s_cons <= 1).all()}")
