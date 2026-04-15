import logging
from typing import Dict, List, Optional

import math
import torch
import torch.nn as nn
from torch import Tensor
from transformers import Dinov2Model

from src.utils.logging import setup_logging
from src.model.base_model import BaseModel
from src.model.detr_loss import HungarianMatcher, SetCriterion
from src.pipeline.train import Train

setup_logging()
logger = logging.getLogger("pipeline")


@Train.register("Detr_DinoV2")
class Detr_DinoV2(BaseModel):
    """
    DETR-style detector with a **DINOv2-Base** vision-transformer encoder
    (frozen by default) and a trainable transformer decoder + prediction head.

    The encoder produces patch embeddings from DINOv2; a standard
    DETR-like query-based decoder cross-attends over these embeddings and
    outputs bounding-box regression + class logits.

    Parameters
    ----------
    num_classes : int
        Number of object classes (background is predicted implicitly via a
        "no-object" class in the DETR fashion).
    """

    _ENCODER_HF_ID = (
        "facebook/dinov2-base"  # use DinoV2 since DinoV3 require agreement for access
    )
    _NUM_QUERIES = 15  # max objects per image (dataset max is 13)
    _HIDDEN_DIM = 256  # transformer hidden dimension
    _NHEADS = 8  # attention heads in decoder
    _NUM_DECODER_LAYERS = 6  # decoder depth

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self._build_model()

        # set the loss criteria to calculate loss when training
        matcher = HungarianMatcher()
        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
        )

    @property
    def encoder_name(self) -> str:
        return "DINOv2-Base (facebook/dinov2-base)"

    def _build_2d_sinusoidal_pos_embed(self, H, W):
        """Build [1, H*W, hidden_dim] sinusoidal positional embedding."""
        y = torch.arange(H, dtype=torch.float32)
        x = torch.arange(W, dtype=torch.float32)
        # Normalize to [0, 2pi]
        y = y / H * 2 * math.pi
        x = x / W * 2 * math.pi

        d = self._HIDDEN_DIM // 4  # split evenly across x/y and sin/cos
        dim_t = 10000 ** (torch.arange(d, dtype=torch.float32) / d)

        pos_x = x[:, None] / dim_t  # [W, d]
        pos_y = y[:, None] / dim_t  # [H, d]

        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).flatten(1)  # [W, 2d]
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).flatten(1)  # [H, 2d]

        # Combine x and y
        pos = torch.cat(
            [
                pos_y.unsqueeze(1).repeat(1, W, 1),  # [H, W, 2d]
                pos_x.unsqueeze(0).repeat(H, 1, 1),  # [H, W, 2d]
            ],
            dim=-1,
        )  # [H, W, hidden_dim]

        return pos.flatten(0, 1).unsqueeze(0)  # [1, H*W, hidden_dim]

    def _build_model(self) -> None:

        # ---- Encoder: DINOv2 ----
        self.encoder = Dinov2Model.from_pretrained(self._ENCODER_HF_ID)
        encoder_dim = self.encoder.config.hidden_size  # 768 for dinov2-base

        # Freeze all encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Project encoder output → hidden_dim
        self.input_proj = nn.Linear(encoder_dim, self._HIDDEN_DIM)

        # DINOv2-Base: 224/14 = 16 patches per side
        self.register_buffer("pos_embed", self._build_2d_sinusoidal_pos_embed(16, 16))

        # ---- Decoder: DETR-style transformer decoder ----
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self._HIDDEN_DIM,
            nhead=self._NHEADS,
            dim_feedforward=self._HIDDEN_DIM * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self._NUM_DECODER_LAYERS,
        )

        # Learnable object queries
        self.query_embed = nn.Embedding(self._NUM_QUERIES, self._HIDDEN_DIM)

        # Prediction heads
        self.class_head = nn.Linear(self._HIDDEN_DIM, self.num_classes + 1)  # +1 no-obj

        # 3-layer MLP for bounding box regression
        self.bbox_head = nn.Sequential(
            nn.Linear(self._HIDDEN_DIM, self._HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(self._HIDDEN_DIM, self._HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(self._HIDDEN_DIM, 4),  # xmin, ymin, xmax, ymax
            nn.Sigmoid(),  # normalised [0, 1]
        )

        logger.info("Model loaded for Detr_DinoV2")

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Dict[str, Tensor] | List[Dict[str, Tensor]]:
        """
        Targets only required when training.

        """

        # Stack images into a batch tensor
        pixel_values = torch.stack(images)  # [B, C, H, W]

        # Encode with DINOv2 — output: [B, num_patches+1, encoder_dim]
        enc_out = self.encoder(pixel_values=pixel_values).last_hidden_state
        # Drop CLS token
        patch_tokens = enc_out[:, 1:, :]  # [B, N, encoder_dim]

        memory = self.input_proj(patch_tokens)

        # add positional embedding
        memory = memory + self.pos_embed  # [B, N, hidden_dim]

        B = pixel_values.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]

        hs = self.decoder(tgt=queries, memory=memory)  # [B, Q, D]

        # Predictions
        pred_logits = self.class_head(hs)  # [B, Q, num_classes+1]
        pred_boxes = self.bbox_head(hs)  # [B, Q, 4]  normalised cxcywh

        # return loss when forward
        # traing is true when the model is set as model.train()
        # follow loss dictonary format as FasterRCNN
        if self.training:
            outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
            return self.criterion(outputs, targets)

        # Inference: convert to per-image output list
        # follow same result as FasterRCNN
        results: List[Dict[str, Tensor]] = []
        probs = pred_logits.softmax(dim=-1)  # [B, Q, C+1]
        scores, labels = probs[..., :-1].max(dim=-1)  # exclude no-obj class

        for b in range(B):
            results.append(
                {
                    "boxes": pred_boxes[b],  # [Q, 4] normalised cxcywh
                    "scores": scores[b],  # [Q]
                    "labels": labels[b],  # [Q]
                }
            )
        return results


if __name__ == "__main__":
    model = Detr_DinoV2(num_classes=3)

    logger.info(model.encoder_name)

    total_trainable_params = model.count_parameters()
    logger.info(f"Total trainable params: {total_trainable_params}")
