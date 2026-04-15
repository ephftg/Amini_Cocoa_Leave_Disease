from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import complete_box_iou, complete_box_iou_loss


# this is for classical DETR head, other DETR variant head use other form of matching for loss calculation
class HungarianMatcher(nn.Module):
    """Matches predictions to ground-truth via optimal assignment using CIoU."""

    # set constant weights for the different loss following DETR paper
    _COST_CLASS = 1.0
    _COST_BBOX = 5.0
    _COST_CIOU = 1.5  # reduced from GIoU's 2.0 — CIoU penalises more aggressively

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, targets):
        B, Q = outputs["pred_logits"].shape[:2]

        pred_logits = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B*Q, C+1]
        pred_boxes = outputs["pred_boxes"].flatten(0, 1)  # [B*Q, 4]

        tgt_labels = torch.cat([t["labels"] for t in targets])  # [total_gt]
        tgt_boxes = torch.cat([t["boxes"] for t in targets])  # [total_gt, 4]

        # Classification cost
        cost_class = -pred_logits[:, tgt_labels]

        # L1 cost on boxes, absolute coordinate distance
        cost_bbox = torch.cdist(pred_boxes, tgt_boxes, p=1)

        # CIoU cost — accounts for centre distance + aspect ratio vs GIoU overlap only
        cost_ciou = -complete_box_iou(pred_boxes, tgt_boxes)

        C = (
            self._COST_CLASS * cost_class
            + self._COST_BBOX * cost_bbox
            + self._COST_CIOU * cost_ciou
        )
        C = C.view(B, Q, -1).cpu()

        sizes = [len(t["boxes"]) for t in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (torch.as_tensor(i, dtype=torch.long), torch.as_tensor(j, dtype=torch.long))
            for i, j in indices
        ]


class SetCriterion(nn.Module):
    """Computes DETR losses after Hungarian matching using CIoU box regression loss."""

    _WEIGHT_CE = 1.0
    _WEIGHT_BBOX = 5.0
    _WEIGHT_CIOU = 1.5  # mirrors HungarianMatcher._COST_CIOU — must stay in sync
    _EOS_COEF = 0.1  # down-weights no-object class to counter query imbalance

    def __init__(self, num_classes: int, matcher: HungarianMatcher) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher

        # Cross-entropy weights: real classes=1.0, no-obj=_EOS_COEF
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = self._EOS_COEF
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)

        # ---- Classification loss ----
        pred_logits = outputs["pred_logits"]  # [B, Q, C+1]
        B, Q = pred_logits.shape[:2]

        tgt_classes = torch.full(
            (B, Q),
            self.num_classes,  # default = no-object
            dtype=torch.long,
            device=pred_logits.device,
        )
        for b, (src_idx, tgt_idx) in enumerate(indices):
            tgt_classes[b, src_idx] = targets[b]["labels"][tgt_idx]

        loss_ce = F.cross_entropy(
            pred_logits.flatten(0, 1),  # [B*Q, C+1]
            tgt_classes.flatten(),  # [B*Q]
            weight=self.empty_weight,
        )

        # ---- Box losses (only matched pairs) ----
        pred_boxes = outputs["pred_boxes"]  # [B, Q, 4]
        src_boxes, tgt_boxes = [], []

        for b, (src_idx, tgt_idx) in enumerate(indices):
            src_boxes.append(pred_boxes[b][src_idx])
            tgt_boxes.append(targets[b]["boxes"][tgt_idx])

        src_boxes = torch.cat(src_boxes)
        tgt_boxes = torch.cat(tgt_boxes)
        num_boxes = max(src_boxes.shape[0], 1)

        # L1 bounding box loss
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / num_boxes

        # CIoU bounding box loss — replaces GIoU
        loss_ciou = (
            complete_box_iou_loss(src_boxes, tgt_boxes, reduction="sum") / num_boxes
        )

        return {
            "loss_ce": self._WEIGHT_CE * loss_ce,
            "loss_bbox": self._WEIGHT_BBOX * loss_bbox,
            "loss_ciou": self._WEIGHT_CIOU * loss_ciou,
        }
