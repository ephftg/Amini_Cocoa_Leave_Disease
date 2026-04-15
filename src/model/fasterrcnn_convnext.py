import logging
from typing import Dict, List, Optional

import torch.nn as nn
from torch import Tensor

import timm
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.feature_pyramid_network import (
    FeaturePyramidNetwork,
    LastLevelMaxPool,
)
from torchvision.ops import MultiScaleRoIAlign

from src.utils.logging import setup_logging
from src.model.base_model import BaseModel
from src.pipeline.train import Train

setup_logging()
logger = logging.getLogger("pipeline")


# register the model
@Train.register("FasterRCNN_ConvNeXtV2")
class FasterRCNN_ConvNeXtV2(BaseModel):
    """
    Faster R-CNN detector with a **ConvNeXtV2-Base** backbone loaded with pre-trained weights
    from HuggingFace ``timm`` (frozen by default) and a trainable detection head.

    The detection head is the standard torchvision Faster-RCNN RPN +
    RoI-head pair.  ``torchvision.ops.FeaturePyramidNetwork`` bridges the
    ConvNeXt feature maps to the RPN.

    Parameters
    ----------
    num_classes : int
        Number of object classes **excluding** background.  The background
        class is added internally (total = ``num_classes + 1``).
    """

    _OUT_CHANNELS = 256  # FPN output channels is usually set at 256

    def __init__(
        self,
        num_classes: int,
    ) -> None:

        super().__init__()

        self.num_classes = num_classes

        # build the model and assign it to self.model
        self._build_model()

    # ------------------------------------------------------------------

    @property
    def encoder_name(self) -> str:
        """Return the name of the encoder."""
        return "ConvNeXtV2-Base (facebook/convnextv2-base-22k-224)"

    # ------------------------------------------------------------------

    def _build_model(self) -> None:
        """Build backbone + FPN + Faster-RCNN head."""

        # ---- Backbone (ConvNeXtV2) ----
        backbone_raw = timm.create_model(
            "convnextv2_base.fcmae_ft_in22k_in1k",
            pretrained=True,
            features_only=True,  # expose intermediate feature maps for FPN
        )

        # ConvNeXtV2-Base stage output channels
        in_channels_list = backbone_raw.feature_info.channels()
        out_channels = self._OUT_CHANNELS

        # Wrap backbone + FPN into a single module torchvision expects
        class _BackboneWithFPN(nn.Module):
            def __init__(
                self, backbone: nn.Module, in_ch: list[int], out_ch: int
            ) -> None:
                super().__init__()
                self.body = backbone
                self.fpn = FeaturePyramidNetwork(
                    in_channels_list=in_ch,
                    out_channels=out_ch,
                    extra_blocks=LastLevelMaxPool(),  ## for large object detection useful for faster RCNN
                )
                self.out_channels = out_ch

            def forward(self, x: Tensor) -> Dict[str, Tensor]:
                features = self.body(x)
                # torchvision FPN expects an OrderedDict with string keys
                feat_dict = {str(i): f for i, f in enumerate(features)}
                return self.fpn(feat_dict)

        self.backbone_with_fpn = _BackboneWithFPN(
            backbone_raw, in_channels_list, out_channels
        )

        # freeze all weights in encoder only, leave the FPN trainable
        for param in self.backbone_with_fpn.body.parameters():
            param.requires_grad = False

        # ---- Anchor generator ----
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),  # match FPN levels
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,  # standard for all levels
        )

        # ---- RoI pooler ----
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2,
        )

        # ---- Full Faster-RCNN model ----
        self.model = FasterRCNN(
            backbone=self.backbone_with_fpn,
            num_classes=self.num_classes + 1,  # +1 for background
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )

        logger.info("Model loaded for FasterRCNN_ConvNeXtV2 ")
        # train the FPN, RPN, ROI

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Dict[str, Tensor] | List[Dict[str, Tensor]]:
        """
        Forward pass of the model.
        Targets are only required when model is in training mode. If in eval model, will ignore targets.
        Return the standard output for train and eval for FasterRCNN.

        """
        return self.model(images, targets)


if __name__ == "__main__":
    model = FasterRCNN_ConvNeXtV2(num_classes=3)

    logger.info(model.encoder_name)

    total_trainable_params = model.count_parameters()
    logger.info(f"Total trainable params: {total_trainable_params}")
