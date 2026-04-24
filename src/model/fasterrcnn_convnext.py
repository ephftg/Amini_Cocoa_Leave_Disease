import logging
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


# register the model to Train class
@Train.register("FasterRCNN_ConvNeXtV2")
class FasterRCNN_ConvNeXtV2(BaseModel):
    """
    Faster R-CNN object detection model with a frozen pre-trained ConvNeXtV2 encoder and a trainable detection head.
    It uses the "FasterRCNN" class from torchvision library.
    Trainable parameters include the FPN, RPN, RoI Align and the two prediction heads.
    """

    # dfault number of FPN output channels
    _OUT_CHANNELS = 256

    def __init__(
        self,
        num_classes: int,
    ) -> None:
        """
        Create model by specifying number of object classes in images.

        Args:
            num_classes: Number of object classes possible in image.
        """
        super().__init__()
        self.num_classes = num_classes

        # build the model and assign it to self.model
        self._build_model()

    @property
    def encoder_name(self) -> str:
        """Return the name of the encoder."""
        return "ConvNeXtV2-Base (facebook/convnextv2-base-22k-224)"

    def _build_model(self) -> None:
        """
        Build the FasterRCNN model and asisgn it to self.model.
        The model consist of a backbone (feature encoder layer and FPN),
        a RPN and RoI Align and detection heads.
        """

        # ---- Backbone (ConvNeXtV2) ----
        # download the pre-trained ConvNeXtV2 encoder layer without the classification head
        # model chosen is base size, with self supervised training, followed by fine tuning with imageNet
        # returns intermediate feature maps from each stage which will be used by FPN for multi-scale feature
        backbone_raw = timm.create_model(
            "convnextv2_base.fcmae_ft_in22k_in1k",
            pretrained=True,
            features_only=True,
        )

        in_channels_list = backbone_raw.feature_info.channels()
        out_channels = self._OUT_CHANNELS

        class _BackboneWithFPN(nn.Module):
            def __init__(
                self,
                backbone: nn.Module,
                in_ch: list[int],
                out_ch: int,
            ) -> None:
                """
                Class for the backbone layer for FasterRCNN which is feature encoder and FPN combined.

                Args:
                    backbone: Model for the feature encoder layer that provides the intermediate feature maps.
                    in_ch: List of channel sizes for each intermediate feature maps provided by backbone.
                    out_ch: Number of output channels of the fetaure maps from FPN.

                """
                super().__init__()
                self.body = backbone

                # create FPN layer
                # create an extra feature map by applying max pooling on the last FPN output. Useful to detect large objects in image.
                self.fpn = FeaturePyramidNetwork(
                    in_channels_list=in_ch,
                    out_channels=out_ch,
                    extra_blocks=LastLevelMaxPool(),
                )
                self.out_channels = out_ch

            def forward(self, x: Tensor) -> dict[str, Tensor]:
                """
                Forward pass of image into the model backbone with FPN.

                Args:
                    x: Input image in tensor form
                Returns:
                    Dictionary of multi-scale feature maps from FPN.
                """
                # pass image into the encoder layer to get feature maps of different scales.
                features = self.body(x)

                # connvert features to OrderedDict with string keys starting from 0, required for torchvision FPN
                feat_dict = {str(i): f for i, f in enumerate(features)}

                # FPN output multi-scale feature maps from input feature maps
                return self.fpn(feat_dict)

        self.backbone_with_fpn = _BackboneWithFPN(
            backbone_raw, in_channels_list, out_channels
        )

        # freeze all weights in encoder only, leave the FPN trainable
        for param in self.backbone_with_fpn.body.parameters():
            param.requires_grad = False

        # size of bounding box for each FPN layer and corresponding aspect ratio
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),  # standard values
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,  # standard values
        )

        # crop and resize feature maps into fixed size for classification head
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2,
        )

        self.model = FasterRCNN(
            backbone=self.backbone_with_fpn,
            num_classes=self.num_classes + 1,  # +1 for background
            rpn_anchor_generator=anchor_generator,  # feed anchor to RPN
            box_roi_pool=roi_pooler,
        )

        logger.info("Model loaded for FasterRCNN_ConvNeXtV2 ")

    def forward(
        self,
        images: list[Tensor],
        targets: list[dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """
        Forward pass of the model that is an abstract method.
        Direct input into the model as the FasterRCNN class handles the process.
        """
        return self.model(images, targets)


if __name__ == "__main__":
    model = FasterRCNN_ConvNeXtV2(num_classes=3)

    logger.info(model.encoder_name)

    total_trainable_params = model.count_parameters()
    logger.info(f"Total trainable params: {total_trainable_params}")
