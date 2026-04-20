import abc
from typing import Dict, List, Optional
import torch.nn as nn
from torch import Tensor


class BaseModel(nn.Module, abc.ABC):
    """
    Abstract base class for pytorch object-detection models.
    """

    @property
    @abc.abstractmethod
    def encoder_name(self) -> str:
        """Return a descriptive name for the encoder used."""

    @abc.abstractmethod
    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Dict[str, Tensor] | List[Dict[str, Tensor]]:
        """
        Forward pass of the data into the model.
        Targets are only required in training mode, in eval mode, it will be ignored even if provided.

        Args:
            images: List of images in tensor form.
            targets: List of ground-truth annotations in dictionary format.
                    Each dictionary must contain "boxes" for the bounding box coordinates and "label" for the class label.

        Returns:
            Changes depending on mode:
                Train: A dictionary of losses
                Eval: A list of prediction dictionary per image.
                      Each dictionary has "boxes" for the bounding box coordinates,
                      "label" for the class label and "scores" for confidence of class prediction.
        """

    def trainable_parameters(self) -> List[nn.Parameter]:
        """
        Filter for trainable parameters (requires gradient) in model.

        Returns:
            List of trainable parameters in model.
        """
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Output total number of parameters in model.
        Can specify only trainable parameters count only.

        Args:
            trainable_only: Boolean variable to specify if only trainable parameter count is required.

        Returns:
            Number of parameters in model or number of trainable parameters in model.
        """
        params = (
            self.trainable_parameters() if trainable_only else list(self.parameters())
        )
        return sum(p.numel() for p in params)
