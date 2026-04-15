import abc
from typing import Dict, List, Optional
import torch.nn as nn
from torch import Tensor


class BaseModel(nn.Module, abc.ABC):
    """
    Abstract base class for all object-detection models in this project.

    Subclasses must implement:
      - :meth:`forward` – inference / training forward pass
      - :meth:`freeze_encoder` – freeze all encoder / backbone parameters
      - :meth:`unfreeze_encoder` – unfreeze encoder parameters (fine-tune)
      - :attr:`encoder_name` – human-readable name of the encoder
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
        Forward pass.

        Parameters
        ----------
        images : List[Tensor]
            Batch of images, each ``[C, H, W]`` in ``[0, 1]``.
        targets : List[Dict[str, Tensor]], optional
            Ground-truth annotations (required during training).
            Each dict must contain:
              - ``"boxes"``  : ``FloatTensor[N, 4]`` – xyxy format
              - ``"labels"`` : ``Int64Tensor[N]``

        Returns
        -------
        During *training*: a dict of scalar losses.
        During *inference*: a list of prediction dicts per image, each
        containing ``"boxes"``, ``"labels"``, ``"scores"``.
        """

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Return only parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Output total number of parameters in model. Can specify only trainable parameters count only."""
        params = (
            self.trainable_parameters() if trainable_only else list(self.parameters())
        )
        return sum(p.numel() for p in params)
