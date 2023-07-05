from typing import Optional
from typing import Tuple
from typing import Dict

from abc import ABCMeta
from abc import abstractmethod

import torch
import torch.nn as nn

from model import MyModel


class GeneClassifier(MyModel, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
            self,
            model_dir: str,
            model_name: str,
            hyperparameter: Dict[str, any],
            weights: Optional[torch.Tensor]
    ):
        super().__init__(model_dir, model_name, hyperparameter, weights)

    @abstractmethod
    def load_data(self, batch, device: torch.device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        pass

    @abstractmethod
    def step(self, inputs: Dict[str, any]) -> any:
        pass

    @abstractmethod
    def compute_loss(self, target: torch.Tensor, *outputs) -> torch.Tensor:
        pass

    @abstractmethod
    def embedding_step(self, inputs: Dict[str, any]) -> any:
        pass

    @abstractmethod
    def get_embedding_layer(self) -> nn.Module:
        pass

    @abstractmethod
    def get_n_classes(self) -> int:
        pass
