from abc import ABC, abstractmethod
import torch


class BaseAttack(ABC):

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def generate(self, x: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError
