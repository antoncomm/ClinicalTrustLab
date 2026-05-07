from abc import ABC, abstractmethod
import torch


class BaseAttack(ABC):

    def __init__(
        self, 
        model,         
        mean: tuple, 
        std: tuple, 
        height: int,
        width: int,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.model.eval()
        self.mean = torch.tensor(mean).to(self.device)
        self.std = torch.tensor(std).to(self.device)
        self.image_size = (height, width)

    @abstractmethod
    def generate(self, x: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError

    def _transform_output(self, x):
        mean = self.mean.view(1, -1, 1, 1).to(self.device)
        std = self.std.view(1, -1, 1, 1).to(self.device)
        return (x - mean) / std
    
    def _transform_input(self, x):
        mean = self.mean.view(1, -1, 1, 1).to(self.device)
        std = self.std.view(1, -1, 1, 1).to(self.device)
        return x * std + mean
