import torch
from .base_attack import BaseAttack
from torch import nn
from tqdm import tqdm


class UAPAttack(BaseAttack):
    """UAP attack on model."""

    def __init__(
        self,
        model: torch.nn.Module,
        mean: tuple, 
        std: tuple, 
        height: int,
        width: int,
        **kwargs
    ) -> None:
        super().__init__(model, mean, std, height, width)

        self.eps = kwargs["eps"]
        self.lr = kwargs["lr"]
        self.max_epochs = kwargs["max_iter"]
        self.image_size = (3, height, width) if kwargs["is_color"] else (1, height, width)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.make_perturbation()

    def get_conv_layers(self, model):
        return [module for module in model.modules() if isinstance(module, nn.Conv2d)]

    def l2_layer_loss(self) -> torch.Tensor:
        activations = []
        remove_handles = []

        def activation_recorder_hook(*args):
            activations.append(args[2])
            return None

        for conv_layer in self.get_conv_layers(self.model):
            handle = conv_layer.register_forward_hook(activation_recorder_hook)
            remove_handles.append(handle)

        self.model.zero_grad()
        self.model(self.perturbation)

        for handle in remove_handles:
            handle.remove()

        loss = -sum([torch.log(torch.sum(torch.square(activation)) / 2) for activation in activations])
        return loss

    def make_perturbation(self) -> None:
        self.model.eval()
        self.perturbation = -2 * self.eps * torch.rand(self.image_size, device=self.device) + self.eps
        self.perturbation.unsqueeze_(0)

        self.perturbation.requires_grad = True
        optimizer = torch.optim.SGD([self.perturbation], lr=self.lr)

        for _ in tqdm(range(self.max_epochs)):
            self.perturbation.requires_grad = True
            optimizer.zero_grad()
            loss = self.l2_layer_loss()
            loss.backward()
            grad = self.perturbation.grad.sign()        # (C, H, W)
            grad_mean = grad.mean(dim=0, keepdim=True) # (1, H, W)
            self.perturbation.grad = grad_mean.repeat(1, grad.shape[0], 1, 1)  # (C, H, W)
            optimizer.step()
            self.perturbation.data.clamp_(-self.eps, self.eps)

        mean_pret = self.perturbation.mean(dim=0, keepdim=True) 
        self.perturbation = mean_pret.repeat(1, grad.shape[0], 1, 1)

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._transform_input(x)
        
        x_adv = (x + self.perturbation).clamp(0, 1)
        
        x_adv_mean = x_adv.mean(dim=1, keepdim=True)   
        x_adv = x_adv_mean.repeat(1, 3, 1, 1)      
        
        x_adv = self._transform_output(x_adv)
        
        return x_adv, y
