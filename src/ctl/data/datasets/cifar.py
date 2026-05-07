import torchvision
import torch
from .basedatasets import BaseDataset
import torchvision.transforms as transforms


class CifarDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = torchvision.datasets.CIFAR10(root=self.root, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
    
    def collate_fn(self, samples):
        images, labels = zip(*samples)
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels, dtype=torch.long).view(-1, 1)
        return images, {"labels": labels}
