import torch
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np
from collections import Counter


class subsetCIFAR10(CIFAR10):
    def __init__(self, sublist, psudoClass, **kwds):
        super().__init__(**kwds)

        if len(sublist) > 0:
            self.data = self.data[sublist]
            self.targets = np.array(self.targets)[sublist].tolist()

        if psudoClass is not None:
            self.targets = psudoClass


class CustomCIFAR10(CIFAR10):
    def __init__(self, sublist, returnLabel=False,  **kwds):
        super().__init__(**kwds)
        self.return_label=returnLabel

        if len(sublist) > 0:
            self.data = self.data[sublist]
            self.targets = np.array(self.targets)[sublist].tolist()

        self.idxsPerClass = [np.where(np.array(self.targets) == idx)[0] for idx in range(10)]
        self.idxsNumPerClass = [len(idxs) for idxs in self.idxsPerClass]
        self.order_of_classes = [i[0] for i in Counter(self.targets).most_common()]

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        if  self.return_label:
            return torch.stack(imgs), self.targets[idx]
        return torch.stack(imgs)


