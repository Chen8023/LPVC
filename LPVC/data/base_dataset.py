
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):


    def __init__(self, opt):
        self.opt = opt


    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):

        pass

