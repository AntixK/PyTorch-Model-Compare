import os
import torch
# from torchvision.models import resnet18, resnet34, resnet50, densenet121
from torchvision.datasets import CIFAR10, Flickr8k, STL10, SBDataset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.nn as nn
from functools import partial
import pprint
from typing import List
from warnings import warn
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from mpl_toolkits import axes_grid1
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models import swin_small_patch4_window7_224, resnetv2_50x1_bitm, resnet34, resnetv2_101x1_bitm
import pandas as pd
from PIL import Image


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)

model1 = swin_small_patch4_window7_224(pretrained=True)
model2 = resnetv2_50x1_bitm(pretrained=True)

layers1=[]
LAYERS = ["norm2", "fc2",  "norm1", "fc1", "act"]
for name, layer in model1.named_modules():
    if len(name.split('.')) > 3:    # Take only
#         if name.split('.')[1] in ['2']: # and name.split('.')[3] in BLOCKS:
        if name.split('.')[-1] in LAYERS:
            layers1 += [name]

