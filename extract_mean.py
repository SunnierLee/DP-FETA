import torchvision
import torch
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

from stylegan3.dataset import ImageFolderDataset
from dataset_tool import CentralDataset


data_name = "mnist"
attr = "Male"
use_labels = True
sample_num_per_cls = 5
sigma = 5
batch_size = 6000

if "celeba" == data_name:
    num_classes = 2
    data_dir = ''
    dataset = ImageFolderDataset(data_dir, split='train', resolution=32, use_labels=use_labels, attr=attr)
elif "camelyon" == data_name:
    num_classes = 2
    data_dir = ''
    dataset = ImageFolderDataset(data_dir, split='train', resolution=32, use_labels=use_labels, attr=attr)
elif data_name == "mnist":
    num_classes = 10
    data_dir = ''
    dataset = torchvision.datasets.MNIST(data_dir, download=True, train=True, transform=transforms.ToTensor())
elif data_name == "fmnist":
    num_classes = 10
    data_dir = ''
    dataset = torchvision.datasets.FashionMNIST(data_dir, download=True, train=True, transform=transforms.ToTensor())
print("number of sensitive data: ", len(dataset))
central_dataset = CentralDataset(dataset, sample_num=sample_num_per_cls, num_classes=num_classes, c_type='mean', sigma=sigma)


q = batch_size / len(dataset)

save_dir = "./mean_{}_{}_{}_{}".format(data_name, sample_num_per_cls, sigma, q)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, 'noisy'), exist_ok=True)

c = 0
for i in range(len(central_dataset)):
    x, y = central_dataset.central_x[i], central_dataset.central_y[i]
    out_dir_noisy = os.path.join(save_dir, 'noisy', str(y).zfill(6))
    os.makedirs(out_dir_noisy, exist_ok=True)

    x = x.permute(1, 2, 0).numpy().astype(np.uint8)
    if x.shape[-1] == 1:
        x = x[..., 0]
    Image.fromarray(x).save(os.path.join(out_dir_noisy, '{}.png'.format(c)))
    c += 1
