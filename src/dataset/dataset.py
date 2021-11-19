
import gdal
from skimage import io
import numpy as np
import torch
from src.utils.config import cfg
import torchvision
"""
 args:
  input path ----- input image to be segmented (.tiff format)
  label path ------- corresponding label (.tiff format)
  batch size 
  shuffle = True for training, True/False for validation and False for inference
return:
  dataset
"""
def unet_dataset(input_path, label_path):
    volumes = torch.Tensor(io.imread(input_path, plugin='pil'))[:, None, :, :]/255
    labels = torch.Tensor(io.imread(label_path, plugin='pil'))[:, None, :, :]/255
    # labels = crop(labels, torch.Size([len(labels), 1, target_shape, target_shape]))
    train_dataset = torch.utils.data.TensorDataset(volumes, labels)
    return train_dataset
def pix2pix_dataset(input_path):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    pix2pix_dataset = torchvision.datasets.ImageFolder(input_path, transform=transform)
    return pix2pix_dataset