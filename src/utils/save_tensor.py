from torchvision.utils import save_image
from torchvision.utils import make_grid
import torch
from torch import nn

def save_tensor_image(image_tensor, num_images =1, path="", size=(1, 28, 28)):
    '''
    Function for saving images: Given a tensor of images, number of images, and
    size per image,save the images in a uniform grid.
    '''
    image_shifted = torch.sigmoid(image_tensor)
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    save_image(image_grid, path)