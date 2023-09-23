**Abstract**: The two major challenges in applying deep learning to develop a computer-aided diagnosis for retinal images is lack of access to enough amount of annotated data and legal concerns regarding patient privacy. Due to the expertise needed to annotate retinal images and the legal issue raised when using patient image data, it is difficult to obtain sufficient amount of labeled retinal images. Various kinds of effort are being made to increase the amount of data either by augmenting training images or by synthesizing real-looking retinal images. However, augmentation is limited to the amount of available data and it doesn't solve the concern of patient privacy. In this paper, we propose a GAN based fundus image synthesis method (FundusGAN) that generates synthetic training images to solve the above problems. The proposed method is an improved way of generating retinal images by following a two-step generation process which involves first training a segmentation network to extract the vessel tree followed by vessel tree to fundus image-to-image translation using unsupervised generative attention networks with adaptive layer-instance normalization. Our experiments result shows that the proposed fundusGAN exceeds state-of-the-art generative models for retinal image synthesis in different GAN metrics. Our method also validates that generated retinal images can be used to train retinal image classifiers for eye diseases diagnosis.





# How to run?
## FundusGAN-I (Colab-version)

Connecting to drive 
```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```python
cd /content/drive/MyDrive/RP/Summer21
```

    /content/drive/MyDrive/RP/Summer21
    


```python
cd Fundus-GAN-version-1/
```

    /content/drive/My Drive/RP/Summer21/Fundus-GAN-version-1
    


**1. Segmentation**



```python
device = 'cuda'
```


```python
from torch.utils.data import DataLoader
import torch
from src.utils.config import *
from src.models.unet import *
from src.models.pix2pix_gen import UNet_pix2pix
from src.dataset.dataset import *
from src.models.discriminator import Discriminator
from train_pix2pix import *
from train import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
```

    cuda
    


```python
train_input_path = datapath_cfg['train_input_path']
train_label_path = datapath_cfg['train_label_path']
batch_size = cfg['batch_size']
print(train_input_path)
```

    /content/drive/MyDrive/RP/Summer21/dataset/newi.tiff
    


```python
train_dataset = unet_dataset(train_input_path, train_label_path)
```


```python
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
```


```python
unet = UNet(input_channels=cfg['input_dim'], output_channels=cfg['output_dim']).to(device)
```


```python
unet_opt = torch.optim.Adam(unet.parameters(), lr=cfg['lr'])
```


```python
unet = UNet(input_channels=cfg['input_dim'], output_channels=cfg['output_dim'])
checkpoint = torch.load('/content/drive/MyDrive/RP/F21/Models/UNET/unet_v_26-09-202114:30:12.pth')
unet.load_state_dict(checkpoint['unet'])
unet.to(device)
```


```python
train(model=unet, optimizer=unet_opt, 
      criterion=cfg['criterion'],
      train_dataloader=train_dataloader, 
      input_dim=cfg['input_dim'], 
      label_dim=cfg['output_dim'], 
      target_shape=cfg['target_shape'],
      device=device, save_dir = "/content/drive/MyDrive/RP/F21/Models/UNET")
```





```python

```

