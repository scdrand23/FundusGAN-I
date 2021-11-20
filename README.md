# FundusGAN-I

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

    Epoch 0: Step 5: U-Net loss: 0.05962532386183739
    


    
![png](images/output_12_1.png)
    



    
![png](images/output_12_2.png)
    



    
![png](images/output_12_3.png)
    


    Epoch 6: Step 205: U-Net loss: 0.04562734067440033
    


    
![png](images/output_12_5.png)
    



    
![png](images/output_12_6.png)
    



    
![png](images/output_12_7.png)
    


    Epoch 11: Step 405: U-Net loss: 0.05267468839883804
    


    
![png](images/output_12_9.png)
    



    
![png](images/output_12_10.png)
    



    
![png](images/output_12_11.png)
    


    Epoch 17: Step 605: U-Net loss: 0.04996743053197861
    
.
.
.
.

    

.
.
.
.  

    
.
.
.
.  


    Epoch 382: Step 13005: U-Net loss: 0.06173957884311676
    


    
![png](images/output_12_261.png)
    



    
![png](images/output_12_262.png)
    



    
![png](images/output_12_263.png)
    


    Epoch 388: Step 13205: U-Net loss: 0.04220299422740936
    


    
![png](images/output_12_265.png)
    



    
![png](images/output_12_266.png)
    



    
![png](images/output_12_267.png)
    



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-16-5dcd1309c74a> in <module>()
          5       label_dim=cfg['output_dim'],
          6       target_shape=cfg['target_shape'],
    ----> 7       device=device, save_dir = "/content/drive/MyDrive/RP/F21/Models/UNET")
    

    /content/drive/My Drive/RP/Summer21/Fundus-GAN-version-1/train.py in train(model, optimizer, criterion, train_dataloader, input_dim, label_dim, target_shape, device, save_dir)
         24             # Flatten the image
         25 
    ---> 26             real = real.to(device)
         27             labels = labels.to(device)
         28             ### Update U-Net ###
    

    KeyboardInterrupt: 





**2. Image generation**





```python
pix2pix_dataset = pix2pix_dataset('/content/drive/MyDrive/RP/Summer21/dataset/maps')
```


```python
dataloader = DataLoader(pix2pix_dataset, batch_size=pix2pix_config['batch_size'], shuffle=True)
```


```python
unet_pix2pix = UNet_pix2pix(input_channels=pix2pix_config['input_dim'],output_channels=pix2pix_config['real_dim']).to(device)
```


```python
unet_opt = torch.optim.Adam(unet_pix2pix.parameters(), lr=pix2pix_config['lr'])
```


```python
disc = Discriminator(pix2pix_config['input_dim'] + pix2pix_config['real_dim']).to(device)
```


```python
disc_opt = torch.optim.Adam(disc.parameters(), lr=pix2pix_config['lr'])
```


```python
model = {'gen':unet_pix2pix, 'disc':disc}
optimizer = {'gen_opt':unet_opt, 'disc_opt':disc_opt}
pix2pix_criteria = {'adv_criterion':pix2pix_config['adv_criterion'], 'recon_criterion':pix2pix_config['recon_criterion']}
```


```python
train_pix2pix(model=model, optimizer=optimizer, criterion=pix2pix_criteria, dataloader = dataloader, input_dim=pix2pix_config['real_dim'],
              label_dim=pix2pix_config['real_dim'], target_shape=pix2pix_config['target_shape'],
              device='cuda')
```

    Loading pretrained initial state
    


    
![png](images/output_21_1.png)
    



    
![png](images/output_21_2.png)
    



    
![png](images/output_21_3.png)
    


    Epoch 0: Step 200: Generator (U-Net) loss: 52.46482770919802, Discriminator loss: 0.6408292359113699
    


    
![png](images/output_21_5.png)
    



    
![png](images/output_21_6.png)
    



    
![png](images/output_21_7.png)
    


    Epoch 0: Step 400: Generator (U-Net) loss: 44.93029767036437, Discriminator loss: 0.3824089297652243
    


    
![png](images/output_21_9.png)
    



    
![png](output_21_10.png)
    



    
![png](output_21_11.png)
    


    Epoch 1: Step 600: Generator (U-Net) loss: 33.83245618820189, Discriminator loss: 0.25708449937403194
    


    
![png](output_21_13.png)
    



    
![png](output_21_14.png)
    



    
![png](output_21_15.png)
    


    Epoch 1: Step 800: Generator (U-Net) loss: 27.038341903686547, Discriminator loss: 0.12598866902291786
    


    
![png](output_21_17.png)
    



    
![png](output_21_18.png)
    



    
![png](output_21_19.png)
    


    Epoch 2: Step 1000: Generator (U-Net) loss: 24.62309978485109, Discriminator loss: 0.06050973799079659
    


    
![png](output_21_21.png)
    



    
![png](output_21_22.png)
    



    
![png](output_21_23.png)
    


    Epoch 2: Step 1200: Generator (U-Net) loss: 23.124661002159122, Discriminator loss: 0.03553965166211128
    


    
![png](output_21_25.png)
    



    
![png](output_21_26.png)
    



    
![png](output_21_27.png)
    


    Epoch 3: Step 1400: Generator (U-Net) loss: 21.25410917282105, Discriminator loss: 0.2585765695944429
    


    
![png](output_21_29.png)
    



    
![png](output_21_30.png)
    



    
![png](output_21_31.png)
    


    Epoch 3: Step 1600: Generator (U-Net) loss: 20.7684798192978, Discriminator loss: 0.18324331607669597
    


    
![png](output_21_33.png)
    



    
![png](output_21_34.png)
    



    
![png](output_21_35.png)
    


    Epoch 3: Step 1800: Generator (U-Net) loss: 21.21658486843109, Discriminator loss: 0.026128428475931286
    


    
![png](output_21_37.png)
    



    
![png](output_21_38.png)
    



    
![png](output_21_39.png)
    


    Epoch 4: Step 2000: Generator (U-Net) loss: 22.123379473686224, Discriminator loss: 0.0170011700410396
    


    
![png](output_21_41.png)
    



    
![png](output_21_42.png)
    



    
![png](output_21_43.png)
    


    Epoch 4: Step 2200: Generator (U-Net) loss: 20.941478252410878, Discriminator loss: 0.06591212144587191
    


    
![png](output_21_45.png)
    



    
![png](output_21_46.png)
    



    
![png](output_21_47.png)
    


    Epoch 5: Step 2400: Generator (U-Net) loss: 18.551384806632992, Discriminator loss: 0.4075392137095334
    


    
![png](output_21_49.png)
    



    
![png](output_21_50.png)
    



    
![png](output_21_51.png)
    


    Epoch 5: Step 2600: Generator (U-Net) loss: 19.37798899650575, Discriminator loss: 0.1957894230447711
    


    
![png](output_21_53.png)
    



    
![png](output_21_54.png)
    



    
![png](output_21_55.png)
    


    Epoch 6: Step 2800: Generator (U-Net) loss: 19.09459172248841, Discriminator loss: 0.20214518136810505
    


    
![png](output_21_57.png)
    



    
![png](output_21_58.png)
    



    
![png](output_21_59.png)
    


    Epoch 6: Step 3000: Generator (U-Net) loss: 19.20126194477082, Discriminator loss: 0.25998362509533757
    


    
![png](output_21_61.png)
    



    
![png](output_21_62.png)
    



    
![png](output_21_63.png)
    


    Epoch 6: Step 3200: Generator (U-Net) loss: 17.592603716850295, Discriminator loss: 0.42042010890319953
    


    
![png](output_21_65.png)
    



    
![png](output_21_66.png)
    



    
![png](output_21_67.png)
    


    Epoch 7: Step 3400: Generator (U-Net) loss: 17.36779361248017, Discriminator loss: 0.44014838546514506
    


    
![png](output_21_69.png)
    



    
![png](output_21_70.png)
    



    
![png](output_21_71.png)
    


    Epoch 7: Step 3600: Generator (U-Net) loss: 16.46069777965546, Discriminator loss: 0.5637747962772841
    


    
![png](output_21_73.png)
    



    
![png](output_21_74.png)
    



    
![png](output_21_75.png)
    


    Epoch 8: Step 3800: Generator (U-Net) loss: 15.786085453033445, Discriminator loss: 0.5743345548957587
    


    
![png](output_21_77.png)
    



    
![png](output_21_78.png)
    



    
![png](output_21_79.png)
    


    Epoch 8: Step 4000: Generator (U-Net) loss: 16.327435817718506, Discriminator loss: 0.5351156285405161
    


    
![png](output_21_81.png)
    



    
![png](output_21_82.png)
    



    
![png](output_21_83.png)
    


    Epoch 9: Step 4200: Generator (U-Net) loss: 15.796210246086117, Discriminator loss: 0.5940861061960462
    


    
![png](output_21_85.png)
    



    
![png](output_21_86.png)
    



    
![png](output_21_87.png)
    


    Epoch 9: Step 4400: Generator (U-Net) loss: 15.309932565689099, Discriminator loss: 0.5718528483062987
    


    
![png](output_21_89.png)
    



    
![png](output_21_90.png)
    



    
![png](output_21_91.png)
    


    Epoch 9: Step 4600: Generator (U-Net) loss: 15.604188284873958, Discriminator loss: 0.5589309097826483
    


    
![png](output_21_93.png)
    



    
![png](output_21_94.png)
    



    
![png](output_21_95.png)
    


    Epoch 10: Step 4800: Generator (U-Net) loss: 15.35258427619935, Discriminator loss: 0.5633312847465274
    


    
![png](output_21_97.png)
    



    
![png](output_21_98.png)
    



    
![png](output_21_99.png)
    


    Epoch 10: Step 5000: Generator (U-Net) loss: 15.272418904304505, Discriminator loss: 0.563543492406607
    


    
![png](output_21_101.png)
    



    
![png](output_21_102.png)
    



    
![png](output_21_103.png)
    


    Epoch 11: Step 5200: Generator (U-Net) loss: 14.911911058425899, Discriminator loss: 0.6039275236427781
    


    
![png](output_21_105.png)
    



    
![png](output_21_106.png)
    



    
![png](output_21_107.png)
    


    Epoch 11: Step 5400: Generator (U-Net) loss: 15.01147954940796, Discriminator loss: 0.543950648754835
    


    
![png](output_21_109.png)
    



    
![png](output_21_110.png)
    



    
![png](output_21_111.png)
    


    Epoch 12: Step 5600: Generator (U-Net) loss: 15.068673787117005, Discriminator loss: 0.5008977087587121
    


    
![png](output_21_113.png)
    



    
![png](output_21_114.png)
    



    
![png](output_21_115.png)
    


    Epoch 12: Step 5800: Generator (U-Net) loss: 15.088388924598696, Discriminator loss: 0.49634317822754365
    


    
![png](output_21_117.png)
    



    
![png](output_21_118.png)
    



    
![png](output_21_119.png)
    


    Epoch 12: Step 6000: Generator (U-Net) loss: 14.925498719215398, Discriminator loss: 0.4236111663281919
    


    
![png](output_21_121.png)
    



    
![png](output_21_122.png)
    



    
![png](output_21_123.png)
    


    Epoch 13: Step 6200: Generator (U-Net) loss: 15.02654624462127, Discriminator loss: 0.349321171361953
    


    
![png](output_21_125.png)
    



    
![png](output_21_126.png)
    



    
![png](output_21_127.png)
    


    Epoch 13: Step 6400: Generator (U-Net) loss: 15.361308097839357, Discriminator loss: 0.30512543072924025
    


    
![png](output_21_129.png)
    



    
![png](output_21_130.png)
    



    
![png](output_21_131.png)
    


    Epoch 14: Step 6600: Generator (U-Net) loss: 15.024779133796695, Discriminator loss: 0.41992817364633084
    


    
![png](output_21_133.png)
    



    
![png](output_21_134.png)
    



    
![png](output_21_135.png)
    


    Epoch 14: Step 6800: Generator (U-Net) loss: 14.677514123916612, Discriminator loss: 0.4989471133798357
    


    
![png](output_21_137.png)
    



    
![png](output_21_138.png)
    



    
![png](output_21_139.png)
    


    Epoch 15: Step 7000: Generator (U-Net) loss: 15.165476856231678, Discriminator loss: 0.36930760577321076
    


    
![png](output_21_141.png)
    



    
![png](output_21_142.png)
    



    
![png](output_21_143.png)
    


    Epoch 15: Step 7200: Generator (U-Net) loss: 15.89894622802734, Discriminator loss: 0.3400735027901828
    


    
![png](output_21_145.png)
    



    
![png](output_21_146.png)
    



    
![png](output_21_147.png)
    


    Epoch 15: Step 7400: Generator (U-Net) loss: 14.142220945358288, Discriminator loss: 0.5264257546514274
    


    
![png](output_21_149.png)
    



    
![png](output_21_150.png)
    



    
![png](output_21_151.png)
    


    Epoch 16: Step 7600: Generator (U-Net) loss: 14.640405068397518, Discriminator loss: 0.45140052106231454
    


    
![png](output_21_153.png)
    



    
![png](output_21_154.png)
    



    
![png](output_21_155.png)
    


    Epoch 16: Step 7800: Generator (U-Net) loss: 14.23081113338471, Discriminator loss: 0.4928330842778087
    


    
![png](output_21_157.png)
    



    
![png](output_21_158.png)
    



    
![png](output_21_159.png)
    


    Epoch 17: Step 8000: Generator (U-Net) loss: 14.365247821807866, Discriminator loss: 0.4132874688878656
    


    
![png](output_21_161.png)
    



    
![png](output_21_162.png)
    



    
![png](output_21_163.png)
    


    Epoch 17: Step 8200: Generator (U-Net) loss: 13.811207842826848, Discriminator loss: 0.49831123992800735
    


    
![png](output_21_165.png)
    



    
![png](output_21_166.png)
    



    
![png](output_21_167.png)
    


    Epoch 18: Step 8400: Generator (U-Net) loss: 13.340272712707515, Discriminator loss: 0.6497187440842389
    


    
![png](output_21_169.png)
    



    
![png](output_21_170.png)
    



    
![png](output_21_171.png)
    


    Epoch 18: Step 8600: Generator (U-Net) loss: 13.199333024024972, Discriminator loss: 0.5171704695373772
    


    
![png](output_21_173.png)
    



    
![png](output_21_174.png)
    



    
![png](output_21_175.png)
    


    Epoch 18: Step 8800: Generator (U-Net) loss: 13.753638467788706, Discriminator loss: 0.537200129926205
    


    
![png](output_21_177.png)
    



    
![png](output_21_178.png)
    



    
![png](output_21_179.png)
    


    Epoch 19: Step 9000: Generator (U-Net) loss: 13.221205992698662, Discriminator loss: 0.5512313174456356
    


    
![png](output_21_181.png)
    



    
![png](output_21_182.png)
    



    
![png](output_21_183.png)
    


    Epoch 19: Step 9200: Generator (U-Net) loss: 12.780668587684643, Discriminator loss: 0.553384855017066
    


    
![png](output_21_185.png)
    



    
![png](output_21_186.png)
    



    
![png](output_21_187.png)
    


    Epoch 20: Step 9400: Generator (U-Net) loss: 12.672384309768676, Discriminator loss: 0.5340735790133477
    


    
![png](output_21_189.png)
    



    
![png](output_21_190.png)
    



    
![png](output_21_191.png)
    


    Epoch 20: Step 9600: Generator (U-Net) loss: 13.009267811775207, Discriminator loss: 0.6000304725766186
    


    
![png](output_21_193.png)
    



    
![png](output_21_194.png)
    



    
![png](output_21_195.png)
    


    Epoch 21: Step 9800: Generator (U-Net) loss: 12.715570578575129, Discriminator loss: 0.5476747442036869
    


    
![png](output_21_197.png)
    



    
![png](output_21_198.png)
    



    
![png](output_21_199.png)
    


    Epoch 21: Step 10000: Generator (U-Net) loss: 12.55386628150941, Discriminator loss: 0.5232586279883981
    


    
![png](output_21_201.png)
    



    
![png](output_21_202.png)
    



    
![png](output_21_203.png)
    


    Epoch 21: Step 10200: Generator (U-Net) loss: 12.833216331005103, Discriminator loss: 0.6019930065423255
    


    
![png](output_21_205.png)
    



    
![png](output_21_206.png)
    



    
![png](output_21_207.png)
    


    Epoch 22: Step 10400: Generator (U-Net) loss: 12.255184249877935, Discriminator loss: 0.5748726470023391
    


    
![png](output_21_209.png)
    



    
![png](output_21_210.png)
    



    
![png](output_21_211.png)
    


    Epoch 22: Step 10600: Generator (U-Net) loss: 12.16266835927963, Discriminator loss: 0.6017312294244763
    


    
![png](output_21_213.png)
    



    
![png](output_21_214.png)
    



    
![png](output_21_215.png)
    


    Epoch 23: Step 10800: Generator (U-Net) loss: 11.873478162288668, Discriminator loss: 0.5276154362410308
    


    
![png](output_21_217.png)
    



    
![png](output_21_218.png)
    



    
![png](output_21_219.png)
    


    Epoch 23: Step 11000: Generator (U-Net) loss: 12.08023518323898, Discriminator loss: 0.5622009706497191
    


    
![png](output_21_221.png)
    



    
![png](output_21_222.png)
    



    
![png](output_21_223.png)
    


    Epoch 24: Step 11200: Generator (U-Net) loss: 12.065670070648196, Discriminator loss: 0.572480154931545
    


    
![png](output_21_225.png)
    



    
![png](output_21_226.png)
    



    
![png](output_21_227.png)
    


    Epoch 24: Step 11400: Generator (U-Net) loss: 12.033194227218626, Discriminator loss: 0.5244859121739863
    


    
![png](output_21_229.png)
    



    
![png](output_21_230.png)
    



    
![png](output_21_231.png)
    


    Epoch 24: Step 11600: Generator (U-Net) loss: 12.01952377557754, Discriminator loss: 0.5497414322197438
    


    
![png](output_21_233.png)
    



    
![png](output_21_234.png)
    



    
![png](output_21_235.png)
    


    Epoch 25: Step 11800: Generator (U-Net) loss: 11.585048260688787, Discriminator loss: 0.5539422848820688
    


    
![png](output_21_237.png)
    



    
![png](output_21_238.png)
    



    
![png](output_21_239.png)
    


    Epoch 25: Step 12000: Generator (U-Net) loss: 11.847832317352301, Discriminator loss: 0.5012211321294309
    


    
![png](output_21_241.png)
    



    
![png](output_21_242.png)
    



    
![png](output_21_243.png)
    


    Epoch 26: Step 12200: Generator (U-Net) loss: 11.87876566410065, Discriminator loss: 0.5430263146758078
    


    
![png](output_21_245.png)
    



    
![png](output_21_246.png)
    



    
![png](output_21_247.png)
    


    Epoch 26: Step 12400: Generator (U-Net) loss: 11.17699061393737, Discriminator loss: 0.5245204541832208
    


    
![png](output_21_249.png)
    



    
![png](output_21_250.png)
    



    
![png](output_21_251.png)
    


    Epoch 27: Step 12600: Generator (U-Net) loss: 11.592132363319404, Discriminator loss: 0.5390707110613586
    


    
![png](output_21_253.png)
    



    
![png](output_21_254.png)
    



    
![png](output_21_255.png)
    


    Epoch 27: Step 12800: Generator (U-Net) loss: 11.470494666099546, Discriminator loss: 0.5289198628440499
    


    
![png](output_21_257.png)
    



    
![png](output_21_258.png)
    



    
![png](output_21_259.png)
    


    Epoch 27: Step 13000: Generator (U-Net) loss: 11.315167849063878, Discriminator loss: 0.5641437524557116
    


    
![png](output_21_261.png)
    



    
![png](output_21_262.png)
    



    
![png](output_21_263.png)
    


    Epoch 28: Step 13200: Generator (U-Net) loss: 11.563940320014959, Discriminator loss: 0.5594559662044046
    


    
![png](output_21_265.png)
    



    
![png](output_21_266.png)
    



    
![png](output_21_267.png)
    


    Epoch 28: Step 13400: Generator (U-Net) loss: 11.338511209487919, Discriminator loss: 0.5244684281200173
    


    
![png](output_21_269.png)
    



    
![png](output_21_270.png)
    



    
![png](output_21_271.png)
    


    Epoch 29: Step 13600: Generator (U-Net) loss: 10.997653491497042, Discriminator loss: 0.5551282133162022
    


    
![png](output_21_273.png)
    



    
![png](output_21_274.png)
    



    
![png](output_21_275.png)
    


    Epoch 29: Step 13800: Generator (U-Net) loss: 10.919450173377987, Discriminator loss: 0.5971719300746916
    


    
![png](output_21_277.png)
    



    
![png](output_21_278.png)
    



    
![png](output_21_279.png)
    


    Epoch 30: Step 14000: Generator (U-Net) loss: 10.948727667331697, Discriminator loss: 0.5408717187494042
    


    
![png](output_21_281.png)
    



    
![png](output_21_282.png)
    



    
![png](output_21_283.png)
    


    Epoch 30: Step 14200: Generator (U-Net) loss: 10.77516280174256, Discriminator loss: 0.544995616450906
    


    
![png](output_21_285.png)
    



    
![png](output_21_286.png)
    



    
![png](output_21_287.png)
    


    Epoch 30: Step 14400: Generator (U-Net) loss: 11.038678736686704, Discriminator loss: 0.5619006096571686
    


    
![png](output_21_289.png)
    



    
![png](output_21_290.png)
    



    
![png](output_21_291.png)
    


    Epoch 31: Step 14600: Generator (U-Net) loss: 10.516819207668307, Discriminator loss: 0.5565231903642418
    


    
![png](output_21_293.png)
    



    
![png](output_21_294.png)
    



    
![png](output_21_295.png)
    


    Epoch 31: Step 14800: Generator (U-Net) loss: 10.490289268493648, Discriminator loss: 0.5651251454651351
    


    
![png](output_21_297.png)
    



    
![png](output_21_298.png)
    



    
![png](output_21_299.png)
    


    Epoch 32: Step 15000: Generator (U-Net) loss: 10.584110949039456, Discriminator loss: 0.5822095587849621
    


    
![png](output_21_301.png)
    



    
![png](output_21_302.png)
    



    
![png](output_21_303.png)
    


    Epoch 32: Step 15200: Generator (U-Net) loss: 10.365930707454682, Discriminator loss: 0.5993096655607226
    


    
![png](output_21_305.png)
    



    
![png](output_21_306.png)
    



    
![png](output_21_307.png)
    


    Epoch 33: Step 15400: Generator (U-Net) loss: 10.105796558856959, Discriminator loss: 0.5619678379595275
    


    
![png](output_21_309.png)
    



    
![png](output_21_310.png)
    



    
![png](output_21_311.png)
    


    Epoch 33: Step 15600: Generator (U-Net) loss: 10.494147965908052, Discriminator loss: 0.5415366851538421
    


    
![png](output_21_313.png)
    



    
![png](output_21_314.png)
    



    
![png](output_21_315.png)
    


    Epoch 33: Step 15800: Generator (U-Net) loss: 10.10233927726746, Discriminator loss: 0.5814128223061563
    


    
![png](output_21_317.png)
    



    
![png](output_21_318.png)
    



    
![png](output_21_319.png)
    


    Epoch 34: Step 16000: Generator (U-Net) loss: 9.661860053539273, Discriminator loss: 0.6183101832866674
    


    
![png](output_21_321.png)
    



    
![png](output_21_322.png)
    



    
![png](output_21_323.png)
    


    Epoch 34: Step 16200: Generator (U-Net) loss: 9.996857042312632, Discriminator loss: 0.5504163563251495
    


    
![png](output_21_325.png)
    



    
![png](output_21_326.png)
    



    
![png](output_21_327.png)
    


    Epoch 35: Step 16400: Generator (U-Net) loss: 9.766125333309173, Discriminator loss: 0.5666019840538503
    


    
![png](output_21_329.png)
    



    
![png](output_21_330.png)
    



    
![png](output_21_331.png)
    


    Epoch 35: Step 16600: Generator (U-Net) loss: 9.898074491024019, Discriminator loss: 0.5808742870390418
    


    
![png](output_21_333.png)
    



    
![png](output_21_334.png)
    



    
![png](output_21_335.png)
    


    Epoch 36: Step 16800: Generator (U-Net) loss: 10.05018318891525, Discriminator loss: 0.5435656126588579
    


    
![png](output_21_337.png)
    



    
![png](output_21_338.png)
    



    
![png](output_21_339.png)
    


    Epoch 36: Step 17000: Generator (U-Net) loss: 9.738629813194281, Discriminator loss: 0.5915006077289581
    


    
![png](output_21_341.png)
    



    
![png](output_21_342.png)
    



    
![png](output_21_343.png)
    


    Epoch 36: Step 17200: Generator (U-Net) loss: 9.734701523780828, Discriminator loss: 0.5810938463360071
    


    
![png](output_21_345.png)
    



    
![png](output_21_346.png)
    



    
![png](output_21_347.png)
    


    Epoch 37: Step 17400: Generator (U-Net) loss: 9.256361927986147, Discriminator loss: 0.596777133792639
    


    
![png](output_21_349.png)
    



    
![png](output_21_350.png)
    



    
![png](output_21_351.png)
    


    Epoch 37: Step 17600: Generator (U-Net) loss: 9.573990066051486, Discriminator loss: 0.5646070724725722
    


    
![png](output_21_353.png)
    



    
![png](output_21_354.png)
    



    
![png](output_21_355.png)
    


    Epoch 38: Step 17800: Generator (U-Net) loss: 9.458319563865661, Discriminator loss: 0.5707471580058334
    


    
![png](output_21_357.png)
    



    
![png](output_21_358.png)
    



    
![png](output_21_359.png)
    


    Epoch 38: Step 18000: Generator (U-Net) loss: 9.510613408088686, Discriminator loss: 0.5644181104749442
    


    
![png](output_21_361.png)
    



    
![png](output_21_362.png)
    



    
![png](output_21_363.png)
    


    Epoch 39: Step 18200: Generator (U-Net) loss: 9.34928073167801, Discriminator loss: 0.5911390881985422
    


    
![png](output_21_365.png)
    



    
![png](output_21_366.png)
    



    
![png](output_21_367.png)
    


    Epoch 39: Step 18400: Generator (U-Net) loss: 9.136145620346067, Discriminator loss: 0.5975072582066063
    


    
![png](output_21_369.png)
    



    
![png](output_21_370.png)
    



    
![png](output_21_371.png)
    


    Epoch 39: Step 18600: Generator (U-Net) loss: 9.004914281368261, Discriminator loss: 0.6259905864298342
    


    
![png](output_21_373.png)
    



    
![png](output_21_374.png)
    



    
![png](output_21_375.png)
    


    Epoch 40: Step 18800: Generator (U-Net) loss: 9.235972428321844, Discriminator loss: 0.6090840595960616
    


    
![png](output_21_377.png)
    



    
![png](output_21_378.png)
    



    
![png](output_21_379.png)
    


    Epoch 40: Step 19000: Generator (U-Net) loss: 8.91376308202744, Discriminator loss: 0.5917609881609674
    


    
![png](output_21_381.png)
    



    
![png](output_21_382.png)
    



    
![png](output_21_383.png)
    


    Epoch 41: Step 19200: Generator (U-Net) loss: 9.268049499988559, Discriminator loss: 0.5506441421806809
    


    
![png](output_21_385.png)
    



    
![png](output_21_386.png)
    



    
![png](output_21_387.png)
    


    Epoch 41: Step 19400: Generator (U-Net) loss: 9.092741999626163, Discriminator loss: 0.584408386349678
    


    
![png](output_21_389.png)
    



    
![png](output_21_390.png)
    



    
![png](output_21_391.png)
    


    Epoch 42: Step 19600: Generator (U-Net) loss: 8.97241967439652, Discriminator loss: 0.5909209805727009
    


    
![png](output_21_393.png)
    



    
![png](output_21_394.png)
    



    
![png](output_21_395.png)
    


    Epoch 42: Step 19800: Generator (U-Net) loss: 8.834741139411923, Discriminator loss: 0.5571738304942846
    


    
![png](output_21_397.png)
    



    
![png](output_21_398.png)
    



    
![png](output_21_399.png)
    


    Epoch 42: Step 20000: Generator (U-Net) loss: 8.683477342128754, Discriminator loss: 0.6295558443665503
    


    
![png](output_21_401.png)
    



    
![png](output_21_402.png)
    



    
![png](output_21_403.png)
    


    Epoch 43: Step 20200: Generator (U-Net) loss: 8.636891114711766, Discriminator loss: 0.6267808697372674
    


    
![png](output_21_405.png)
    



    
![png](output_21_406.png)
    



    
![png](output_21_407.png)
    


    Epoch 43: Step 20400: Generator (U-Net) loss: 8.548097536563874, Discriminator loss: 0.5963472524285319
    


    
![png](output_21_409.png)
    



    
![png](output_21_410.png)
    



    
![png](output_21_411.png)
    


    Epoch 44: Step 20600: Generator (U-Net) loss: 8.740780816078189, Discriminator loss: 0.5823457930237058
    


    
![png](output_21_413.png)
    



    
![png](output_21_414.png)
    



    
![png](output_21_415.png)
    


    Epoch 44: Step 20800: Generator (U-Net) loss: 8.568996372222902, Discriminator loss: 0.5826686953008172
    


    
![png](output_21_417.png)
    



    
![png](output_21_418.png)
    



    
![png](output_21_419.png)
    


    Epoch 45: Step 21000: Generator (U-Net) loss: 8.657251267433166, Discriminator loss: 0.5825450591742993
    


    
![png](output_21_421.png)
    



    
![png](output_21_422.png)
    



    
![png](output_21_423.png)
    


    Epoch 45: Step 21200: Generator (U-Net) loss: 8.360435013771056, Discriminator loss: 0.6020312575250867
    


    
![png](output_21_425.png)
    



    
![png](output_21_426.png)
    



    
![png](output_21_427.png)
    


    Epoch 45: Step 21400: Generator (U-Net) loss: 8.493312695026395, Discriminator loss: 0.5671911882609129
    


    
![png](output_21_429.png)
    



    
![png](output_21_430.png)
    



    
![png](output_21_431.png)
    


    Epoch 46: Step 21600: Generator (U-Net) loss: 8.361584975719454, Discriminator loss: 0.6076765740662813
    


    
![png](output_21_433.png)
    



    
![png](output_21_434.png)
    



    
![png](output_21_435.png)
    


    Epoch 46: Step 21800: Generator (U-Net) loss: 8.473430678844451, Discriminator loss: 0.6104270809888843
    


    
![png](output_21_437.png)
    



    
![png](output_21_438.png)
    



    
![png](output_21_439.png)
    


    Epoch 47: Step 22000: Generator (U-Net) loss: 8.482928881645202, Discriminator loss: 0.5948556883633137
    


    
![png](output_21_441.png)
    



    
![png](output_21_442.png)
    



    
![png](output_21_443.png)
    


    Epoch 47: Step 22200: Generator (U-Net) loss: 8.154896392822268, Discriminator loss: 0.6210964620113376
    


    
![png](output_21_445.png)
    



    
![png](output_21_446.png)
    



    
![png](output_21_447.png)
    


    Epoch 48: Step 22400: Generator (U-Net) loss: 8.453690104484556, Discriminator loss: 0.6007121166586878
    


    
![png](output_21_449.png)
    



    
![png](output_21_450.png)
    



    
![png](output_21_451.png)
    


    Epoch 48: Step 22600: Generator (U-Net) loss: 8.286154940128325, Discriminator loss: 0.6035485128313303
    


    
![png](output_21_453.png)
    



    
![png](output_21_454.png)
    



    
![png](output_21_455.png)
    


    Epoch 48: Step 22800: Generator (U-Net) loss: 8.133513925075532, Discriminator loss: 0.6001232493668794
    


    
![png](output_21_457.png)
    



    
![png](output_21_458.png)
    



    
![png](output_21_459.png)
    


    Epoch 49: Step 23000: Generator (U-Net) loss: 8.435312154293062, Discriminator loss: 0.6366416804492472
    


    
![png](output_21_461.png)
    



    
![png](output_21_462.png)
    



    
![png](output_21_463.png)
    


    Epoch 49: Step 23200: Generator (U-Net) loss: 7.931170160770418, Discriminator loss: 0.6146333284676073
    


    
![png](output_21_465.png)
    



    
![png](output_21_466.png)
    



    
![png](output_21_467.png)
    


    Epoch 50: Step 23400: Generator (U-Net) loss: 8.167942233085634, Discriminator loss: 0.5987573345005511
    


    
![png](output_21_469.png)
    



    
![png](output_21_470.png)
    



    
![png](output_21_471.png)
    


    Epoch 50: Step 23600: Generator (U-Net) loss: 8.144440662860871, Discriminator loss: 0.58312982223928
    


    
![png](output_21_473.png)
    



    
![png](output_21_474.png)
    



    
![png](output_21_475.png)
    


    Epoch 51: Step 23800: Generator (U-Net) loss: 7.981751627922053, Discriminator loss: 0.6259574723988774
    


    
![png](output_21_477.png)
    



    
![png](output_21_478.png)
    



    
![png](output_21_479.png)
    


    Epoch 51: Step 24000: Generator (U-Net) loss: 8.032577598094942, Discriminator loss: 0.5882394541800023
    


    
![png](output_21_481.png)
    



    
![png](output_21_482.png)
    



    
![png](output_21_483.png)
    


    Epoch 51: Step 24200: Generator (U-Net) loss: 7.986506166458131, Discriminator loss: 0.6034535127878191
    


    
![png](output_21_485.png)
    



    
![png](output_21_486.png)
    



    
![png](output_21_487.png)
    


    Epoch 52: Step 24400: Generator (U-Net) loss: 7.898453428745268, Discriminator loss: 0.6116383324563504
    


    
![png](output_21_489.png)
    



    
![png](output_21_490.png)
    



    
![png](output_21_491.png)
    


    Epoch 52: Step 24600: Generator (U-Net) loss: 7.982549576759339, Discriminator loss: 0.6078894490003586
    


    
![png](output_21_493.png)
    



    
![png](output_21_494.png)
    



    
![png](output_21_495.png)
    


    Epoch 53: Step 24800: Generator (U-Net) loss: 7.840874252319342, Discriminator loss: 0.6013284813612699
    


    
![png](output_21_497.png)
    



    
![png](output_21_498.png)
    



    
![png](output_21_499.png)
    


    Epoch 53: Step 25000: Generator (U-Net) loss: 7.8039536070823665, Discriminator loss: 0.5806936464458708
    


    
![png](output_21_501.png)
    



    
![png](output_21_502.png)
    



    
![png](output_21_503.png)
    


    Epoch 54: Step 25200: Generator (U-Net) loss: 7.643192324638368, Discriminator loss: 0.5882720114290719
    


    
![png](output_21_505.png)
    



    
![png](output_21_506.png)
    



    
![png](output_21_507.png)
    


    Epoch 54: Step 25400: Generator (U-Net) loss: 7.722957003116606, Discriminator loss: 0.6218619191646576
    


    
![png](output_21_509.png)
    



    
![png](output_21_510.png)
    



    
![png](output_21_511.png)
    


    Epoch 54: Step 25600: Generator (U-Net) loss: 7.543678371906287, Discriminator loss: 0.6441167706251144
    


    
![png](output_21_513.png)
    



    
![png](output_21_514.png)
    



    
![png](output_21_515.png)
    


    Epoch 55: Step 25800: Generator (U-Net) loss: 7.7378231263160755, Discriminator loss: 0.5903166390210391
    


    
![png](output_21_517.png)
    



    
![png](output_21_518.png)
    



    
![png](output_21_519.png)
    


    Epoch 55: Step 26000: Generator (U-Net) loss: 7.579036321640016, Discriminator loss: 0.644935392588377
    


    
![png](output_21_521.png)
    



    
![png](output_21_522.png)
    



    
![png](output_21_523.png)
    


    Epoch 56: Step 26200: Generator (U-Net) loss: 7.531041111946104, Discriminator loss: 0.6087161214649677
    


    
![png](output_21_525.png)
    



    
![png](output_21_526.png)
    



    
![png](output_21_527.png)
    


    Epoch 56: Step 26400: Generator (U-Net) loss: 7.614216177463534, Discriminator loss: 0.6210830762982369
    


    
![png](output_21_529.png)
    



    
![png](output_21_530.png)
    



    
![png](output_21_531.png)
    


    Epoch 57: Step 26600: Generator (U-Net) loss: 7.418335602283475, Discriminator loss: 0.623887641429901
    


    
![png](output_21_533.png)
    



    
![png](output_21_534.png)
    



    
![png](output_21_535.png)
    


    Epoch 57: Step 26800: Generator (U-Net) loss: 7.413139448165891, Discriminator loss: 0.6195812321454283
    


    
![png](output_21_537.png)
    



    
![png](output_21_538.png)
    



    
![png](output_21_539.png)
    


    Epoch 57: Step 27000: Generator (U-Net) loss: 7.5577192306518555, Discriminator loss: 0.5993185633420943
    


    
![png](output_21_541.png)
    



    
![png](output_21_542.png)
    



    
![png](output_21_543.png)
    


    Epoch 58: Step 27200: Generator (U-Net) loss: 7.453517360687256, Discriminator loss: 0.6276462447643278
    


    
![png](output_21_545.png)
    



    
![png](output_21_546.png)
    



    
![png](output_21_547.png)
    


    Epoch 58: Step 27400: Generator (U-Net) loss: 7.246000390052795, Discriminator loss: 0.604410396143794
    


    
![png](output_21_549.png)
    



    
![png](output_21_550.png)
    



    
![png](output_21_551.png)
    


    Epoch 59: Step 27600: Generator (U-Net) loss: 7.388180525302884, Discriminator loss: 0.6215704169869426
    


    
![png](output_21_553.png)
    



    
![png](output_21_554.png)
    



    
![png](output_21_555.png)
    


    Epoch 59: Step 27800: Generator (U-Net) loss: 7.225581369400022, Discriminator loss: 0.6246555268019436
    


    
![png](output_21_557.png)
    



    
![png](output_21_558.png)
    



    
![png](output_21_559.png)
    


    Epoch 60: Step 28000: Generator (U-Net) loss: 7.437454600334169, Discriminator loss: 0.6196156658232208
    


    
![png](output_21_561.png)
    



    
![png](output_21_562.png)
    



    
![png](output_21_563.png)
    


    Epoch 60: Step 28200: Generator (U-Net) loss: 7.230567984580992, Discriminator loss: 0.6210672846436501
    


    
![png](output_21_565.png)
    



    
![png](output_21_566.png)
    



    
![png](output_21_567.png)
    


    Epoch 60: Step 28400: Generator (U-Net) loss: 7.450877540111537, Discriminator loss: 0.6399757929891348
    


    
![png](output_21_569.png)
    



    
![png](output_21_570.png)
    



    
![png](output_21_571.png)
    


    Epoch 61: Step 28600: Generator (U-Net) loss: 7.2039382362365725, Discriminator loss: 0.6419419240951538
    


    
![png](output_21_573.png)
    



    
![png](output_21_574.png)
    



    
![png](output_21_575.png)
    


    Epoch 61: Step 28800: Generator (U-Net) loss: 7.173999564647674, Discriminator loss: 0.6197565411031246
    


    
![png](output_21_577.png)
    



    
![png](output_21_578.png)
    



    
![png](output_21_579.png)
    


    Epoch 62: Step 29000: Generator (U-Net) loss: 7.101286325454714, Discriminator loss: 0.6064659295976166
    


    
![png](output_21_581.png)
    



    
![png](output_21_582.png)
    



    
![png](output_21_583.png)
    


    Epoch 62: Step 29200: Generator (U-Net) loss: 7.131996757984159, Discriminator loss: 0.6203482546657324
    


    
![png](output_21_585.png)
    



    
![png](output_21_586.png)
    



    
![png](output_21_587.png)
    


    Epoch 63: Step 29400: Generator (U-Net) loss: 7.265620002746581, Discriminator loss: 0.6186644597351548
    


    
![png](output_21_589.png)
    



    
![png](output_21_590.png)
    



    
![png](output_21_591.png)
    


    Epoch 63: Step 29600: Generator (U-Net) loss: 6.9575718522071845, Discriminator loss: 0.631886478215456
    


    
![png](output_21_593.png)
    



    
![png](output_21_594.png)
    



    
![png](output_21_595.png)
    


    Epoch 63: Step 29800: Generator (U-Net) loss: 7.163659732341764, Discriminator loss: 0.6153849122673273
    


    
![png](output_21_597.png)
    



    
![png](output_21_598.png)
    



    
![png](output_21_599.png)
    


    Epoch 64: Step 30000: Generator (U-Net) loss: 6.9297455811500575, Discriminator loss: 0.6341720418632032
    


    
![png](output_21_601.png)
    



    
![png](output_21_602.png)
    



    
![png](output_21_603.png)
    


    Epoch 64: Step 30200: Generator (U-Net) loss: 6.917034459114073, Discriminator loss: 0.623916437327862
    


    
![png](output_21_605.png)
    



    
![png](output_21_606.png)
    



    
![png](output_21_607.png)
    


    Epoch 65: Step 30400: Generator (U-Net) loss: 6.860779623985289, Discriminator loss: 0.6243686932325365
    


    
![png](output_21_609.png)
    



    
![png](output_21_610.png)
    



    
![png](output_21_611.png)
    


    Epoch 65: Step 30600: Generator (U-Net) loss: 6.793545405864719, Discriminator loss: 0.637522247955203
    


    
![png](output_21_613.png)
    



    
![png](output_21_614.png)
    



    
![png](output_21_615.png)
    


    Epoch 66: Step 30800: Generator (U-Net) loss: 7.016542189121245, Discriminator loss: 0.623966294229031
    


    
![png](output_21_617.png)
    



    
![png](output_21_618.png)
    



    
![png](output_21_619.png)
    


    Epoch 66: Step 31000: Generator (U-Net) loss: 6.7345853638649, Discriminator loss: 0.6175943006575105
    


    
![png](output_21_621.png)
    



    
![png](output_21_622.png)
    



    
![png](output_21_623.png)
    



```python

```

