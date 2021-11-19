from src.utils.show_tensor import *
from src.utils.crop import *
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.nn import functional as F
from datetime import datetime

def dice_loss(input, target):
    smooth = 1.
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()    
    return 1 - ((2. * intersection + smooth)/(iflat.sum() + tflat.sum() + smooth))

unet_loss_list = []
n_epochs = 2000
display_step = 200
def train(model, optimizer, criterion, train_dataloader, input_dim, label_dim, target_shape, device, save_dir = ""): 
    cur_step = 0   
    for epoch in range(n_epochs):
        for real, labels in tqdm(train_dataloader, disable=True):
            cur_batch_size = len(real)
            # Flatten the image
            
            real = real.to(device)
            labels = labels.to(device)
            ### Update U-Net ###
            optimizer.zero_grad()
            pred = model(real)
            unet_loss = criterion(pred, labels)
            dice_l = dice_loss(pred,labels)
            # print(dice_l)
            
            total_loss = unet_loss + dice_l
            # print(total_loss)
            unet_loss_arr = total_loss.detach().cpu().numpy()
            unet_loss_list.append(unet_loss_arr)
            total_loss.backward()
            optimizer.step()
        
            if cur_step % display_step == 5:
              print(f"Epoch {epoch}: Step {cur_step}: U-Net loss: {unet_loss}") 
              show_tensor_images(crop(real, torch.Size([len(real), 1, target_shape, target_shape])), size=(input_dim, target_shape, target_shape))      
              show_tensor_images(labels, size=(label_dim, target_shape, target_shape))
              show_tensor_images(torch.sigmoid(pred), size=(label_dim, target_shape, target_shape))       
                      
            cur_step += 1
          
        if save_dir != "" and epoch%99 == 0:
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y%-H:%M:%S")
            torch.save({'unet': model.state_dict(),  
                 'opt': optimizer.state_dict(),                    
            }, save_dir+"/unet_v_"+dt_string+".pth")
           

    
