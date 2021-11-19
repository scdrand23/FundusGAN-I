import numpy as np
import os
import cv2
from src.utils.save_tensor_image import *
def parse_data(datadir):
    for root,_, filenames in os.walk(datadir):  #root: median/1
        img_list = []      
        for filename in filenames:               
                filei = os.path.join(root, filename)
                img_list.append(str(filei))                
        return img_list

def inference(model, label_dim, input_path, output_path, device):
    img_list = parse_data(input_path)
    for real in  img_list:
        s = str(real[-(real[::-1].find("/")):-4])
        img = cv2.imread(real)
        resized_img = cv2.resize(img,(512, 512))
        resized_rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(resized_rgb_img, cv2.COLOR_BGR2GRAY)    
        volume = torch.Tensor(gray_img)[None, None,: , :] / 255
        volume = volume.to(device)
        pred = model(volume)
        save_tensor_image(pred,path=output_path+"/"+s+".jpg", size=(label_dim, 512, 512))   