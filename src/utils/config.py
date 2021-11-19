from torch import nn
import torch
datapath_cfg = {
                  'train_input_path':'/content/drive/MyDrive/RP/Summer21/dataset/newi.tiff',
                  'train_label_path':'/content/drive/MyDrive/RP/Summer21/dataset/newL.tiff',
                  'validation_input_path':'',
                  'validation_label_path':'',
                  'test_input_path':'/root/gdrive/MyDrive/preprocessed_images/',                
                  'test_output_path':'/root/gdrive/MyDrive/RP/S21/fake vessel trees/fake_vessel_trees',
                }
cfg = {
    'lr': 0.0002,
    'criterion': nn.BCEWithLogitsLoss(),
    'input_dim': 1,
    'output_dim': 1,
    'display_step': 2000,
    'batch_size': 4,
    'initial_shape': 512,
    'target_shape': 512,
    'device': 'cuda',
   }

pix2pix_config ={
                  'adv_criterion':nn.BCEWithLogitsLoss(),
                  'recon_criterion':nn.L1Loss(),
                  'lambda_recon':200,
                  'n_epochs':500,
                  'input_dim':3,
                  'real_dim':3,
                  'display_step':200,
                  'batch_size':4,
                  'lr':0.0002,
                  'target_shape':512,
                  'device':'cuda'
                  }