from datetime import datetime
import torch
import gc
from src.utils.config import *
from torch import nn
from src.utils.show_tensor import *
from src.utils.crop import *
from tqdm.auto import tqdm

n_epochs = 200
lambda_recon = 200
display_step = 200
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
        
def get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon):
    '''

    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator; takes the condition and returns potential images
        disc: the discriminator; takes images and the condition and
          returns real/fake prediction matrices
       
        adv_criterion: the adversarial loss function; takes the discriminator 
                  predictions and the true labels and returns a adversarial 
                  loss (which you aim to minimize)
        recon_criterion: the reconstruction loss function; takes the generator 
                    outputs and the real images and returns a reconstructuion 
                    loss (which you aim to minimize)
        lambda_recon: the degree to which the reconstruction loss should be weighted in the sum
    '''

    fake = gen(condition)
    disc_ = disc(fake, condition)
    adv_loss = adv_criterion(disc_, torch.ones_like(disc_))
    gen_rec_loss = recon_criterion(fake, real)
    gen_loss = adv_loss + lambda_recon*gen_rec_loss
    return gen_loss


disc_loss_list = []
gen_loss_list = []
def train_pix2pix(model, optimizer, criterion,
    dataloader, input_dim, label_dim, 
    target_shape, device, save_model_path="",
    pretrained_model_path = ""): 
    # pretrained = False
    gen = model['gen']
    disc = model['disc']
    gen_opt = optimizer['gen_opt']
    disc_opt = optimizer['disc_opt']
    adv_criterion = criterion['adv_criterion']
    recon_criterion = criterion['recon_criterion']
    if pretrained_model_path != "":
        loaded_state = torch.load(pretrained_model_path)
        gen.load_state_dict(loaded_state["gen"])
        gen_opt.load_state_dict(loaded_state["gen_opt"])
        disc.load_state_dict(loaded_state["disc"])
        disc_opt.load_state_dict(loaded_state["disc_opt"])
    else:
        gen = gen.apply(weights_init)
        disc = disc.apply(weights_init)


    mean_generator_loss = 0
    mean_discriminator_loss = 0
    cur_step = 0
    for epoch in range(n_epochs):
        # Dataloader returns the batches
        gc.collect()
        torch.cuda.empty_cache()
        for image, _ in tqdm(dataloader, disable=True):
            image_width = image.shape[3]
            condition = image[:, :, :, :image_width // 2]
            # noise = torch.randn_like(condition)
            # condition = condition
            condition = nn.functional.interpolate(condition, size=target_shape)
            real = image[:, :, :, image_width // 2:]
            real = nn.functional.interpolate(real, size=target_shape)
            cur_batch_size = len(condition)
            
            condition = condition.to(device)
            real = real.to(device)
            # print(condition.shape)
            ### Update discriminator ###
            disc_opt.zero_grad() # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake = gen(condition)
            # print(fake.shape)
            # print(condition.shape)
            disc_fake_hat = disc(fake.detach(), condition) # Detach generator
            # print(disc_fake_hat.shape)
            # print(type(disc_fake_hat))
            # print(adv_criterion)
            disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
            disc_real_hat = disc(real, condition)
            disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True) # Update gradients
            disc_opt.step() # Update optimizer
       
            ### Update generator ###
            gen_opt.zero_grad()
            # gc.collect()
            # torch.cuda.empty_cache()
            gen_loss = get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon)

            gen_loss.backward() # Update gradients
            gen_opt.step() # Update optimizer
            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step
            disc_loss_list.append(disc_loss.detach().cpu().numpy())
            gen_loss_list.append(gen_loss.detach().cpu().numpy())
            gc.collect()
            torch.cuda.empty_cache()
         
            ### Visualization code ###
            # print(curr)
            if cur_step % display_step*3 == 0:
                if cur_step > 0:
                    print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                else:
                    print("Pretrained initial state")
                show_tensor_images(real, size=(input_dim, target_shape, target_shape), label = "real images")
                show_tensor_images(condition, size=(input_dim, target_shape, target_shape), label = "Vessel tree images")
                show_tensor_images(fake, size=(input_dim, target_shape, target_shape), label = "Fake generated images")
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1
            del real
            del condition
            del disc_fake_hat
            del disc_fake_loss
            del disc_loss
   
                # You can change save_model to True if you'd like to save the model
        if save_model_path != "" and mean_generator_loss < 3 and epoch > 20:
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y%-H:%M:%S")
            torch.save({'gen': gen.state_dict(),  
                 'gen_opt': gen_opt.state_dict(),                    
                'disc': disc.state_dict(),
                'disc_opt': disc_opt.state_dict()
            }, model_path+"/pix2pix_version_"+dt_string+".pth")