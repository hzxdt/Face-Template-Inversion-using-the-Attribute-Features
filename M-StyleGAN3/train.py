"""
Training code for the following paper:
    Hatef Otroshi Shahreza and Sébastien Marcel, "Face Reconstruction from Facial Templates by
    Learning Latent Space of a Generator Network", Thirty-seventh Conference on Neural Information 
    Processing Systems (NeurIPS), 2023.

Author: Hatef Otroshi
Institution: Idiap Research Institute, Martigny, Switzerland.

Copyright (C) 2023 Hatef Otroshi

This software is distributed under the terms described in the LICENSE file 
located in the parent directory of this source code repository. 

For inquiries, please contact the author at hatef.otroshi@idiap.ch
"""
import argparse
parser = argparse.ArgumentParser(description='Train face reconstruction network')
parser.add_argument('--path_stylegan_repo', metavar='<path_stylegan_repo>', type= str, default='./stylegan3',
                    help='./sytlegan3')
parser.add_argument('--path_stylegan_checkpoint', metavar='<path_stylegan_checkpoint>', type= str, default='./stylegan3-r-ffhq-1024x1024.pkl',
                    help='./stylegan3-r-ffhq-1024x1024.pkl')
parser.add_argument('--path_ffhq_dataset', metavar='<path_ffhq_dataset>', type= str, default='/home/PublicData/FFHQ',
                    help='FFHQ directory`')
parser.add_argument('--FR_system', metavar='<FR_system>', type= str, default='ArcFace',
                    help='ArcFace/ElasticFace (FR system from whose database the templates are leaked)')
parser.add_argument('--FR_loss', metavar='<FR_loss>', type= str, default='ElasticFace',
                    help='ArcFace/ElasticFace (same model as FR_loss in whitebox and a different proxy model in blackbox attacks)')
parser.add_argument('--epoch_to_resume', metavar='<epoch_to_resume>', type= int, default=0,
                    help='Epoch to resume')
parser.add_argument('--use-checkpoint', metavar='<use_checkpoint>', type= str, default='./checkpoints/ArcFace-ElasticFace_loss.pth')
args = parser.parse_args()


import os,sys
sys.path.append(os.getcwd()) # import src
sys.path.append(args.path_stylegan_repo) # import stylegan files

import pickle
import torch
import torch_utils

import random
import numpy as np
import cv2


seed=2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print("************ NOTE: The torch device is:", device)


#=================== import Dataset ======================
from src.Dataset import MyDataset
from torch.utils.data import DataLoader

training_dataset = MyDataset(dataset_dir=args.path_ffhq_dataset, FR_system= args.FR_system, train=True,  device=device)
testing_dataset  = MyDataset(dataset_dir=args.path_ffhq_dataset, FR_system= args.FR_system, train=False, device=device)

train_dataloader = DataLoader(training_dataset, batch_size=2, shuffle=True)
test_dataloader  = DataLoader(testing_dataset,  batch_size=2, shuffle=False)
#========================================================

#=================== import Network =====================
path_stylegan = args.path_stylegan_checkpoint

with open(path_stylegan, 'rb') as f:
    StyleGAN = pickle.load(f)['G_ema']
    StyleGAN.to(device)
    StyleGAN.eval()
    StyleGAN_synthesis = StyleGAN.synthesis
    StyleGAN_mapping   = StyleGAN.mapping
    StyleGAN_synthesis.eval()
    StyleGAN_mapping.eval()


from src.Network import Discriminator, MappingNetwork 
model_Discriminator = Discriminator()
model_Discriminator.to(device)

model_Generator = MappingNetwork(z_dim = 8,                      # Input latent (Z) dimensionality.
                                 c_dim = 512,                        # Conditioning label (C) dimensionality, 0 = no labels.
                                 w_dim = 512,                      # Intermediate latent (W) dimensionality.
                                 num_ws = 16,                      # Number of intermediate latents to output.
                                 num_layers = 2,                   # Number of mapping layers.
                                 )
model_Generator.to(device)
z_dim_Generator = model_Generator.z_dim
z_dim_StyleGAN = StyleGAN_mapping.z_dim
#========================================================
# print('hhhhhhhhhhhhhhhh')
#=================== import Loss ========================
# ***** ID_loss
from src.loss.FaceIDLoss import ID_Loss
ID_loss = ID_Loss(FR_loss= args.FR_loss, device=device)

# ***** Other losses
Pixel_loss = torch.nn.MSELoss()
#========================================================
sys.path.append('/home/ljh/teststylegan/mystylegan/BFIRGB')
from FaceAttr.FaceAttrSolver import FaceAttrSolver
attr_model = FaceAttrSolver(epoches=100,batch_size=32,learning_rate=1e-2,model_type='Resnet18',optim_type='SGD',momentum=0.9,pretrained=True,loss_type='BCE_loss',exp_version='v7',device=device)
# print('hhhhhhhhhhhhhhh')
import torch.nn as nn
#=================== Optimizers =========================
# ***** optimizer_Generator
for param in model_Generator.parameters():
    param.requires_grad = True

# ***** optimizer_Generator
optimizer1_Generator    = torch.optim.Adam(model_Generator.parameters(), lr=1e-1)
scheduler1_Generator    = torch.optim.lr_scheduler.StepLR(optimizer1_Generator, step_size=3, gamma=0.5)

optimizer2_Generator    = torch.optim.Adam(model_Generator.parameters(), lr=1e-1)
scheduler2_Generator    = torch.optim.lr_scheduler.StepLR(optimizer2_Generator, step_size=3, gamma=0.5)
# ***** optimizer_Discriminator
optimizer_Discriminator = torch.optim.Adam(model_Discriminator.parameters(), lr=1e-1)
scheduler_Discriminator = torch.optim.lr_scheduler.StepLR(optimizer_Discriminator, step_size=3, gamma=0.5)
#========================================================



#=================== Save models and logs ===============
import os
os.makedirs('training_files_attr',exist_ok=True)
os.makedirs('training_files_attr/models',exist_ok=True)
os.makedirs('training_files_attr/Generated_images',exist_ok=True)
os.makedirs('training_files_attr/logs_train',exist_ok=True)

with open('training_files_attr/logs_train/generator.csv','w') as f:
    f.write("epoch,Pixel_loss_Gen,ID_loss_Gen,Attr_loss_Gen,total_loss,score_Disc_real,score_Disc_fake\n")

with open('training_files_attr/logs_train/log.txt','w') as f:
    pass


for embedding, real_image, real_image_HQ in test_dataloader:
    noise = torch.randn(embedding.size(0), z_dim_Generator, device=device)
    w_fake = model_Generator(z=noise, c=embedding).detach()
    pass

real_image = real_image.cpu()
real_image_HQ = real_image_HQ.cpu()
for i in range(real_image.size(0)):
    os.makedirs(f'training_files_attr/Generated_images/{i}', exist_ok=True)
    
    img = real_image[i].squeeze()
    im = (img.numpy().transpose(1,2,0)*255).astype(int)
    cv2.imwrite(f'training_files_attr/Generated_images/{i}/real_image_cropped.jpg',np.array([im[:,:,2],im[:,:,1],im[:,:,0]]).transpose(1,2,0))

    img = real_image_HQ[i].squeeze()
    im = (img.numpy().transpose(1,2,0)*255).astype(int)
    cv2.imwrite(f'training_files_attr/Generated_images/{i}/real_image_HQ.jpg',np.array([im[:,:,2],im[:,:,1],im[:,:,0]]).transpose(1,2,0))
#========================================================
if args.use_checkpoint !='':
    model_Generator.load_state_dict(torch.load(args.use_checkpoint))
epoch_to_resume = args.epoch_to_resume  # 你想要从哪个epoch开始恢复训练，这个值你需要提前知道
if epoch_to_resume>0:
    model_Generator.load_state_dict(torch.load('training_files_attr/models/Generator_{}.pth'.format(epoch_to_resume)))
    model_Discriminator.load_state_dict(torch.load('training_files_attr/models/Discriminator_{}.pth'.format(epoch_to_resume)))
#=================== Train ==============================
num_epochs=20-epoch_to_resume
#=================== Train ==============================
for epoch in range(num_epochs):  
    iteration=0
    
    print(f'epoch: {epoch+epoch_to_resume}, \t learning rate: {optimizer1_Generator.param_groups[0]["lr"]}')
    
    for embedding, real_image, real_image_HQ in train_dataloader:
        
        if iteration % 4 == 0:   
            """
            ******************* GAN: Update Discriminator *******************
            """
            model_Generator.eval()
            model_Discriminator.train()

            # Generate batch of latent vectors
            noise = torch.randn(embedding.size(0), z_dim_Generator, device=device)
            w_fake = model_Generator(z=noise, c=embedding).detach()

            noise = torch.randn(embedding.size(0), z_dim_StyleGAN, device=device)
            w_real = StyleGAN_mapping(z=noise, c=None).detach()
            # ==================forward==================
            # disc should give lower score for real and high for gnerated (fake)
            output_discriminator_real = model_Discriminator(w_real)
            errD_real  = output_discriminator_real

            output_discriminator_fake  = model_Discriminator(w_fake)
            errD_fake  = (-1) * output_discriminator_fake

            loss_GAN_Discriminator = (errD_fake + errD_real).mean()
            # ==================backward=================
            optimizer_Discriminator.zero_grad()
            loss_GAN_Discriminator.backward()
            optimizer_Discriminator.step()

            for param in model_Discriminator.parameters():
                param.data.clamp_(-0.01, 0.01)
        
        
        if iteration % 2 == 0:   
            model_Generator.train()
            model_Discriminator.eval()           
            """
            ******************* GAN: Update Generator *******************
            """
            # Generate batch of latent vectors
            noise = torch.randn(embedding.size(0), z_dim_Generator, device=device)
            w_fake = model_Generator(z=noise, c=embedding)
            # ==================forward==================
            output_discriminator_fake  = model_Discriminator(w_fake)
            loss_GAN_Generator  = output_discriminator_fake.mean()
            # ==================backward=================
            optimizer1_Generator.zero_grad()
            loss_GAN_Generator.backward()
            optimizer1_Generator.step()
    
        model_Generator.train()
        """
        ******************* Train Generator *******************
        """
        # ==================forward==================
        noise = torch.randn(embedding.size(0), z_dim_Generator, device=device)
        w = model_Generator(z=noise, c=embedding)
        generated_image = StyleGAN_synthesis(w)

        ID  = ID_loss(generated_image, real_image)
        Pixel = Pixel_loss( ( torch.clamp(generated_image, min=-1, max=1) + 1) / 2.0 ,real_image_HQ)
        Attr=nn.L1Loss()(attr_model.predict(generated_image), attr_model.predict(real_image_HQ))
        loss_train_Generator = Pixel + ID+Attr*10
        
        # ==================backward=================
        optimizer2_Generator.zero_grad()
        loss_train_Generator.backward()#(retain_graph=True)
        optimizer2_Generator.step()

    

        # ==================log======================
        iteration +=1
        if iteration % 200 == 0:
            with open('training_files_attr/logs_train/log.txt','a') as f:
                f.write(f'epoch:{epoch+1+epoch_to_resume}, \t iteration: {iteration}, \t loss_train_Generator:{loss_train_Generator.data.item()}, \t loss_GAN_Discriminator:{loss_GAN_Discriminator.data.item()}, \t loss_GAN_Generator:{loss_GAN_Generator.data.item()}\n')
            pass
        
    # ******************** Eval Genrator ********************
    model_Generator.eval()
    model_Discriminator.eval()
    ID_loss_Gen_test = Pixel_loss_Gen_test = Attr_loss_Gen_test = total_loss_Gen_test = score_Discriminator_fake_test = score_Discriminator_real_test = 0
    iteration =0
    for embedding, real_image, real_image_HQ in test_dataloader:
        iteration +=1
        # ==================forward==================
        with torch.no_grad():
            noise = torch.randn(embedding.size(0), z_dim_Generator, device=device)
            w = model_Generator(z=noise, c=embedding)
            generated_image = StyleGAN_synthesis(w)
            ID  = ID_loss(generated_image, real_image)
            Pixel = Pixel_loss(( torch.clamp(generated_image, min=-1, max=1) + 1) / 2.0 ,real_image_HQ)
            Attr=nn.L1Loss()(attr_model.predict(generated_image), attr_model.predict(real_image_HQ))
            total_loss_Generator = Pixel + ID+Attr*10
            ####
            ID_loss_Gen_test  += ID.item()
            Pixel_loss_Gen_test += Pixel.item()
            Attr_loss_Gen_test += Attr.item()*10
            total_loss_Gen_test += total_loss_Generator.item()

            # Eval Discriminator (GAN)
            output_discriminator_fake  = model_Discriminator(w)
            score_Discriminator_fake_test   += output_discriminator_fake.mean().item()

            noise = torch.randn(embedding.size(0), z_dim_StyleGAN, device=device)
            w_real = StyleGAN_mapping(z=noise, c=None).detach()
            output_discriminator_real = model_Discriminator(w_real)
            score_Discriminator_real_test  += output_discriminator_real.mean().item()

    with open('training_files_attr/logs_train/generator.csv','a') as f:
        f.write(f"{epoch+1}, {Pixel_loss_Gen_test/iteration}, {ID_loss_Gen_test/iteration}, {Attr_loss_Gen_test/iteration},{total_loss_Gen_test/iteration}, {score_Discriminator_real_test/iteration},{score_Discriminator_fake_test/iteration}\n")

    noise = torch.randn(embedding.size(0), z_dim_Generator, device=device)
    w = model_Generator(z=noise, c=embedding)
    generated_image = StyleGAN_synthesis(w).detach()
    for i in range(generated_image.size(0)):
        img = generated_image[i].squeeze()
        img =  (torch.clamp(img, min=-1, max=1) + 1) / 2.0
        im = (img.cpu().numpy().transpose(1,2,0))
        im = (im * 255).astype(int)
        cv2.imwrite(f'training_files_attr/Generated_images/{i}/epoch_{epoch+1}.jpg',np.array([im[:,:,2],im[:,:,1],im[:,:,0]]).transpose(1,2,0))
    # *******************************************************
    
    # Save model_Generator
    torch.save(model_Generator.state_dict(), 'training_files_attr/models/Generator_{}.pth'.format(epoch+1+epoch_to_resume))
    torch.save(model_Discriminator.state_dict(), 'training_files_attr/models/Discriminator_{}.pth'.format(epoch+1+epoch_to_resume))
    
    # Update schedulers
    scheduler1_Generator.step()
    scheduler2_Generator.step()
    scheduler_Discriminator.step()
