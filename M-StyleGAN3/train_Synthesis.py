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
parser.add_argument('--FR_loss', metavar='<FR_loss>', type= str, default='ArcFace',
                    help='ArcFace/ElasticFace (same model as FR_loss in whitebox and a different proxy model in blackbox attacks)')
parser.add_argument('--epoch_to_resume', metavar='<epoch_to_resume>', type= int, default=20,
                    help='Epoch to resume')
args = parser.parse_args()

import os, sys

sys.path.append(os.getcwd())  # import src
sys.path.append(args.path_stylegan_repo)  # import stylegan files

import pickle
import torch
import torch_utils

import random
import numpy as np
import cv2

seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
batch_size=2
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
print("************ NOTE: The torch device is:", device)

# =================== import Dataset ======================
from src.Dataset import MyDataset
from torch.utils.data import DataLoader

training_dataset = MyDataset(dataset_dir=args.path_ffhq_dataset, FR_system=args.FR_system, train=True, device=device)
testing_dataset = MyDataset(dataset_dir=args.path_ffhq_dataset, FR_system=args.FR_system, train=False, device=device)

train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
# ========================================================

# =================== import Network =====================
path_stylegan = args.path_stylegan_checkpoint

# with open(path_stylegan, 'rb') as f:
#     StyleGAN = pickle.load(f)['G_ema']
#     StyleGAN.to(device)
#     StyleGAN.train()
#     StyleGAN_synthesis = StyleGAN.synthesis
#     # StyleGAN_mapping = StyleGAN.mapping
#     StyleGAN_synthesis.train()
#     # StyleGAN_mapping.eval()

# from src.Network import Discriminator, MappingNetwork
import sys
sys.path.append('/home/ljh/teststylegan/mystylegan/BFIRGB')
from model_blocks import DGWGAN160
import torch.nn as nn
model_Discriminator = DGWGAN160(in_dim=3, dim=1024)
model_Discriminator.to(device)

sys.path.append('/home/gank/bob.paper.neurips2023_face_ti/stylegan3/training')
from networks_stylegan3 import SynthesisNetwork
StyleGAN_synthesis=SynthesisNetwork(w_dim=512,img_resolution=1024, img_channels=3 )
StyleGAN_synthesis.to(device)
StyleGAN_synthesis.train()
# model_Generator = MappingNetwork(z_dim=8,  # Input latent (Z) dimensionality.
#                                  c_dim=512,  # Conditioning label (C) dimensionality, 0 = no labels.
#                                  w_dim=512,  # Intermediate latent (W) dimensionality.
#                                  num_ws=16,  # Number of intermediate latents to output.
#                                  num_layers=2,  # Number of mapping layers.
#                                  )
# model_Generator.to(device)
# z_dim_Generator = model_Generator.z_dim
# z_dim_StyleGAN = StyleGAN_mapping.z_dim
# ========================================================

# =================== import Loss ========================
# ***** ID_loss
from src.loss.FaceIDLoss import ID_Loss

ID_loss = ID_Loss(FR_loss=args.FR_loss, device=device)

# ***** Other losses
Pixel_loss = torch.nn.MSELoss()
# ========================================================


# =================== Optimizers =========================
# ***** optimizer_Generator
for param in StyleGAN_synthesis.parameters():
    param.requires_grad = True
for param in model_Discriminator.parameters():
    param.requires_grad = True

# ***** optimizer_Generator
optimizer_Synthesis = torch.optim.Adam(StyleGAN_synthesis.parameters(), lr=1e-1)
scheduler_Synthesis = torch.optim.lr_scheduler.StepLR(optimizer_Synthesis, step_size=3, gamma=0.5)
# optimizer1_Generator = torch.optim.Adam(model_Generator.parameters(), lr=1e-1)
# scheduler1_Generator = torch.optim.lr_scheduler.StepLR(optimizer1_Generator, step_size=3, gamma=0.5)
# 
# optimizer2_Generator = torch.optim.Adam(model_Generator.parameters(), lr=1e-1)
# scheduler2_Generator = torch.optim.lr_scheduler.StepLR(optimizer2_Generator, step_size=3, gamma=0.5)
# ***** optimizer_Discriminator
optimizer_Discriminator = torch.optim.Adam(model_Discriminator.parameters(), lr=1e-1)
scheduler_Discriminator = torch.optim.lr_scheduler.StepLR(optimizer_Discriminator, step_size=3, gamma=0.5)
# ========================================================


# =================== Save models and logs ===============
import os

os.makedirs('training_files_Synthesis2', exist_ok=True)
os.makedirs('training_files_Synthesis2/models', exist_ok=True)
os.makedirs('training_files_Synthesis2/Generated_images', exist_ok=True)
os.makedirs('training_files_Synthesis2/logs_train', exist_ok=True)

# with open('training_files_Synthesis2/logs_train/generator.csv', 'w') as f:
#     f.write("epoch,Pixel_loss_Gen,ID_loss_Gen,total_loss,score_Disc_real,score_Disc_fake\n")

with open('training_files_Synthesis2/logs_train/log.txt', 'w') as f:
    pass

for embedding, real_image, real_image_HQ in test_dataloader:
    pass

real_image = real_image.cpu()
real_image_HQ = real_image_HQ.cpu()
for i in range(real_image.size(0)):
    os.makedirs(f'training_files_Synthesis2/Generated_images/{i}', exist_ok=True)

    img = real_image[i].squeeze()
    im = (img.numpy().transpose(1, 2, 0) * 255).astype(int)
    cv2.imwrite(f'training_files_Synthesis2/Generated_images/{i}/real_image_cropped.jpg',
                np.array([im[:, :, 2], im[:, :, 1], im[:, :, 0]]).transpose(1, 2, 0))

    img = real_image_HQ[i].squeeze()
    im = (img.numpy().transpose(1, 2, 0) * 255).astype(int)
    cv2.imwrite(f'training_files_Synthesis2/Generated_images/{i}/real_image_HQ.jpg',
                np.array([im[:, :, 2], im[:, :, 1], im[:, :, 0]]).transpose(1, 2, 0))
# ========================================================
epoch_to_resume = args.epoch_to_resume  # 你想要从哪个epoch开始恢复训练，这个值你需要提前知道
if epoch_to_resume>0:
    StyleGAN_synthesis.load_state_dict(torch.load('training_files_Synthesis2/models/Synthesis_{}.pth'.format(epoch_to_resume)))
    model_Discriminator.load_state_dict(torch.load('training_files_Synthesis2/models/Discriminator_{}.pth'.format(epoch_to_resume)))
#=================== Train ==============================
num_epochs=40-epoch_to_resume
# =================== Train ==============================
for epoch in range(num_epochs):
    iteration = 0

    print(f'epoch: {epoch+epoch_to_resume}, \t learning rate: {optimizer_Synthesis.param_groups[0]["lr"]}')

    for embedding, real_image, real_image_HQ in train_dataloader:

        if iteration % 4 == 0:
            """
            ******************* GAN: Update Discriminator *******************
            """
            StyleGAN_synthesis.eval()
            model_Discriminator.train()
            img_fake=StyleGAN_synthesis(embedding.unsqueeze(1).repeat([1,16,1]))
            img_fake = torch.clamp(img_fake, min=-1, max=1)
            img_fake = (img_fake + 1) / 2.0  # range: (0,1)
            # img_fake = img_fake.transpose(0, 2, 1)
            optimizer_Discriminator.zero_grad()
            label = torch.full((batch_size,), 1, dtype=img_fake.dtype, device=device)
            real_prob = nn.Sigmoid()(model_Discriminator(real_image_HQ))
            loss3_real = nn.BCELoss()(real_prob, label)
            loss3_real.backward()
            label.fill_(0)
            fake_prob = nn.Sigmoid()(model_Discriminator(img_fake))
            loss3_fake = nn.BCELoss()(fake_prob, label)
            loss3_fake.backward()
            optimizer_Discriminator.step()
            for param in model_Discriminator.parameters():
                param.data.clamp_(-0.01, 0.01)
        StyleGAN_synthesis.train()
        model_Discriminator.eval()
        """
        ******************* Train Generator *******************
        """
        # ==================forward==================
        generated_image = StyleGAN_synthesis(embedding.unsqueeze(1).repeat([1,16,1]))

        ID = ID_loss(generated_image, real_image)
        Pixel = Pixel_loss((torch.clamp(generated_image, min=-1, max=1) + 1) / 2.0, real_image_HQ)

        loss_train_Generator = Pixel + ID

        # ==================backward=================
        optimizer_Synthesis.zero_grad()
        loss_train_Generator.backward()  # (retain_graph=True)
        optimizer_Synthesis.step()

        # ==================log======================
        iteration += 1
        if iteration % 200 == 0:
            with open('training_files_Synthesis2/logs_train/log.txt', 'a') as f:
                f.write(
                    f'epoch:{epoch + 1+epoch_to_resume}, \t iteration: {iteration}, \t loss_train_Generator:{loss_train_Generator.data.item()}, \t loss_real_Discriminator:{loss3_real.data.item()}, \t loss_fake_Generator:{loss3_fake.data.item()}\n')
            pass

    # ******************** Eval Genrator ********************
    StyleGAN_synthesis.eval()
    model_Discriminator.eval()
    # ID_loss_Gen_test = Pixel_loss_Gen_test = total_loss_Gen_test = score_Discriminator_fake_test = score_Discriminator_real_test = 0
    # iteration =0
    for embedding, real_image, real_image_HQ in test_dataloader:
        pass
    #     iteration +=1
    #     # ==================forward==================
    #     with torch.no_grad():
    #         noise = torch.randn(embedding.size(0), z_dim_Generator, device=device)
    #         w = model_Generator(z=noise, c=embedding)
    #         generated_image = StyleGAN_synthesis(w)
    #         ID  = ID_loss(generated_image, real_image)
    #         Pixel = Pixel_loss(( torch.clamp(generated_image, min=-1, max=1) + 1) / 2.0 ,real_image_HQ)
    #
    #         total_loss_Generator = Pixel + ID
    #         ####
    #         ID_loss_Gen_test  += ID.item()
    #         Pixel_loss_Gen_test += Pixel.item()
    #         total_loss_Gen_test += total_loss_Generator.item()
    #
    #         # Eval Discriminator (GAN)
    #         output_discriminator_fake  = model_Discriminator(w)
    #         score_Discriminator_fake_test   += output_discriminator_fake.mean().item()
    #
    #         noise = torch.randn(embedding.size(0), z_dim_StyleGAN, device=device)
    #         w_real = StyleGAN_mapping(z=noise, c=None).detach()
    #         output_discriminator_real = model_Discriminator(w_real)
    #         score_Discriminator_real_test  += output_discriminator_real.mean().item()
    #
    # with open('training_files_Synthesis2/logs_train/generator.csv','a') as f:
    #     f.write(f"{epoch+1}, {Pixel_loss_Gen_test/iteration}, {ID_loss_Gen_test/iteration}, {total_loss_Gen_test/iteration}, {score_Discriminator_real_test/iteration},{score_Discriminator_fake_test/iteration}\n")
    #
    # noise = torch.randn(embedding.size(0), z_dim_Generator, device=device)
    # w = model_Generator(z=noise, c=embedding)
    generated_image = StyleGAN_synthesis(embedding.unsqueeze(1).repeat([1,16,1])).detach()
    for i in range(generated_image.size(0)):
        img = generated_image[i].squeeze()
        img =  (torch.clamp(img, min=-1, max=1) + 1) / 2.0
        im = (img.cpu().numpy().transpose(1,2,0))
        im = (im * 255).astype(int)
        cv2.imwrite(f'training_files_Synthesis2/Generated_images/{i}/epoch_{epoch+1+epoch_to_resume}.jpg',np.array([im[:,:,2],im[:,:,1],im[:,:,0]]).transpose(1,2,0))
    # *******************************************************

    # Save model_Generator
    torch.save(StyleGAN_synthesis.state_dict(), 'training_files_Synthesis2/models/Synthesis_{}.pth'.format(epoch + 1+epoch_to_resume))
    torch.save(model_Discriminator.state_dict(), 'training_files_Synthesis2/models/Discriminator_{}.pth'.format(epoch + 1+epoch_to_resume))

    # Update schedulers
    scheduler_Synthesis.step()
    scheduler_Discriminator.step()
