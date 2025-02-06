from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision.transforms import Resize


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x


class Classifier(nn.Module):
    def __init__(self, nc, ndf, nz):
        super(Classifier, self).__init__()

        self.nc = nc  # number of channels
        self.ndf = ndf
        self.nz = nz

        self.encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 3, 1, 1),
            nn.BatchNorm2d(ndf),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*8) x 4 x 4
        )

        self.fc = nn.Sequential(
            nn.Linear(ndf * 8 * 4 * 4, nz * 5),
            nn.Dropout(0.5),
            nn.Linear(nz * 5, nz),
        )

    def forward(self, x, release=False):

        x = x.view(-1, 1, 64, 64)
        x = self.encoder(x)
        x = x.view(-1, self.ndf * 8 * 4 * 4)
        x = self.fc(x)

        if release:
            return F.softmax(x, dim=1)
        else:
            return F.log_softmax(x, dim=1)


class Inversion(nn.Module):
    def __init__(self, nc, ngf, nz, truncation, c):
        super(Inversion, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.nz = nz
        self.truncation = truncation
        self.c = c

        self.decoder = nn.Sequential(
            # input is Z
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=nz, out_channels=ngf * 32,
                        kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 32),
            nn.Tanh(),
            # state size. (ngf*16) x 4 x 4
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 32, ngf * 16, 3, 1, 1),
            nn.BatchNorm2d(ngf * 16),
            nn.Tanh(),
            # state size. (ngf*8) x 8 x 8
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 16, ngf * 8, 3, 1, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.Tanh(),
            # state size. (ngf*4) x 16 x 16
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),
            # state size. (ngf*2) x 64 x 64
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 64 x 64
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf * 1, 3, 1, 1),
            nn.BatchNorm2d(ngf * 1),
            nn.Tanh(),
            # state size. (ngf) x 64 x 64
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 1, nc, 3, 1, 1),
            nn.Sigmoid()
            # state size. (nc) x 128 x 128
        )

    def forward(self, x):
        x = x.view(-1, self.nz, 1, 1)
        x = self.decoder(x.float())
        return x

class InversionBLUR(nn.Module):
    def __init__(self, nc, ngf, nz, truncation, c):
        super(Inversion, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.nz = nz
        self.truncation = truncation
        self.c = c

        self.decoder = nn.Sequential(
               # input is Z
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 32), nn.Tanh(),
               # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 32, ngf * 16, 3, 1, 1), nn.BatchNorm2d(ngf * 16), nn.Tanh(),
               # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 3, 1, 1), nn.BatchNorm2d(ngf * 8), nn.Tanh(),
               # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 1, 1), nn.BatchNorm2d(ngf * 4), nn.Tanh(),
               # state size. (ngf*2) x 64 x 64
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 1, 1), nn.BatchNorm2d(ngf * 2), nn.Tanh(),
               # state size. (ngf*2) x 64 x 64
            nn.ConvTranspose2d(ngf * 2, ngf * 1, 3, 1, 1), nn.BatchNorm2d(ngf * 1), nn.Tanh(),
               # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf * 1, nc, 3, 1, 1),
            nn.Sigmoid()
               # state size. (nc) x 128 x 128
            )

    def forward(self, x):
        x = x.view(-1, self.nz, 1, 1)
        x = self.decoder(x.float())
        return x


class DGWGAN(nn.Module):
    def __init__(self, in_dim=3, dim=64):
        super(DGWGAN, self).__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                #nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2),
            nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            nn.Conv2d(dim*2, 1, 32),
            #conv_ln_lrelu(dim * 4, dim * 8),
            #nn.Conv2d(dim * 8, 1, 4),
            nn.Sigmoid())

    def forward(self, x):
        y = self.ls(x.float())
        y = y.view(-1)
        return y

class DGWGAN160(nn.Module):
    def __init__(self, in_dim=3, dim=64):
        super(DGWGAN160, self).__init__()

        self.torch_resize = Resize([128, 128])  # 定义Resize类对象

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                #nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2),
            nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            nn.Conv2d(dim*2, 1, 32),
            #conv_ln_lrelu(dim * 4, dim * 8),
            #nn.Conv2d(dim * 8, 1, 4),
            nn.Sigmoid())

    def forward(self, x):
        x = self.torch_resize(x.float())
        y = self.ls(x.float())
        y = y.view(-1)
        return y



class DGWGAN2(nn.Module):
    def __init__(self, in_dim=3, dim=64):
        super(DGWGAN2, self).__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                #nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2),
            nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            #nn.Conv2d(dim*2, 1, 32),
            #conv_ln_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 8),
            nn.Sigmoid())

    def forward(self, x):
        y = self.ls(x.float())
        y = y.view(-1)
        return y

class VGG16(nn.Module):
    def __init__(self, n_classes):
        super(VGG16, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes

        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)
        self.fc_layer = nn.Linear(
            self.feat_dim, self.n_classes)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)

        return [feature, res]

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return out
