import numpy as np
import os
import pretty_midi
import shutil
import muspy 
from mido import MidiFile
import numpy as np
import torch 
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, input_nc, batch_size=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_bias=False):
            self.batch_size = batch_size

    def forward(self, input):
          return
    

class Generator(nn.Module):
    def __init__(self, input_nc, batch_size=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_bias=False):
            self.batch_size = batch_size

    def forward(self, input):
        return input

class Classifier(nn.Module):
    def __init__(self, dim=64):
          self.cla = dim

    def forward(self, input):
          return input
    

class CycleGAN(nn.Module):
        def __init__(self, sigma=0.01, sample_size=50, lamb=10, mode='train'):
              super(CycleGAN, self).__init__()
              assert mode in ["train", "A2B", "B2A"]
              self.G_A2B = Generator()
              self.G_B2A = Generator()
              self.D_A = Discriminator()
              self.D_B = Discriminator()
              self.D_A_all = Discriminator()
              self.D_B_all = Discriminator()
              self.l2loss = nn.MSELoss(reduction="mean")
              self.mode = mode
        def forward(self, real_A, real_B, x_m):
              fake_B = self.G_A2B(real_A)
              cycle_A = self.G_B2A(fake_B)

              fake_A = self.G_B2A(real_B)
              cycle_B = self.G_A2B(fake_A)
              if self.mode == "train":
                    [sample_fake_A, sample_fake_B] = self.sampler([fake_A, fake_B])
                    gauss_noise = nn.ones_like(real_A)

                    DA_real = self.D_A(real_A + gauss_noise)
                    DB_real = self.D_B(real_B + gauss_noise)

                    DA_fake = self.D_A(fake_A + gauss_noise)
                    DB_fake = self.D_B(fake_B + gauss_noise)

                    DA_fake_sample = self.D_A(sample_fake_A + gauss_noise)
                    DB_fake_sample = self.D_B(sample_fake_B + gauss_noise)

                    DA_real_all = self.D_A_all(x_m + gauss_noise)
                    DB_real_all = self.D_B_all(x_m + gauss_noise)

                    DA_fake_all = self.D_A_all(sample_fake_A + gauss_noise)
                    DB_fake_all = self.D_B_all(sample_fake_B + gauss_noise)   

                    #https://github.com/Asthestarsfalll/Symbolic-Music-Genre-Transfer-with-CycleGAN-for-pytorch/blob/main/model.py      