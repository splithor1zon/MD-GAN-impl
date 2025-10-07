"""
%run main.py -h
%run main.py
"""

# Imports
import os
import time
from copy import deepcopy
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

import datatools
import models

# Arguments
parser = argparse.ArgumentParser(prog='MD-GAN implementation', description='Default values in <> brackets')
parser.add_argument('--dataset', type=str, default='mnist', help='<mnist> | fmnist | cifar10')
parser.add_argument('--data_distribution', type=str, default='iid', help='<iid> | dirichlet_[alpha: float] | sort-part_[shards-per-part: int]')

parser.add_argument('--g_model', type=str, default='cnn', help='<cnn> | mlp')
parser.add_argument('--g_lr', type=float, default=0.001, help='learning rate for generator <0.001>')
parser.add_argument('--g_beta1', type=float, default=0.0, help='beta1 for Adam optimizer of generator <0.0>')
parser.add_argument('--g_beta2', type=float, default=0.9, help='beta2 for Adam optimizer of generator <0.9>')
parser.add_argument('--g_batch_size', type=int, default=100, help='batch size of generated and trained data per round <100>')
parser.add_argument('--g_batch_num', type=int, default=3, help='number of different batches generated each round <3>')
parser.add_argument('--g_latent_dim', type=int, default=100, help='dimensionality of the latent space - input noise <100>')

parser.add_argument('--d_model', type=str, default='cnn', help='<cnn> | mlp')
parser.add_argument('--d_lr', type=float, default=0.004, help='learning rate for discriminator <0.004>')
parser.add_argument('--d_beta1', type=float, default=0.0, help='beta1 for Adam optimizer of discriminator <0.0>')
parser.add_argument('--d_beta2', type=float, default=0.9, help='beta2 for Adam optimizer of discriminator <0.9>')
parser.add_argument('--d_epochs', type=int, default=1, help='number of discriminator epochs per round <1>')
parser.add_argument('--d_swap_each', type=int, default=5, help='swap discriminators every n rounds (0 = never) <5>')
parser.add_argument('--d_real_batch_variety', type=str, default='static', help='<static> | dynamic - same/different real batches each epoch of a training round')


parser.add_argument('--label_smoothing', type=float, default=0.0, help='label smoothing at client [0.0, 0.5) <0.0>')
parser.add_argument('--n_clients', type=int, default=10, help='number of clients <10>')
parser.add_argument('--n_rounds', type=int, default=5000, help='number of complete rounds <5000>')
parser.add_argument('--sample_images_each', type=int, default=100, help='save generated image samples every n rounds (0 = never) <100>')
parser.add_argument('--save_model_each', type=int, default=500, help='save model checkpoints every n rounds (0 = never) <500>')
parser.add_argument('--client_sample_rate', type=float, default=1.0, help='proportion of clients sampled each round <1.0>')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device to use <cuda if available else cpu>')

parser.add_argument('-f', '--file', help='IPython file argument (ignore)', default='')
args = parser.parse_args()

# Dataset
trainset, testset, data_shape, data_mean, data_std = datatools.load_dataset(args.dataset)
args.n_classes, args.data_shape, args.data_mean, args.data_std = len(trainset.classes), data_shape, data_mean, data_std

# Data distribution
match args.data_distribution.split('_'):
    case ['iid']:
        partition_buckets = datatools.iid_partition(args, trainset)
    case ['dirichlet', alpha]:
        alpha = float(alpha)
        partition_buckets = datatools.dirichlet_partition(trainset, args.n_clients, alpha)
    case ['sort-part', shards_per_part]:
        shards_per_part = int(shards_per_part)
        partition_buckets = datatools.sort_partition(trainset, args.n_clients, shards_per_part)
    case _:
        raise ValueError(f"Data distribution '{args.data_distribution}' is not supported.")

# Model initialization functions
def weights_init_normal(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)

# Client class
class Client:
    def __init__(self, args, name, dataloader):
        self.name = name
        self.args = args
        self.dataloader = dataloader
        self.netD = models.select_discriminator(args).to(args.device)
        self.netD.apply(weights_init_normal)
        self.optimD = torch.optim.Adam(self.netD.parameters(), lr=args.d_lr, betas=(args.d_beta1, args.d_beta2))
        self.loss_adv = nn.BCEWithLogitsLoss()
        self.loss_aux = nn.CrossEntropyLoss()
        # Vectors for adversarial ground truths, with label smoothing if defined
        self.one_vector = torch.full((args.g_batch_size,), 1.0 - args.label_smoothing, device=args.device)
        self.zero_vector = torch.full((args.g_batch_size,), args.label_smoothing, device=args.device)

    def train(self, gen_data_d, gen_labels_d, gen_data_g, gen_labels_g):
        self.netD.train()
        gen_data_d, gen_labels_d = gen_data_d.to(self.args.device, non_blocking=True), gen_labels_d.to(self.args.device, non_blocking=True)
        gen_data_g, gen_labels_g = gen_data_g.to(self.args.device, non_blocking=True), gen_labels_g.to(self.args.device, non_blocking=True)
        epochs_left = self.args.d_epochs
        while epochs_left > 0: # Loop if there are less real batches than epochs
            for real_data, real_labels in self.dataloader:
                real_data, real_labels = real_data.to(self.args.device, non_blocking=True), real_labels.to(self.args.device, non_blocking=True)
                while epochs_left > 0:
                    self.optimD.zero_grad()
                    # Real data loss
                    real_adv, real_aux = self.netD(real_data)
                    real_loss = self.loss_adv(real_adv, self.one_vector) + self.loss_aux(real_aux, real_labels)
                    # Generated data loss
                    gen_adv_d, gen_aux_d = self.netD(gen_data_d)
                    gen_loss_d = self.loss_adv(gen_adv_d, self.zero_vector) + self.loss_aux(gen_aux_d, gen_labels_d)
                    # Discriminator update
                    loss_d = real_loss + gen_loss_d
                    loss_d.backward()
                    self.optimD.step()
                    epochs_left -= 1
                    if self.args.d_real_batch_variety == 'dynamic':
                        break  # Use a different real batch next epoch
                if epochs_left <= 0:
                    break  # Completed all epochs
        
        # Prepare generator feedback
        #self.netD.eval()
        gen_adv_g, gen_aux_g = self.netD(gen_data_g)
        gen_loss_g = self.loss_adv(gen_adv_g, self.one_vector) + self.loss_aux(gen_aux_g, gen_labels_g)

        error_feedback = torch.autograd.grad(gen_loss_g, gen_data_g)[0]
        print(f"{self.name} - Real D loss: {real_loss.item():.4f}, Generated D loss: {gen_loss_d.item():.4f}, Total D loss: {loss_d.item():.4f}, G Loss: {gen_loss_g.item():.4f}")
        return error_feedback.detach()
    
    def get_model_state(self):
        return deepcopy(self.netD.state_dict())
    
    def set_model_state(self, state_dict):
        self.netD.load_state_dict(deepcopy(state_dict))
        self.netD.to(self.args.device)
    
# Server-side code

