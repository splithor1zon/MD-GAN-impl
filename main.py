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

parser.add_argument('--g_model', type=str, default='cnn-v2', help='cnn-v1 | <cnn-v2> | cnn-v3 | mlp')
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

parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing at client [0.0, 0.5) <0.0>')
parser.add_argument('--n_clients', type=int, default=10, help='number of clients <10>')
parser.add_argument('--n_rounds', type=int, default=5000, help='number of complete rounds <5000>')
parser.add_argument('--sample_images_each', type=int, default=10, help='save generated image samples every n rounds (0 = never) <100>')
parser.add_argument('--save_model_each', type=int, default=500, help='save model checkpoints every n rounds (0 = never) <500>')
parser.add_argument('--client_sample_rate', type=float, default=1.0, help='proportion of clients sampled each round <1.0>')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device to use <cuda if available else cpu>')

parser.add_argument('-f', '--file', help='IPython file argument (ignore)', default='')
args = parser.parse_args()

# Dataset
trainset, testset, data_shape, data_mean, data_std = datatools.load_dataset(args.dataset)
args.n_classes, args.data_shape, args.data_mean, args.data_std = len(trainset.classes), data_shape, data_mean, data_std

denormalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(data_mean, data_std)],
    std=[1.0/s for s in data_std])

# TODO: magic numbers
args.d_dropout = 0.5
args.d_relu_slope = 0.2

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
        self.one_vector = torch.full((args.g_batch_size, 1), 1.0 - args.label_smoothing, device=args.device)
        self.zero_vector = torch.full((args.g_batch_size, 1), args.label_smoothing, device=args.device)

    def train(self, gen_data_d, gen_labels_d, gen_data_g, gen_labels_g):
        self.netD.train()
        gen_data_d, gen_labels_d = gen_data_d.to(self.args.device, non_blocking=True), gen_labels_d.to(self.args.device, non_blocking=True)
        gen_data_g, gen_labels_g = gen_data_g.to(self.args.device, non_blocking=True), gen_labels_g.to(self.args.device, non_blocking=True)
        gen_data_g.requires_grad_()
        # Train discriminator
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
        # print(f"{self.name} - Real D loss: {real_loss.item():.4f}, Generated D loss: {gen_loss_d.item():.4f}, Total D loss: {loss_d.item():.4f}, G Loss: {gen_loss_g.item():.4f}")
        return error_feedback.detach(), loss_d.item(), gen_loss_g.item()
    
    def get_model_state(self):
        return deepcopy(self.netD.state_dict())
    
    def set_model_state(self, state_dict):
        self.netD.load_state_dict(deepcopy(state_dict))
        self.netD.to(self.args.device)

    def set_optim(self, optim):
        self.optimD = optim
    
# Server-side code

# Initialize clients
clients: List[Client] = []
for i, bucket in enumerate(partition_buckets):
    client_name = f"Client_{i+1:03d}"
    client_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset, bucket),
        batch_size=args.g_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True)
    clients.append(Client(args, client_name, client_dataloader))
    print(f"Init {client_name}: {len(bucket)} samples")

# Initialize generator
netG = models.select_generator(args).to(args.device)
netG.apply(weights_init_normal)
optimG = torch.optim.Adam(netG.parameters(), lr=args.g_lr, betas=(args.g_beta1, args.g_beta2))

# Fixed noise for image sampling
fixed_noise = torch.randn(10*10, args.g_latent_dim, device=args.device)
fixed_labels = torch.arange(0, 10, device=args.device).repeat(10)
out_dir = f"./output_{int(time.time())}"
os.makedirs(out_dir, exist_ok=True)
print(f"Output directory: {out_dir}")

np_rng = np.random.default_rng()

# Training loop
for server_round in range(1, args.n_rounds + 1):
    # print(f"\n--- Round {server_round} ---")
    optimG.zero_grad()

    # Generate data for clients
    gen_data_batches = []
    for _ in range(args.g_batch_num):
        noise = torch.randn(args.g_batch_size, args.g_latent_dim, device=args.device)
        labels = torch.randint(0, args.n_classes, (args.g_batch_size,), device=args.device)
        gen_data = netG(noise, labels)
        gen_data_batches.append((gen_data, labels))

    # Select clients
    n_sampled_clients = max(1, int(args.client_sample_rate * args.n_clients))
    sampled_clients = np_rng.choice(clients, n_sampled_clients, replace=False)
    batch_assignments = [(i % args.g_batch_num, (i+1) % args.g_batch_num) for i in range(n_sampled_clients)]
    d_loss_cum, g_loss_cum = 0.0, 0.0
    for client, (batch_d_idx, batch_g_idx) in zip(sampled_clients, batch_assignments):
        gen_data_d, gen_labels_d = gen_data_batches[batch_d_idx]
        gen_data_g, gen_labels_g = gen_data_batches[batch_g_idx]
        # Train discriminator and get feedback for generator
        error_feedback, d_loss, g_loss = client.train(
            gen_data_d.clone().detach(),
            gen_labels_d,
            gen_data_g.clone().detach(),
            gen_labels_g)
        d_loss_cum += d_loss
        g_loss_cum += g_loss
        
        # Update generator
        gen_data_g.backward(error_feedback, retain_graph=True)
    print(f"Round {server_round} - Avg Discriminator loss: {d_loss_cum / n_sampled_clients:.4f}, Avg Generator loss: {g_loss_cum / n_sampled_clients:.4f}")   
    optimG.step()

    # Swap discriminators
    if args.d_swap_each > 0 and server_round % args.d_swap_each == 0:
        shuffled_clients = np_rng.permutation(clients)
        half = len(clients) // 2
        swap_pairs = []
        for i in range(half):
            c1, c2 = shuffled_clients[i], shuffled_clients[i + half]
            c1_state, c2_state = c1.get_model_state(), c2.get_model_state()
            c1.set_model_state(c2_state)
            c2.set_model_state(c1_state)
            # TODO: Swap or init new optimizers?
            c1_optim = torch.optim.Adam(c1.netD.parameters(), lr=args.d_lr, betas=(args.d_beta1, args.d_beta2))
            c2_optim = torch.optim.Adam(c2.netD.parameters(), lr=args.d_lr, betas=(args.d_beta1, args.d_beta2))
            c1.set_optim(c1_optim)
            c2.set_optim(c2_optim)
            swap_pairs.append((c1.name, c2.name))
        print(f"Swapped discriminators between pairs: {swap_pairs}")

    # Save generated image samples
    if args.sample_images_each > 0 and server_round % args.sample_images_each == 0:
        with torch.no_grad():
            sampled_images = netG(fixed_noise, fixed_labels).cpu()
        sampled_images = denormalize(sampled_images)
        sampled_images = torch.clamp(sampled_images, 0.0, 1.0)
        grid = torchvision.utils.make_grid(sampled_images, nrow=10, padding=2)
        torchvision.utils.save_image(grid, os.path.join(out_dir, f"round_{server_round}.png"))
