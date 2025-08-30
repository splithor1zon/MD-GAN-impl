#%% Imports
import os
import time
from typing import Dict
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

#%% Config variables
CONFIG = {
    "batch_size": 100,
    "lr_g": 1e-4,
    "lr_d": 4e-4,
    "betas_g": (0.0, 0.9),
    "betas_d": (0.0, 0.9),
    "latent_dim": 100,
    "epochs": 5000,
    "n_clients": 10,
    "client_sample_rate": 1.0,
    "swap_each": 5
}
CONFIG["clients_per_epoch"] = int(CONFIG["n_clients"] * CONFIG["client_sample_rate"])
CONFIG["generated_batches_per_epoch"] = max(2, int(np.log2(CONFIG["n_clients"]) * CONFIG["client_sample_rate"]))
DEVICE = torch.device('cuda:0')
OUT_DIR = 'output'

#%% Convenience functions
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

#%% Dataset preparation
mnist_train = torchvision.datasets.MNIST(
    root=os.path.join(os.path.dirname(__file__), 'datasets'),
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
)

def iid_partition(dataset, num_partitions):
    partition_buckets = [[] for _ in range(num_partitions)]
    indices = list(range(len(dataset)))

    rng = np.random.default_rng()
    rng.shuffle(indices)
    
    for i in range(num_partitions):
        partition_buckets[i] = indices[i::num_partitions]
    return partition_buckets

#%% Model definition
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.n_classes = 10
        self.color_ch = 1
        self.latent_dim = CONFIG["latent_dim"]
        self.img_size = 32

        self.label_emb = nn.Embedding(self.n_classes, self.latent_dim)

        self.init_size = self.img_size // 4
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 128 * self.init_size ** 2)
        )
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, self.color_ch, 3, 1, 1),
            nn.Tanh()
        )
    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.fc(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        n_classes = 10
        color_ch = 1
        img_size = 32

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(color_ch, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = img_size // 2 ** 4

        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)
        self.aux_layer = nn.Linear(128 * ds_size ** 2, n_classes)
    
    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

#%% Client class
class MDGANClient():
    def __init__(self, name, dataloader, discriminator, discriminator_optim, adversarial_loss_fn, auxiliary_loss_fn, device='cpu'):
        self.name = name
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.discriminator = discriminator.to(device)
        self.discriminator_optim = discriminator_optim
        self.adversarial_loss_fn = adversarial_loss_fn
        self.auxiliary_loss_fn = auxiliary_loss_fn
        self.device = device
        # Use label smoothing for better training stability
        self.real_label_value = 0.9
        self.fake_label_value = 0.1
        self.ones_vector = torch.full((self.batch_size, 1), self.real_label_value, dtype=torch.float, device=self.device, requires_grad=False)
        self.zeros_vector = torch.full((self.batch_size, 1), self.fake_label_value, dtype=torch.float, device=self.device, requires_grad=False)
    
    def train(self, data_d, labels_d, data_g, labels_g, epochs=1):
        self.discriminator.train()
        # Prepare generated data for discriminator training
        data_d = data_d.to(self.device, non_blocking=True)
        labels_d = labels_d.to(self.device, non_blocking=True)
        # Prepare generated data for generator training
        data_g = data_g.to(self.device, non_blocking=True).requires_grad_()
        labels_g = labels_g.to(self.device, non_blocking=True)

        # Only use a single batch of real data for training
        for data, labels in self.dataloader:
            real_data_batch = data.to(self.device, non_blocking=True)
            real_labels_batch = labels.to(self.device, non_blocking=True)
            break

        # Discriminator train loop
        self.discriminator.train()  # Ensure discriminator is in training mode
        for epoch in range(epochs):
            # Train discriminator
            self.discriminator_optim.zero_grad()
            
            # Real data loss
            real_adversarial, real_auxiliary = self.discriminator(real_data_batch)
            real_loss = (self.adversarial_loss_fn(real_adversarial, self.ones_vector) +
                         self.auxiliary_loss_fn(real_auxiliary, real_labels_batch))
            
            # Generated data loss
            generated_adversarial_d, generated_auxiliary_d = self.discriminator(data_d)
            generated_loss_d = (self.adversarial_loss_fn(generated_adversarial_d, self.zeros_vector) +
                                    self.auxiliary_loss_fn(generated_auxiliary_d, labels_d))
            
            # Discriminator update
            discriminator_loss = real_loss + generated_loss_d
            discriminator_loss.backward()
            self.discriminator_optim.step()

        # Feedback for generator (separate forward pass)
        self.discriminator.eval()  # Set to eval mode for generator feedback
        generated_adversarial_g, generated_auxiliary_g = self.discriminator(data_g)
        generated_loss_g = (self.adversarial_loss_fn(generated_adversarial_g, self.ones_vector) +
                            self.auxiliary_loss_fn(generated_auxiliary_g, labels_g))
        
        # Compute gradients for generator
        error_feedback = torch.autograd.grad(generated_loss_g, data_g)[0]
        
        print(f"{self.name} - Real D loss: {real_loss.item():.4f}, Generated D loss: {generated_loss_d.item():.4f}, Total D loss: {discriminator_loss.item():.4f}, G Loss: {generated_loss_g.item():.4f}")
        return error_feedback.detach()
    
    def get_model_state(self):
        return deepcopy(self.discriminator.state_dict())

    def set_model_state(self, model_state):
        """Set the model and optimizer state from the provided state dictionaries"""
        self.discriminator.load_state_dict(model_state)
        self.discriminator.to(self.device)

    def set_optimizer(self, optimizer):
        self.discriminator_optim = optimizer

#%% Server logic
adversarial_loss_fn = nn.BCEWithLogitsLoss()
auxiliary_loss_fn = nn.CrossEntropyLoss()

# Initialize server-side components
generator = Generator()
generator.apply(weights_init_normal)
generator.to(DEVICE)
generator_optim = torch.optim.Adam(generator.parameters(), lr=CONFIG["lr_g"], betas=CONFIG["betas_g"])

# Initialize the clients
client_list = []
partition_buckets = iid_partition(mnist_train, CONFIG["n_clients"])
for client_id in range(CONFIG["n_clients"]):
    partition_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(mnist_train, partition_buckets[client_id]),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    model = Discriminator()
    model.apply(weights_init_normal)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr_d"], betas=CONFIG["betas_d"])
    client = MDGANClient(
        name=f"Client_{client_id:03d}",
        dataloader=partition_dataloader,
        discriminator=model,
        discriminator_optim=optimizer,
        adversarial_loss_fn=adversarial_loss_fn,
        auxiliary_loss_fn=auxiliary_loss_fn,
        device=DEVICE
    )
    client_list.append(client)

# Server-side training loop
rng = np.random.default_rng(42)
fixed_noise = torch.randn(10 * 10, CONFIG["latent_dim"], device=DEVICE)
out_dir_timestamped = f"{OUT_DIR}_{int(time.time())}"

for epoch in range(CONFIG["epochs"]):
    print(f"Epoch {epoch + 1}/{CONFIG['epochs']}")

    generator_optim.zero_grad()
    
    # Generate data for this epoch
    generated_batches = []
    gbpe = CONFIG["generated_batches_per_epoch"]
    
    for j_batch in range(gbpe):
        noise_z = torch.randn(CONFIG["batch_size"], CONFIG["latent_dim"], device=DEVICE)
        labels_z = torch.randint(0, 10, (CONFIG["batch_size"],), device=DEVICE)
        generated_batches.append((noise_z, labels_z, generator(noise_z, labels_z)))
    
    # Sample clients for this epoch
    sampled_clients = rng.choice(client_list, size=CONFIG["clients_per_epoch"], replace=False)
    batch_assignments = [(i%gbpe, (i+1)%gbpe) for i in range(CONFIG["clients_per_epoch"])]

    # Prepare tasks
    class Task():
        def __init__(self, client, generated_batch_d, generated_batch_g):
            self.client = client
            self.generated_batch_d = generated_batch_d
            self.generated_batch_g = generated_batch_g

    client_tasks = []
    for i, client in enumerate(sampled_clients):
        task = Task(
            client=client,
            generated_batch_d=generated_batches[batch_assignments[i][0]],
            generated_batch_g=generated_batches[batch_assignments[i][1]]
        )
        client_tasks.append(task)
    
    # Accumulate gradients from all clients
    for task in client_tasks:
        # Train each client with the generated samples
        error_feedback = task.client.train(
            task.generated_batch_d[2].clone().detach(),
            task.generated_batch_d[1],
            task.generated_batch_g[2].clone().detach(),
            task.generated_batch_g[1],
            epochs=1
        )
        
        # Apply the error feedback as gradients
        task.generated_batch_g[2].backward(error_feedback, retain_graph=True)
    
    # Step the generator optimizer
    generator_optim.step()

    # Swap client models in defined intervals
    if (epoch + 1) % CONFIG["swap_each"] == 0:
        # Shuffle client list, divide into two halves and swap models
        shuffled_clients = rng.permutation(client_list)
        half = len(shuffled_clients) // 2
        swap_pairs = []
        for i in range(half):
            client_a = shuffled_clients[i]
            client_b = shuffled_clients[i + half]
            # Swap model states
            model_a_state = client_a.get_model_state()
            model_b_state = client_b.get_model_state()
            client_a.set_model_state(model_b_state)
            client_b.set_model_state(model_a_state)
            # Create new optimizers for both clients
            optimizer_a = torch.optim.Adam(client_a.discriminator.parameters(), lr=CONFIG["lr_d"], betas=CONFIG["betas_d"])
            optimizer_b = torch.optim.Adam(client_b.discriminator.parameters(), lr=CONFIG["lr_d"], betas=CONFIG["betas_d"])
            client_a.set_optimizer(optimizer_a)
            client_b.set_optimizer(optimizer_b)
            swap_pairs.append((client_a.name, client_b.name))
        print(f"Swapped models between the following client pairs: {swap_pairs}")


    # Save generated images for visualization
    if (epoch + 1) % 10 == 0 or epoch == 0:
        if not os.path.exists(out_dir_timestamped):
            os.makedirs(out_dir_timestamped)
        with torch.no_grad():
            fixed_labels = torch.arange(0, 10, device=DEVICE).repeat(10)
            fixed_images = generator(fixed_noise, fixed_labels).cpu()
            fixed_images = fixed_images * 0.5 + 0.5
            torchvision.utils.save_image(fixed_images, os.path.join(out_dir_timestamped, f"generated_epoch_{epoch + 1}.png"), nrow=10, normalize=True)
        
