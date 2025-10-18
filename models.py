import torch
import torch.nn as nn
import numpy as np

class Gen_MLP(nn.Module):
    """
    Simple MLP-based Generator for MNIST
    Architecture: 3 fully-connected layers with 512, 512, and 784 neurons
    Total parameters: 716,560 (latent_dim=100)

    Modeled after the model proposed in the MD-GAN paper
    """
    def __init__(self, latent_dim=100, num_classes=10, img_dim=(1, 28, 28)):
        super(Gen_MLP, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.label_emb = nn.Embedding(num_classes, latent_dim)

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(512, np.prod(img_dim)),
            nn.Tanh()
        )
        
    def forward(self, noise, labels):
        # Element-wise multiplication of noise and label embedding
        x = torch.mul(self.label_emb(labels), noise)
        x = self.model(x)
        x = x.view(x.size(0), *self.img_dim)
        return x
    
class Dis_MLP_ACGAN(nn.Module):
    """
    MLP-based Discriminator for MNIST with ACGAN output
    Architecture: 3 fully-connected layers with 512, 512, and 11 neurons
    Total parameters: 670,219
    Output: 1 neuron for real/fake, 10 neurons for class prediction

    Modeled after the model proposed in the MD-GAN paper
    """
    def __init__(self, num_classes=10, img_dim=(1, 28, 28)):
        super(Dis_MLP_ACGAN, self).__init__()
        self.num_classes = num_classes
        self.img_dim = img_dim

        self.model = nn.Sequential(
            nn.Linear(np.prod(img_dim), 512),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.fc_adv = nn.Linear(512, 1)
        self.fc_aux = nn.Linear(512, num_classes)
        
    def forward(self, input):
        # Flatten the input image
        x = input.view(input.size(0), -1)
        x = self.model(x)
        adversrial = self.fc_adv(x)
        auxiliary = self.fc_aux(x)
        return adversrial, auxiliary

class Gen_CNN_MNIST_v1(nn.Module):
    """
    Simple CNN-based Generator for MNIST.
    Modeled after the model proposed in the MD-GAN paper, parameter count does not match exactly (mistake in paper?).
    Built using transposed convolution layers.
    DOES NOT WORK WELL - CHECKERBOARD ARTIFACTS!

    Architecture: 1 FC layer (6,272 neurons) + 2 transposed conv layers (32 and 1 kernels, 5x5)
    Total parameters: 736,769 (latent_dim=100)
    """
    def __init__(self, latent_dim=100):
        super(Gen_CNN_MNIST_v1, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = 10
        self.img_dim = (1, 28, 28)
        self.label_emb = nn.Embedding(self.num_classes, latent_dim)
        
        # Initial size after FC layer: 7x7x128 = 6272
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)
        self.conv1 = nn.Sequential(
            # 7x7x128 -> 14x14x32
            nn.ConvTranspose2d(128, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            # 14x14x32 -> 28x28x1
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )
        
    def forward(self, noise, labels):
        x = torch.mul(self.label_emb(labels), noise)
        x = self.fc(x)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Gen_CNN_MNIST_v2(nn.Module):
    """
    CNN-based Generator for MNIST.
    Based on Gen_CNN_MNIST_1.
    Improved version using upsampling+convolution to reduce checkerboard artifacts.
    """
    def __init__(self, latent_dim=100):
        super(Gen_CNN_MNIST_v2, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = 10
        self.img_dim = (1, 28, 28)
        self.label_emb = nn.Embedding(self.num_classes, latent_dim)
        
        # Initial size after FC layer: 7x7x128 = 6272
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)
        self.conv1 = nn.Sequential(
            # 7x7x128 -> 14x14x32
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            # 14x14x32 -> 28x28x1
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Element-wise multiplication of noise and label embedding
        x = torch.mul(self.label_emb(labels), noise)
        x = self.fc(x)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Gen_CNN_MNIST_v3(nn.Module):
    """
    CNN-based Generator for MNIST.
    More complex architecture with 3 convolution layers.
    """
    def __init__(self, latent_dim=100):
        super(Gen_CNN_MNIST_v3, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = 10
        self.img_dim = (1, 28, 28)
        self.label_emb = nn.Embedding(self.num_classes, latent_dim)

        # Initial size after FC layer: 7x7x128 = 6272
        self.fc = nn.Linear(self.latent_dim, 7 * 7 * 128)
        self.conv1 = nn.Sequential(
            # 7x7x128 -> 14x14x128
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            # 14x14x128 -> 28x28x64
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            # 28x28x64 -> 28x28x1
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.mul(self.label_emb(labels), noise)
        x = self.fc(x)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class Gen_CNN_CIFAR10(nn.Module):
    """
    CNN-based Generator for CIFAR10
    Architecture: 1 FC layer (6,144 neurons) + 3 transposed conv layers (192, 96, 3 kernels, 5x5)
    Total parameters: 2,932,035 (latent_dim=100)

    Modeled after the model proposed in the MD-GAN paper, parameter count does not match exactly
    """
    def __init__(self, latent_dim=100, num_classes=10, img_dim=(3, 32, 32)):
        super(Gen_CNN_CIFAR10, self).__init__()
        assert img_dim == (3, 32, 32), "CIFAR10 images must have dimensions (3, 32, 32)"
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        # Initial size after FC layer: 4x4x384 = 6144
        self.fc = nn.Linear(latent_dim, 4 * 4 * 384)
        
        self.conv_layers = nn.Sequential(
            # 4x4x384 -> 8x8x192
            nn.ConvTranspose2d(384, 192, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            # 8x8x192 -> 16x16x96
            nn.ConvTranspose2d(192, 96, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            # 16x16x96 -> 32x32x3
            nn.ConvTranspose2d(96, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )
        
    def forward(self, noise, labels):
        # Element-wise multiplication of noise and label embedding
        x = torch.mul(self.label_emb(labels), noise)
        x = self.fc(x)
        x = x.view(x.size(0), 384, 4, 4)
        x = self.conv_layers(x)
        return x

class MinibatchDiscrimination(nn.Module):
    """
    Minibatch discrimination layer as described in the paper "Improved Techniques for Training GANs"
    (https://arxiv.org/abs/1606.03498)
    """
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super(MinibatchDiscrimination, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        nn.init.normal_(self.T, 0, 1)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)   # NxB, subtract 1 to exclude self
        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1)
        return x

class Dis_CNN_ACGAN(nn.Module):
    """
    Unified CNN-based Discriminator for both MNIST and CIFAR10 with ACGAN output
    Architecture: 6 conv layers (16, 32, 64, 128, 256, 512 kernels, 3x3) + minibatch discrimination + FC layer (11 neurons)
    
    """
    def __init__(self, num_classes=10, img_dim=(1, 28, 28), dropout=0.5, relu_slope=0.2, use_minibatch_disc=True):
        super(Dis_CNN_ACGAN, self).__init__()
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.dropout = dropout
        self.relu_slope = relu_slope
        self.use_minibatch_disc = use_minibatch_disc
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv layer
            nn.Conv2d(img_dim[0], 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Dropout2d(dropout),
            
            # Second conv layer
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Dropout2d(dropout),
            
            # Third conv layer
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Dropout2d(dropout),
            
            # Fourth conv layer
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Dropout2d(dropout),
            
            # Fifth conv layer
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Dropout2d(dropout),
            
            # Sixth conv layer
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Dropout2d(dropout),
        )
        # Calculate the size after conv layers
        if img_dim == (1, 28, 28):  # MNIST
            conv_out_size = 512 * 4 * 4  # After 3 stride-2 layers: 28->14->7->4
        elif img_dim == (3, 32, 32):  # CIFAR10
            conv_out_size = 512 * 4 * 4  # After 3 stride-2 layers: 32->16->8->4
        else:
            raise ValueError(f"Unsupported image dimensions: {img_dim}")
        
        # Minibatch discrimination layer
        if self.use_minibatch_disc:
            self.minibatch_disc = MinibatchDiscrimination(conv_out_size, 50, 5)
            fc_input_size = conv_out_size + 50
        else:
            fc_input_size = conv_out_size
            
        
        self.fc_adv = nn.Linear(fc_input_size, 1)
        self.fc_aux = nn.Linear(fc_input_size, num_classes)
        
    def forward(self, input):
        x = self.conv_layers(input)
        x = x.view(x.size(0), -1)  # Flatten
        
        if self.use_minibatch_disc:
            x = self.minibatch_disc(x)
            
        adversarial = self.fc_adv(x)  # Real/Fake output
        auxiliary = self.fc_aux(x)  # Class prediction output
        return adversarial, auxiliary

class Dis_CNN_MNIST_1(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5, relu_slope=0.2):
        super(Dis_CNN_MNIST_1, self).__init__()
        self.num_classes = num_classes
        self.color_ch = 1
        self.img_size = 32
        self.dropout = dropout
        self.relu_slope = relu_slope

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(self.relu_slope, inplace=True), nn.Dropout2d(self.dropout)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(self.color_ch, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = self.img_size // 2 ** 4

        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)
        self.aux_layer = nn.Linear(128 * ds_size ** 2, self.num_classes)
    
    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

def select_discriminator(args):
    match (args.d_model, args.dataset):
        case ('mlp', 'mnist' | 'fmnist'):
            netD = Dis_MLP_ACGAN(args.n_classes, args.data_shape)
        case ('cnn', 'mnist' | 'fmnist' | 'cifar10'):
            netD = Dis_CNN_ACGAN(args.n_classes, args.data_shape, args.d_dropout, args.d_relu_slope)
        case ('cnn1', 'mnist' | 'fmnist'):
            netD = Dis_CNN_MNIST_1(args.n_classes, args.d_dropout, args.d_relu_slope)
        case _:
            raise ValueError(f'Discriminator model {args.d_model} with dataset {args.dataset} not supported')
    return netD

def select_generator(args):
    match (args.g_model.lower(), args.dataset.lower()):
        case ('mlp', 'mnist' | 'fmnist' | 'cifar10'):
            netG = Gen_MLP(args.g_latent_dim, args.n_classes, args.data_shape)
        case ('cnn-v1', 'mnist' | 'fmnist'):
            netG = Gen_CNN_MNIST_v1(args.g_latent_dim)
        case ('cnn-v2', 'mnist' | 'fmnist'):
            netG = Gen_CNN_MNIST_v2(args.g_latent_dim)
        case ('cnn-v3', 'mnist' | 'fmnist'):
            netG = Gen_CNN_MNIST_v3(args.g_latent_dim)
        case ('cnn', 'cifar10'):
            netG = Gen_CNN_CIFAR10(args.g_latent_dim, args.n_classes, args.data_shape)
        case _:
            raise ValueError(f'Generator model {args.g_model} with dataset {args.dataset} not supported')
    return netG