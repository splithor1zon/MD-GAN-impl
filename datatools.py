import torchvision.transforms as tvt
import torchvision.datasets as tvd
import numpy as np

def load_dataset(name, data_dir = "./datasets", normalize=True):
    transform_list = []
    #transform_list.append(tvt.Resize((32, 32)))  # Resize all images to 32x32 for consistency
    transform_list.append(tvt.ToTensor())
    match name.lower():
        case 'mnist':
            # mean, std = (0.1307,), (0.3081,)
            mean, std = (0.5,), (0.5,) # Normalize to [-1, 1] for tanh activation
            shape = (1, 28, 28)
            if normalize:
                transform_list.append(tvt.Normalize(mean, std))
            transform = tvt.Compose(transform_list)
            trainset = tvd.MNIST(data_dir, train=True, download=True, transform=transform)
            testset = tvd.MNIST(data_dir, train=False, download=True, transform=transform)
        case 'fmnist' | 'fashionmnist':
            # mean, std = (0.2860,), (0.3530,)
            mean, std = (0.5,), (0.5,) # Normalize to [-1, 1] for tanh activation
            shape = (1, 28, 28)
            if normalize:
                transform_list.append(tvt.Normalize(mean, std))
            transform = tvt.Compose(transform_list)
            trainset = tvd.FashionMNIST(data_dir, train=True, download=True, transform=transform)
            testset = tvd.FashionMNIST(data_dir, train=False, download=True, transform=transform)
        case 'cifar10':
            # mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) # Normalize to [-1, 1] for tanh activation
            shape = (3, 32, 32)
            if normalize:
                transform_list.append(tvt.Normalize(mean, std))
            transform = tvt.Compose(transform_list)
            trainset = tvd.CIFAR10(data_dir, train=True, download=True, transform=transform)
            testset = tvd.CIFAR10(data_dir, train=False, download=True, transform=transform)
        case _:
            raise ValueError(f"Dataset '{name}' is not supported.")
    return trainset, testset, shape, mean, std

def iid_partition(args, dataset):
    partition_buckets = [[] for _ in range(args.n_clients)]
    indices = list(range(len(dataset)))

    rng = np.random.default_rng()
    rng.shuffle(indices)
    
    for i in range(args.n_clients):
        partition_buckets[i] = indices[i::args.n_clients]
    return partition_buckets