import torch
import torchvision
import torchvision.transforms as transforms

from places365classes import places365_classes
from tiny_img import download_tinyImg200

import os


def load_data(name, img_size,
              batch_size=4, num_workers=2,
              train_size=None, test_size=None):
    """Load a dataset given its name.

    Parameters
    ----------
    name : [str]
        One of ['CIFAR10', 'MNIST', 'Places365', 'ImageNet200']

    img_size : tuple(int)
        (width: int, height: int), dimensions of images in the dataloader
    batch_size : int, optional
        _description_, by default 4
    num_workers : int, optional
        _description_, by default 2
    train_size : _type_, optional
        _description_, by default None
    test_size : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    TypeError
        _description_
    ValueError
        _description_
    """
    if not isinstance(name, str):
        raise TypeError('Name of dataset should be str. '
                        f'Got {type(name)} instead.')
    name = name.lower()
    if name not in [
        'cifar10', 'mnist', 'places365', 'imagenet200'
    ]:
        raise ValueError(f'Unsupported dataset: {name}')

    if name == 'cifar10':
        return cifar10(
            img_size=img_size,
            batch_size=batch_size, num_workers=num_workers,
            train_size=train_size, test_size=test_size
        )
    elif name == 'mnist':
        return mnist(
            img_size=img_size,
            batch_size=batch_size, num_workers=num_workers,
            train_size=train_size, test_size=test_size
        )
    elif name == 'places365':
        return places365(
            img_size=img_size,
            batch_size=batch_size, num_workers=num_workers,
            train_size=train_size, test_size=test_size
        )
    elif name == 'imagenet200':
        return imagenet200(
            img_size=img_size,
            batch_size=batch_size, num_workers=num_workers
        )
    return 1


def cifar10(
    img_size=(32, 32),
    batch_size=4, num_workers=2,
    train_size=None,
    test_size=None
    ):
    # TODO
    # parametrize: output path for torchvision.datasets.CIFAR10(root=path)
    # why are train transforms not the same as test transforms?

    # Prepare train loader
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root='../data', train=True,
        download=True, transform=train_transform
    )
    if train_size is not None:
        # Randomly sample a subset of the training set
        indices = torch.randperm(len(trainset))[:train_size]
        trainset = torch.utils.data.Subset(trainset, indices)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )

    # Why are these transforms different?
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    testset = torchvision.datasets.CIFAR10(
        root='../data', train=False,
        download=True, transform=test_transform
    )
    if test_size is not None:
        # Randomly sample a subset of the test set
        indices = torch.randperm(len(testset))[:test_size]
        testset = torch.utils.data.Subset(testset, indices)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )

    classes = (
        'plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse',
        'ship', 'truck'
    )
    return trainloader, testloader, classes


def mnist(
    img_size=(28, 28),
    batch_size=4, num_workers=2,
    train_size=None, test_size=None
    ):
    # TODO
    # parametrize: '../data'
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB by replicating channels
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.Normalize((0.5,), (0.5,))
        ]
    )

    trainset = torchvision.datasets.MNIST(
        root='../data', train=True,
        download=True, transform=train_transform
    )
    if train_size is not None:
        # Randomly sample a subset of the training set
        indices = torch.randperm(len(trainset))[:train_size]
        trainset = torch.utils.data.Subset(trainset, indices)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.Normalize((0.5,), (0.5,))
        ]
    )
    testset = torchvision.datasets.MNIST(
        root='../data', train=False,
        download=True, transform=test_transform
    )
    if test_size is not None:
        # Randomly sample a subset of the test set
        indices = torch.randperm(len(testset))[:test_size]
        testset = torch.utils.data.Subset(testset, indices)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    
    return trainloader, testloader, classes


def places365(
    img_size=(64, 64),
    batch_size=4, num_workers=2,
    train_size=100000, test_size=25000
    ):
    # TODO
    # parametrize: (64, 64)
    # parametrize: data path
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Resize((256, 256)),
            transforms.Resize(img_size), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(
                (64, 64),
                scale=(0.8, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            )
        ]
    )

    trainset = torchvision.datasets.Places365(
        root='./data', split='train-standard',
        # split='val',
        small= True, transform = train_transform, download= True
    )
    
    if train_size is not None:
        # Randomly sample a subset of the training set
        indices = torch.randperm(len(trainset))[:train_size]
        trainset = torch.utils.data.Subset(trainset, indices)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )


    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            )
        ]
    )

    testset = torchvision.datasets.Places365(
        root='./data', split='val', 
        small= True, transform = test_transform, download= True
    )
    if test_size is not None:
        # Randomly sample a subset of the test set
        indices = torch.randperm(len(testset))[:test_size]
        testset = torch.utils.data.Subset(testset, indices)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )

    classes = places365_classes
    
    return trainloader, testloader, classes


def imagenet200(img_size=(64, 64), batch_size=4, num_workers=2):

    if not os.path.exists('./tiny-imagenet-200/'):
        download_tinyImg200('.')

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(img_size), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ]
    )
            
    train_dataset = torchvision.datasets.ImageFolder(
        'tiny-imagenet-200/train', transform=train_transform
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Resize((224, 224)),
            transforms.Resize(img_size),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ]
    )
    
    test_dataset = torchvision.datasets.ImageFolder(
        'tiny-imagenet-200/val', transform=test_transform
    )

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )

    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )

    classes = list(range(0, 200))
    
    return trainloader, testloader, classes
