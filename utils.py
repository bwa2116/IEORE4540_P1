import json, os, math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms

from vit import ViTForClassfication


def save_experiment(experiment_name, config, model, train_losses, test_losses, accuracies, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    
    # Save the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)
    
    # Save the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)
    
    # Save the model
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)


def save_checkpoint(experiment_name, model, epoch, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch}.pt')
    torch.save(model.state_dict(), cpfile)


def load_experiment(experiment_name, checkpoint_name="model_final.pt", base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    # Load the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'r') as f:
        config = json.load(f)
    # Load the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    accuracies = data['accuracies']
    # Load the model
    model = ViTForClassfication(config)
    cpfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile))
    return config, model, train_losses, test_losses, accuracies


def visualize_images(dataset):
    if dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True)
        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    elif dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True)
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    elif dataset == 'Places365':
        from places365classes import places365_classes
        trainset = torchvision.datasets.Places365(root='./data', split='val', 
                                                  small= True, download= True)
        classes = places365_classes

    elif dataset == 'ImageNet200':
        import os
        from tiny_img import download_tinyImg200
        if not os.path.exists('./tiny-imagenet-200/'):
            download_tinyImg200('.')
        trainset = torchvision.datasets.ImageFolder('tiny-imagenet-200/train')

        classes = list(range(0, 200))

    # Pick 30 samples randomly
    indices = torch.randperm(len(trainset))[:30]
    images = [np.asarray(trainset[i][0]) for i in indices]
    labels = [trainset[i][1] for i in indices]
    # Visualize the images using matplotlib
    fig = plt.figure(figsize=(10, 10))
    for i in range(30):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        ax.imshow(images[i])
        ax.set_title(classes[labels[i]])


@torch.no_grad()
def visualize_attention(model, dataset, output=None, device="cuda"):
    """
    Visualize the attention maps of the first 4 images.
    """
    model.eval()
    
    # Load random images
    num_images = 30
    if dataset == 'CIFAR10':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        image_size = (32,32)
        # Pick 30 samples randomly
        indices = torch.randperm(len(testset))[:num_images]
        raw_images = [np.asarray(testset[i][0]) for i in indices]
        labels = [testset[i][1] for i in indices]
        # Convert the images to tensors
        test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        images = torch.stack([test_transform(image) for image in raw_images])
   
    elif dataset == 'MNIST':
        image_size = (32,32)
        testset = torchvision.datasets.MNIST(root='./data',
                                             train=False, download=True)
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        
        # Pick 30 samples randomly
        indices = torch.randperm(len(testset))[:num_images]
        raw_images = [np.asarray(testset[i][0]) for i in indices]
        labels = [testset[i][1] for i in indices]
        # Convert the images to tensors
        test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.Normalize((0.5,), (0.5,))])
        images = torch.stack([test_transform(image) for image in raw_images])
    
    elif dataset == 'Places365':
        from places365classes import places365_classes
        testset = torchvision.datasets.Places365(root='./data', split='val', 
                                                  small= True, download= True)
        classes = places365_classes
        # image_size = (256,256)
        image_size = (64,64)
        # Pick 30 samples randomly
        indices = torch.randperm(len(testset))[:num_images]
        raw_images = [np.asarray(testset[i][0]) for i in indices]
        labels = [testset[i][1] for i in indices]
        # Convert the images to tensors
        test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        images = torch.stack([test_transform(image) for image in raw_images])

    elif dataset == 'ImageNet200':
        # from imagenetclasses import imagenet_classes
        testset = torchvision.datasets.ImageFolder('tiny-imagenet-200/val')
        
        classes = list(range(0, 200))
        
        # image_size = (224,224)
        image_size = (64,64)
        # Pick 30 samples randomly
        indices = torch.randperm(len(testset))[:num_images]
        raw_images = [np.asarray(testset[i][0]) for i in indices]
        labels = [testset[i][1] for i in indices]
        # Convert the images to tensors
        test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        images = torch.stack([test_transform(image) for image in raw_images])

    # Move the images to the device
    images = images.to(device)
    model = model.to(device)
    # Get the attention maps from the last block
    logits, attention_maps = model(images, output_attentions=True)
    # Get the predictions
    predictions = torch.argmax(logits, dim=1)
    # Concatenate the attention maps from all blocks
    attention_maps = torch.cat(attention_maps, dim=1)
    # select only the attention maps of the CLS token
    attention_maps = attention_maps[:, :, 0, 1:]
    # Then average the attention maps of the CLS token over all the heads
    attention_maps = attention_maps.mean(dim=1)
    # Reshape the attention maps to a square
    num_patches = attention_maps.size(-1)
    size = int(math.sqrt(num_patches))
    attention_maps = attention_maps.view(-1, size, size)
    # Resize the map to the size of the image
    attention_maps = attention_maps.unsqueeze(1)
    attention_maps = F.interpolate(attention_maps, size=image_size, mode='bilinear', align_corners=False)
    attention_maps = attention_maps.squeeze(1)
    # Plot the images and the attention maps
    fig = plt.figure(figsize=(20, 10))
    mask = np.concatenate([np.ones(image_size), np.zeros(image_size)], axis=1)
    for i in range(num_images):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        img = np.concatenate((raw_images[i], raw_images[i]), axis=1)
        ax.imshow(img)
        # Mask out the attention map of the left image
        extended_attention_map = np.concatenate((np.zeros(image_size), attention_maps[i].cpu()), axis=1)
        extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
        ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')
        # Show the ground truth and the prediction
        gt = classes[labels[i]]
        pred = classes[predictions[i]]
        ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt==pred else "red"))
    if output is not None:
        plt.savefig(output)
    plt.show()
