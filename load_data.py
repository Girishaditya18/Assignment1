import os
import torch
import torchvision


def load_data(batch_size=16):
    # should_download = False if os.path.exists('./data/') else True
    trainset = torchvision.datasets.CIFAR10(root='./data/', download=True, train=True, transform=torchvision.transforms.ToTensor())
    testset = torchvision.datasets.CIFAR10(root='./data/', download=True, train=False, transform=torchvision.transforms.ToTensor())

    # The images in CIFAR-10 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.
    # The train dataset contains 50000 images, the test dataset contains 10000 images.
    print("# of images for training: ", len(trainset))
    print("# of images for testing: ", len(testset))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
