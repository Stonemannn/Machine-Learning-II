from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import os
from functions import *
from train import *
from resnet import resnet18


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define path
    traindir = f"data/grocery_data_mini/train"
    validdir = f"data/grocery_data_mini/val"
    # Change to fit hardware
    num_workers = 0
    batch_size = 8
    n_epochs = 5
    learning_rate = 0.001
    max_epochs_stop = 3
    print_every = 1
    # define image transformations
    # define transforms
    image_transforms = {
        # Train uses data augmentation
        'train':
            transforms.Compose([
                # You can use transforms.RandomResizedCrop() for crop
                # You can use transforms.RandomRotation() for rotation
                # You can use transforms.ColorJitter() for colorjitter
                # You can use transforms.RandomHorizontalFlip() for flip

                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),

                transforms.CenterCrop(size=224),  # Image net standards
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet standards
            ]),
        # Validation does not use augmentation
        'valid':
            transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    }
    # Datasets from folders

    data = {
        'train':
            datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
        'valid':
            datasets.ImageFolder(root=validdir, transform=image_transforms['valid'])
    }

    # Dataloader iterators, make sure to shuffle

    dataloaders = {
        'train': torch.utils.data.DataLoader(data['train'], batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers),
        'valid': torch.utils.data.DataLoader(data['valid'], batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers)
    }

    # Iterate through the dataloader once
    trainiter = iter(dataloaders['train'])
    validationiter = iter(dataloaders['valid'])

    categories = []
    for d in os.listdir(traindir):
        categories.append(d)

    n_classes = len(categories)
    print('Length train: {} Length test: {}'.format(len(data['train']), len(data['valid'])))

    # get some random training images
    dataiter = iter(dataloaders['train'])
    images, labels = iter(dataiter).__next__()

    # show images
    imshow_tensor(torchvision.utils.make_grid(images), title=[categories[x] for x in labels])
    # print labels
    print(' '.join('%5s' % categories[labels[j]] for j in range(batch_size)))

    # Define the network with pretrained models
    model = models.resnet18(weights="IMAGENET1K_V1")

    model = resnet18()
    # load pretrain weights
    model_weight_path = "resnet18_Weights_IMAGENET1K_V1.pth"

    model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    # freeze the pretrained parameters except for the last fully connected layer
    for param in model.parameters():
        param.requires_grad = False

    # create a customized fully connected layer to substitute the previous one
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, n_classes)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    # covert class to index
    model.class_to_idx = data['train'].class_to_idx
    model.idx_to_class = {
        idx: class_
        for class_, idx in model.class_to_idx.items()
    }

    # Set up your loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)

    save_file_name = 'resnet18_model_best_model_ft.pt'

    model, history = train(model,
                           criterion,
                           optimizer,
                           dataloaders['train'],
                           dataloaders['valid'],
                           save_file_name,
                           max_epochs_stop,
                           n_epochs,
                           print_every)

    # """
    # dataiter = iter(dataloaders['valid'])
    # images, labels = iter(dataiter).__next__()
    # print images
    # #imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join('%5s' % categories[labels[j]] for j in range(batch_size)))
    # outputs = model(images)
    # _, predicted = torch.max(outputs, 1)
    # print('Predicted: ', ' '.join('%5s' % categories[predicted[j]] for j in range(batch_size)))

    save_model_path = 'resnet18_saved_model_ft.pth'
    torch.save(model, save_model_path)
    # """
    

# Compared to training from scratch,
# fine-tuning the existing model save 2/3 of training time
# and have a much higher accuracy.
