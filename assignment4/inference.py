from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import os
from functions import *
from train import *
from fine_tune import *
from resnet import resnet18

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define path
    saved_model_path = 'resnet18_saved_model_ft.pth'
    validdir = f"data/grocery_data_mini/val"

    # Change to fit hardware
    num_workers = 0
    batch_size = 8

    # define image transformations
    # define transforms
    valid_transforms = transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    # Datasets from folders
    valid_data = datasets.ImageFolder(root=validdir, transform=valid_transforms)

    # Dataloader iterators, make sure not to shuffle
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    categories = []
    for d in os.listdir(validdir):
        categories.append(d)
    n_classes = len(categories)

    #load model
    model = torch.load(saved_model_path)
    model.eval()

    # dataiter = iter(valid_loader)
    # images, labels = iter(dataiter).__next__()
    # print images
    #imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join('%5s' % categories[labels[j]] for j in range(batch_size)))
    # outputs = model(images)
    # _, predicted = torch.max(outputs, 1)
    # print('Predicted: ', ' '.join('%5s' % categories[predicted[j]] for j in range(batch_size)))


   # Final Prediction
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Valid Accuracy: %d %%' % (correct/total*100))

