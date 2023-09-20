from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    ### This code defines a neural network class called Net using PyTorch.
    # The __init__ method initializes the network's layers:
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 and self.conv2 are 2D convolutional layers.
        # conv1 takes a single-channel (grayscale) input and outputs 32 channels.
        # conv2 takes these 32 channels and outputs 64 channels.
        # Both use a 3x3 filter and a stride of 1.
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 and self.dropout2 are dropout layers for regularization.
        # They randomly set a fraction (25% and 50%, respectively)
        # of the input units to 0 during training to prevent overfitting.
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # self.fc1 and self.fc2 are fully connected (dense) layers.
        # fc1 takes a flattened version of the last convolutional layer with 9216 units and outputs 128 units.
        # fc2 takes these 128 units and outputs 10 units,
        # suitable for a classification task with 10 classes.
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    # The forward function defines the forward pass of the neural network,
    # specifying how the input data flows through the layers to produce an output.
    # This function takes an input tensor x and transforms it step by step:
    def forward(self, x):
        # The input x first passes through the 1st convolutional layer (conv1).
        # This applies 32 different 3x3 filters to the input, generating 32 output channels.
        x = self.conv1(x)
        # The output from conv1 is then passed through a ReLU (Rectified Linear Unit) activation function,
        # which replaces all negative values in the tensor with zeros.
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # Max pooling is applied to reduce the spatial dimensions (width & height) of the input volume.
        # The number 2 specifies that the size of the pooling window is 2x2.
        x = F.max_pool2d(x, 2)
        # The dropout layer randomly sets 25% of its input units to 0
        # to prevent overfitting during training.
        x = self.dropout1(x)
        # The 2D output tensor from the dropout layer is flattened into a 1D tensor,
        # so it can be fed into a fully connected layer.
        x = torch.flatten(x, 1)
        # The flattened tensor is passed through the first fully connected layer (fc1),
        # transforming it into a 128-unit tensor.
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # a log-softmax activation function is applied.
        # This squashes the output values so that they lie in the range (0, 1),
        # and the sum is 1.
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        ### Moves the data and target labels to the computing device specified.
        data, target = data.to(device), target.to(device)
        ### Clears old gradients, essential before performing a new optimization step.
        optimizer.zero_grad()
        ### Feeds the data through the model to get the output (predictions).
        output = model(data)
        ### Computes the negative log-likelihood loss between the model output and the true labels.
        # This is the function you are trying to minimize.
        loss = F.nll_loss(output, target)
        ### Computes the gradient of the loss with respect to the model parameters.
        # This is used to update the weights.
        loss.backward()
        ### Updates the model parameters using the optimizer algorithm (e.g., SGD, Adam, etc.).
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
