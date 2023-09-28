import numpy as np
import matplotlib.pyplot as plt
import torchvision


def imshow_tensor(image, title=None):
    """Imshow for Tensor."""

    # Set the color channel as the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Reverse the preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    image = np.clip(image, 0, 1)
    plt.figure(figsize=(24, 24))

    plt.imshow(image)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = npimg.transpose(1,2,0)

    plt.imshow(npimg)
    plt.show()
