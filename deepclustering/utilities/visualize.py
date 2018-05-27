import matplotlib.pyplot as plt
import numpy as np
import PIL.Image


def plot_image(image_array):
    """
    Plot numpy array as image.
    :param image_array: Numpy array that represent an image.
    """
    plt.axis('off')
    plt.imshow(image_array)


def plot_image_file(image_path):

    img = np.array(PIL.Image.open(image_path, 'r'))
    plt.axis('off')
    plt.imshow(img)


def plot_labels(labels_array):
    """
    Plot numpy array as image.
    :param labels_array: Numpy array that represent the labels.
    """
    plt.axis('off')
    plt.imshow(labels_array, alpha=0.5)


def plot_labels_file(labels_path):

    img = np.array(PIL.Image.open(labels_path, 'r'))
    plt.axis('off')
    plt.imshow(img, alpha=0.5)

