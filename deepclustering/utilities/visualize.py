import matplotlib.pyplot as plt


def plot_image(image_array):
    """
    Plot numpy array as image.
    :param image_array: Numpy array that represent an image.
    """
    print("plotting image")
    plt.axis('off')
    plt.imshow(image_array)
