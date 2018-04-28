import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from deepclustering.config.config import (get_train_labels_dir, get_train_images_dir,
                                          get_validation_images_dir, get_validation_labels_dir,
                                          resize_height, resize_width,
                                          get_train_dataset_json_file, get_validation_dataset_json_file)
import tensorflow as tf
from deepclustering.dataset.coco import get_dataset
from deepclustering.utilities.visualize import plot_image
import matplotlib.pyplot as plt


# Temporary
def iterate_dataset(dataset):
    iterator = dataset.make_one_shot_iterator()
    next_pair = iterator.get_next()

    with tf.Session() as sess:
        # while True:
        for i in range(3):
            try:
                image, labels = sess.run(next_pair)
                print(type(image), type(labels))
                print(image.shape, labels.shape)
                plt.figure()
                plot_image(image)
            except tf.errors.OutOfRangeError:
                print("Done")
                break

    plt.show()


def try_dataset(image_dir, labels_dir, dataset_json_path):
    dataset = get_dataset(image_dir, labels_dir, dataset_json_path, (500, 500))
    iterate_dataset(dataset)



# create_labels("E:\\data\\mlproj_dataset\\coco\\annotations\\instances_train2014.json",
#               "E:\\data\\mlproj_dataset\\coco\\labels\\train2014",
#               "E:\\data\\mlproj_dataset\\coco\\labels\\train2014_dataset.json")


# create_labels("E:\\data\\mlproj_dataset\\coco\\annotations\\instances_val2014.json",
#               "E:\\data\\mlproj_dataset\\coco\\labels\\val2014",
#               "E:\\data\\mlproj_dataset\\coco\\labels\\val2014_dataset.json")


# try_dataset("E:\\data\\mlproj_dataset\\coco\\images\\train2014",
#             "E:\\data\\mlproj_dataset\\coco\\labels\\train2014",
#             "E:\\data\\mlproj_dataset\\coco\\labels\\train2014_dataset.json")

try_dataset(get_train_images_dir("coco2014"),
            get_train_labels_dir("coco2014"),
            get_train_dataset_json_file("coco2014"))
