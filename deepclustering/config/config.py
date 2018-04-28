import yaml
import os.path

config_yaml_path = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(config_yaml_path, "r") as f:
    config_dict = yaml.load(f)


def get_train_images_dir(dataset_name):
    """
    Return the train images directory of the given dataset.
    :param dataset_name: Name of the dataset.
    :return: Path to images directory.
    """
    return config_dict['dataset'][dataset_name]['train']['images_dir']


def get_train_labels_dir(dataset_name):
    """
    Return the train labels directory of the given dataset.
    :param dataset_name: Name of the dataset.
    :return: Path to labels directory.
    """
    return config_dict['dataset'][dataset_name]['train']['labels_dir']


def get_validation_images_dir(dataset_name):
    """
    Return the validation images directory of the given dataset.
    :param dataset_name: Name of the dataset.
    :return: Path to images directory.
    """
    return config_dict['dataset'][dataset_name]['validation']['images_dir']


def get_validation_labels_dir(dataset_name):
    """
    Return the validation labels directory of the given dataset.
    :param dataset_name: Name of the dataset.
    :return: Path to labels directory.
    """
    return config_dict['dataset'][dataset_name]['validation']['labels_dir']


def get_train_dataset_json_file(dataset_name):
    """
    Return the train json file of the given dataset.
    :param dataset_name: Name of the dataset.
    :return: Path to dataset's json file.
    """
    return config_dict['dataset'][dataset_name]['train']['json_file']


def get_validation_dataset_json_file(dataset_name):
    """
    Return the validation json file of the given dataset.
    :param dataset_name: Name of the dataset.
    :return: Path to dataset's json file.
    """
    return config_dict['dataset'][dataset_name]['validation']['json_file']


resize_height = config_dict['model']['resize_height']
resize_width = config_dict['model']['resize_width']
