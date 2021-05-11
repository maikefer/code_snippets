import os

from torch.utils.data import DataLoader
from data_loading.npz_dataset import NpzDataset
from utilities.logger import status


def load_data(config, label_name='l'):
    """
    Load data as dataloader
    :param config: the configuration dict, containing:
        "data": "batch_size": The batch size of this training
                "shuffle": If the data should be shuffled or not
                "train_geometries": the name of the train dataset
                "val_geometries": the name of the validation dataset
                "tmp": If the data is stored in tmp directory on server
                "data_path": Data path for local training
    :param label_name: The name of the desired label in the datafiles
    :return: A torch data loader for the train and one for the validation dataset
    """

    batch_size = config["data"]["batch_size"]
    train_data_path, val_data_path = retrieve_data_path(config)

    train_data_loader = DataLoader(
        NpzDataset(train_data_path, geometries=(config["data"]["train_geometries"]), label=label_name),
        shuffle=config["data"]["shuffle"], batch_size=batch_size, pin_memory=True, drop_last=True)
    val_data_loader = DataLoader(
        NpzDataset(val_data_path, geometries=(config["data"]["val_geometries"]), label=label_name),
        shuffle=False, batch_size=batch_size, pin_memory=True, drop_last=False)

    return val_data_loader, train_data_loader


def retrieve_data_path(config):
    """
    Determine path where data is stored for tain and val dataset.
    Data path varies if the network is trained locally or on the unicluster.
    :param config: Config-dictionary containing
        "data": "tmp": If the data is stored in tmp directory on server
                "data_path": Data path for local training
                "train_geometries": the name of the train dataset
                "val_geometries": the name of the test dataset
    :return: the retrieved data paths of the train and validation set
    """

    data_path = os.environ["TMP"] + '/' if config["data"]["tmp"] else config["data"]["data_path"]
    status('\t Load Data from ', data_path)

    train_data_path = data_path + 'train/' if config["data"]["train_geometries"] == 'train_reduced' \
        else data_path + config["data"]["train_geometries"] + '/'
    val_data_path = data_path + 'val/' if config["data"]["val_geometries"] == 'val_reduced' \
        else data_path + config["data"]["val_geometries"] + '/'

    return train_data_path, val_data_path
