import numpy as np
from torch.utils.data import Dataset
from data_loading.npz_dataset_helper import get_foci_lookups
from utilities.logger import error


class NpzDataset(Dataset):
    """
    Implements my customized Dataset for the BSPM data.
    The dataset is saved as npz file per geometry per focus.
    One npz file contains the label as well as signal of all three ventricle transforms.

    The dataloading is optimized by loading a static lookup table from index to specific focus and ventricle transform.
    """

    def __init__(self, data_path, geometries, label='l'):
        self.data_path = data_path
        self.index_to_file, self.index_to_transform = get_foci_lookups(geometries, data_path)
        self.amount_foci = self.index_to_file.shape[0]
        self.label_name = label

    def __len__(self):
        return self.amount_foci

    def __getitem__(self, idx):
        # return the idx-th sample
        try:
            focus_file = np.load(self.data_path + self.index_to_file[idx])
            transform_no = self.index_to_transform[idx]
            bsp_data = focus_file[transform_no]
            label = focus_file[self.label_name]

            return bsp_data, label

        except:
            error('Error loading file!', self.data_path, idx, self.index_to_file[idx])
