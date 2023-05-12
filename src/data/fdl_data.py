import os
import re
import time
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
import pytorch_lightning as pl
from collections import defaultdict, namedtuple
import torch
from torch.utils.data import random_split, Dataset, DataLoader, Subset
from matplotlib import pyplot as plt

from torchvision import transforms

from typing import Optional, List, Tuple

# from data.custom_transforms import GalaxyClassificationTargetTransform, \
#                                    StandardizationTransform


class FDLFormatDatasetV1(Dataset):
    """Dataset format matches that of FDL implementation for non-recurrent model.
    Minor difference: Each simulation has its own input and output blob file.
    Where as the original FDL implementation has a single input and a single output file containing all the simulations as a larger tensor blob.
    
    TODO: Possible speed improvements:
    - Store input and output blob as a single file. (not scalable though)

    TODO: Transforms aren't currently implemented.

    Inherits:
        Dataset (torch.utils.data.Dataset): Inherits from the torch Dataset type.

    Args:
        data_dir (int): Data containing all the sample files.
        layer_num (int, optional): Layer of interest. Defaults to None.
        transform (Callable, optional): Transformation to the input sample. Defaults to None.
        target_transform (Callable, optional): Transformation to the output/target sample. Defaults to None.
        logger_level (int, optional): Level of logs to generate. Defaults to logging.WARNING.
    """

    def __init__(
        self, 
        data_dir, 
        layer_num=None,
        num_years=None,
        transform=None, 
        target_transform=None,
        meta_data_file_path=None,
        logger_level=logging.WARNING,
    ):
        # Defining logger
        self.logger_ = logging.getLogger(self.__class__.__name__)
        self.logger_.setLevel(logger_level)

        self.data_dir = data_dir
        self.layer_num = layer_num
        self.num_years = num_years

        file_list = os.listdir(self.data_dir)
        file_list = [f for f in file_list if f.startswith("sim")]

        sim_pattern = re.compile("sim[0-9]{1,3}")
        self.sims_list = list(set([sim_pattern.findall(f)[0] for f in file_list]))
        self.sims_list.sort()

        self.transform = transform
        self.target_transform = target_transform

        # Fetch meta data information
        self.meta_data_path = meta_data_file_path
        if self.meta_data_file_path is not None:
            self.meta_data_path = meta_data_file_path
            self.meta_data = namedtuple('meta_data', [
                'input_names',
                'time_steps',
                'input_min',
                'input_max',
                'output_names',
            ])
            with open(self.meta_data_path) as f:
                lines = f.readlines()

            self.meta_data.input_names = str(lines[0]).strip().split(", ")
            self.meta_data.time_steps = np.array(str(lines[1]).strip().split(", "), dtype = 'float64')
            self.meta_data.time_steps = torch.from_numpy(np.array(self.meta_data.time_steps, dtype = 'int64'))    # FIXME: Seems redundant.
            self.meta_data.input_min = torch.from_numpy(np.array(str(lines[2]).strip().split(", "), dtype = 'float64'))
            self.meta_data.input_max = torch.from_numpy(np.array(str(lines[3]).strip().split(", "), dtype = 'float64'))
            self.meta_data.output_names = str(lines[4]).strip().split(", ")

    def __len__(self):
        return len(self.sims_list)

    def __getitem__(self, idx):
        if not self.layer_num:
            input_blob_path = f"{self.data_dir}/{self.sims_list[idx]}_input_blob.npy"
            output_blob_path = f"{self.data_dir}/{self.sims_list[idx]}_output_blob.npy"
        else:
            input_blob_path = f"{self.data_dir}/{self.sims_list[idx]}_layer{self.layer_num}_input_blob.npy"
            output_blob_path = f"{self.data_dir}/{self.sims_list[idx]}_layer{self.layer_num}_output_blob.npy"

        input_blob = torch.from_numpy(np.load(input_blob_path))
        output_blob = torch.from_numpy(np.load(output_blob_path))

        if self.num_years is not None:
            input_blob = input_blob[:, :, :self.num_years, :]
            output_blob = output_blob[:, :, :self.num_years, :]

        return (input_blob, output_blob)


class FDLFormatDatasetV2(Dataset):
    """Dataset format matches that of FDL implementation for non-recurrent model.
    Addresses the minor difference between the original FDL dataset and FDLFormatDatasetV1. (Check FDLFormatDatasetV1 docstring)
    
    TODO: Transforms aren't currently implemented.

    Inherits:
        Dataset (torch.utils.data.Dataset): Inherits from the torch Dataset type.

    Args:
        data_dir (int): Data containing all the sample files.
        layer_num (int, optional): Layer of interest. Defaults to None.
        transform (Callable, optional): Transformation to the input sample. Defaults to None.
        target_transform (Callable, optional): Transformation to the output/target sample. Defaults to None.
        logger_level (int, optional): Level of logs to generate. Defaults to logging.WARNING.
    """

    def __init__(
        self, 
        data_dir, 
        layer_num=None,
        transform=None, 
        target_transform=None,
        logger_level=logging.WARNING,
    ):
        # Defining logger
        self.logger_ = logging.getLogger(self.__class__.__name__)
        self.logger_.setLevel(logger_level)

        self.data_dir = data_dir
        self.layer_num = layer_num

        if not self.layer_num:
            input_blob_path = f"{self.data_dir}/input_blob.npy"
            output_blob_path = f"{self.data_dir}/output_blob.npy"
        else:
            input_blob_path = f"{self.data_dir}/layer{self.layer_num}_input_blob.npy"
            output_blob_path = f"{self.data_dir}/layer{self.layer_num}_output_blob.npy"

        self.input_blob = np.load(input_blob_path)        
        self.output_blob = np.load(output_blob_path)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Returns number of simulations in the dataset.

        Returns:
            int: number of simulations in the dataset.
        """
        return len(self.output_blob)

    def __getitem__(self, idx):
        return (self.input_blob[idx], self.output_blob[idx])


# if __name__ == "__main__":
#     data_dir = "/global/cfs/projectdirs/m1012/satyarth/Data/ensemble_simulation_runs/ensemble_simulation_run2/training_data/v2"

#     ts = time.time()
#     ds = FDLFormatDatasetV2(
#         data_dir=data_dir,
#         layer_num=7,
#     )
#     read_time = time.time() - ts

#     exec_times = []
#     for i in tqdm(range(len(ds))):
#         ts = time.time()
#         ds[i]
#         exec_times.append(time.time() - ts)
    
#     np.savetxt("exec_times_v2.txt", exec_times)
#     print(f"{read_time = } seconds")
#     print(f"{np.mean(exec_times) = } seconds")


