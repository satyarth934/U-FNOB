import os
import re
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
import pytorch_lightning as pl
from collections import defaultdict
import torch
from torch.utils.data import random_split, Dataset, DataLoader, Subset
from matplotlib import pyplot as plt

from torchvision import transforms

from typing import Optional, List, Tuple

# from data.custom_transforms import GalaxyClassificationTargetTransform, \
#                                    StandardizationTransform


class FDLFormatDataset(Dataset):
    """Dataset format matches that of FDL implementation for non-recurrent model.
    
    TODO: Possible speed improvements:
    - Store input and output blob as a single file. (not scalable though)

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

        file_list = os.listdir(self.data_dir)
        file_list = [f for f in file_list if f.startswith("sim")]

        sim_pattern = re.compile("sim[0-9]{1,3}")
        self.sims_list = list(set([sim_pattern.findall(f)[0] for f in file_list]))
        self.sims_list.sort()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sims_list)

    def __getitem__(self, idx):
        if not self.layer_num:
            input_blob_path = f"{self.data_dir}/{self.sims_list[idx]}_input_blob.npy"
            output_blob_path = f"{self.data_dir}/{self.sims_list[idx]}_output_blob.npy"
        else:
            input_blob_path = f"{self.data_dir}/{self.sims_list[idx]}_layer{self.layer_num}_input_blob.npy"
            output_blob_path = f"{self.data_dir}/{self.sims_list[idx]}_layer{self.layer_num}_output_blob.npy"

        input_blob = np.load(input_blob_path)
        output_blob = np.load(output_blob_path)

        return (input_blob, output_blob)


if __name__ == "__main__":
    data_dir = "/global/cfs/projectdirs/m1012/satyarth/Data/ensemble_simulation_runs/ensemble_simulation_run2/training_data/v1"

    ds = FDLFormatDataset(
        data_dir=data_dir,
        layer_num=7,
    )


    import time
    from tqdm import tqdm

    exec_times = []
    for i in tqdm(range(len(ds))):
        ts = time.time()
        ds[i]
        exec_times.append(time.time() - ts)
    
    np.savetxt("exec_times.txt", exec_times)
    print(f"{np.mean(exec_times) = }")


# class FDLFormatDataModule(pl.LightningDataModule):
#     def __init__(
#         self, 
#         data_dir: str="./", 
#         batch_size: int=32,
#         num_workers: int=None,
#         logger_level=logging.WARNING,
#     ):
#         # Defining logger
#         self.logger_ = logging.getLogger(self.__class__.__name__)
#         self.logger_.setLevel(logger_level)

#         super().__init__()
#         self.data_dir = data_dir
#         self.transform = transforms.Compose(
#             [
#                 # transforms.ToTensor(), 
#                 transforms.RandomRotation(degrees=(0, 360)),
#                 transforms.CenterCrop(101),
#                 # transforms.Normalize(mean=(7.3798e+13,), std=(3.2842e+16,)),
#                 # NanToNumTransform(),
#                 # NormalizationTransform(multiplier=500),
#                 StandardizationTransform(),
#             ]
#         )

#         if num_workers is None:
#             self.num_workers = max(1, multiprocessing.cpu_count() // 2)
#         else:
#             self.num_workers = num_workers
#         self.batch_size = batch_size

#         self.prepare_data_done = False
#         self.setup_done = False
    

#     def init_check(self):
#         """Prepare and setup data if not already done.
#         """

#         if not self.prepare_data_done:
#             self.prepare_data()
        
#         if not self.setup_done:
#             self.setup()


#     def get_category_counts(self, dataset="full") -> dict:
#         """Returns the number of samples for each category. Majorly used for debugging and sanity checks.

#         Args:
#             dataset (Dataset): Input dataset.
        
#         Returns:
#             dict: dictionary of count of each category.
#         """

#         if isinstance(dataset, Dataset):
#             pass
#         elif dataset.lower() == "full":
#             dataset = self.full_dataset
#         elif dataset.lower() == "train":
#             dataset = self.train_dataset
#         elif dataset.lower() == "validation":
#             dataset = self.val_dataset
#         elif dataset.lower() in ["predict", "test"]:
#             dataset = self.test_dataset

#         value_counts = defaultdict(lambda: 0)

#         for _, label in tqdm(dataset):
#             value_counts[label] += 1

#         return value_counts


#     def get_category_indices(self, dataset="full") -> dict:
#         """This function is used to diversify the datasets along categories.

#         Returns:
#             dict: dictionary of a list of all the indices for each category.
#         """
#         if isinstance(dataset, Dataset):
#             pass
#         elif dataset.lower() == "full":
#             dataset = self.full_dataset
#         elif dataset.lower() == "train":
#             dataset = self.train_dataset
#         elif dataset.lower() == "validation":
#             dataset = self.val_dataset
#         elif dataset.lower() in ["predict", "test"]:
#             dataset = self.test_dataset

#         category_labels = defaultdict(list)
#         # for i, img_path_label in enumerate(dataset.img_labels):
#         #     label = GalaxyClassificationTargetTransform()(img_path_label)
#         #     category_labels[label].append(i)
#         for i, data_sample in tqdm(enumerate(dataset)):
#             data_tensor, data_label = data_sample
#             category_labels[data_label].append(i)
        
#         return category_labels


#     def prepare_data(self):
#         self.full_dataset = GalaxyModelDataset(
#             data_dir=self.data_dir, 
#             transform=self.transform,
#             target_transform=GalaxyClassificationTargetTransform(),
#         )
#         self.prepare_data_done = True


#     def setup(self, stage: Optional[str] = None):
#         """Setting up the dataset maintaining the categorical diversity across train, val, and test datasets.

#         Args:
#             stage (str, optional): train, validation, test, predict stages. Implicitly used by pytorch. Defaults to None.
#         """
#         # initialize split portions
#         train_portion = 0.7
#         val_portion = 0.2
#         test_portion = 0.1    # Unused. Just mentioning here for readability.

#         # train_split = int(len(self.full_dataset) * train_portion)
#         # val_split = int(len(self.full_dataset) * val_portion)
#         # test_split = int(len(self.full_dataset) - train_split - val_split)
#         # self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.full_dataset, [train_split, val_split, test_split])

#         # Get index values maintaining categorical diversity
#         train_indices = list()
#         val_indices = list()
#         test_indices = list()
#         self.category_indices = self.get_category_indices(dataset=self.full_dataset)
#         # self.category_indices.pop(0)    # Removing category 0.
#         # self.category_indices.pop(1)    # Removing category 1 as it might hurt training. TODO: Find a better solution.
#         # self.category_indices.pop(2)    # Removing category 2.
#         # self.category_indices.pop(3)    # Removing category 3 as it might hurt training TODO: Find a better solution.

#         for category, indices in self.category_indices.items():
#             train_split = int(len(indices) * train_portion)
#             val_split = int(len(indices) * val_portion)
#             test_split = int(len(indices) - train_split - val_split)

#             ctrain_idxs, cval_idxs, ctest_idxs = random_split(indices, [train_split, val_split, test_split])
            
#             train_indices.extend([_ for _ in ctrain_idxs])
#             val_indices.extend([_ for _ in cval_idxs])
#             test_indices.extend([_ for _ in ctest_idxs])
        
#         # get dataset subsets for the defined indices
#         self.train_dataset = Subset(self.full_dataset, train_indices)
#         self.val_dataset = Subset(self.full_dataset, val_indices)
#         self.test_dataset = Subset(self.full_dataset, test_indices)

#         self.setup_done = True

    
#     def get_data_shape(self):
#         self.init_check()
#         return self.train_dataset[0][0].shape

#     def get_full_dataset(self):
#         self.init_check()
#         return self.full_dataset
    
#     def get_train_dataset(self):
#         self.init_check()
#         return self.train_dataset
    
#     def get_val_dataset(self):
#         self.init_check()
#         return self.val_dataset

#     def get_test_dataset(self):
#         self.init_check()
#         return self.test_dataset

#     def train_dataloader(self):
#         self.init_check()
#         return DataLoader(
#             self.train_dataset, 
#             batch_size=self.batch_size, 
#             shuffle=True, 
#             num_workers=self.num_workers,
#         )

#     def val_dataloader(self):
#         self.init_check()
#         return DataLoader(
#             self.val_dataset, 
#             batch_size=self.batch_size, 
#             shuffle=False, 
#             num_workers=self.num_workers,
#         )

#     def test_dataloader(self):
#         self.init_check()
#         return DataLoader(
#             self.test_dataset, 
#             batch_size=self.batch_size, 
#             shuffle=False, 
#             num_workers=self.num_workers,
#         )

#     def predict_dataloader(self):
#         self.init_check()
#         return DataLoader(
#             self.test_dataset, 
#             batch_size=self.batch_size, 
#             shuffle=False, 
#             num_workers=self.num_workers,
#         )
    
#     def plot_samples(
#         self, 
#         stage: str, 
#         num_samples: int, 
#         sample_randomly: bool=False, 
#         title_str: str=None, 
#         grid_size: Tuple=None, 
#         seed: int=453, 
#         save_fig_filename: str=None,
#         sampled_idxs: List=None,
#         **kwargs,
#     ) -> None:
#         """Plot the data samples in a grid for visual inspection.

#         Args:
#             stage (str): Dataset to be visualized. Options are "full" | "train" | "validation" | "predict" | "test".
#             num_samples (int): Number of samples to be visualized.
#             sample_randomly (bool, optional): Whether to fetch the data samples randomly. Defaults to False.
#             title_str (str, optional): Title for the plot. Defaults to None.
#             grid_size (Tuple, optional): Grid size for the figure in the format (nrows, ncols). The nrows * ncols count should be <= num_samples. Defaults to None.
#             seed (int, optional): Numpy random seed. Defaults to 453.
#             save_fig_filename (str, optional): Saves the plot as a file if a value for this parameter is provided. Defaults to None, i.e. does not save a figure.
#             sampled_idxs (List, optional): List of indices that are to be visualized. If not specified, it is computed internally. Specifying this manually can break the code so be cautious while doing it manually. Defaults to None.
#         """
    
#         # Prepare and setup the data
#         self.prepare_data()
#         self.setup()

#         # Set variables based on the input parameters
#         if stage.lower() == "train":
#             dataset = self.train_dataset
#         elif stage.lower() == "validation":
#             dataset = self.val_dataset
#         elif stage.lower() in ["predict", "test"]:
#             dataset = self.test_dataset
#         elif stage.lower() == "full":
#             dataset = self.full_dataset

#         # np.random.seed(seed)    # Already defined at the beginning.

#         if num_samples is None:
#             num_samples = len(dataset)
        
#         if num_samples > len(dataset):
#             self.logger_.critical("num_samples should be <= len(dataset)!!")
#             return
        
#         if sampled_idxs is None:
#             if sample_randomly:
#                 sampled_idxs = np.random.randint(0, len(dataset), num_samples)
#             else:
#                 sampled_idxs = np.arange(0, num_samples)

#         if grid_size is None:
#             ncols = 4
#             nrows = int(np.ceil(num_samples/ncols))
#         else:
#             nrows, ncols = grid_size

#         fig, ax = plt.subplots(nrows=nrows, ncols=ncols, **kwargs)

#         for i in range(nrows):
#             for j in range(ncols):
#                 dataset_idx = sampled_idxs[((i * ncols) + j)]
#                 if dataset_idx < len(dataset):
#                     data_img = dataset[dataset_idx][0].squeeze()
#                     ax[i,j].imshow(data_img)
#                     ax[i,j].title.set_text(f"Category {dataset[dataset_idx][1]}")
        
#         fig.suptitle(stage if title_str is None else title_str)
#         fig.tight_layout()
#         if save_fig_filename:
#             plt.savefig(save_fig_filename)
#         else:
#             plt.show()


#     def get_pixel_value_stats(
#         self, 
#         stage: str,
#         num_samples: int=None,
#         sample_randomly: bool=False,
#         sampled_idxs: List=None,
#         title_str: str=None,
#         save_fig_filename: str=None,
#     ):
#         self.init_check()

#         if stage.lower() == "train":
#             dataset = self.train_dataset
#         elif stage.lower() == "validation":
#             dataset = self.val_dataset
#         elif stage.lower() in ["predict", "test"]:
#             dataset = self.test_dataset
#         elif stage.lower() == "full":
#             dataset = self.full_dataset

#         if num_samples is None:
#             num_samples = len(dataset)
        
#         if num_samples > len(dataset):
#             self.logger_.critical("num_samples should be <= len(dataset)!!")
#             return
        
#         if sampled_idxs is None:
#             if sample_randomly:
#                 sampled_idxs = np.random.randint(0, len(dataset), num_samples)
#             else:
#                 sampled_idxs = np.arange(0, num_samples)
        
#         min_list = list()
#         mean_list = list()
#         max_list = list()
#         for si in tqdm(sampled_idxs):
#             min_list.append(dataset[si][0].min())
#             mean_list.append(dataset[si][0].mean())
#             max_list.append(dataset[si][0].max())

#         print(f"{len(min_list) = }")
#         print(f"{len(mean_list) = }")
#         print(f"{len(max_list) = }")
        
#         plt.figure()
#         plt.scatter(range(len(min_list)), min_list, s=1, color="green", label="min")
#         plt.scatter(range(len(mean_list)), mean_list, s=1, color="blue", label="mean")
#         plt.scatter(range(len(max_list)), max_list, s=1, color="red", label="max")

#         plt.legend()

#         plt.title(stage if title_str is None else title_str)
#         if save_fig_filename:
#             plt.savefig(save_fig_filename)
#         else:
#             plt.show()