import os
import re
import glob
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from joblib import Parallel, delayed

parallel_function = Parallel(n_jobs=-1, verbose=5)


def downsample_df(original_df, x_resolution=10, y_resolution=10):
    """Downsamples simulation output data based on the given resolution.

    Args:
        original_df (pd.DataFrame): Input data frame containing x, y parameters.
        x_resolution (int, optional): Resolution across x-axis. Defaults to 10 meters.
        y_resolution (int, optional): Resolution across y-axis. Defaults to 10 meters.
    """

    copy_df = copy.deepcopy(original_df)

    copy_df.x = (copy_df.x - copy_df.x.min()) // x_resolution
    copy_df.y = (copy_df.y - copy_df.y.min()) // y_resolution

    downsampled_df = copy_df[~copy_df[["x", "y"]].duplicated(keep="last")]

    return downsampled_df


def downsample_file(input_parq_path, x_resolution=10, y_resolution=10):
    """Downsamples simulation output data based on the given resolution.

    Args:
        input_parq_path (pd.DataFrame): Input parquet file path containing the data frame containing x, y columns.
        x_resolution (int, optional): Resolution across x-axis. Defaults to 10 meters.
        y_resolution (int, optional): Resolution across y-axis. Defaults to 10 meters.
    """
    
    # Read file
    parq_df = pd.read_parquet(input_parq_path)

    # Downsample the data
    downsampled_df = downsample_df(
        original_df=parq_df, 
        x_resolution=x_resolution, 
        y_resolution=y_resolution,
    )

    # Get the output path
    dest_parq_path = input_parq_path.replace("/identity/", "/identity_downsampled/")

    # Create directory if does not exist
    dest_parq_parent = os.path.dirname(dest_parq_path)
    if not os.path.exists(dest_parq_parent):
        os.makedirs(dest_parq_parent, exist_ok=True)

    # Write downsampled file
    downsampled_df.to_parquet(dest_parq_path)


def main():
    print("Fetching data files...")
    data_root_dir = "/global/cfs/projectdirs/m1012/satyarth/Data/ensemble_simulation_runs/ensemble_simulation_run2"
    glob_str = f"{data_root_dir}/sim*/separate_2d_layer_info/identity/layer7_*.parq"

    sim_data_paths = glob.glob(glob_str)

    print("Downsampling...")
    parallel_function(
        delayed(downsample_file)(
            sim_data_path,
            x_resolution=10,
            y_resolution=10,
        )
        for sim_data_path in sim_data_paths
    )

if __name__ == "__main__":
    main()