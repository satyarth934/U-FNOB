import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
parallel_function = Parallel(n_jobs=-1, verbose=5)

import generate_data_blob_utils as gdutils


PLOT_FLAG = False


#############################
# INPUT
#############################
def generate_input_data_blob(
    df_row,
    data_mode,
    layer_of_interest,
    write_dir,
    return_generated_tensor=False,
):
    """Generates input data blob.
    Dataset version 1:
        #sims files
        Dimensions => y * x * t * #variables
    Dataset version 2:
        1 file
        Dimensions => #sims * y * x * t * #variables

    Write the generated files to a destination directory.

    Args:
        df_row (pd.Series): Single simulation row from the meta data DataFrame.
        data_mode (str): Type of preprocessed data to use. Points to a subdirectory withing the simulation directory.
        layer_of_interest (int): Layer of interest.
    """

    # Read df row values
    # ------------------
    sim_path = df_row["@sim_path@"]
    
    variables_dict = dict(
        perm=df_row["@Perm@"],
        por=df_row["@Por@"],
        alpha=df_row["@alpha@"],
        rech=df_row["@Rech@"],
        seepage=df_row["@seepage@"],
    )
    variables_names = variables_dict.keys()
    variables_of_interest = variables_dict.values()

    # Get path to each simulation file for the specified layer
    # --------------------------------------------------------
    sim_layer_globstr = f"{sim_path}/{data_mode}/layer{layer_of_interest}*.parq"
    sim_layer_paths = glob.glob(sim_layer_globstr)
    sim_layer_paths.sort()    # To make sure that the years are in order

    # Read the first simulation file to get the grid size
    single_sim_df = pd.read_parquet(sim_layer_paths[0])

    # breakpoint()
    single_sim_grid = gdutils._point_to_grid(single_sim_df)
    nan_idxs = np.isnan(single_sim_grid)
    # nan_idxs = gdutils.fill_holes(nan_idxs)    # Not being used currently

    nx = int(single_sim_df["x"].max() - single_sim_df["x"].min() + 1)
    ny = int(single_sim_df["y"].max() - single_sim_df["y"].min() + 1)

    # Fetch years from the simulation filenames
    # -----------------------------------------
    year_pattern = re.compile("[0-9]{4}y")
    years = [
        int(year_pattern.findall(slpath)[0].strip("y")) 
        for slpath in sim_layer_paths
    ]

    # Return tensor placeholder - this is the tensor that will be used as input to the training model
    # -----------------------------------------
    num_inp_variables = len(variables_of_interest) + 3    # x, y, t as the extra 3 input variables
    return_tensor = np.zeros([ny, nx, len(years), num_inp_variables])

    # Storing variables in the return tensor
    # -----------------------------------------
    for vi, var_i in enumerate(variables_of_interest):
        return_tensor[:, :, :, vi] = var_i
    
    # Storing x, y in return tensor
    # -----------------------------------------
    grid_x = np.ones([ny, nx]) * np.arange(nx) / nx
    grid_y = np.ones([ny, nx]) * (np.arange(ny) / ny).reshape(ny, -1)

    grid_x_var_i = len(variables_of_interest)
    grid_y_var_i = grid_x_var_i + 1
    return_tensor[:, :, :, grid_x_var_i] = grid_x[:, :, np.newaxis].repeat(
        repeats=len(years), 
        axis=2,
    )
    return_tensor[:, :, :, grid_y_var_i] = grid_y[:, :, np.newaxis].repeat(
        repeats=len(years), 
        axis=2,
    )

    # Storing year in return tensor
    # -----------------------------------------
    # breakpoint()
    years_np = np.array(years)
    years_normed = (years_np - years_np.min()) / (years_np.max() - years_np.min())
    for yi, year in enumerate(years_normed):
        grid_t = np.ones([ny, nx]) * year
        return_tensor[:, :, yi, -1] = grid_t

    # NaN out all the pixels that are NaNs in the concentration grid
    # --------------------------------------------------------------
    for time_i in range(0, len(years)):
        for var_j in range(0, num_inp_variables):
            return_tensor[:,:, time_i, var_j][nan_idxs] = np.nan

    # PLOT the return tensor as subplots of dimension variables_of_interest x 10 as a sanity check. Plot data every 10 years.
    # -----------------------------------------
    if PLOT_FLAG:
        var_mins = [variables_dict[vname] for vname in variables_names] + [0, 0, 0]
        var_maxs = [variables_dict[vname] for vname in variables_names] + [1, 1, 1]
        variables_names = list(variables_names) + ["grid_x", "grid_y", "grid_t"]    # FIXIT: Remove/Fix later
        fig, axes = plt.subplots(nrows=12, ncols=num_inp_variables, figsize=(30,24))    # shape = years x variables
        years_freq = 5
        for time_i in range(0, axes.shape[0]):
            for var_j in range(axes.shape[1]):
                if time_i == 0:
                    axes[time_i, var_j].set_title(variables_names[var_j], fontweight="bold")

                year_idx = (time_i * years_freq)
                axes[time_i, var_j].imshow(
                    return_tensor[:, :, year_idx, var_j],
                    vmin=var_mins[var_j],
                    vmax=var_maxs[var_j],
                    origin="lower",
                )

                if variables_names[var_j] in ["grid_x", "grid_y"]:
                    img_caption = f"min={np.nanmin(return_tensor[:, :, year_idx, var_j]):.8e}\nmax{np.nanmax(return_tensor[:, :, year_idx, var_j]):.8e}"
                else:
                    img_caption = f"mean={np.nanmean(return_tensor[:, :, year_idx, var_j]):.8e}\nval00={return_tensor[50, 100, year_idx, var_j]:.8e}"

                if variables_names[var_j] == "grid_t":
                    img_caption = f"{img_caption}\nyear={years[year_idx]}"
                
                axes[time_i, var_j].text(
                    0.5, -0.4, 
                    img_caption, 
                    ha='center', 
                    va='center', 
                    transform=axes[time_i, var_j].transAxes,
                )

        fig.tight_layout()

        # Save the plot
        # -----------------------------------------
        plt.savefig("inp_plot.png")
        # breakpoint()

    if return_generated_tensor:
        return return_tensor
    else:
        # Save return_tensor as the input_blob npy file
        # -----------------------------------------
        # save the 3D blob within the simulation dir
        # train_data_dir = "/global/cfs/projectdirs/m1012/satyarth/Data/ensemble_simulation_runs/ensemble_simulation_run2/training_data/v1"
        dest_path = f"{write_dir}/{os.path.basename(sim_path)}_layer{layer_of_interest}_input_blob.npy"

        # print(f"{dest_path = }\t{return_tensor.shape = }")
        np.save(dest_path, return_tensor)


#############################
# OUTPUT
#############################
def generate_output_data_blob(
    df_row,
    data_mode,
    layer_of_interest,
    write_dir,
    return_generated_tensor=False,
):

    # Read df row values
    # ------------------
    sim_path = df_row["@sim_path@"]
    
    # Get path to each simulation file for the specified layer
    # --------------------------------------------------------
    sim_layer_globstr = f"{sim_path}/{data_mode}/layer{layer_of_interest}*.parq"
    sim_layer_paths = glob.glob(sim_layer_globstr)
    sim_layer_paths.sort()    # To make sure that the years are in order

    # Create return tensor placeholder
    # -----------------------------------------
    single_sim_df = pd.read_parquet(sim_layer_paths[0])
    single_sim_grid = gdutils._point_to_grid(single_sim_df)

    nx = int(single_sim_df["x"].max() - single_sim_df["x"].min() + 1)
    ny = int(single_sim_df["y"].max() - single_sim_df["y"].min() + 1)

    # Fetch years from the simulation filenames
    year_pattern = re.compile("[0-9]{4}y")
    years = [
        int(year_pattern.findall(slpath)[0].strip("y")) 
        for slpath in sim_layer_paths
    ]
    year_start = np.min(years)

    num_inp_variables = 1    # because only predicting concentration at the moment
    return_tensor = np.zeros([ny, nx, len(years), num_inp_variables])

    # Populating the return tensor
    for sim_layer_path in sim_layer_paths:

        sim_df = pd.read_parquet(sim_layer_path)
        sim_grid = gdutils._point_to_grid(sim_df)

        year_idx = int(year_pattern.findall(sim_layer_path)[0].strip("y")) - year_start

        return_tensor[:, :, year_idx, 0] = sim_grid
    
    # PLOT
    # ------
    if PLOT_FLAG:
        gdutils.animate_sim_over_years(
            sim_blob_3d=return_tensor[:, :, :, 0],
            filename="sim_over_years.gif",
        )

    if return_generated_tensor:
        return return_tensor
    else:
        # WRITE return_tensor as the output_blob npy file
        # -----------------------------------------
        # train_data_dir = "/global/cfs/projectdirs/m1012/satyarth/Data/ensemble_simulation_runs/ensemble_simulation_run2/training_data/v1"
        dest_path = f"{write_dir}/{os.path.basename(sim_path)}_layer{layer_of_interest}_output_blob.npy"

        # print(f"{dest_path = }\t{return_tensor.shape = }")
        np.save(dest_path, return_tensor)


def generate_input_output_blobs(
    df_row,
    data_mode,
    layer_of_interest,
    write_dir,
):
    generate_input_data_blob(
        df_row=df_row,
        data_mode=data_mode,
        layer_of_interest=layer_of_interest,
        write_dir=write_dir,
    )
    generate_output_data_blob(
        df_row=df_row,
        data_mode=data_mode,
        layer_of_interest=layer_of_interest,
        write_dir=write_dir,
    )


def main_v1():
    csv_path = "/global/cfs/cdirs/m1012/satyarth/Data/ensemble_simulation_runs/ensemble_simulation_run2/sampled_params_succ_status.csv"

    data_mode = "separate_2d_layer_info/identity_downsampled"
    layer_of_interest = 7

    sims_meta_df = pd.read_csv(csv_path)
    sims_meta_df = sims_meta_df[sims_meta_df["@successful@"]]

    # # Used for debugging
    # for i, row in sims_meta_df.iterrows():       
    #     generate_input_output_blobs(
    #         df_row=row,
    #         data_mode=data_mode,
    #         layer_of_interest=layer_of_interest,
    #     )
        
    #     import sys
    #     sys.exit()

    parallel_function(
        delayed(generate_input_output_blobs)(
            df_row=row,
            data_mode=data_mode,
            layer_of_interest=layer_of_interest,
            write_dir="/global/cfs/projectdirs/m1012/satyarth/Data/ensemble_simulation_runs/ensemble_simulation_run2/training_data/v1",
        )
        for i, row in sims_meta_df.iterrows()
    )

################################################################################





def main_v2():
    layer_of_interest = 7
    v1_dir = "/global/cfs/projectdirs/m1012/satyarth/Data/ensemble_simulation_runs/ensemble_simulation_run2/training_data/v1"

    write_dir = "/global/cfs/projectdirs/m1012/satyarth/Data/ensemble_simulation_runs/ensemble_simulation_run2/training_data/v2"

    v1_glob_str = f"{v1_dir}/sim*.npy"
    data_filepaths = glob.glob(v1_glob_str)
    data_filepaths.sort(
        key=lambda x: int(os.path.basename(x).split("_")[0].strip("sim"))
    )

    input_filepaths = [f for f in data_filepaths if "input_blob" in f]
    output_filepaths = [f for f in data_filepaths if "output_blob" in f]
    # small_dataset_size = 32
    # input_filepaths = input_filepaths[:small_dataset_size]
    # output_filepaths = output_filepaths[:small_dataset_size]

    def write_npy_to_placeholder(npy_path, idx, placeholder):
        placeholder[idx] = np.load(npy_path)

    from tqdm import tqdm
    # INPUT
    # ------
    input_tensor = [None] * len(input_filepaths)
    for i, inp_f in tqdm(enumerate(input_filepaths), desc="Input filepaths"):
        input_tensor[i] = np.load(inp_f)
        # write_npy_to_placeholder(
        #     npy_path=inp_f, 
        #     idx=i, 
        #     placeholder=input_tensor,
        # )
    # parallel_function(
    #     delayed(write_npy_to_placeholder)(
    #         npy_path=inp_f, 
    #         idx=i, 
    #         placeholder=input_tensor,
    #     )
    #     for i, inp_f in enumerate(input_filepaths)
    # )
    input_tensor = np.stack(input_tensor)
    input_tensor_writepath = f"{write_dir}/layer{layer_of_interest}_input_blob.npy"
    np.save(input_tensor_writepath, input_tensor)
    
    # OUTPUT
    # ------
    output_tensor = [None] * len(output_filepaths)
    for i, out_f in tqdm(enumerate(output_filepaths), desc="Output filepaths"):
        output_tensor[i] = np.load(out_f)
        # write_npy_to_placeholder(
        #     npy_path=outf, 
        #     idx=i, 
        #     placeholder=output_tensor,
        # )
    # parallel_function(
    #     delayed(write_npy_to_placeholder)(
    #         npy_path=out_f, 
    #         idx=i, 
    #         placeholder=output_tensor,
    #     )
    #     for i, out_f in enumerate(output_filepaths)
    # )
    output_tensor = np.stack(output_tensor)
    output_tensor_writepath = f"{write_dir}/layer{layer_of_interest}_output_blob.npy"
    np.save(output_tensor_writepath, output_tensor)


if __name__ == "__main__":
    main_v2()