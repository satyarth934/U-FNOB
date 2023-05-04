import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import generate_data_blob_utils as gdutils


# def animate_sim_over_years(sim_blob_3d, filename="sim_over_years.gif"):
#     sim_blob_3d_norm = ((sim_blob_3d - np.nanmin(sim_blob_3d)) / (np.nanmax(sim_blob_3d) - np.nanmin(sim_blob_3d)))
    
#     sim_blob_3d_gif = sim_blob_3d_norm.transpose([2, 0, 1])

#     sim_blob_3d_gif_flist = [np.flipud(layer) for layer in sim_blob_3d_gif]
#     sim_blob_3d_gif_cmap = [plt.get_cmap()(layer) for layer in sim_blob_3d_gif_flist]
    
#     import imageio
#     imageio.mimsave(filename, sim_blob_3d_gif_cmap)


# def save_simulation_blob(
#     sim_dir,
#     data_mode,
#     layer_of_interest,
# ) -> None:
#     # get a layer for all years
#     year_data_paths_globstr = f"{sim_dir}/{data_mode}/layer{layer_of_interest}_*.parq"
#     year_data_paths = glob.glob(year_data_paths_globstr)
#     year_data_paths.sort()    # to make sure that the years are sorted

#     year_data_grid_list = list()
#     # for each year:
#     for year_data_path in year_data_paths:
#         year_data_df = pd.read_parquet(year_data_path)

#         # convert data to grid
#         year_data_grid = _point_to_grid(year_data_df)

#         # append grid to the larger 3D blob
#         year_data_grid_list.append(year_data_grid)

#     sim_blob_3d = np.array(year_data_grid_list)
#     sim_blob_3d = sim_blob_3d.transpose([1,2,0])
#     assert sim_blob_3d.shape[-1] == 65    # make sure time dim has 65 values

#     # # Animate simulation over all the years and save as a GIF.
#     # animate_sim_over_years(
#     #     sim_blob_3d=sim_blob_3d, 
#     #     filename=f"{os.path.basename(sim_dir)}_layer{layer_of_interest}.gif",
#     # )

#     # save the 3D blob within the simulation dir
#     train_data_dir = "/global/cfs/projectdirs/m1012/satyarth/Data/ensemble_simulation_runs/ensemble_simulation_run2/training_data/v1"
#     # dest_path = os.path.dirname(year_data_paths_globstr).replace(data_mode, f"{data_mode}_layer{layer_of_interest}_blob3D.npy")
#     dest_path = f"{train_data_dir}/{os.path.basename(sim_dir)}_layer{layer_of_interest}_output_blob.npy"

#     print(f"{dest_path = }\t{sim_blob_3d.shape = }")
#     # np.save(sim_blob_3d)


# def generate_output_data_blob():
#     layer_of_interest = 7
#     data_mode = "separate_2d_layer_info/identity_downsampled"

#     # sim_root
#     sim_root_dir = "/global/cfs/projectdirs/m1012/satyarth/Data/ensemble_simulation_runs/ensemble_simulation_run2"

#     # list of all sims
#     sim_dirs = glob.glob(f"{sim_root_dir}/sim*")

#     # for each sim:
#     for sim_dir in sim_dirs:
#         save_simulation_blob(
#             sim_dir=sim_dir,
#             data_mode=data_mode,
#             layer_of_interest=layer_of_interest,
#         )


# def main_old():
#     generate_output_data_blob()












































#############################
# INPUT
#############################
def generate_input_data_blob(
    df_row,
    data_mode,
    layer_of_interest,
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

    PLOT_FLAG = False

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
    single_sim_grid = _point_to_grid(single_sim_df)
    nan_idxs = np.isnan(single_sim_grid)
    # nan_idxs = fill_holes(nan_idxs)    # Not being used currently

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

    # Save return_tensor as the input_blob npy file
    # -----------------------------------------
    # save the 3D blob within the simulation dir
    train_data_dir = "/global/cfs/projectdirs/m1012/satyarth/Data/ensemble_simulation_runs/ensemble_simulation_run2/training_data/v1"
    dest_path = f"{train_data_dir}/{os.path.basename(sim_path)}_layer{layer_of_interest}_input_blob.npy"

    print(f"{dest_path = }\t{return_tensor.shape = }")
    np.save(dest_path, return_tensor)


#############################
# OUTPUT
#############################
def generate_output_data_blob(
    df_row,
    data_mode,
    layer_of_interest,
):
    PLOT_FLAG = True

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
    single_sim_grid = _point_to_grid(single_sim_df)

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
        sim_grid = _point_to_grid(sim_df)

        year_idx = int(year_pattern.findall(sim_layer_path)[0].strip("y")) - year_start

        return_tensor[:, :, year_idx, 0] = sim_grid
    
    # PLOT
    # ------
    if PLOT_FLAG:
        animate_sim_over_years(
            sim_blob_3d=return_tensor[:, :, :, 0],
            filename="sim_over_years.gif",
        )

    # WRITE return_tensor as the output_blob npy file
    # -----------------------------------------
    train_data_dir = "/global/cfs/projectdirs/m1012/satyarth/Data/ensemble_simulation_runs/ensemble_simulation_run2/training_data/v1"
    dest_path = f"{train_data_dir}/{os.path.basename(sim_path)}_layer{layer_of_interest}_output_blob.npy"

    print(f"{dest_path = }\t{return_tensor.shape = }")
    np.save(dest_path, return_tensor)


def main():
    csv_path = "/global/cfs/cdirs/m1012/satyarth/Data/ensemble_simulation_runs/ensemble_simulation_run2/sampled_params_succ_status.csv"

    data_mode = "separate_2d_layer_info/identity_downsampled"
    layer_of_interest = 7

    sims_meta_df = pd.read_csv(csv_path)
    sims_meta_df = sims_meta_df[sims_meta_df["@successful@"]]

    for i, row in sims_meta_df.iterrows():
        # generate_input_data_blob(
        #     df_row=row,
        #     data_mode=data_mode,
        #     layer_of_interest=layer_of_interest,
        # )
        generate_output_data_blob(
            df_row=row,
            data_mode=data_mode,
            layer_of_interest=layer_of_interest,
        )

        
        import sys
        sys.exit()



if __name__ == "__main__":
    main()