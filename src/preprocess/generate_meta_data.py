import re
import os
import glob
import numpy as np
import pandas as pd


def generate_meta_data():
    csv_path = "/global/cfs/cdirs/m1012/satyarth/Data/ensemble_simulation_runs/ensemble_simulation_run2/sampled_params_succ_status.csv"
    csv_df = pd.read_csv(csv_path, index_col=None)
    csv_df = csv_df[csv_df['@successful@']]
    
    inp_var_cols = ["@Perm@","@Por@","@alpha@","@Rech@","@seepage@"]
    inp_var_names = ["permeability", "porosity", "alpha", "recharge", "seepage"]
    out_var_names = ["total_component_concentration.cell.Tritium conc"]

    # Get YEARS
    data_mode = "separate_2d_layer_info/identity_downsampled"
    layer_of_interest = 7
    sim_path = csv_df['@sim_path@'][0]

    sim_layer_globstr = f"{sim_path}/{data_mode}/layer{layer_of_interest}*.parq"
    sim_layer_paths = glob.glob(sim_layer_globstr)
    sim_layer_paths.sort()

    year_pattern = re.compile("[0-9]{4}y")
    years = [
        str(int(year_pattern.findall(slpath)[0].strip("y"))) 
        for slpath in sim_layer_paths
    ]

    # Formatting meta data to string
    inp_var_names_str = ", ".join(inp_var_names)
    years_str = ", ".join(years)
    inp_var_min_str = ", ".join(csv_df[inp_var_cols].min().apply(lambda x: str(x)))
    inp_var_max_str = ", ".join(csv_df[inp_var_cols].max().apply(lambda x: str(x)))
    out_var_names_str = ", ".join(out_var_names)

    meta_str = f"{inp_var_names_str}\n{years_str}\n{inp_var_min_str}\n{inp_var_max_str}\n{out_var_names_str}"

    # Write meta_data.txt    
    meta_file = "meta_data.txt"
    with open(meta_file, "w") as mf:
        mf.write(meta_str)


def main():
    generate_meta_data()


if __name__ == "__main__":
    main()