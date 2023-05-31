#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
# from ufno_3D import *
from models.ufno_3D import *
# from lploss import *
from models.lploss import *
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# import seaborn as sns
from io import BytesIO
# import tensorflow as tf
# from tensorflow.python.lib.io import file_io
from tqdm import tqdm
import torch.utils.data as data
import os
from io import StringIO
# from google.cloud import storage
import imageio
import gc


# Empty cache before starting
gc.collect()
torch.cuda.empty_cache()


# LOAD INPUT PARAMETER DATA
# --------------------------
csv_path = "/global/cfs/cdirs/m1012/satyarth/Data/ensemble_simulation_runs/ensemble_simulation_run2/sampled_params_succ_status.csv"
# bucket_name = "us-digitaltwiner-landing"
# storage_client = storage.Client()
# bucket = storage_client.get_bucket(bucket_name)

# blob = bucket.get_blob("ensemble_simulation_run4_fdl2d_allvars/sampled_params_succ_status.csv")
# params_string = blob.download_as_text()
# params_string = StringIO(params_string)
sim_parameters = pd.read_csv(csv_path, index_col=0)
sim_parameters = sim_parameters.rename({"@serial_number@": "sim_id","@sim_path@": "path", "@Perm@": "permeability","@Por@": "porosity", "@alpha@": "alpha","@sr@":"residual water content","@m@":"m", "@Rech@": "recharge", "@Rech_hist@": "recharge history","@Rech_mid@": "recharge mid-century", "@Rech_late@": "recharge late-century", "@seepage@": "seepage","@seepage_conc@":"seepage concentration","@cap_rate@":"cap rate",  "@successful@":"successful"}, axis=1)
sim_parameters.index = sim_parameters["sim_id"]
sim_parameters.drop("sim_id", axis=1, inplace=True)
# sim_parameters['path'] = sim_parameters['path'].str.replace(r'/global/scratch/users/satyarth/Projects/ensemble_simulation_runs/', '')
sim_parameters['path'] = sim_parameters['path'].str.replace(r'/global/cfs/projectdirs/m1012/satyarth/Data/ensemble_simulation_runs/', '')    # XXX: Might be useless. Delete if this command serves no purpose.
sim_parameters = sim_parameters[sim_parameters["successful"]==True].drop("successful", axis=1)
# for recharge combination
# order = ['path','permeability','porosity','alpha','residual water content','m','recharge history','seepage','seepage concentration','cap rate', 'recharge mid-century','recharge late-century']
# sim_parameters = sim_parameters[order]

# print(f"{sim_parameters.columns = }")
# print(f"{sim_parameters.shape = }")
# print(sim_parameters)

# import sys; sys.exit(0)

# data_path = 'us-digitaltwiner-dev-features/data-sim-test-2D-1000_run4/'
# f_input = BytesIO(file_io.read_file_to_string("gs://" + data_path + 'input_top_layer.npy', binary_mode=True)) 
# f_output = BytesIO(file_io.read_file_to_string("gs://" + data_path + 'output.npy', binary_mode=True))
# input_array = torch.from_numpy(np.load(f_input)) 
# output_array = torch.from_numpy(np.load(f_output))

data_path = '/global/cfs/projectdirs/m1012/satyarth/Data/ensemble_simulation_runs/ensemble_simulation_run2/training_data/v3'
f_input = f"{data_path}/layer7_input_blob_small.npy"
f_output = f"{data_path}/layer7_output_blob_small.npy"
    # f_input = f"{data_path}/layer7_input_blob.npy"
    # f_output = f"{data_path}/layer7_output_blob.npy"

print("Read files from paths")
input_array = torch.from_numpy(np.load(f_input)) 
output_array = torch.from_numpy(np.load(f_output))
input_array = input_array[:, :, :, :64, :]
output_array = output_array[:, :, :, :64, :]

# size of array from the input
ns, nz, nx, nt, nc = input_array.shape
no = output_array.shape[-1]
nc = nc - 3

# calculate the statistics of output_array
output_array_mean = np.mean(output_array.numpy(),axis = 0)
output_array_std = np.std(output_array.numpy(),axis = 0)
output_array_max = np.max(output_array.numpy(),axis = 0)
output_array_min = np.min(output_array.numpy(),axis = 0)

# # meta_data
# f = (BytesIO(file_io.read_file_to_string("gs://" + data_path + 'meta_data.txt', binary_mode=True)))
# lines = f.readlines()
# input_names = str(lines[0]).split('\'')[1].split('\\n')[0].split(', ')
# time_steps = np.array(str(lines[1]).split('\'')[1].split('\\n')[0].split(', '),dtype = 'float64')
# time_steps = np.array(time_steps, dtype = 'int64')
# input_min = np.array(str(lines[2]).split('\'')[1].split('\\n')[0].split(', '),dtype = 'float64')
# input_max = np.array(str(lines[3]).split('\'')[1].split('\\n')[0].split(', '),dtype = 'float64')
# output_names = str(lines[4]).split('\'')[1].split('\\n')[0].split(', ')

# tritium_MCL = 7e-13
# # Custom min and max values per variable for rescaling
# rescale_factors = {
#     0 : {
#         'min': np.nanmin(output_array[:,:,:,:,0]),
#         'max': np.nanmax(output_array[:,:,:,:,0])/2
#     },
#     1 : {
#         'min': np.nanmin(output_array[:,:,:,:,1]),
#         'max': np.nanmax(output_array[:,:,:,:,1])/5
#     },
#     2 : {
#         'min': np.nanmin(output_array[:,:,:,:,2]),
#         'max': np.nanmax(output_array[:,:,:,:,2])
#     },
#     3 : {
#         'min': np.nanmin(output_array[:,:,:,:,3]),
#         'max': np.nanmax(output_array[:,:,:,:,3])
#     },
#     4 : {
#         'min': np.nanmin(output_array[:,:,:,:,4]),
#         'max': np.nanmax(output_array[:,:,:,:,4])
#     },
#     5 : {
#         'min': tritium_MCL*0.2,
#         'max': 9e-9
#     },
#     6 : {
#         'min': np.nanmin(output_array[:,:,:,:,6]),
#         'max': np.nanmax(output_array[:,:,:,:,6])
#     }
# }

# meta_data
print("Read meta data")

meta_data_path = f"{data_path}/meta_data.txt"
    
with open(meta_data_path) as f:
    lines = f.readlines()
# lines = f.readlines()

# input_names = str(lines[0]).split('\'')[1].split('\\n')[0].split(', ')
input_names = str(lines[0]).strip().split(", ")
# time_steps = np.array(str(lines[1]).split('\'')[1].split('\\n')[0].split(', '),dtype = 'float64')
time_steps = np.array(str(lines[1]).strip().split(", "), dtype = 'float64')
time_steps = np.array(time_steps, dtype = 'int64')
# input_min = np.array(str(lines[2]).split('\'')[1].split('\\n')[0].split(', '),dtype = 'float64')
input_min = np.array(str(lines[2]).strip().split(", "), dtype = 'float64')
# input_max = np.array(str(lines[3]).split('\'')[1].split('\\n')[0].split(', '),dtype = 'float64')
input_max = np.array(str(lines[3]).strip().split(", "), dtype = 'float64')
# output_names = str(lines[4]).split('\'')[1].split('\\n')[0].split(', ')
output_names = str(lines[4]).strip().split(", ")
print(f"Sanity check {output_names = }")

# rescale output
print(f"Rescaling")
tritium_MCL = 7e-13
# Custom min and max values per variable for rescaling
rescale_factors = {
    0 : {
        'min': tritium_MCL*0.2,
        'max': 9e-9
    },
}

# # Rescale input
# input_max_values = np.nanmax(input_array.reshape(-1,nc+3),axis = 0).reshape(1,1,1,1,-1)
# input_array = input_array/input_max_values

# # Input nan -> 0
# input_array[np.isnan(input_array)] = 0

# # Rescale output_array between 0 and 1.

# scaled_output = output_array.detach().clone()

# for i in range(no):
#     scaled_output[:,:,:,:,i][scaled_output[:,:,:,:,i]<rescale_factors[i]['min']] = rescale_factors[i]['min']
#     #if(i==5):
#     scaled_output[:,:,:,:,i][scaled_output[:,:,:,:,i]>rescale_factors[i]['max']] = rescale_factors[i]['max']
#     scaled_output[:,:,:,:,i] = (scaled_output[:,:,:,:,i] - rescale_factors[i]['min'])/(rescale_factors[i]['max']-rescale_factors[i]['min'])

# scaled_output[np.isnan(scaled_output)] = 0

# Rescale input
input_max_values = np.nanmax(input_array.reshape(-1,nc+3),axis = 0).reshape(1,1,1,1,-1)
input_array = input_array/input_max_values

# Input nan -> 0
print(f"Set nan -> 0")
input_array[input_array.isnan()] = 0

# Rescale output_array between 0 and 1.

scaled_output = output_array.detach().clone()

for i in range(no):
    scaled_output[:,:,:,:,i][scaled_output[:,:,:,:,i]<rescale_factors[i]['min']] = rescale_factors[i]['min']
    scaled_output[:,:,:,:,i][scaled_output[:,:,:,:,i]>rescale_factors[i]['max']] = rescale_factors[i]['max']
    scaled_output[:,:,:,:,i] = (scaled_output[:,:,:,:,i] - rescale_factors[i]['min'])/(rescale_factors[i]['max']-rescale_factors[i]['min'])

scaled_output[scaled_output.isnan()] = 0


# # Current training
# selected_idx = np.array([0,1,2,5])
# scaled_output_4 = scaled_output[:,:,:,:,selected_idx]
# output_names_4 = list(np.array(output_names)[selected_idx])

# Current training
# selected_idx = np.array([0,1,2,5])
selected_idx = np.array([0])    # 0th index is the output variable index
scaled_output_4 = scaled_output[:,:,:,:,selected_idx]
output_names_4 = list(np.array(output_names)[selected_idx])
print(f"{output_names_4 = }")


# LOAD MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torch.load('saved_models/dP_UFNO3D_UNet_False_2ep_36width_10m1_10m2_664train_1.00e-03lr_b1_0b2_0b3_0b4_0b5_0b6_0b7_0')
model = torch.load('saved_models/dP_UFNO3D_UNet_True_30ep_36width_10m1_10m2_32train_3.21e-04lr_b1_0b2_0b3_0b4_0b5_0b6_0b7_0')
model.to(device)

# Printing model summary
from torchsummary import summary
summary(model, input_size=(119, 171, 8, 8))    # 8 because we did not consider all the years worth of data. We selected only few years at regular intervals. Check dataset v3 datacard for more information.


batch_size = 8
torch_dataset = torch.utils.data.TensorDataset(input_array, scaled_output_4)
dataset_sizes = [
    np.int32(np.int32(ns*0.8)/batch_size)*batch_size, 
    np.int32((ns-np.int32(np.int32(ns*0.8)/batch_size)*batch_size)/2), 
    np.int32((ns-np.int32(np.int32(ns*0.8)/batch_size)*batch_size)/2),
]
train_data, val_data, test_data = data.random_split(torch_dataset, dataset_sizes ,generator=torch.Generator().manual_seed(0))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader  = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

params_train = sim_parameters.iloc[train_data.indices]
params_val = sim_parameters.iloc[val_data.indices]
params_test = sim_parameters.iloc[test_data.indices]


# output_names_4 = ['Darcy velocity (x)', 'Darcy velocity (z)', 'Hydraulic head', 'Tritium concentration']
# output_units_4 = [ 'm/s', 'm/s', 'm','mol/L']
output_names_4 = ['Tritium concentration']
output_units_4 = ['mol/L']    # FIXME: Check if the unit is correct for tritium concentration. Correct it if not.


# PERFORM INFERENCE
data_type = 'Test'
# data_type = 'Train'

if(data_type=='Train'):
    data_loader = train_loader
    samples = dataset_sizes[0]
    input_params = params_train
    bs = batch_size
if(data_type=='Val'):
    data_loader = val_loader
    samples = dataset_sizes[1]
    input_params = params_val
    bs = batch_size
if(data_type=='Test'):
    data_loader = test_loader
    samples = dataset_sizes[2]
    input_params = params_test
    bs = 1


print(f"{samples = }")
print(f"{nz = }")
print(f"{nx = }")
print(f"{nt = }")
print(f"{bs = }")
# num_output_variables = 4
num_output_variables = 1
y_true_all = np.zeros((samples, nz, nx, nt, num_output_variables))
y_pred_all = np.zeros((samples, nz, nx, nt, num_output_variables))
i = 0
for x, y in data_loader:
    x, y = x.to(device), y.to(device)
    y_pred = model(x.float())
    try:
        if len(y_pred.shape) == 3:    # missing batch dimension and output variable dimension
            y_pred = y_pred.unsqueeze(0)
        if len(y_pred.shape) == 4:    # missing output variable dimension
            y_pred = y_pred.unsqueeze(-1)
        y_true_all[i:(i+bs),:] = y.cpu().detach().numpy()
        y_pred_all[i:(i+bs),:] = y_pred.cpu().detach().numpy()
    except ValueError as e:
        raise e
    i = i+bs

# REVERSE RESCALE
# selected_idx = np.array([0,1,2,5])
selected_idx = np.array([0])
for i in range(len(selected_idx)):
    # if(i==5):
    #     i=3
    y_true_all[:,:,:,:,i] = y_true_all[:,:,:,:,i] * (rescale_factors[selected_idx[i]]['max']-rescale_factors[selected_idx[i]]['min']) + rescale_factors[selected_idx[i]]['min']
    y_pred_all[:,:,:,:,i] = y_pred_all[:,:,:,:,i] * (rescale_factors[selected_idx[i]]['max']-rescale_factors[selected_idx[i]]['min']) + rescale_factors[selected_idx[i]]['min']

diff_all = y_true_all - y_pred_all


vis_mask = ((x[0,:,:,0,0]!=0).cpu().detach().numpy())*1.0
vis_mask[vis_mask==0] = np.nan

mask = (input_array[0,:,:,0:1,0]!=0).reshape(nz, nx, 1).repeat(1,1,nt).cpu().detach().numpy()



def visualize_sample_at_time_t(
    sample_num, 
    output_var, 
    time, 
    var_min=None, 
    var_max=None, 
    save=False, 
    path=".", 
    diff_min=None, 
    diff_max=None, 
    show_param=False, 
    param_name= "",
) -> None:
    fig, ax = plt.subplots(1,3, figsize=(20,5), dpi=200)
    font_size = 15
    aspect = 1
    
    if(var_min==None):
        var_min = np.nanmin(y_true_all[sample_num,:,:,:,output_var]*mask)
    if(var_max==None):
        var_max = np.nanmax(y_true_all[sample_num,:,:,:,output_var]*mask)
    if(diff_min==None):
        diff_min = np.nanmin(diff_all[sample_num,:,:,:,output_var]*mask)
    if(diff_max==None):
        diff_max = np.nanmax(diff_all[sample_num,:,:,:,output_var]*mask)

    left = ax[0].imshow(
        (y_true_all[sample_num, :,:,time, output_var]*vis_mask).reshape(nz,nx), 
        aspect=aspect, 
        origin ='lower', 
        vmin=var_min, 
        vmax=var_max,
    )
    middle = ax[1].imshow(
        (y_pred_all[sample_num, :,:,time, output_var]*vis_mask).reshape(nz,nx), 
        aspect=aspect, 
        origin ='lower', 
        vmin=var_min, 
        vmax=var_max,
    )
    right = ax[2].imshow(
        (diff_all[sample_num, :,:,time, output_var]*vis_mask).reshape(nz,nx), 
        aspect=aspect, 
        origin ='lower', 
        cmap = 'RdBu_r', 
        vmin=diff_min, 
        vmax=diff_max,
    )
    
    # TEXT BOX SETTINGS
    if(show_param):
        value = input_params.iloc[sample_num][param_name]
        textstr = "$\\bf{"+param_name.replace(' ', '\ ').capitalize()+"}$" + f":\n {value:.4E}"
        props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
        ax[0].text(0.01, 1.2, textstr, transform=ax[0].transAxes, fontsize=font_size-2,
                verticalalignment='top', bbox=props)


    # TITLES
    fig.suptitle("{}\n\nSample {} of {} simulations\n{}".format(time_steps[time],sample_num+1, samples, output_names_4[output_var]), fontsize=font_size, fontweight="bold", y=1)

    subtitles = ["Ground Truth", "Prediction", "Difference"]
    for i in range(ax.shape[0]):
        ax[i].axis('off')
        ax[i].set_title(subtitles[i], y=-0.1)

    # COLORBAR SETTINGS
    p0 = ax[0].get_position().get_points().flatten()
    p1 = ax[1].get_position().get_points().flatten()
    p2 = ax[2].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p0[0], 0.0, p1[2]-p0[0], 0.05])
    fig.colorbar(
        left, cax=ax_cbar, orientation='horizontal',
    ).set_label(
        label=f"{output_names_4[output_var]} ({output_units_4[output_var]})", 
        size=font_size, 
        labelpad=15,
    )
    ax_cbar1 = fig.add_axes([p2[0], 0.0, p2[2]-p2[0], 0.05])
    fig.colorbar(right, cax=ax_cbar1, orientation='horizontal').set_label(label="{} ({})".format(output_names_4[output_var], output_units_4[output_var]), size=font_size, labelpad=15)
    if(save):
        sub_folder = str("{}__Sample_{}_of_{}".format(output_names_4[output_var], sample_num, samples)).replace(" ", "")
        file_name = str("Time_{}__{}__Sample_{}_of_{}.png".format(time_steps[time], output_names_4[output_var], sample_num, samples)).replace(" ", "")
        if not os.path.exists(path+'/'+sub_folder):
            os.makedirs(path+'/'+sub_folder)
        plt.ioff()
        plt.savefig(path+'/'+sub_folder+'/'+file_name, bbox_inches='tight')



def create_sim_sequence(sample_num, output_var, path, var_min=None, var_max=None, save=False, diff_min=None, diff_max=None, show_param=False, param_name= ""):
    for t in tqdm(range(time_steps.shape[0])):
        visualize_sample_at_time_t(sample_num = sample_num, output_var = output_var, time = t, save=True, path=path,var_min=var_min, var_max=var_max, diff_min=diff_min, diff_max=diff_max, show_param=show_param, param_name= param_name)
def create_sim_animation(source_path, destination_path, fps=4):
    images = []
    for file_name in sorted(os.listdir(source_path)):
        if file_name.endswith('.png'):
            file_path = os.path.join(source_path, file_name)
            images.append(imageio.v2.imread(file_path))
            
    # imageio.mimsave(destination_path, images, fps=fps)
    imageio.mimsave(
        destination_path, 
        images, 
        duration=(1000 * (1/fps)),
        loop=0,
    )


# sample_num = 52
# output_var = 3
sample_num = 0
output_var = 0
print(f"{sample_num = }\t\t{output_var = }")
print(f"{input_params.columns = }")
# visualize_sample_at_time_t(sample_num=sample_num, output_var = output_var,time = 4, show_param=True, param_name='seepage concentration', diff_min=None, diff_max=None)
visualize_sample_at_time_t(
    sample_num=sample_num, 
    output_var=output_var, 
    time=4, 
    show_param=True, 
    param_name='seepage', 
    diff_min=None, 
    diff_max=None,
    save=True,
)


# In[14]:


parent_folder = 'Visualizations'
figures_folder = 'Figures'
animation_folder = 'Animations'

if not os.path.exists(parent_folder):
    os.makedirs(parent_folder)
if not os.path.exists(parent_folder+'/'+figures_folder):
    os.makedirs(parent_folder+'/'+figures_folder)
if not os.path.exists(parent_folder+'/'+animation_folder):
    os.makedirs(parent_folder+'/'+animation_folder)


# In[15]:


# sample_num = 50
# output_var = 3
sample_num = 0
output_var = 0
diff_min = -1.5e-9
diff_max = 1.5e-9 

path = parent_folder+'/'+figures_folder
sub_folder = str("{}__Sample_{}_of_{}".format(output_names_4[output_var], sample_num, samples)).replace(" ", "")
animation_file_name = ("UFNO-3D_{}__Sample_{}_of_{}.gif".format(output_names_4[output_var], sample_num, samples)).replace(" ", "")
# create_sim_sequence(sample_num, output_var, path, show_param=True, param_name='seepage concentration', diff_min=diff_min, diff_max=diff_max)
create_sim_sequence(sample_num, output_var, path, show_param=True, param_name='seepage', diff_min=diff_min, diff_max=diff_max)
create_sim_animation(source_path=path+'/'+sub_folder, destination_path=parent_folder+'/'+animation_folder+'/'+animation_file_name)


# Inference with different parameters
def change_input_sample_value(
    sample_num=30, 
    values=[3e-12],
    name=['permeability'],
) -> torch.Tensor:
    """Setting new values for the specified input variable.

    Args:
        sample_num (int, optional): Sample number that is to be modified. Defaults to 30.
        values (list, optional): Value that is to be set for the corresponding input variable name. Defaults to [3e-12].
        name (list, optional): Name of the input variable that is to be modified. Corresponding value in the `values` list is used against the respective variable name. Defaults to ['permeability'].

    Returns:
        torch.Tensor: `sample_num`th sample with new `values` for the variables specified in the `name` list.
    """
    input_max_recover = np.zeros(sim_parameters.shape[1]-1)
    # num_input_variables = 8
    # input_max_recover[:8] = input_max_values.reshape(-1)[:8]
    # input_max_recover[8] = input_max_recover[6]    # QUESTION: Seems useless at the moment. Check why it is important.
    # input_max_recover[9:] = input_max_recover[5]    # QUESTION: Seems useless at the moment. Check why it is important.
    num_inp_vars = 5
    input_max_recover[:num_inp_vars] = input_max_values.reshape(-1)[:num_inp_vars]
    # input_max_recover[num_inp_vars] = input_max_recover[num_inp_vars-2]
    # input_max_recover[num_inp_vars+1:] = input_max_recover[num_inp_vars-3]

    default_input_values = np.array(sim_parameters.values[:,1:sim_parameters.shape[1]],dtype = 'float64')[sample_num,:]/input_max_recover
    default_input_values = pd.DataFrame(default_input_values.reshape(1,-1),columns = sim_parameters.columns[1:])
    
    test_input = torch.clone(input_array[sample_num:(sample_num+1),:,:,:,:]).numpy()
    for idx,changed_name in enumerate(name):
        i = np.where(default_input_values.columns==changed_name)[0][0]
        test_input[test_input==default_input_values[default_input_values.columns[i]].values] = values[idx]/input_max_recover[i]

    return torch.from_numpy(test_input)



def visualization_change_parameters(
    sample_num=30, 
    output_var=3, 
    time=5, 
    static_val=1e-5, 
    var_val_max=2e-5, 
    interval=1e-6, 
    parameter_to_change='recharge mid-century', 
    path=path, 
    diff_min=-1e-10, 
    diff_max=1e-10, 
    fps=4,
) -> None:
    variable_val = np.arange(static_val+interval, var_val_max, interval)
    print(f"Number of frames: {len(variable_val)}, \n{variable_val}")

    x_static = change_input_sample_value(
        sample_num=sample_num, 
        values=[
            # 1e-5, 
            static_val, 
            # 1e-5,
        ],
        name=[
            # 'recharge history', 
            parameter_to_change,
            # 'recharge late-century',
        ],
    )
    y_static =  model(x_static.float().to(device)).cpu().detach()
    if len(y_static.shape) == 3:
        y_static = y_static.unsqueeze(-1).numpy()

    # x = torch.zeros((1,nz,nx,nt,sim_parameters.columns[1:].shape[0], len(variable_val)))
    x = torch.zeros((1,nz,nx,nt,sim_parameters.columns[1:].shape[0]+3, len(variable_val)))
    # y = torch.zeros((nz,nx,nt,4, len(variable_val)))
    y = torch.zeros((nz,nx,nt,num_output_variables, len(variable_val)))
    for i in range(len(variable_val)):
        x[:,:,:,:,:,i] = change_input_sample_value(
            sample_num=sample_num, 
            values=[
                # 1e-5, 
                variable_val[i], 
                # 1e-5,
            ],
            name=[
                # 'recharge history', 
                parameter_to_change,
                # 'recharge late-century',
            ],
        )
        y_pred_temp = model(
            (x[:,:,:,:,:,i]).float().to(device)
        ).cpu().detach()
        try:
            if y_pred_temp.shape == 4:
                y[:,:,:,:,i] = y_pred_temp
            elif y_pred_temp.shape == 3:
                y[:,:,:,:,i] = y_pred_temp.unsqueeze(-1)
        except Exception as e:
            raise e

    # REVERSE RESCALE
    # selected_idx = np.array([0,1,2,5])
    selected_idx = np.array([0])
    for j in range(len(selected_idx)):
        if(i==5):
            i=3
        y_static[:,:,:,j] = y_static[:,:,:,j] * (rescale_factors[selected_idx[j]]['max']-rescale_factors[selected_idx[j]]['min']) + rescale_factors[selected_idx[j]]['min']
        y[:,:,:,j,:] = y[:,:,:,j,:] * (rescale_factors[selected_idx[j]]['max']-rescale_factors[selected_idx[j]]['min']) + rescale_factors[selected_idx[j]]['min']
    
    
    def visualize_sample(sample_num, output_var, time, static_val, change_val, y_static, y_change, diff, path='', var_min=None, var_max=None, save=False, diff_min=None, diff_max=None, show_param=False, param_name= "", fps=4):
        fig, ax = plt.subplots(1,3, figsize=(20,5), dpi=200)
        font_size = 15
        # aspect = 6
        aspect = 1
        
        if(var_min==None):
            var_min = np.nanmin(y[ :,:,time, output_var,:])
        if(var_max==None):
            var_max = np.nanmax(y[ :,:,time, output_var,:])


        left = ax[0].imshow((y_static[ :,:,time, output_var]*vis_mask).reshape(nz,nx), aspect=aspect, origin ='lower', vmin=var_min, vmax=var_max)
        middle = ax[1].imshow((y_change[ :,:,time, output_var]*vis_mask).reshape(nz,nx), aspect=aspect, origin ='lower',vmin=var_min, vmax=var_max)
        right = ax[2].imshow((diff[ :,:,time, output_var]*vis_mask).reshape(nz,nx), aspect=aspect, origin ='lower', cmap = 'RdBu_r', vmin=diff_min, vmax=diff_max)
        
        # TEXT BOX SETTINGS
        if(show_param):
            value = input_params.iloc[sample_num][param_name]
            textstr = "$\\bf{"+param_name.replace(' ', '\ ').capitalize()+"}$" + f":\n {value:.4E}"
            props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
            ax[0].text(0.01, 1.2, textstr, transform=ax[0].transAxes, fontsize=font_size-2,
                    verticalalignment='top', bbox=props)


        # TITLES
        fig.suptitle("{}\n\n {}".format(time_steps[time], output_names_4[output_var]), fontsize=font_size, fontweight="bold", y=1)

        subtitles = [f"{parameter_to_change}:\n{static_val: .2E}", f"{parameter_to_change}:\n{change_val:.2E}", "Difference\n"]
        for i in range(ax.shape[0]):
            ax[i].axis('off')
            ax[i].set_title(subtitles[i], y=-0.2)

        # COLORBAR SETTINGS
        p0 = ax[0].get_position().get_points().flatten()
        p1 = ax[1].get_position().get_points().flatten()
        p2 = ax[2].get_position().get_points().flatten()
        # ax_cbar = fig.add_axes([p0[0], 0.1, p1[2]-p0[0], 0.05])
        ax_cbar = fig.add_axes([p0[0], 0.0, p1[2]-p0[0], 0.05])
        fig.colorbar(left, cax=ax_cbar, orientation='horizontal').set_label(label="{} ({})".format(output_names_4[output_var], output_units_4[output_var]), size=font_size, labelpad=15)
        # ax_cbar1 = fig.add_axes([p2[0], 0.1, p2[2]-p2[0], 0.05])
        ax_cbar1 = fig.add_axes([p2[0], 0.0, p2[2]-p2[0], 0.05])
        fig.colorbar(right, cax=ax_cbar1, orientation='horizontal').set_label(label="{} ({})".format(output_names_4[output_var], output_units_4[output_var]), size=font_size, labelpad=15)
        if(save):
            sub_folder = str("UFNO-3D__time_{}__{}__{}__max_{}".format(time_steps[time],output_names_4[output_var], parameter_to_change, var_val_max)).replace(" ", "")
            file_name = str("Time_{}__{}__Sample_{}_of_{}__val_{}.png".format(time_steps[time], output_names_4[output_var], sample_num, samples, change_val)).replace(" ", "")
            if not os.path.exists(path+'/'+sub_folder):
                os.makedirs(path+'/'+sub_folder)
            plt.ioff()
            plt.savefig(path+'/'+sub_folder+'/'+file_name, bbox_inches='tight')
    
    def create_sim_animation(source_path, destination_path, fps=4):
        images = []
        for file_name in sorted(os.listdir(source_path)):
            if file_name.endswith('.png'):
                file_path = os.path.join(source_path, file_name)
                images.append(imageio.v2.imread(file_path))
                
        imageio.mimsave(
            destination_path, 
            images, 
            # fps=4,
            duration=(1000 * (1/fps)),
            loop=0,
        )

    for i in tqdm(range(len(variable_val))):
    # for i in range(1):
        change_val = variable_val[i]
        y_change = y[:,:,:,:,i]
        diff = y_change - y_static
        # print(static_val, change_val)
        visualize_sample(sample_num, output_var, time, static_val, change_val, y_static, y_change, diff, path, save=True,diff_min=diff_min, diff_max=diff_max)
    

    path = parent_folder+'/'+figures_folder
    sub_folder = str("UFNO-3D__time_{}__{}__{}__max_{}".format(time_steps[time],output_names_4[output_var], parameter_to_change, var_val_max)).replace(" ", "")
    animation_file_name = str("UFNO-3D__time_{}__{}__{}__max_{}.gif".format(time_steps[time], output_names_4[output_var], parameter_to_change, var_val_max)).replace(" ", "")
    create_sim_animation(source_path=path+'/'+sub_folder, destination_path=parent_folder+'/'+animation_folder+'/'+animation_file_name, fps=fps)


# FIXME, THINK: Investigate more on this part! Maybe it's working but I am not sure if the plots are as expected. Investigate more! Think and draft the expectations first before changing the code.
visualization_change_parameters(
    # sample_num=30, 
    # output_var=3, 
    sample_num=0,
    output_var=0, 
    time=3,
    static_val=1e-5, var_val_max=6e-5, interval=2e-6, 
    # parameter_to_change='recharge mid-century', 
    parameter_to_change='recharge', 
    path=path,
    diff_min=-1e-10, diff_max=1e-10,
    fps=8)

