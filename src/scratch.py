import torch
import numpy as np
from models.ufno_3D import *
from models.lploss import *


def output_size_0_issue_in_view():
    mode1 = 10
    mode2 = 10
    mode3 = 4
    width = 36
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[SCRATCH] Prepping model")
    model = UFNO3d(mode1, mode2, mode3, width, UNet = True)
    model.to(device)

    # Printing model summary
    print("[SCRATCH] Prepping model summary")
    from torchsummary import summary
    summary(model, input_size=(119, 171, 65, 8))
    # import sys
    # sys.exit(0)

    print("[SCRATCH] Prepping loss")
    myloss = LpLoss(size_average=True) # relative lp loss

    print("[SCRATCH] Prepping dummy data")
    y = torch.rand((2, 119, 171, 65, 1))
    y = y[:, :, :, :64, :]
    no = y.shape[-1]
    x = torch.rand((2, 119, 171, 65, 8))
    x = x[:, :, :, :64, :]
    nz = x.shape[1]
    nx = x.shape[2]
    nt = x.shape[-2]

    mask = (x[:,:,:,0:1,0]!=0).reshape(x.shape[0], nz, nx, 1, 1).repeat(1,1,1,nt,no) # deactivate those input values with 0, i.e. outside the concentration observing zone

    print("[SCRATCH] Prediction")
    pred = model(x.cuda().float())
    pred = pred.contiguous()
    pred = pred.view(-1, nz, nx, nt, no)

    print("[SCRATCH] Computing loss")
    i = 0
    loss_val = myloss(pred[i,...][mask[i,...]].reshape(1, -1).cuda(), y[i,...][mask[i,...]].reshape(1, -1).cuda(),)
    print(f"{loss_val = }")


def loading_custom_dataset():
    ##################################
    # FDL DATA
    ##################################
    fdlinp_npy_path = "/global/cfs/cdirs/m1012/satyarth/Data/ensemble_simulation_runs/FDL2022_data/us-digitaltwiner-dev-features/data-sim-test-2D-1000_run4/input.npy"
    fdlout_npy_path = "/global/cfs/cdirs/m1012/satyarth/Data/ensemble_simulation_runs/FDL2022_data/us-digitaltwiner-dev-features/data-sim-test-2D-1000_run4/output.npy"

    fdlinp_np_data = np.load(fdlinp_npy_path)
    fdlout_np_data = np.load(fdlout_npy_path)

    fdl_tdataset = torch.utils.data.TensorDataset(
        torch.from_numpy(fdlinp_np_data),
        torch.from_numpy(fdlout_np_data),
    )

    ##################################
    # Custom DATA
    ##################################
    custinp_dir_path = "/global/cfs/cdirs/m1012/satyarth/Data/ensemble_simulation_runs/ensemble_simulation_run2/training_data/v1/"

    from data.fdl_data import FDLFormatDatasetV1
    cust_dataset = FDLFormatDatasetV1(
        data_dir=custinp_dir_path,
        layer_num=7,
    )

    print(f"{len(cust_dataset) = }")
    print(f"{len(cust_dataset[0]) = }")
    print(f"{cust_dataset[0][0].shape = }")
    print(f"{cust_dataset[0][1].shape = }")


def scaling_data():
    data_path = '/global/cfs/projectdirs/m1012/satyarth/Data/ensemble_simulation_runs/ensemble_simulation_run2/training_data/v1'
    # f_input = f"{data_path}/layer7_input_blob_small.npy"
    # f_output = f"{data_path}/layer7_output_blob_small.npy"
    
    # print("Read files from paths")
    # input_array = torch.from_numpy(np.load(f_input)) 
    # output_array = torch.from_numpy(np.load(f_output))
    
    # input_array = input_array[:, :, :, :64, :]
    # output_array = output_array[:, :, :, :64, :]
    

    # from collections import namedtuple
    # meta_data_path = f"{data_path}/meta_data.txt"
    # meta_data = namedtuple(
    #     'meta_data', 
    #     [
    #         'input_names',
    #         'time_steps',
    #         'input_min',
    #         'input_max',
    #     ]
    # )
    # with open(meta_data_path) as f:
    #     lines = f.readlines()

    # meta_data.input_names = str(lines[0]).strip().split(", ")
    # meta_data.time_steps = np.array(str(lines[1]).strip().split(", "), dtype = 'float64')
    # meta_data.time_steps = np.array(meta_data.time_steps, dtype = 'int64')    # FIXME: Seems redundant.
    # meta_data.input_min = np.array(str(lines[2]).strip().split(", "), dtype = 'float64')
    # meta_data.input_max = np.array(str(lines[3]).strip().split(", "), dtype = 'float64')

    # print(f"{meta_data.input_max.reshape(1,1,1,1,5) = }")

    from data import custom_transforms as ct
    from data.fdl_data import FDLFormatDatasetV1
    sit = ct.ScaleInputTransform(meta_data_file_path=f"{data_path}/meta_data.txt")

    ds = FDLFormatDatasetV1(
        data_dir=data_path,
        layer_num=7,
        num_years=64,
    )

    data, target = ds[1]
    sit(data)



if __name__ == "__main__":
    # output_size_0_issue_in_view()
    # loading_custom_dataset()
    scaling_data()