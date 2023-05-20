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


import imageio
def util_create_gif(tensor, axis, filename):
    # Normalize tensor values between 0 and 255
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    tensor = 255 * (tensor - tensor_min) / (tensor_max - tensor_min)

    # Convert tensor to uint8
    tensor = tensor.astype(np.uint8)

    # Transpose tensor to align the specified axis with the first axis
    tensor = np.transpose(tensor, (axis, *range(axis), *range(axis+1, tensor.ndim)))

    # Create a list of image frames
    frames = [tensor[i] for i in range(tensor.shape[0])]

    # Save the frames as a GIF file
    imageio.mimsave(filename, frames, duration=0.3)


import imageio
import matplotlib.pyplot as plt
import numpy as np

def util_create_gif_with_annotations(tensor, axis, filename):

    frames = []
    
    for i in range(tensor.shape[axis]):
        frame = np.take(tensor, i, axis=axis)
        fig, ax = plt.subplots()
        ax.imshow(frame)
        
        ax.text(10, 10, f"Axis: {i}", color='white', fontsize=12, fontweight='bold')
        ax.axis('off')

        # plt.savefig(f"del_dy_pred_dx_0_frame{i}.png")
        
        fig.canvas.draw()
        fig_buffer = fig.canvas.tostring_rgb()
        image = np.frombuffer(fig_buffer, dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        frames.append(image)
        plt.close()
    
    imageio.mimsave(filename, frames, loop=0, duration=0.5)




def loss_nan_issue():
    # MODEL
    # --------
    mode1 = 10
    mode2 = 10
    mode3 = 4
    width = 36
    model = UFNO3d(mode1, mode2, mode3, width, UNet=True)
    model.load_state_dict(torch.load('del_model_ckpt.pth'))
    
    # from torchsummary import summary
    # summary(model.cuda(), input_size=(119, 171, 8, 8))    # 64 because we ignored the 

    # FUNCTION ARGS
    # --------------
    beta3 = 0
    beta4 = 0
    axis = 0

    x = torch.from_numpy(np.load("del_x.npy"))
    y = torch.from_numpy(np.load("del_y.npy"))

    breakpoint()

    myloss = LpLoss(size_average=True) # relative lp loss

    # PREPARE VARIABLES
    # ------------------
    grid_dx = torch.from_numpy(np.load("del_grid_dx.npy"))
    grid_dz = torch.from_numpy(np.load("del_grid_dz.npy"))

    current_ns = x.shape[0]
        
    if len(x.shape)==5: # 3D
        nt = x.shape[-2]
    else: #2D
        nt = 1

    nz = x.shape[1]
    nx = x.shape[2]

    if len(x.shape)<5:
        mask = (x[:,:,:,0]!=0).reshape(-1,nz,nx,1).repeat(1,1,1,nt) 
    elif len(x.shape)==5:   
        mask = (x[:,:,:,0:1,0]!=0).repeat(1,1,1,nt) # deactivate those input values with 0, i.e. above the surface

    tritium_MCL = 7e-13
    rescale_factors = {
        0 : {
            'min': tritium_MCL*0.2,
            'max': 9e-9
        },
    }
    selected_idx = np.array([0])    # 0th index is the output variable index
    MCL_threshold = (tritium_MCL-rescale_factors[selected_idx[axis]]['min'])/(rescale_factors[selected_idx[axis]]['max']-rescale_factors[selected_idx[axis]]['min'])
    y = (y>MCL_threshold)*1

    dy_dx = (y[:,:,2:,:] - y[:,:,:-2,:])/grid_dx[:,:,:,:,0]
    dy_dx[dy_dx.isnan()] = 0
    dy_dz = (y[:,2:,:,:] - y[:,:-2,:,:])/grid_dz[:,:,:,:,0]

    # num_output_variables = 4
    num_output_variables = 1
    pred = model(x.float()).view(-1, nz, nx, nt, num_output_variables)[:,:,:,:,axis]
    pred = (pred>MCL_threshold)*1

    der_x_loss = 0
    der_z_loss = 0

    # 1st derivative loss
    # dx
    dy_pred_dx = (pred[:,:,2:,:] - pred[:,:,:-2,:])/grid_dx[:,:,:,:,0]
    dy_pred_dx[dy_pred_dx.isnan()] = 0
    mask_dy_dx = mask[:,:,:(nx-2),:]

    # util_create_gif_with_annotations(dy_pred_dx[0,...].numpy(), axis=-1, filename='del_dy_pred_dx_0.gif')
    # util_create_gif_with_annotations(mask_dy_dx[0,...].numpy(), axis=-1, filename='del_mask_dy_dx_0.gif')

    breakpoint()

    for i in range(current_ns):
        der_x_loss += myloss(dy_pred_dx[i,...][mask_dy_dx[i,...]].reshape(1, -1), dy_dx[i,...][mask_dy_dx[i,...]].view(1, -1))
        print(f"for {i = } => {der_x_loss = }")

    # 1st derivative loss
    # dz
    dy_pred_dz = (pred[:,2:,:,:] - pred[:,:-2,:,:])/grid_dz[:,:,:,:,0]
    mask_dy_dz = mask[:,:(nz-2),:,:]

    for i in range(current_ns):
        der_z_loss += myloss(dy_pred_dz[i,...][mask_dy_dz[i,...]].reshape(1, -1), dy_dz[i,...][mask_dy_dz[i,...]].view(1, -1))

    loss = beta3 * der_x_loss + beta4 * der_z_loss

    print(f"{loss = }")


if __name__ == "__main__":
    # output_size_0_issue_in_view()
    # loading_custom_dataset()
    # scaling_data()
    loss_nan_issue()