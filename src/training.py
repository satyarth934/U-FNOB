# %%
import torch
import numpy as np
from models.ufno_3D import *
from models.ufno_2D_recurrent import *
from models.lploss import *
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# from io import BytesIO
# import tensorflow as tf
# from tensorflow.python.lib.io import file_io
from tqdm import tqdm
import torch.utils.data as data
import gc

torch.manual_seed(0)
np.random.seed(0)

import wandb
wandb.init(project="FDL2022-PSSM", entity="satyarth934")

def main(epochs, batch_size, learning_rate, ufno_model, UNet, beta1, beta2, beta3, beta4, beta5, beta6, beta7, dataset = 'even_interval'):
    
    # dataset = 'even_interval', dataset = 'uneven_interval'
    
    # Empty cache before starting
    print("Empty cache before starting")
    gc.collect()
    torch.cuda.empty_cache()
        
    # PARAMETERS
    print("Set params")
    e_start = 0
    scheduler_step = 4
    scheduler_gamma = 0.85
    
    print("Set paths")
    # data_path = 'us-digitaltwiner-dev-features/data-sim-test-2D-1000_run4/'
    # data_path = '/global/cfs/cdirs/m1012/satyarth/Data/ensemble_simulation_runs/FDL2022_data/us-digitaltwiner-dev-features/data-sim-test-2D-1000_run4'
    data_path = '/global/cfs/projectdirs/m1012/satyarth/Data/ensemble_simulation_runs/ensemble_simulation_run2/training_data/v2'
    if dataset == 'even_interval':
        # f_input = BytesIO(file_io.read_file_to_string("gs://" + data_path + 'input_recurrent.npy', binary_mode=True)) 
        # f_output = BytesIO(file_io.read_file_to_string("gs://" + data_path + 'output_recurrent.npy', binary_mode=True))
        f_input = f"{data_path}/input_recurrent.npy"
        f_output = f"{data_path}/output_recurrent.npy"
    elif dataset == 'uneven_interval':
        # f_input = BytesIO(file_io.read_file_to_string("gs://" + data_path + 'input_top_layer.npy', binary_mode=True)) 
        # f_output = BytesIO(file_io.read_file_to_string("gs://" + data_path + 'output.npy', binary_mode=True))
        # f_input = f"{data_path}/input_top_layer.npy"
        # f_output = f"{data_path}/output.npy"
        f_input = f"{data_path}/layer7_input_blob.npy"
        f_output = f"{data_path}/layer7_output_blob.npy"
  
    print("Read files from paths")
    input_array = torch.from_numpy(np.load(f_input)) 
    output_array = torch.from_numpy(np.load(f_output))
    
    # size of array from the input
    ns, nz, nx, nt, nc = input_array.shape
    no = output_array.shape[-1]
    nc = nc - 3    # QUESTION: Why -3??

    # meta_data
    print("Read meta data")
    if dataset == 'even_interval':
        # f = (BytesIO(file_io.read_file_to_string("gs://" + data_path + 'meta_data_recurrent.txt', binary_mode=True)))
        meta_data_path = f"{data_path}/meta_data_recurrent.txt"
        
    elif dataset == 'uneven_interval':
        # f = (BytesIO(file_io.read_file_to_string("gs://" + data_path + 'meta_data.txt', binary_mode=True)))
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
            'min': np.nanmin(output_array[:,:,:,:,0]),
            'max': np.nanmax(output_array[:,:,:,:,0])/2
        },
        1 : {
            'min': np.nanmin(output_array[:,:,:,:,1]),
            'max': np.nanmax(output_array[:,:,:,:,1])/5
        },
        2 : {
            'min': np.nanmin(output_array[:,:,:,:,2]),
            'max': np.nanmax(output_array[:,:,:,:,2])
        },
        3 : {
            'min': np.nanmin(output_array[:,:,:,:,3]),
            'max': np.nanmax(output_array[:,:,:,:,3])
        },
        4 : {
            'min': np.nanmin(output_array[:,:,:,:,4]),
            'max': np.nanmax(output_array[:,:,:,:,4])
        },
        5 : {
            'min': tritium_MCL*0.2,
            'max': 9e-9
        },
        6 : {
            'min': np.nanmin(output_array[:,:,:,:,6]),
            'max': np.nanmax(output_array[:,:,:,:,6])
        }
    }
    
    
    # Rescale input
    input_max_values = np.nanmax(input_array.reshape(-1,nc+3),axis = 0).reshape(1,1,1,1,-1)
    input_array = input_array/input_max_values

    # Input nan -> 0
    print(f"Set nan -> 0")
    input_array[np.isnan(input_array)] = 0

    # Rescale output_array between 0 and 1.

    scaled_output = output_array.detach().clone()

    for i in range(no):
        scaled_output[:,:,:,:,i][scaled_output[:,:,:,:,i]<rescale_factors[i]['min']] = rescale_factors[i]['min']
        scaled_output[:,:,:,:,i][scaled_output[:,:,:,:,i]>rescale_factors[i]['max']] = rescale_factors[i]['max']
        scaled_output[:,:,:,:,i] = (scaled_output[:,:,:,:,i] - rescale_factors[i]['min'])/(rescale_factors[i]['max']-rescale_factors[i]['min'])

    scaled_output[np.isnan(scaled_output)] = 0
    
    # Current training
    selected_idx = np.array([0,1,2,5])
    scaled_output_4 = scaled_output[:,:,:,:,selected_idx]
    output_names_4 = list(np.array(output_names)[selected_idx])
    
    # Build U-FNO model
    # QUESTION: How are these values chosen?
    # %%
    mode1 = 10
    mode2 = 10
    mode3 = 4
    width = 36
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if ufno_model == '2D':
        if UNet:
            model = UFNO2d(mode1, mode2, width, UNet = True)
            # model_head = UFNO2d(mode1, mode2, width, UNet = True)    # XXX: Uncomment this
            model_head_path = "/global/cfs/cdirs/m1012/satyarth/Projects/digitaltwin-pssm/saved_models/dP_UFNO2D_UNet_True_200ep_36width_10m1_10m2_664model_model_headtrain_1.00e-03lr"
            model_head = torch.load(model_head_path)    # XXX: Comment this
            #model = torch.load('saved_models/dP_UFNO2D_UNet_True_50ep_36width_10m1_10m2_664model_modeltrain_1.42e-04lr')
        else:
            model = UFNO2d(mode1, mode2, width, UNet = False)
            model_head = UFNO2d(mode1, mode2, width, UNet = False)
    elif ufno_model == '3D':
        if UNet: 
            model = UFNO3d(mode1, mode2, mode3, width, UNet = True)
        else:
            model = UFNO3d(mode1, mode2, mode3, width, UNet = False)

    model.to(device)
    
    if ufno_model == '2D':
        model_head.to(device)
        
    # prepare derivatives
    
    grid_x = input_array[0,8,:,0,-3]
    grid_dx =  - grid_x[:-2] + grid_x[2:]
    grid_dx = grid_dx[None, None, :, None, None].to(device)

    grid_z = input_array[0,:,:,0,-2]
    grid_dz =  - grid_z[:-2,:] + grid_z[2:,:] 
    grid_dz[grid_dz==0] = 1/nz # to avoid divide by 0
    grid_dz = grid_dz[None, :, :, None, None].to(device)

    # bottom_z location
    bottom_z = np.zeros(nx)
    for idx_x in range(nx):
        nan_idx = np.where(np.isnan(output_array[0,:10,idx_x,0,0])==1)[0]
        if len(nan_idx)>0:
            bottom_z[idx_x] = np.max(nan_idx)+1
        else:
            bottom_z[idx_x] = 0
    bottom_z = np.array(bottom_z,dtype = 'float64')

    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "e_start": e_start,
        "scheduler_step": scheduler_step,
        "scheduler_gamma": scheduler_gamma,
        "model": ufno_model,
        "UNet": UNet,
        "beta1":beta1,
        "beta2":beta2,
        "beta3":beta3,
        "beta4":beta4,
        "beta5":beta5,
        "beta6":beta6,
        "beta7":beta7,
        "dataset":dataset
    }
    wandb.init(config=wandb.config)

   
    # Split dataset into training, val and test set
    
    torch_dataset = torch.utils.data.TensorDataset(input_array, scaled_output_4)

    dataset_sizes = [
        np.int(np.int(ns*0.8)/batch_size)*batch_size, 
        np.int((ns-np.int(np.int(ns*0.8)/batch_size)*batch_size)/2),
        np.int((ns-np.int(np.int(ns*0.8)/batch_size)*batch_size)/2),
    ]

    train_data, val_data, test_data = data.random_split(torch_dataset, dataset_sizes ,generator=torch.Generator().manual_seed(0))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader  = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    myloss = LpLoss(size_average=True) # relative lp loss
    
    if ufno_model == '2D':
        optimizer_head = torch.optim.Adam(model_head.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler_head = torch.optim.lr_scheduler.StepLR(optimizer_head, step_size=scheduler_step, gamma=scheduler_gamma)
    
    # loss functions
    def loss_function(x,y, model, beta1, beta2):
        no = y.shape[-1]
        current_ns = x.shape[0]

        if len(x.shape)==5: # 3D
            nt = x.shape[-2]
        else: #2D
            nt = 1

        nz = x.shape[1]
        nx = x.shape[2]

        if len(x.shape)<5:
            mask = (x[:,:,:,0]!=0).reshape(x.shape[0], nz, nx, 1, 1).repeat(1,1,1,nt,no) 
        elif len(x.shape)==5:   
            mask = (x[:,:,:,0:1,0]!=0).reshape(x.shape[0], nz, nx, 1, 1).repeat(1,1,1,nt,no) # deactivate those input values with 0, i.e. above the surface
        dy_dx = (y[:,:,2:,:,:] - y[:,:,:-2,:,:])/grid_dx
        dy_dz = (y[:,2:,:,:,:] - y[:,:-2,:,:,:])/grid_dz

        pred = model(x.float()).view(-1, nz, nx, nt, no)

        ori_loss = 0
        der_x_loss = 0
        der_z_loss = 0


        # original loss
        for i in range(current_ns):
            ori_loss += myloss(pred[i,...][mask[i,...]].reshape(1, -1), y[i,...][mask[i,...]].reshape(1, -1))

        # 1st derivative loss
        # dx
        dy_pred_dx = (pred[:,:,2:,:,:] - pred[:,:,:-2,:,:])/grid_dx
        mask_dy_dx = mask[:,:,:(nx-2),:,:]

        for i in range(current_ns):
            der_x_loss += myloss(dy_pred_dx[i,...][mask_dy_dx[i,...]].reshape(1, -1), dy_dx[i,...][mask_dy_dx[i,...]].view(1, -1))


        # 1st derivative loss
        # dz
        dy_pred_dz = (pred[:,2:,:,:,:] - pred[:,:-2,:,:,:])/grid_dz
        mask_dy_dz = mask[:,:(nz-2),:,:,:]

        for i in range(current_ns):
            der_z_loss += myloss(dy_pred_dz[i,...][mask_dy_dz[i,...]].reshape(1, -1), dy_dz[i,...][mask_dy_dz[i,...]].view(1, -1))

        loss = ori_loss + beta1 * der_x_loss + beta2 * der_z_loss
        return loss


    def loss_function_boundary(x, y,  model, beta3, beta4, axis=3):

        # This is for the plume part
        # y should be the plume slice
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

        MCL_threshold = (tritium_MCL-rescale_factors[selected_idx[axis]]['min'])/(rescale_factors[selected_idx[axis]]['max']-rescale_factors[selected_idx[axis]]['min'])
        y = (y>MCL_threshold)*1

        dy_dx = (y[:,:,2:,:] - y[:,:,:-2,:])/grid_dx[:,:,:,:,0]
        dy_dz = (y[:,2:,:,:] - y[:,:-2,:,:])/grid_dz[:,:,:,:,0]

        pred = model(x.float()).view(-1, nz, nx, nt, 4)[:,:,:,:,axis]
        pred = (pred>MCL_threshold)*1

        der_x_loss = 0
        der_z_loss = 0

        # 1st derivative loss
        # dx
        dy_pred_dx = (pred[:,:,2:,:] - pred[:,:,:-2,:])/grid_dx[:,:,:,:,0]
        mask_dy_dx = mask[:,:,:(nx-2),:]

        for i in range(current_ns):
            der_x_loss += myloss(dy_pred_dx[i,...][mask_dy_dx[i,...]].reshape(1, -1), dy_dx[i,...][mask_dy_dx[i,...]].view(1, -1))

        # 1st derivative loss
        # dz
        dy_pred_dz = (pred[:,2:,:,:] - pred[:,:-2,:,:])/grid_dz[:,:,:,:,0]
        mask_dy_dz = mask[:,:(nz-2),:,:]

        for i in range(current_ns):
            der_z_loss += myloss(dy_pred_dz[i,...][mask_dy_dz[i,...]].reshape(1, -1), dy_dz[i,...][mask_dy_dz[i,...]].view(1, -1))

        loss = beta3 * der_x_loss + beta4 * der_z_loss
        return loss

    def loss_function_PINN_BC1(x, model, axis = 0): 

        # This is for darcy 0, x direction
        current_ns = x.shape[0]
        if len(x.shape)==5: # 3D
            nt = x.shape[-2]
        else: #2D
            nt = 1

        nz = x.shape[1]
        nx = x.shape[2]

        zero_value = (0-rescale_factors[selected_idx[axis]]['min'])/(rescale_factors[selected_idx[axis]]['max']-rescale_factors[selected_idx[axis]]['min'])

        if len(x.shape)<5:
            mask = (x[:,:,:,0]!=0).reshape(-1,nz,nx,1).repeat(1,1,1,nt) 
        elif len(x.shape)==5:   
            mask = (x[:,:,:,0:1,0]!=0).repeat(1,1,1,nt) # deactivate those input values with 0, i.e. above the surface

        pred = model(x.float()).view(-1, nz, nx, nt, 4)
        pred = pred[:,:,:,:,axis]

        pinn_BC_loss = 0

        # boundary condition loss
        for i in range(current_ns):
            pinn_BC_loss += torch.norm(torch.abs(pred[i,:,-1,:][mask[i,:,-1,:]]-zero_value),2)/nz # right x boundary
            pinn_BC_loss += torch.norm(torch.abs(pred[i,:,0,:][mask[i,:,0,:]]-zero_value),2)/nz# left x boundary       
        return pinn_BC_loss/(current_ns)


    def loss_function_PINN_BC2(x,  model, axis = 1): 
        # This is for darcy 1, z direction
        current_ns = x.shape[0]
        if len(x.shape)==5: # 3D
            nt = x.shape[-2]
        else: #2D
            nt = 1

        nz = x.shape[1]
        nx = x.shape[2]

        zero_value = (0-rescale_factors[selected_idx[axis]]['min'])/(rescale_factors[selected_idx[axis]]['max']-rescale_factors[selected_idx[axis]]['min'])

        if len(x.shape)<5:
            mask = (x[:,:,:,0]!=0).reshape(-1,nz,nx,1).repeat(1,1,1,nt) 
        elif len(x.shape)==5:   
            mask = (x[:,:,:,0:1,0]!=0).repeat(1,1,1,nt) # deactivate those input values with 0, i.e. above the surface

        pred = model(x.float()).view(-1, nz, nx, nt, 4)
        pred = pred[:,:,:,:,axis]

        pinn_BC_loss = 0

        # boundary condition loss
        for i in range(current_ns):
            pinn_BC_loss += torch.norm(torch.abs(pred[i,bottom_z,np.arange(nx),:][mask[i,bottom_z,np.arange(nx),:]]-zero_value),2)/nx # bottom z boundary

        return pinn_BC_loss/(current_ns)


    def loss_function_PINN_BC3(x, model, axis = 2): 
        # This is for hydraulic head 
        current_ns = x.shape[0]

        sx = 10
        sz = 2.5

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

        pred = model(x.float()).view(-1, nz, nx, nt, 4)
        pred = pred[:,:,:,:,axis]

        pinn_BC_loss = 0

        # boundary condition loss
        for i in range(current_ns):
            pinn_BC_loss += torch.norm(torch.abs((pred[i,:,-2,:]-pred[i,:,-1,:])/sx)[mask[i,:,-1,:]],2)/nz # right x boundary
            pinn_BC_loss += torch.norm(torch.abs((pred[i,:,1,:]- pred[i,:,0,:])/sx)[mask[i,:,0,:]],2)/nz# left x boundary
            pinn_BC_loss += torch.norm(torch.abs((pred[i,bottom_z+1,np.arange(nx),:]-pred[i,bottom_z,np.arange(nx),:])/sz)[mask[i,bottom_z,np.arange(nx),:]],2)/nx # bottom z boundary

        return pinn_BC_loss/(current_ns)


    # Training
    def training_loop_2D(model_current = 'model_head',epochs = 100):

        plume_axis = np.where(np.array(output_names_4) == "total_component_concentration.cell.Tritium conc")[0][0]
        darcy_x_axis = np.where(np.array(output_names_4) == "darcy_velocity.cell.0")[0][0]
        darcy_z_axis = np.where(np.array(output_names_4) == "hydraulic_head.cell.0")[0][0]
        hh_axis = np.where(np.array(output_names_4) == "hydraulic_head.cell.0")[0][0]

        train_l2 = 0.0
        train_loss_array = np.zeros(epochs)
        val_loss_array = np.zeros(epochs)

        im_init = torch.mean(scaled_output_4[:,:,:,0,:],axis = 0).to(device)

        #mask = (scaled_output_4[0:1,:,:,0:1,0]!=0).reshape(1, nz, nx, 1).repeat(batch_size,1,1,4)

        for ep in range(1,epochs+1):
            if model_current == 'model_head':
                model_head.train()
                num_time = 1
            else:
                model.train()
                num_time = nt

            train_l2 = 0
            val_l2 = 0
            counter = 0

            for xx, yy in train_loader:

                im = im_init.repeat(xx.shape[0],1,1,1) 

                num_current_batch = xx.shape[0]

                loss = 0

                xx = xx.to(device)
                yy = yy.to(device)

                for t in np.arange(num_time):
                    x = torch.cat((xx[:,:,:,t,:], im), dim=-1).to(device)

                    y = yy[:,:,:,t:(t+1),:]

                    if t == 0:
                        im = model_head(x.float())
                        loss += loss_function(x,y, model_head, beta1, beta2) + loss_function_boundary(x, y[:,:,:,:,plume_axis],model_head, beta3, beta4, axis=plume_axis) \
                            + beta5*loss_function_PINN_BC1(x,model_head, axis=darcy_x_axis)+ beta6*loss_function_PINN_BC2(x,model_head, axis=darcy_z_axis)+ beta7*loss_function_PINN_BC3(x,model_head,axis=hh_axis)
                        pred = im
                    else:
                        im = model(x.float())
                        loss += loss_function(x,y, model, beta1, beta2) + loss_function_boundary(x, y[:,:,:,:,plume_axis],model, beta3, beta4, axis=plume_axis) \
                            + beta5*loss_function_PINN_BC1(x,model, axis=darcy_x_axis)+ beta6*loss_function_PINN_BC2(x,model, axis=darcy_z_axis)+ beta7*loss_function_PINN_BC3(x,model,axis=hh_axis)
                        pred = torch.cat((pred, im), -1)

                if model_current == 'model_head':
                    optimizer_head.zero_grad()
                else:
                    optimizer.zero_grad()

                loss.backward()

                if model_current == 'model_head':
                    optimizer_head.step()   
                else:
                    optimizer.step()       

                counter += 1
                if counter % 100 == 0:
                    print(f'epoch: {ep}, batch: {counter}/{len(train_loader)}, train loss: {loss.item()/batch_size:.4f}')

                train_l2 += loss.item()
                # if model_current == 'model':
                #     print(loss.item())

            for xx, yy in val_loader:
                im = im_init.repeat(xx.shape[0],1,1,1) 

                num_current_batch = xx.shape[0]

                loss = 0

                xx = xx.to(device)
                yy = yy.to(device)

                for t in np.arange(num_time):

                    x = torch.cat((xx[:,:,:,t,:], im), dim=-1).to(device)

                    y = yy[:,:,:,t:(t+1),:]

                    if t == 0:
                        im = model_head(x.float())
                        loss += loss_function(x,y, model_head, beta1, beta2) + loss_function_boundary(x, y[:,:,:,:,plume_axis],model_head, beta3, beta4, axis=plume_axis) \
                            + beta5*loss_function_PINN_BC1(x,model_head, axis=darcy_x_axis)+ beta6*loss_function_PINN_BC2(x,model_head, axis=darcy_z_axis)+ beta7*loss_function_PINN_BC3(x,model_head,axis=hh_axis)
                        pred = im
                    else:
                        im = model(x.float())
                        loss += loss_function(x,y, model, beta1, beta2) + loss_function_boundary(x, y[:,:,:,:,plume_axis],model, beta3, beta4, axis=plume_axis) \
                            + beta5*loss_function_PINN_BC1(x,model, axis=darcy_x_axis)+ beta6*loss_function_PINN_BC2(x,model, axis=darcy_z_axis)+ beta7*loss_function_PINN_BC3(x,model,axis=hh_axis)
                        pred = torch.cat((pred, im), -1)

                val_l2 += loss.item()

            train_loss = train_l2/dataset_sizes[0]
            val_loss = val_l2/dataset_sizes[1]
            print(f'epoch: {ep}, train loss: {train_loss:.4f}')
            print(f'epoch: {ep}, val loss:   {val_loss:.4f}')

            train_loss_array[ep-1] = train_loss
            val_loss_array[ep-1] = val_loss

            if model_current == 'model_head':
                scheduler_head.step()   
            else:
                scheduler.step()    

            lr_ = optimizer.param_groups[0]['lr']
            if ep % 5 == 0:
                PATH = f'saved_models/dP_UFNO2D_UNet_{UNet}_{ep}ep_{width}width_{mode1}m1_{mode2}m2_{input_array.shape[0]}model_{model_current}train_{lr_:.2e}lr'
                if model_current == 'model_head':
                    torch.save(model_head, PATH)
                else:
                    torch.save(model, PATH)

            # Framework agnostic / custom metrics
            wandb.log({"epoch": ep, "loss": train_loss, "val_loss": val_loss})

                    
        
    def training_loop_3D():

        plume_axis = np.where(np.array(output_names_4) == "total_component_concentration.cell.Tritium conc")[0][0]
        darcy_x_axis = np.where(np.array(output_names_4) == "darcy_velocity.cell.0")[0][0]
        darcy_z_axis = np.where(np.array(output_names_4) == "hydraulic_head.cell.0")[0][0]
        hh_axis = np.where(np.array(output_names_4) == "hydraulic_head.cell.0")[0][0]

        train_loss_array = np.zeros(epochs)
        val_loss_array = np.zeros(epochs)

        for ep in range(1,epochs+1):
            model.train()
            train_l2 = 0
            val_l2 = 0
            counter = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                
                loss = loss_function(x,y, model, beta1, beta2)+loss_function_boundary(x, y[:,:,:,:,plume_axis], model, beta3, beta4, axis=plume_axis) \
                    + beta5*loss_function_PINN_BC1(x, model, axis=darcy_x_axis)+ beta6*loss_function_PINN_BC2(x,model, axis=darcy_z_axis)+ beta7*loss_function_PINN_BC3(x,model, axis=hh_axis)
                
                loss.backward()
                optimizer.step()
                train_l2 += loss.item()
                
                counter += 1
                if counter % 100 == 0:
                    print(f'epoch: {ep}, batch: {counter}/{len(train_loader)}, train loss: {loss.item()/batch_size:.4f}')
                #print(loss.item())
                
            scheduler.step()

            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                loss = loss_function(x,y, model,  beta1, beta2) + loss_function_boundary(x, y[:,:,:,:,plume_axis], model, beta3, beta4, axis=plume_axis) \
                    + beta5*loss_function_PINN_BC1(x, model, axis=darcy_x_axis)+ beta6*loss_function_PINN_BC2(x, model, axis=darcy_z_axis) + beta7*loss_function_PINN_BC3(x, model, axis=hh_axis)
                
                val_l2 += loss.item()
                
            train_loss = train_l2/dataset_sizes[0]
            val_loss = val_l2/dataset_sizes[1]
            print(f'epoch: {ep}, train loss: {train_loss:.4f}')
            print(f'epoch: {ep}, val loss:   {val_loss:.4f}')
            
            train_loss_array[ep-1] = train_loss
            val_loss_array[ep-1] = val_loss
            
            lr_ = optimizer.param_groups[0]['lr']
            if ep % 5 == 0:
                PATH = f'saved_models/dP_UFNO3D_UNet_{UNet}_{ep}ep_{width}width_{mode1}m1_{mode2}m2_{input_array.shape[0]}train_{lr_:.2e}lr_b1_{beta1}b2_{beta2}b3_{beta3}b4_{beta4}b5_{beta5}b6_{beta6}b7_{beta7}'
                torch.save(model, PATH)
            # Framework agnostic / custom metrics
            wandb.log({"epoch": ep, "loss": train_loss, "val_loss": val_loss})
        
    
    print(f"Running {ufno_model = }")
    if ufno_model == '2D':
        # training_loop_2D(model_current = 'model_head',epochs = 200)    # XXX: Uncomment
        training_loop_2D(model_current = 'model',epochs = epochs)
    elif ufno_model == '3D':
        training_loop_3D()
    

    mask = (input_array[0,:,:,0:1,0]!=0).reshape(1,nz, nx, 1, 1).repeat(1,1,1,nt,4)
    mse_function = torch.nn.MSELoss()

    def r2_function(y_pred, y):
        target_mean = torch.mean(y)
        ss_tot = torch.sum((y - target_mean) ** 2)
        ss_res = torch.sum((y - y_pred) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2
    
    def measure_metric(data_loader, sample_size, batch_size, metric='mse'):
        result = np.zeros(sample_size)
        i = 0
        for x, y in data_loader:
            nt = x.shape[-2]
            x, y = x.to(device), y.to(device)
            if ufno_model == '3D':
                y_pred = model(x.float())

            elif ufno_model == '2D': # inference with 2D model
                im_mean = torch.mean(scaled_output_4[:,:,:,0,:],axis = 0).to(device)
                im = im_mean.to(device).reshape(-1,nz,nx,4).repeat(x.shape[0],1,1,1) 
                for t in np.arange(nt): 
                    x_ = torch.cat((x[:,:,:,t,:], im), dim=-1).to(device)
                    if t == 0:
                        im = model_head(x_.float())
                    else:
                        im = model(x_.float())
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                y_pred = pred.reshape(x.shape[0],nz,nx,nt,4)
                if x.shape[0]==1:
                    y_pred = y_pred[0,:,:,:,:]

            if(batch_size==1):
                if(metric=='mse'):
                    result[i] = mse_function(y_pred[mask[0,...]], y[0,...][mask[0,...]])
                if(metric=='mre'):
                    result[i] = (torch.norm(y[0,...][mask[0,...]]-y_pred[mask[0,...]], 2)/torch.norm(y[0,...][mask[0,...]],2)).cpu().detach().numpy()
                if(metric=='r2'):
                    result[i] = r2_function(y_pred[mask[0,...]], y[0,...][mask[0,...]])
                i = i+1
            else:
                for b in range(batch_size):
                    try:
                        if(metric=='mse'):
                            result[i+b] = mse_function(y_pred[b,...][mask[0,...]], y[b,...][mask[0,...]])
                        if(metric=='mre'):
                            result[i+b] = (torch.norm(y[b,...][mask[0,...]]-y_pred[b,...][mask[0,...]], 2)/torch.norm(y[b,...][mask[0,...]],2)).cpu().detach().numpy()
                        if(metric=='r2'):
                            result[i+b] = r2_function(y_pred[b,...][mask[0,...]], y[b,...][mask[0,...]])
                    except:
                        pass
                i = i+batch_size
        return result
    
    mode_settings = {
        'Train':{
            'data': train_loader,
            'sample_size': dataset_sizes[0],
            'batch_size': batch_size
        },
        'Validation':{
            'data': val_loader,
            'sample_size': dataset_sizes[1],
            'batch_size': batch_size
        },
        'Test':{
            'data': test_loader,
            'sample_size': dataset_sizes[2],
            'batch_size': 1
        }
    }

    import os
    if(os.path.exists('./evaluations_UFNO.json')==False):
        f = open('evaluations_UFNO.json', 'w')

    import json
    # OPEN PREVIOUS RESULTS
    with open('evaluations_UFNO.json') as json_file:
        eval_file = json.load(json_file)

    # APPEND NEW RESULTS
    try:
        eval_file[str(wandb.config.as_dict())] = {}
    except Exception as e:
        print(f"Ignoring Exception => {e}")
        eval_file[str(wandb.config)] = {}

    for mode in mode_settings.keys():
        loader = mode_settings[mode]['data']
        sample_size = mode_settings[mode]['sample_size']
        batch_size = mode_settings[mode]['batch_size']

        mre = measure_metric(data_loader=loader, sample_size=sample_size, batch_size=batch_size, metric='mre').mean()
        mse = measure_metric(data_loader=loader, sample_size=sample_size, batch_size=batch_size, metric='mse').mean()
        r2 = measure_metric(data_loader=loader, sample_size=sample_size, batch_size=batch_size, metric='r2').mean()

        # print(mre, mse, r2)
        try:
            wandb_config_dict = str(wandb.config.as_dict())
        except Exception as e:
            print(f"Ignoring Exception => {e}")
            wandb_config_dict = str(wandb.config)

        eval_file[wandb_config_dict][mode] = {
            'MRE': mre,
            'MSE': mse,
            'R^2': r2
        }

        print(mode + ": \nMRE: " + str(mre) + "\nMSE: " + str(mse) + "\nR2: " + str(r2) + "\n")

    # SAVE NEW RESULTS TO FILE
    file = json.dumps(eval_file)
    f = open("evaluations_UFNO.json","w")
    f.write(file)
    f.close()
