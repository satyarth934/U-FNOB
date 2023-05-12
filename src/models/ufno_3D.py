import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.util_module import PrintShape
# import logging
# logging.basicConfig(filename='model_investigation.log', level=logging.DEBUG)

import operator
from functools import reduce
from functools import partial

torch.manual_seed(0)

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(U_net, self).__init__()
        self.input_channels = input_channels
        self.conv1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        self.conv3 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv3_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        
        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels*2, output_channels)
        self.deconv0 = self.deconv(input_channels*2, output_channels)
    
        self.output_layer = self.output(input_channels*2, output_channels, 
                                         kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)


    def forward(self, x):
        # PrintShape(msg="x")(x)
        out_conv1 = self.conv1(x)
        # PrintShape(msg="out_conv1")(out_conv1)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        # PrintShape(msg="out_conv2")(out_conv2)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        # PrintShape(msg="out_conv3")(out_conv3)
        out_deconv2 = self.deconv2(out_conv3)
        # PrintShape(msg="out_deconv2")(out_deconv2)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        # PrintShape(msg="concat2")(concat2)
        out_deconv1 = self.deconv1(concat2)
        # PrintShape(msg="out_deconv1")(out_deconv1)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        # PrintShape(msg="concat1")(concat1)
        out_deconv0 = self.deconv0(concat1)
        # PrintShape(msg="out_deconv0")(out_deconv0)
        concat0 = torch.cat((x, out_deconv0), 1)
        # PrintShape(msg="concat0")(concat0)
        out = self.output_layer(concat0)
        # PrintShape(msg="out")(out)

        return out

    def conv(self, in_planes, output_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv3d(in_planes, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias = False),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(input_channels, output_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size,
                         stride=stride, padding=(kernel_size - 1) // 2)



class SimpleBlock3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, UNet=True):
        super(SimpleBlock3d, self).__init__()
        """
        U-FNO contains 3 Fourier layers and 3 U-Fourier layers.
        
        input shape: (batchsize, x=200, y=96, t=24, c=12)
        output shape: (batchsize, x=200, y=96, t=24, c=1)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.UNet = UNet
        # num_variables = 11    # FDL data has 11 variables
        num_variables = 8
        self.fc0 = nn.Linear(num_variables, self.width)
        """        
        12 channels for [kx, kz, porosity, inj_loc, inj_rate, 
                         pressure, temperature, Swi, Lam, 
                         grid_x, grid_y, grid_t]
        """
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv4 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv5 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.w5 = nn.Conv1d(self.width, self.width, 1)
        self.unet3 = U_net(self.width, self.width, 3, 0)
        self.unet4 = U_net(self.width, self.width, 3, 0)
        self.unet5 = U_net(self.width, self.width, 3, 0)
        self.fc1 = nn.Linear(self.width, 128)
        # self.fc2 = nn.Linear(128, 4)    # Since FDL code is predicting 4 variables.
        self.fc2 = nn.Linear(128, 1)    # Right now predicting only 1 output variable.


    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]
        
        # # CHECKING MEMORY USAGE
        # max_memory_used = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
        # print(f"[SimpleBlock3d - entering forward] Max memory used during forward pass: {max_memory_used:.2f} MB")

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        # # CHECKING MEMORY USAGE
        # max_memory_used = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
        # print(f"[SimpleBlock3d - after fc0] Max memory used during forward pass: {max_memory_used:.2f} MB")

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2 
        x = F.relu(x)

        # # CHECKING MEMORY USAGE
        # max_memory_used = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
        # print(f"[SimpleBlock3d - after conv0] Max memory used during forward pass: {max_memory_used:.2f} MB")
        
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2 
        x = F.relu(x)

        # # CHECKING MEMORY USAGE
        # max_memory_used = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
        # print(f"[SimpleBlock3d - after conv1] Max memory used during forward pass: {max_memory_used:.2f} MB")
        
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2 
        x = F.relu(x)

        # # CHECKING MEMORY USAGE
        # max_memory_used = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
        # print(f"[SimpleBlock3d - after conv2] Max memory used during forward pass: {max_memory_used:.2f} MB")
        
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet3(x) 
        
        # # CHECKING MEMORY USAGE
        # max_memory_used = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
        # print(f"[SimpleBlock3d - after conv3] Max memory used during forward pass: {max_memory_used:.2f} MB")

        if self.UNet: 
            x = x1 + x2 + x3
        else:
            x = x1 + x2
        x = F.relu(x)
        
        x1 = self.conv4(x)
        x2 = self.w4(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet4(x)

        # # CHECKING MEMORY USAGE
        # max_memory_used = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
        # print(f"[SimpleBlock3d - after conv4] Max memory used during forward pass: {max_memory_used:.2f} MB")
        
        if self.UNet: 
            x = x1 + x2 + x3
        else:
            x = x1 + x2
        x = F.relu(x)
        
        x1 = self.conv5(x)
        x2 = self.w5(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet5(x)

        # # CHECKING MEMORY USAGE
        # max_memory_used = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
        # print(f"[SimpleBlock3d - after conv5] Max memory used during forward pass: {max_memory_used:.2f} MB")
        
        if self.UNet: 
            x = x1 + x2 + x3
        else:
            x = x1 + x2
        x = F.relu(x)
        
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        # # CHECKING MEMORY USAGE
        # max_memory_used = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
        # print(f"[SimpleBlock3d - after fc1 and fc2 - exiting forward] Max memory used during forward pass: {max_memory_used:.2f} MB")
        
        return x

class UFNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, UNet=True):
        super(UFNO3d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock3d(modes1, modes2, modes3, width, UNet)


    def forward(self, x):
        # x = x[:, :, :, :64, :]    # considering only first 64 years for numerical sanity. 64 is a power of 2. total 65 yrs available. TODO: Move this step to the point where the data is being fed to the model.
        batchsize = x.shape[0]
        # breakpoint()

        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]
        # if size_z == 11:
        #     padding_para = [5,7,8] #z->t, y, x
        # elif size_z == 30:
        #     padding_para = [2,7,8] #z->t, y, x
        padding_para = [0,5,9]
        
        # num_output_variables = 4
        num_output_variables = 1
        x = F.pad(F.pad(x, (0,0,0,padding_para[0],0,padding_para[1]), "replicate"), (0,0,0,0,0,0,0,padding_para[2]), 'constant', 0)
        x = self.conv1(x)    # QUESTION: Why is the last dimension=4?
        x = x.view(batchsize, size_x+padding_para[2], size_y+padding_para[1], size_z+padding_para[0], num_output_variables)[..., :-padding_para[2],:-padding_para[1],:(-padding_para[0] if padding_para[0]!=0 else None), :]
        return x.squeeze()


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c
