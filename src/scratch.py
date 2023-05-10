import torch
import numpy as np
from models.ufno_3D import *

mode1 = 10
mode2 = 10
mode3 = 4
width = 36
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UFNO3d(mode1, mode2, mode3, width, UNet = True)
model.to(device)

# Printing model summary
from torchsummary import summary
summary(model, input_size=(119, 171, 65, 8))
import sys
sys.exit(0)