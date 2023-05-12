import pandas as pd
import numpy as np
from tqdm import tqdm
import training as training    # HACK: Sim-wise dataloader using training_2.py
from timeit import default_timer as timer


epochs = 30
# epochs = 3
learning_rate = 0.001
beta1 = 0
beta2 = 0
beta3 = 0
beta4 = 0
beta5 = 0
beta6 = 0
beta7 = 0


batch_size = 8
# ufno_model = '2D'
ufno_model = '3D'
UNet = True
dataset = 'uneven_interval'

# Head has been trained for a fixed 200 epoch
# The rest depends on the epochs value we specified earlier

start = timer()

training.main(epochs, batch_size, learning_rate, ufno_model, UNet, beta1, beta2, beta3, beta4, beta5, beta6, beta7, dataset)

end = timer()

print(str((end - start)/60)+str(' mins'))
