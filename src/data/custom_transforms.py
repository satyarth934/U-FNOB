import torch
import numpy as np
from pprint import pprint
from collections import namedtuple


class ScaleInputTransform:
    """Scale input.
    """
    def __init__(
        self,
        meta_data_file_path,
        set_nan_to_zero=True,
    ) -> None:
        
        self.set_nan_to_zero = set_nan_to_zero
        
        self.meta_data_path = meta_data_file_path
        self.meta_data = namedtuple('meta_data', [
            'input_names',
            'time_steps',
            'input_min',
            'input_max',
        ])
        with open(self.meta_data_path) as f:
            lines = f.readlines()

        self.meta_data.input_names = str(lines[0]).strip().split(", ")
        self.meta_data.time_steps = np.array(str(lines[1]).strip().split(", "), dtype = 'float64')
        self.meta_data.time_steps = torch.from_numpy(np.array(self.meta_data.time_steps, dtype = 'int64'))    # FIXME: Seems redundant.
        self.meta_data.input_min = torch.from_numpy(np.array(str(lines[2]).strip().split(", "), dtype = 'float64'))
        self.meta_data.input_max = torch.from_numpy(np.array(str(lines[3]).strip().split(", "), dtype = 'float64'))
        # XXX: pprint(f"self.meta_data = {self.meta_data}")

    
    def __call__(self, data):
        nz, nx, nt, nc = data.shape
        nc = nc - 3    # NOTE: last 3 variables => grid_x, grid_y, and grid_t
                
        # Rescale input
        input_max_values = np.append(
            self.meta_data.input_max,
            np.nanmax(data[:, :, :, nc:].reshape(-1, 3), axis=0),
        ).reshape(1,1,1,-1)
        # XXX: print(f"{input_max_values = }")
        ret_data = data/input_max_values

        # Input nan -> 0
        if self.set_nan_to_zero:
            # XXX: print(f"Set nan -> 0")
            ret_data[ret_data.isnan()] = 0

        return ret_data

    def __repr__(self):
        return f"{self.__class__.__name__}"


class ScaleOutputTransform:
    """Scale output.
    """
    def __init__(
        self,
        meta_data_file_path,
        set_nan_to_zero=True,
    ) -> None:
        self.set_nan_to_zero = set_nan_to_zero
        
        self.meta_data_path = meta_data_file_path
        self.meta_data = namedtuple('meta_data', [
            'input_names',
            'time_steps',
            'input_min',
            'input_max',
            'output_names',
        ])
        with open(self.meta_data_path) as f:
            lines = f.readlines()

        self.meta_data.input_names = str(lines[0]).strip().split(", ")
        self.meta_data.time_steps = np.array(str(lines[1]).strip().split(", "), dtype = 'float64')
        self.meta_data.time_steps = torch.from_numpy(np.array(self.meta_data.time_steps, dtype = 'int64'))    # FIXME: Seems redundant.
        self.meta_data.input_min = torch.from_numpy(np.array(str(lines[2]).strip().split(", "), dtype = 'float64'))
        self.meta_data.input_max = torch.from_numpy(np.array(str(lines[3]).strip().split(", "), dtype = 'float64'))
        self.meta_data.output_names = str(lines[4]).strip().split(", ")
        pprint(f"self.meta_data = {self.meta_data}")

        # rescale output
        print(f"Rescaling")
        tritium_MCL = 7e-13
        # Custom min and max values per variable for rescaling
        self.rescale_factors = {
            0 : {
                'min': tritium_MCL*0.2,
                'max': 9e-9
            },
        }
        

    def __call__(self, data):
        no = data.shape[-1]
        
        # Rescale data between 0 and 1.
        scaled_output = data.detach().clone()

        for i in range(no):
            scaled_output[:,:,:,i][
                scaled_output[:,:,:,i]<self.rescale_factors[i]['min']
            ] = self.rescale_factors[i]['min']
            scaled_output[:,:,:,i][
                scaled_output[:,:,:,i]>self.rescale_factors[i]['max']
            ] = self.rescale_factors[i]['max']
            scaled_output[:,:,:,i] = (scaled_output[:,:,:,i] - self.rescale_factors[i]['min']) / (self.rescale_factors[i]['max']-self.rescale_factors[i]['min'])

        scaled_output[scaled_output.isnan()] = 0

        # Select the output variables of interest
        selected_idx = np.array([0])
        return scaled_output[:,:,:,selected_idx]


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"