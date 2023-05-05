import imageio
import numpy as np
import matplotlib.pyplot as plt


def _point_to_grid(data):
        """Here it is assumed that the data will have x and y downsampled and in integer cartesian coordinates and not the world coordinates. Will need to rescale if it is in world coordinates. (just for data consistency)
        Proxy for Target Transform.
        TODO: Convert to a transform.

        Args:
            data (pd.DataFrame): DataFrame containing x, y, concentration.

        Returns:
            torch.Tensor: Data grid built from the sparse matrix
        """
        nx = int(data["x"].max() - data["x"].min() + 1)
        ny = int(data["y"].max() - data["y"].min() + 1)

        data_grid = np.zeros((nx, ny))
        data_grid[:] = np.nan
        data_grid[data["x"].astype(int), data["y"].astype(int)] = data.values[:,2]
        return data_grid.transpose()


def fill_holes(data):
    # Find the indices of the True values
    indices = np.argwhere(data)

    # Loop over each True value index
    for i, j in indices:
        # Get the surrounding indices
        surrounding_indices = np.argwhere(data[max(0, i-1):i+2, max(0, j-1):j+2])
        
        # Get the surrounding values
        surrounding_values = data[max(0, i-1):i+2, max(0, j-1):j+2][surrounding_indices[:,0], surrounding_indices[:,1]]
        
        if surrounding_values.dtype == bool:
            # Set the current value to True if more than half the surrounding values are not nan
            data[i, j] = np.sum(surrounding_values) > surrounding_values.size//2
        else:
            # Set the current value to the average of the surrounding values
            data[i, j] = surrounding_values.mean()
    
    return data


def animate_sim_over_years(sim_blob_3d, filename="sim_over_years.gif"):
    sim_blob_3d_norm = ((sim_blob_3d - np.nanmin(sim_blob_3d)) / (np.nanmax(sim_blob_3d) - np.nanmin(sim_blob_3d)))
    
    sim_blob_3d_t = sim_blob_3d_norm.transpose([2, 0, 1])


    sim_blob_3d_flist = [np.flipud(layer) for layer in sim_blob_3d_t]
    sim_blob_3d_cmap = [plt.get_cmap()(layer) for layer in sim_blob_3d_flist]
    
    imageio.mimsave(filename, sim_blob_3d_cmap)