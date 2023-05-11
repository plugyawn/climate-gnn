import xarray as xr
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import torch
from torch_geometric.data import Data

# Load the netCDF file into an xarray Dataset
ds = xr.open_dataset('era5-india.nc')

# Convert the Dataset into a pandas DataFrame
df = ds.to_dataframe().reset_index()

# Extract the latitude, longitude, and time variables from the DataFrame
lats = df['latitude'].unique()
lons = df['longitude'].unique()
times = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

# Create a 2D grid of latitude and longitude coordinates
lon_grid, lat_grid = np.meshgrid(lons, lats)

# Flatten the grids into 1D arrays
lon_flat = lon_grid.flatten()
lat_flat = lat_grid.flatten()

# Create a KDTree for efficient nearest-neighbor searches
tree = KDTree(np.column_stack((lat_flat, lon_flat)))

# Create a list of node indices corresponding to the grid points
node_idx = []
for lat, lon in zip(lats, lons):
    dist, idx = tree.query([lat, lon])
    node_idx.append(idx)

# Create a dictionary of node features
node_features = {
    'temperature': [],
    'precipitation': [],
}
for i, row in df.iterrows():
    node_features['temperature'].append(row['t2m'])
    node_features['precipitation'].append(row['tp'])

# Create a list of edges between adjacent nodes
edge_index = []
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        if j < len(lons) - 1:
            edge_index.append([node_idx[i][j], node_idx[i][j+1]])
        if i < len(lats) - 1:
            edge_index.append([node_idx[i][j], node_idx[i+1][j]])

# Create a PyTorch Geometric Data object
data = Data(
    x=torch.tensor(list(zip(node_features['temperature'], node_features['precipitation'])), dtype=torch.float),
    edge_index=torch.tensor(edge_index).T,
    num_nodes=len(node_idx),
    time=torch.tensor(times.astype(int) // 10**9, dtype=torch.long), # convert to UNIX timestamps
)

# Save the data object to a file
torch.save(data, 'era5-india.pt')