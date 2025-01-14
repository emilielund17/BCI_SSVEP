import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt #for plotting, optional

# Path to your .mat file
mat_file_path = '/Users/HannahWolfe_1/Desktop/SSVEP-BCI-Data/BCI_SSVEP/data/S1.mat' # hannah
#mat_file_path = 'path/to/your/data/S01.mat' # emilie

# Load the .mat file
mat_contents = sio.loadmat(mat_file_path)

# Access the data (assuming it's named 'data' inside the .mat file)
data = mat_contents['data'] #adapt if your data variable name is different

#Data dimensions
print(data.shape) # Check the shape to ensure it matches the README info

# Accessing a specific trial (example):
# electrode index 10, time point 500, target index 5, block index 2
trial_data = data[10, 500, 5, 2] 
print(trial_data)