import numpy as np

# Define the .dat file path
file_path = 'your_file_path.dat'

# Load the .dat file
data = np.fromfile(file_path, dtype=np.float32)

# Display the numpy array
print(data)