import numpy as np
import matplotlib.pyplot as plt

def load_show(path, show=True):
    data = np.fromfile(path, dtype=np.uint16).reshape((256, 256))
    data = np.clip(data, 0, 200)
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
    print(f"Maximum data value: {np.max(data_norm)}")
    print(f"Median data value: {np.median(data_norm)}")

    if show:
        plt.imshow(data_norm,cmap='viridis')
        plt.show()

file = r"C:\Users\drend\OneDrive\Plocha\VU\FeO-Shell_Cimc\FeO-Shell_Cimc\03\000_038.dat"
load_show(file)