import numpy as np
import matplotlib.pyplot as plt

def load_show(path, show=True):
    data = np.fromfile(path, dtype=np.uint16).reshape((256, 256))
    if show:
        plt.imshow(data, vmin=0, vmax=300,cmap='viridis')
        plt.show()

file = r"C:\Users\drend\OneDrive\Plocha\VU\FeO-Shell_Cimc\FeO-Shell_Cimc\03\000_038.dat"
load_show(file)