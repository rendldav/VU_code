import stemdiff as sd
import dbase
import numpy as np
from Deconv_class import RichardsonLucy
import matplotlib.pyplot as plt
import os
import cupy as cp
import cv2

img_path = r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA"
df_path = r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA\resultsdbase_sum.zip"
psf = np.load(r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA\resultspsf.npy")

print(psf.shape)
df = dbase.read_database(df_path)
print(df.head())
filenames = df['DatafileName'].tolist()

deconv = RichardsonLucy(iterations=300)
sum_deconvolved = cp.zeros((256, 256))

for files in filenames:
    filepath = os.path.join(img_path, str(files))
    data = np.fromfile(filepath, dtype='uint16').reshape((256,256))
    data = data.astype(np.float32)
    deconvolved = deconv.deconvRL(data, psf)
    sum_deconvolved += deconvolved

plt.imshow(sum_deconvolved.get(), cmap='viridis', vmax=20000)
plt.show()

