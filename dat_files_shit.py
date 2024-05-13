import dbase
import numpy as np
from Deconv_class import RichardsonLucy
import matplotlib.pyplot as plt
import os
from PSF_fit import fit_and_sample_voigt_2d, fit_and_sample_gaussian_2d, smooth_psf, detect_and_extract_peak

img_path = r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA"
df_path = r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA\resultsdbase_sum.zip"
psf = np.load(r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA\resultspsf.npy")
psf = fit_and_sample_voigt_2d(psf, [100,100])

print(psf.shape)
df = dbase.read_database(df_path)
filenames = df['DatafileName'].tolist()

deconv = RichardsonLucy(iterations=30)
filepath = os.path.join(img_path, str(filenames[124]))
data = np.fromfile(filepath, dtype='uint16').reshape((256,256))
data = data.astype(np.float32)
peak_region = detect_and_extract_peak(data)
psf = fit_and_sample_voigt_2d(peak_region, [40,40])
plt.imshow(peak_region, cmap='viridis')
plt.show()
plt.imshow(data, cmap='viridis', vmax=200)
plt.show()
deconvolved = deconv.deconvRLTV(data, psf, 0.07)



plt.imshow(deconvolved, cmap='viridis', vmax=3)
plt.show()
