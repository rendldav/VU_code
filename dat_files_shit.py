import dbase
import numpy as np
from Deconv_class import RichardsonLucy
import matplotlib.pyplot as plt
import os
import stemdiff as sd
import ediff
from PSF_fit import fit_and_sample_voigt_2d, fit_and_sample_gaussian_2d, smooth_psf, detect_and_extract_peak

img_path = r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA"
df_path = r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA\resultsdbase_sum.zip"
psf = np.load(r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA\resultspsf.npy")
psf_voigt = fit_and_sample_voigt_2d(psf, [100,100])
psf_gauss = fit_and_sample_gaussian_2d(psf, [100,100])
psf_kernel = smooth_psf(psf)

print(psf.shape)
df = dbase.read_database(df_path)
filenames = df['DatafileName'].tolist()

deconv = RichardsonLucy(iterations=30)
filepath = os.path.join(img_path, str(filenames[124]))
data = np.fromfile(filepath, dtype='uint16').reshape((256,256))
plt.imshow(data, cmap='viridis', vmax=200)
plt.show()
data = (data - np.min(data)) / (np.max(data) - np.min(data))
data = data.astype(np.float64)

SDATA = sd.gvars.SourceData(
    detector=sd.detectors.TimePix(),
    data_dir=img_path,
    filenames=r'*.dat')
DIFFIMAGES = sd.gvars.DiffImages()
sum_data_no_dec = sd.summ.sum_datafiles(SDATA, DIFFIMAGES,df, deconv=0, psf=psf, iterate=10)

sum_data10 = sd.summ.sum_datafiles(SDATA, DIFFIMAGES,df, deconv=1, psf=psf_voigt, iterate=10)
sd.io.Arrays.show(sum_data10, icut=200, cmap='viridis')

sum_data30 = sd.summ.sum_datafiles(SDATA, DIFFIMAGES,df, deconv=1, psf=psf_kernel, iterate=10)
sd.io.Arrays.show(sum_data30, icut=200, cmap='viridis')

sum_data50 = sd.summ.sum_datafiles(SDATA, DIFFIMAGES,df, deconv=1, psf=psf_gauss, iterate=10)
sd.io.Arrays.show(sum_data50, icut=200, cmap='viridis')
sum_data60 = sd.summ.sum_datafiles(SDATA, DIFFIMAGES,df, deconv=1, psf=psf, iterate=10)
sd.io.Arrays.show(sum_data50, icut=200, cmap='viridis')

ediff.io.plot_radial_distributions(data_to_plot=[[sum_data_no_dec,'k--','Sum w/o deconvolution'], [sum_data10,'r--','Voigt fit PSF'],[sum_data30,'b--','Kernel estimate PSF'],[sum_data50,'g--','Gauss fit PSF'],[sum_data60,'k-','Central peak PSF']], xlimit=250, ylimit=250)
