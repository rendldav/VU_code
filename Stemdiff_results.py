import dbase
import numpy as np
from PSF_fit import fit_and_sample_voigt_2d, fit_and_sample_gaussian_2d, smooth_psf
import stemdiff as sd
import sum

img_path = r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA"
df_path = r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA\resultsdbase_sum.zip"
psf = np.load(r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA\resultspsf.npy")
psf = fit_and_sample_voigt_2d(psf,[50,50])
print(psf)
print(psf.shape)
df = dbase.read_database(df_path)
filenames = df['DatafileName'].tolist()

SDATA = sd.gvars.SourceData(
    detector=sd.detectors.TimePix(),
    data_dir=img_path,
    filenames=r'*.dat')
DIFFIMAGES = sd.gvars.DiffImages()
sum_data = sum.sum_datafiles(SDATA, DIFFIMAGES,df, deconv=2, psf=psf, iterate=30)
sd.io.Arrays.show(sum_data, icut=3, cmap='viridis')