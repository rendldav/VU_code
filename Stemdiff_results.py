import dbase
import numpy as np
from PSF_fit import fit_and_sample_voigt_2d, fit_and_sample_gaussian_2d, smooth_psf, average_datafiles
import stemdiff as sd
import sum
import sum_Mirek
import ediff
#test
img_path = r"C:\Users\drend\Desktop\VU\1_AU\1_AU\DATA"
df_path = r"C:\Users\drend\Desktop\VU\1_AU\1_AU\DATA\resultsdbase_sum.zip"
psf = np.load(r"C:\Users\drend\Desktop\VU\1_AU\1_AU\DATA\resultspsf.npy")


df_sum = dbase.read_database(df_path)
filenames = df_sum['DatafileName'].tolist()

SDATA = sd.gvars.SourceData(
    detector=sd.detectors.TimePix(),
    data_dir=img_path,
    filenames=r'*.dat')
DIFFIMAGES = sd.gvars.DiffImages()

sum_data_Mirek = sum_Mirek.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=1, psf=psf, iterate=10)
sd.io.Arrays.show(sum_data_Mirek, icut=300, cmap='viridis')
ediff.io.plot_radial_distributions(data_to_plot=[[sum_data_Mirek,'k--','10it']], xlimit=250, ylimit=180)

sum_data_David = sum.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=1, psf=psf, iterate=10)
sd.io.Arrays.show(sum_data_David, icut=300, cmap='viridis')
ediff.io.plot_radial_distributions(data_to_plot=[[sum_data_David,'k--','10it_pico']], xlimit=250, ylimit=180)

