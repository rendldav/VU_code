import dbase
import numpy as np
from PSF_fit import fit_and_sample_voigt_2d, fit_and_sample_gaussian_2d, smooth_psf, average_datafiles
import stemdiff as sd
import sum
import sum_Mirek
import summ
import summ_Mirek
import ediff

img_path = r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA"
df_path = r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA\resultsdbase_sum.zip"
psf = np.load(r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA\resultspsf.npy")


df_sum = dbase.read_database(df_path)
filenames = df_sum['DatafileName'].tolist()

SDATA = sd.gvars.SourceData(
    detector=sd.detectors.TimePix(),
    data_dir=img_path,
    filenames=r'*.dat')
DIFFIMAGES = sd.gvars.DiffImages()

sum_data_Mirek = summ_Mirek.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=0, psf=psf, iterate=10)
sd.io.Arrays.show(sum_data_Mirek, icut=300, cmap='viridis')
#ediff.io.plot_radial_distributions(data_to_plot=[[sum_data_Mirek,'k--','10it']], xlimit=250, ylimit=180)

sum_data_David_None = sum.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=1, psf=psf, iterate=60, regularization=None)
sd.io.Arrays.show(sum_data_David_None, icut=300, cmap='viridis')
sum_data_David_TM = sum.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=1, psf=psf, iterate=120, regularization='TM', lambda_reg=0.02)
sd.io.Arrays.show(sum_data_David_TM, icut=300, cmap='viridis')
sum_data_David_TV = sum.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=1, psf=psf, iterate=120, regularization='TV', lambda_reg=0.1)
sd.io.Arrays.show(sum_data_David_TV, icut=300, cmap='viridis')
ediff.io.plot_radial_distributions(data_to_plot=[[sum_data_Mirek,'k-','No_deconv'],[sum_data_David_None,'r--','RL no reg'],[sum_data_David_TM,'g:','RLTM'],[sum_data_David_TV,'b-.','RLTV']], xlimit=250, ylimit=220)

