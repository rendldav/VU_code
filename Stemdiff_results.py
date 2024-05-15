import dbase
import numpy as np
from PSF_fit import fit_and_sample_voigt_2d, fit_and_sample_gaussian_2d, smooth_psf, average_datafiles
import stemdiff as sd
import sum
import ediff

img_path = r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA"
df_path = r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA\resultsdbase_sum.zip"
df_all_path = r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA\resultsdbase_all.zip"
df_psf_path = r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA\resultsdbase_for_psf.zip"
psf = np.load(r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA\resultspsf.npy")

psf_smoothed = smooth_psf(psf, sigma=2)
psf_voigt_fit = fit_and_sample_voigt_2d(psf,[50,50])
psf_gauss_fit = fit_and_sample_gaussian_2d(psf, [50,50])

df_psf = dbase.read_database(df_psf_path)
filenames_psf = df_psf['DatafileName'].tolist()

df_sum = dbase.read_database(df_path)
filenames = df_sum['DatafileName'].tolist()

SDATA = sd.gvars.SourceData(
    detector=sd.detectors.TimePix(),
    data_dir=img_path,
    filenames=r'*.dat')
DIFFIMAGES = sd.gvars.DiffImages()
sum_data10 = sum.sum_datafiles(SDATA, DIFFIMAGES,df_sum, deconv=1, psf=psf, iterate=10)
sd.io.Arrays.show(sum_data10, icut=4, cmap='viridis')

sum_data30 = sum.sum_datafiles(SDATA, DIFFIMAGES,df_sum, deconv=1, psf=psf, iterate=30)
sd.io.Arrays.show(sum_data30, icut=4, cmap='viridis')

sum_data50 = sum.sum_datafiles(SDATA, DIFFIMAGES,df_sum, deconv=1, psf=psf, iterate=50)
sd.io.Arrays.show(sum_data50, icut=4, cmap='viridis')
ediff.io.plot_radial_distributions(data_to_plot=[[sum_data10,'k--','10it'], [sum_data30,'r--','30it'],[sum_data50,'b--','50it']], xlimit=250, ylimit=5)

sum_data10 = sum.sum_datafiles(SDATA, DIFFIMAGES,df_sum, deconv=1, psf=psf, iterate=20)
sd.io.Arrays.show(sum_data10, icut=4, cmap='viridis')

sum_data30 = sum.sum_datafiles(SDATA, DIFFIMAGES,df_sum, deconv=1, psf=psf, iterate=20, regularization='TM')
sd.io.Arrays.show(sum_data30, icut=4, cmap='viridis')

sum_data50 = sum.sum_datafiles(SDATA, DIFFIMAGES,df_sum, deconv=1, psf=psf, iterate=20, regularization='TV')
sd.io.Arrays.show(sum_data50, icut=4, cmap='viridis')
ediff.io.plot_radial_distributions(data_to_plot=[[sum_data10,'k--','No reg'], [sum_data30,'r--','TM reg'],[sum_data50,'b--','TV reg']], xlimit=250, ylimit=5)