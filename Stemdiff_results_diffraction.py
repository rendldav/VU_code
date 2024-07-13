import cv2
import ediff.io
import matplotlib.pyplot as plt
import dbase
import numpy as np
from PSF_fit import fit_and_sample_voigt_2d, fit_and_sample_gaussian_2d, smooth_psf, average_datafiles
from Deconv_class import RichardsonLucy
import stemdiff as sd
import sum
import sum_Mirek
import summ
import summ_Mirek
import ediff as ed

img_path = r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA"
df_path = r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA\resultsdbase_sum.zip"
psf = np.load(r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA\resultspsf.npy")
print(np.max(psf))
print(psf.shape)
print(np.count_nonzero(psf == np.max(psf)))
psf = psf[30:90, 30:90]
flat_psf = psf.flatten()

num_elements_to_set_zero = int(0.6 * flat_psf.size)
threshold_value = np.partition(flat_psf, num_elements_to_set_zero)[num_elements_to_set_zero]
mask = psf <= threshold_value
psf[mask] = 0
psf_kernel = smooth_psf(psf)
psf_gauss = fit_and_sample_gaussian_2d(psf,[50, 50], display=False)
psf_voigt = fit_and_sample_voigt_2d(psf, [50, 50], display=False)

df_sum = dbase.read_database(df_path)
filenames = df_sum['DatafileName'].tolist()
SDATA = sd.gvars.SourceData(
    detector=sd.detectors.TimePix(),
    data_dir=img_path,
    filenames=r'*.dat')
DIFFIMAGES = sd.gvars.DiffImages()


#PSF dependence plot
sum_data_Mirek = summ_Mirek.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=0, psf=psf, iterate=50)
sum_data_orig = summ_Mirek.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=1, psf=psf, iterate=50)
sum_data_smoothed = sum.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=1, psf=psf_kernel, iterate=50)
sum_data_gauss = sum.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=1, psf=psf_gauss, iterate=50)
sum_data_voigt = sum.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=1, psf=psf_voigt, iterate=50)

ed.io.plot_radial_distributions(data_to_plot=[[sum_data_Mirek,'k--','No_deconv'],[sum_data_orig,'r-','PSF from center'],[sum_data_smoothed,'g-','Kernel estimate of PSF'],[sum_data_gauss,'b-.','Gauss fit PSF'],[sum_data_voigt, 'y-', 'Voigt fit PSF']], xlimit=250, ylimit=220, output_file='Au_PSF_dependence_corrected.png')

# TM plot
sum_data_TM1 = sum.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=1, psf=psf, iterate=50, regularization='TM', lambda_reg=0.01)
sum_data_TM2 = sum.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=1, psf=psf, iterate=50, regularization='TM', lambda_reg=0.02)
sum_data_TM3 = sum.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=1, psf=psf, iterate=50, regularization='TM', lambda_reg=0.03)
sum_data_TM4 = sum.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=1, psf=psf, iterate=50, regularization='TM', lambda_reg=0.04)

ed.io.plot_radial_distributions(data_to_plot=[[sum_data_Mirek,'k--','No_deconv'],[sum_data_TM1,'r-','TM 0.01'],[sum_data_TM2,'g-','TM 0.02'],[sum_data_TM3,'b-.','TM 0.03'],[sum_data_TM4, 'y-', 'TM 0.04']], xlimit=250, ylimit=220, output_file='Au_TM_dependence_corrected.png')

#TV plot

sum_data_TV1 = sum.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=1, psf=psf, iterate=50, regularization='TV', lambda_reg=0.03)
sum_data_TV2 = sum.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=1, psf=psf, iterate=50, regularization='TV', lambda_reg=0.06)
sum_data_TV3 = sum.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=1, psf=psf, iterate=50, regularization='Tv', lambda_reg=0.09)
sum_data_TV4 = sum.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=1, psf=psf, iterate=50, regularization='TV', lambda_reg=0.12)

ed.io.plot_radial_distributions(data_to_plot=[[sum_data_Mirek,'k--','No_deconv'],[sum_data_TV1,'r-','TV 0.03'],[sum_data_TV2,'g-','TV 0.06'],[sum_data_TV3,'b-.','TV 0.09'],[sum_data_TV4, 'y-', 'TV 0.12']], xlimit=250, ylimit=220, output_file='Au_TV_dependence_corrected.png')

#Deconv type2
sum_data_dcv2 = sum.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=2, iterate=50)
ed.io.plot_radial_distributions(data_to_plot=[[sum_data_Mirek,'k--','No_deconv'],[sum_data_orig,'r-','PSF from center peaks'],[sum_data_dcv2,'g-','PSF from center peaks type 2']], xlimit=250, ylimit=220, output_file='Au_deconv_type_comp.png')