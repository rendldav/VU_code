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

img_path = r"C:\Users\drend\OneDrive\Plocha\VU\4_MARUSKA_LAF3\MARUSKA_LAF3.DATA\D_MARUSKA_C214"
df_path = r"C:\Users\drend\OneDrive\Plocha\VU\4_MARUSKA_LAF3\MARUSKA_LAF3.DATA\D_MARUSKA_C214\resultsdbase_sum.zip"
psf = np.load(r"C:\Users\drend\OneDrive\Plocha\VU\4_MARUSKA_LAF3\MARUSKA_LAF3.DATA\D_MARUSKA_C214\resultspsf.npy")
print(np.max(psf))
print(psf.shape)
print(np.count_nonzero(psf == np.max(psf)))
psf = psf[30:90, 30:90]
flat_psf = psf.flatten()

# Find the number of elements to set to 0 (10% of the total number of elements)
num_elements_to_set_zero = int(0.6 * flat_psf.size)

# Find the threshold value which separates the smallest 10% values
threshold_value = np.partition(flat_psf, num_elements_to_set_zero)[num_elements_to_set_zero]

# Create a boolean mask for the elements that are smaller than or equal to the threshold value
mask = psf <= threshold_value
# Set the elements corresponding to the mask to 0
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


#počty iterací
#sum_data_Mirek = summ_Mirek.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=0, psf=psf, iterate=10)
#sd.io.Arrays.show(sum_data_Mirek, icut=300, cmap='viridis')

#sum_data_David_10iter = sum_Mirek.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv=2, psf=psf, iterate=50)
#sd.io.Arrays.save_as_image(sum_data_David_10iter, 'deconv2_laf3.png', itype='16bit', icut=200)
#sd.io.Arrays.show(sum_data_David_10iter, icut=300, cmap='viridis')
sum_data_David_10iter = ediff.io.read_image(r"C:\Users\drend\OneDrive\Plocha\VU\pythonProject\FeO_sum_segment3_16bit_for_pxrd.png", itype='16bit')
sd.io.Arrays.show(sum_data_David_10iter, icut=300, cmap='viridis')
#sum_data_David_Segment3 = sum.sum_datafiles(SDATA, DIFFIMAGES, df_sum, deconv='Segment3', psf=psf, iterate=10, regularization=None)
#sd.io.Arrays.show(sum_data_David_Segment3, icut=300, cmap='viridis')
#sum_data_David_10iter = ediff.io.read_image(r"C:\Users\drend\OneDrive\Plocha\VU\pythonProject\segment3_laf3.png", itype='16bit')

#Deconv
EDIR = r'./'
CIF_FILE = r"C:\Users\drend\OneDrive\Plocha\VU\3_FEO_PURE\FEO_PURE.DATA\Fe3O4.cif"
XRD_FILE = EDIR + 'xrd_au.txt'
ediff.io.set_plot_parameters(size=(10,5), dpi=120, fontsize=8)
XTAL = ediff.pxrd.Crystal(structure=CIF_FILE, temp_factors=0.8)
EPAR = ediff.pxrd.Experiment(wavelength=0.71, two_theta_range=(2,140))
PPAR = ediff.pxrd.PlotParameters(x_axis='q', xlim=(0.5,8.5))
PXDR = ediff.pxrd.PXRDcalculation(XTAL,EPAR,PPAR,peak_profile_sigma=0.01)

PXDR.save_diffractogram(XRD_FILE)
PXDR.save_diffractions(XRD_FILE+'.diff')
PXDR.plot_diffractogram(XRD_FILE+'.png')

ED_DIFFRACTOGRAM = sum_data_David_10iter
ediff.io.set_plot_parameters(size=(9,9), dpi=100, fontsize=10)
diffractogram = sum_data_David_10iter

ED_FILE1 = EDIR + 'ed1_raw.txt'
ediff.io.set_plot_parameters(size=(10,5), dpi=120, fontsize=8)
profile = ediff.radial.calc_radial_distribution(diffractogram)
plt.title('ED: raw data')
plt.plot(profile[0], profile[1])
plt.xlabel('Distance from center [pixels]')
plt.ylabel('Intensity')
plt.xlim(0,300)
plt.ylim(0,300)
plt.grid()
plt.tight_layout()
plt.show()

np.savetxt(ED_FILE1, np.transpose(profile), fmt=['%4d', '%8.2f'], header='Columns: Pixels, Intensity')

ED_FILE2 = EDIR + 'ed2_bcorr.txt'
DATA = ediff.background.InputData(ED_FILE1, usecols=[0,1], unpack=True)
PPAR = ediff.background.PlotParams(ED_FILE2, 'Pixels', 'Intensity', xlim=[0,300], ylim=[0,140])
IPLOT = ediff.background.InteractivePlot(DATA, PPAR, CLI=False, messages=True)
IPLOT.run()

eld = np.loadtxt(ED_FILE2, unpack=True)
xrd = np.loadtxt(XRD_FILE, unpack=True)

max_eld = float(eld[0,(eld[2]==np.max(eld[2]))])
max_xrd = float(xrd[2,(xrd[3]==np.max(xrd[3]))])

calibration_constant = max_xrd/max_eld
eld[0] = eld[0] * calibration_constant
eld[1] = eld[1]/np.max(eld[1])
eld[2] = eld[2]/np.max(eld[2])

# (e) Plot the calibrated ED profile with subtracted background
plt.title('ED: background correction + calibration')
plt.plot(eld[0], eld[2])
plt.xlabel('q [1/A]')
plt.ylabel('Intensity')
plt.xlim(1, 10)
plt.ylim(-0.1, 1.1)
plt.grid()
plt.tight_layout()
plt.show()

ED_FILE3 = EDIR + 'ed_final.txt'
fine_tuning = 1

# (c) Plot and save final result graph
# (we use the {xrd} and {eld} profiles calculated above
plt.plot(xrd[2], xrd[3], label=r'PXRD of FeO fcc')
plt.plot(eld[0]*fine_tuning, eld[2], color='red', label='ED experiment')
plt.xlim(1, 10)
plt.xlabel(r'$q$ [1/\u212B]')
plt.ylabel('Intensity')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(ED_FILE3+'.png', facecolor='white', dpi=300)
plt.show()








