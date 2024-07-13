import cv2
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
import ediff
from skimage import transform, measure, morphology
from skimage.feature import peak_local_max
import bcorr
from skimage.filters import threshold_otsu
from skimage import exposure
from skimage import io
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from sklearn.mixture import GaussianMixture

RL = RichardsonLucy(iterations=150)
img_path = r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA"
df_path = r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA\resultsdbase_sum.zip"

SDATA = sd.gvars.SourceData(
    detector=sd.detectors.TimePix(),
    data_dir=img_path,
    filenames=r'*.dat')
DIFFIMAGES = sd.gvars.DiffImages()

df_sum = dbase.read_database(df_path)
datafiles = [datafile[1] for datafile in df_sum.iterrows()]
datafile = df_sum.loc[df_sum['S'].idxmax(), 'DatafileName']
datafile_name = SDATA.data_dir.joinpath(datafile)
arr = sd.io.Datafiles.read(SDATA, datafile_name)
preprocessed_image = bcorr.rolling_ball(arr, radius=1)
sd.io.Arrays.show(preprocessed_image, icut=150)

reshaped_image = preprocessed_image.reshape(-1, 1)

# Apply Gaussian Mixture Model
gmm = GaussianMixture(n_components=3)
gmm.fit(reshaped_image)
gmm_labels = gmm.predict(reshaped_image)

# Reshape the labels back to the original image shape
segmented_image = gmm_labels.reshape(preprocessed_image.shape)

# Create a binary image from the segmented image
# Assume that the peaks are in the component with the higher mean value
peak_component = np.argmax(gmm.means_)
binary_image_gmm = (segmented_image == peak_component).astype(np.uint8)
binary_image = cv2.morphologyEx(binary_image_gmm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)))
binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)))

labeled_image = label(binary_image)
new_image = np.zeros_like(arr, dtype=np.float64)
regions = regionprops(labeled_image)
for region in regions:
    if region.area >= 3:
        coords = region.coords
        sum_intensity = arr[coords[:, 0], coords[:, 1]].sum()
        centroid = region.centroid
        new_image[int(centroid[0]), int(centroid[1])] = sum_intensity

# Display the binary image
plt.imshow(np.where(arr>150,150, arr), cmap='gray')
plt.title("Original image")
plt.axis('off')
plt.show()

plt.imshow(binary_image_gmm, cmap='gray')
plt.title("Binary image from GMM")
plt.axis('off')
plt.show()

plt.imshow(binary_image, cmap='gray')
plt.title("Cleaned binary Image with Peaks")
plt.axis('off')
plt.show()

plt.imshow(np.where(new_image>150,150, new_image), cmap='gray')
plt.title("New Image with Peaks")
plt.axis('off')
plt.show()


