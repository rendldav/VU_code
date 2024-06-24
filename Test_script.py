import numpy as np
from Img_blur import ImageBlurring
from Deconv_class import RichardsonLucy
import cv2

blurer = ImageBlurring()
deblurer = RichardsonLucy(800, True, display=True, turn_off_progress_bar=False)
lena = blurer.load_image(r"C:\Users\drend\OneDrive\Plocha\VU\Lena.png")
img, kernel = blurer.gaussian_blur(lena, [20,20], 5)
deblurred_img = deblurer.deconvRLTV(img, kernel, 0.06)
cv2.imwrite('deblurred_lena_RLTV_Gauss.png', (deblurred_img*255).astype(np.uint8))
cv2.imwrite('lena_Gauss.png', (img*255).astype(np.uint8))



