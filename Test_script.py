import numpy as np
from Img_blur import ImageBlurring
from Deconv_class import RichardsonLucy
import cv2

blurer = ImageBlurring()
deblurer = RichardsonLucy(800, True, display=True, turn_off_progress_bar=False)
lena = blurer.load_image(r"C:\Users\drend\OneDrive\Plocha\VU\Lena.png")
#test2
img, kernel = blurer.gaussian_blur(lena, [15,15], 5)
deblurred_img = deblurer.deconvRLTV(img, kernel, 0.03)
