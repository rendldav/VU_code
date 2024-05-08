import numpy as np
from Img_blur import ImageBlurring
from Deconv_class import RichardsonLucy
import cv2

blurer = ImageBlurring()
deblurer = RichardsonLucy(500, True, display=True)
lena = blurer.load_image(r"C:\Users\drend\OneDrive\Plocha\VU\Lena.png")
#test2
img, kernel = blurer.motion_blur(lena, 30, 45)
deblurred_img = deblurer.deconvRLTV(img, kernel, 0.08)
