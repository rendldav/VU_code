import numpy as np
from Img_blur import ImageBlurring
from Deconv_class import RichardsonLucy
import cv2

blurer = ImageBlurring()
deblurer = RichardsonLucy(1000, True, display=True)
lena = blurer.load_image(r"C:\Users\drend\OneDrive\Plocha\VU\Lena.png")
#test2
img, kernel = blurer.motion_blur(lena, 35, 3)
deblurred_img = deblurer.deconvRL(img, kernel)
