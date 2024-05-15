import numpy as np
from Img_blur import ImageBlurring
from Deconv_class import RichardsonLucy
import cv2

blurer = ImageBlurring()
deblurer = RichardsonLucy(100, True, display=True, turn_off_progress_bar=False)
lena = blurer.load_image(r"C:\Users\drend\OneDrive\Plocha\VU\lena.png")
img, kernel = blurer.gaussian_blur(lena, [10,10], 3)
deblurred_img = deblurer.deconvRL(img, kernel)
cv2.imwrite("deblurred_img.png", deblurred_img)


