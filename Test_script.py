import numpy as np

from Img_blur import ImageBlurring
from Deconv_class import RichardsonLucy
import cv2

blurer = ImageBlurring()
deblurer = RichardsonLucy(500, True, display=True)
lena = blurer.load_image(r"C:\Users\drend\Desktop\VU\Lena.png")

img, kernel = blurer.motion_blur(lena, 180, 0)
img_write = (img*255).astype(np.uint8)
cv2.imwrite("Lena_Blurred.png", img_write)
deblurred_img = deblurer.deconvRLTM(img, kernel, 0.01)
cv2.imwrite("Deconv_lena.png", deblurred_img)