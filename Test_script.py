import numpy as np
from Img_blur import ImageBlurring
from Deconv_class import RichardsonLucy
import cv2

blurer = ImageBlurring()
deblurer = RichardsonLucy(500, True, display=True, turn_off_progress_bar=False)
lena = blurer.load_image(r"C:\Users\drend\OneDrive\Plocha\VU\lena.png")
img, kernel = blurer.motion_blur(lena, 80, 0)
cv2.imwrite("lena_blurred.png", (img*255).astype(np.uint8))
deblurred_img = deblurer.deconvRLTV(img, kernel, 0.08)
cv2.imwrite("deblurred_img_quokka.png", deblurred_img)

