from Img_blur import ImageBlurring
from Deconv_class import RichardsonLucy


blurer = ImageBlurring()
deblurer = RichardsonLucy(300, True, display=True)
lena = blurer.load_image(r'C:\Users\drend\OneDrive\Plocha\VU\Lena.png')
img, kernel = blurer.gaussian_blur(lena, [20,20], 5)
deblurred_img = deblurer.deconvRLTM(img, kernel, 0.01)
