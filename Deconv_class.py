import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from cupyx.scipy.ndimage import convolve
from scipy.ndimage import convolve as np_convolve
import time
import cv2
from scipy.signal import convolve2d
from skimage.color import rgb2ycbcr, ycbcr2rgb

class RichardsonLucy:
    """
    tohle je test jestli všechno jede jak má 2.0
    """

    def __init__(self, iterations=30, cuda=True, display=False):
        if not isinstance(iterations, int) or iterations < 1:
            raise ValueError("iterations should be a positive integer.")
        self.iterations = iterations
        self.cuda = cuda
        self.display = display

    def display_image(self, image, title):
        """
            Display an image with a given title.

            :param image: A numpy array representing the image to be displayed.
            :param title: A string representing the title of the image.
            :return: None

        """
        if self.cuda:
            image = cp.asnumpy(image)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.show()

    def fspecial_laplacian(self, alpha=0.2):
        """
        Creates a Laplacian filter based on the specified alpha value.

        :param alpha: A float value between 0 and 1 representing the strength of the Laplacian filter.
        :return: A Laplacian filter matrix.
        :raises ValueError: If alpha is not between 0 and 1.
        """
        # Validate alpha
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")

        if self.cuda:
            laplacian_4 = cp.array([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]], dtype=float)

            laplacian_8 = cp.array([[1, 1, 1],
                                    [1, -8, 1],
                                    [1, 1, 1]], dtype=float)

            laplacian_filter = (alpha / (alpha + 1)) * laplacian_8 + (1 / (alpha + 1)) * laplacian_4
        else:
            laplacian_4 = np.array([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]], dtype=float)

            laplacian_8 = np.array([[1, 1, 1],
                                    [1, -8, 1],
                                    [1, 1, 1]], dtype=float)

            laplacian_filter = (alpha / (alpha + 1)) * laplacian_8 + (1 / (alpha + 1)) * laplacian_4

        #print(laplacian_filter)

        return laplacian_filter

    def deconvRL(self, image, psf):

        if self.cuda:
            print('CUDA is available and in use.')
            I = cp.asarray(image)
            O_k = cp.asarray(image)
            P = cp.asarray(psf)/cp.sum(psf)
            convolve_func = convolve
        else:
            convolve_func = np_convolve

        for i in range(1, self.iterations):
            ratio = I / convolve_func(O_k, P, mode='reflect')
            O_k = O_k * (convolve_func(ratio, cp.rot90(P, k=2), mode='reflect'))

        O_k = (O_k.get() * 255).astype(np.uint8)

        if self.display:
            plt.imshow(O_k, cmap='gray')
            plt.show()

        return O_k


    def deconvRLTM(self, image, psf, lambda_reg):
        laplacian = self.fspecial_laplacian(0.33)
        if self.cuda:
            print('CUDA is available and in use.')
            I = cp.asarray(image)
            O_k = cp.asarray(image)
            P = cp.asarray(psf)/cp.sum(psf)
            laplacian = cp.asarray(laplacian)
            convolve_func = convolve
        else:
            convolve_func = np_convolve


        for i in range(1, self.iterations):
            ratio = I / convolve_func(O_k, P, mode='reflect')
            O_k = O_k * (convolve_func(ratio, cp.rot90(P, k=2), mode='reflect'))
            laplacian_image = convolve_func(O_k, laplacian, mode='reflect')
            regularization_term = 1/(1+2*lambda_reg*laplacian_image)
            O_k = O_k * regularization_term
            O_k = cp.clip(O_k, 0, 1)

        O_k = (O_k.get() * 255).astype(np.uint8)

        if self.display:
            plt.imshow(O_k, cmap='gray')
            plt.show()

        return O_k




    def deconvLandweber(self, image, psf, lambda_=0.4):
        """
        Deconvolution using the Landweber method.

        :param image: The input image to be deconvolved.
        :param psf: The point spread function (PSF) used for convolution.
        :param lambda_: The regularization parameter for the Landweber method. Defaults to 0.4.
        :return: The deconvolved image.

        """
        if self.cuda:
            image = cp.asarray(image)
            psf = cp.asarray(psf)
            convolve_func = convolve
        else:
            image = np.asarray(image)
            psf = np.asarray(psf)
            convolve_func = np_convolve

        img_dec = image
        start_time = time.time()
        for i in range(self.iterations):
            error_image = convolve_func(img_dec, psf, mode='reflect', cval=0.0) - image  # using convolve from cupyx
            img_dec = img_dec - lambda_ * (convolve_func(error_image, psf.T, mode='reflect', cval=0.0))  # using convolve from cupyx
            img_dec = cp.clip(img_dec, 0, cp.max(image))
            print(f"Iteration {i + 1}, Mean Intensity: {img_dec.mean()}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Deconvolution took: {elapsed_time} seconds')
        img_dec = (img_dec - img_dec.min()) / (img_dec.max() - img_dec.min())
        img_dec = (img_dec * cp.max(image)).astype(image.dtype)
        if self.display == 1:
            self.display_image(img_dec, 'deconvolved')