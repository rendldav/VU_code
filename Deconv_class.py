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
    tohle je test jestli všechno jede jak má
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
        """
        DeconvRL Method Documentation:

        :param image: The input image to be deconvolved.
        :param psf: The point spread function (PSF) used for deconvolution.
        :return: The deconvolved image.

        The deconvRL method performs a Richardson-Lucy deconvolution on the input image using the given PSF.
        If the "cuda" flag is set to True, the method utilizes GPU acceleration using the CuPy library for faster computation.
        Otherwise, it uses the NumPy library.

        The PSF is normalized by dividing it by the sum of its elements to ensure its total intensity is equal to 1.

        The deconvolution algorithm iterates over a specified number of iterations.
        In each iteration, it performs the following steps:
        1. Convolve the current deconvolved image with the PSF using the convolve_func function.
        2. Calculate the ratio between the input image and the convolution result, adding a small constant to avoid division by zero.
        3. Convolve the ratio with the rotated PSF using the convolve_func function.
        4. Multiply the current deconvolved image by the convolved ratio.
        5. Clip the values of the deconvolved image to maintain stability and prevent negative values.
        6. Normalize the deconvolved image to the range 0-1 and then scale it to match the original image range.

        After the deconvolution process is completed, the deconvolved image can be optionally displayed using the display_image method,
        if the "display" flag is set to 1.

        Finally, the deconvolved image is returned.

        Example usage:

        image = ...
        psf = ...
        deconvolved_image = deconvRL(image, psf)
        """
        if self.cuda:
            image = cp.asarray(image)
            psf = cp.asarray(psf)
            convolve_func = convolve
        else:
            convolve_func = np_convolve

        psf = psf / cp.sum(psf)  # Normalize PSF
        img_dec = cp.copy(image)
        start_time = time.time()
        for i in range(self.iterations):
            den = convolve_func(img_dec, psf, mode='reflect')
            ratio = image / (den + 1e-6)  # Add small constant to avoid division by zero
            temp = convolve_func(ratio, cp.rot90(psf, 2), mode='reflect')
            img_dec *= temp
            # Clip values to maintain stability and prevent negative values
            img_dec = cp.clip(img_dec, 0, cp.max(image))
            #print(f"Iteration {i + 1}, Mean Intensity: {img_dec.mean()}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Deconvolution took: {elapsed_time} seconds')
        # Normalize to 0-1 and then scale to original range
        #img_dec = (img_dec - img_dec.min()) / (img_dec.max() - img_dec.min())
        #img_dec = (img_dec * cp.max(image)).astype(image.dtype)

        # Display the deconvolved image
        if self.display == 1:
            self.display_image(img_dec, f'Result for {self.iterations} iterations')

        return img_dec

    def deconvRLTM(self, image, psf, lambda_reg):
        """
        :param image: Input image to be deconvolved.
        :param psf: Point spread function.
        :param lambda_reg: Regularization parameter for deconvolution.
        :return: Deconvolved image.

        This method performs Richardson-Lucy Tikhonov (RLTM) deconvolution on an input image using a given point spread function and regularization parameter. The deconvolution process
        * iteratively estimates the original image by correcting the blurring effects caused by the PSF.

        The method accepts the following parameters:
        - image: Numpy array representing the input image.
        - psf: Numpy array representing the point spread function.
        - lambda_reg: Regularization parameter for controlling the trade-off between resolution and noise amplification in the deconvolution process.

        The method returns the deconvolved image as a Numpy array.

        Example Usage:
        ```
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        psf = np.array([[0.1, 0.1, 0.1], [0.1, 0.5, 0.1], [0.1, 0.1, 0.1]])
        lambda_reg = 0.1

        result = deconvRLTM(image, psf, lambda_reg)
        ```

        """
        if self.cuda:
            image = cp.asarray(image)
            psf = cp.asarray(psf)
            convolve_func = convolve
        else:
            convolve_func = np_convolve

        psf = psf / cp.sum(psf)  # Normalize PSF
        img_dec = cp.copy(image)
        start_time = time.time()

        laplacian_filter = self.fspecial_laplacian(0.00001)

        for i in range(self.iterations):
            laplacian_image = convolve_func(img_dec, laplacian_filter, mode='reflect')
            den = convolve_func(img_dec, psf, mode='reflect')
            ratio = image / (den + 1e-6)  # Add small constant to avoid division by zero
            temp = convolve_func(ratio, cp.rot90(psf, 2), mode='reflect')
            img_dec *= temp*(1/(2-lambda_reg*laplacian_image+1e-6))
            img_dec = cp.clip(img_dec, 0, cp.max(image))
            #img_dec = (img_dec/cp.sum(img_dec))*cp.sum(image)
            #print(f"Iteration {i + 1}, Mean Intensity: {img_dec.mean()}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Deconvolution took: {elapsed_time} seconds for {self.iterations} iterations')

        # Display the deconvolved image
        if self.display == 1:
            self.display_image(img_dec, f'Result for {self.iterations} iterations, lambda_TM = {lambda_reg}')

        return img_dec




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