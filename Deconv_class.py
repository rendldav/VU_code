import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from cupyx.scipy.ndimage import convolve
from scipy.ndimage import convolve as np_convolve
import time
from tqdm import tqdm
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

        for i in tqdm(range(1, self.iterations)):
            ratio = I / convolve_func(O_k, P, mode='reflect')
            O_k = O_k * (convolve_func(ratio, cp.rot90(P, k=2), mode='reflect'))

        O_k = (O_k.get() * 255).astype(np.uint8)

        if self.display:
            plt.imshow(O_k, cmap='gray')
            plt.show()

        return O_k


    def deconvRLTM(self, image, psf, lambda_reg):
        if self.cuda:
            print('CUDA is available and in use.')
            laplacian = cp.array([[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]], dtype=np.float32)
            I = cp.asarray(image)
            O_k = cp.asarray(image)
            P = cp.asarray(psf)/cp.sum(psf)
            laplacian = cp.asarray(laplacian)
            convolve_func = convolve
        else:
            laplacian = np.array([[0, -1, 0],
                                  [-1, 4, -1],
                                  [0, -1, 0]], dtype=np.float32)
            I = np.asarray(image)
            O_k = np.asarray(image)
            P = np.asarray(psf)/cp.sum(psf)
            laplacian = np.asarray(laplacian)
            convolve_func = np_convolve


        for i in tqdm(range(1, self.iterations)):
            ratio = I / (convolve_func(O_k, P, mode='reflect') + 1e-6)
            O_k = O_k * (convolve_func(ratio, cp.rot90(P, k=2), mode='reflect'))
            laplacian_image = convolve_func(O_k, laplacian, mode='reflect')
            regularization_term = 1/(1-2*lambda_reg*laplacian_image)
            O_k = O_k * regularization_term
            O_k = cp.clip(O_k, 0, 1)

        O_k = (O_k.get() * 255).astype(np.uint8)

        if self.display:
            plt.imshow(O_k, cmap='gray')
            plt.show()

        return O_k

    def mL1(self, DU, alpha, beta):
        """
        Applies the half-quadratic algorithm (multiplicative version) for the prior function phi(s) = |s|.
        If |DU| < alpha/beta, returns beta/2.
        """
        V = 2 / alpha * np.maximum(DU, alpha / beta)
        return 1.0 / V

    def msetupLnormPrior(self, q, alpha, beta):
        """
        Sets up the normalization prior for a given 'q'.
        For q=1, uses the mL1 function to define the behavior of the prior.
        """
        if q == 1:
            P = {'fh': lambda x: self.mL1(x, alpha, beta)}
            return P
        else:
            raise ValueError("Only q=1 is implemented.")

    def compute_prior(self, O_k, P):
        usize = list(O_k.shape)  # Extract the size of the image
        usize.append(1)  # Add a singleton third dimension
        L = np.zeros(usize + [5])  # Shape: (height, width, 1, 5)

        # Compute the vertical and horizontal differences (adjusting for zero-based indexing)
        vertical_diff = np.sqrt(np.sum((O_k[1:, :, np.newaxis] - O_k[:-1, :, np.newaxis]) ** 2, axis=2))
        horizontal_diff = np.sqrt(np.sum((O_k[:, 1:, np.newaxis] - O_k[:, :-1, np.newaxis]) ** 2, axis=2))

        # Apply the P.fh function to the differences
        VV = np.repeat(P['fh'](vertical_diff)[:, :, np.newaxis], usize[2], axis=2)
        VH = np.repeat(P['fh'](horizontal_diff)[:, :, np.newaxis], usize[2], axis=2)

        L[:, :, :, 0] = np.concatenate([VV, np.zeros((1, usize[1], usize[2]))], axis=0)  # VV with an extra row
        L[:, :, :, 1] = np.roll(L[:, :, :, 0], shift=1, axis=0)  # Circular shift down
        L[:, :, :, 2] = np.concatenate([VH, np.zeros((usize[0], 1, usize[2]))], axis=1)  # VH with an extra column
        L[:, :, :, 3] = np.roll(L[:, :, :, 2], shift=1, axis=1)  # Circular shift right
        L[:, :, :, 4] = -np.sum(L[:, :, :, 0:4], axis=3)

        # Apply circular shifts to `O_k` and add a singleton third dimension
        img_dec_shifted_1 = np.roll(O_k, shift=-1, axis=0)[:, :, np.newaxis]  # Shift up
        img_dec_shifted_2 = np.roll(O_k, shift=1, axis=0)[:, :, np.newaxis]   # Shift down
        img_dec_shifted_3 = np.roll(O_k, shift=-1, axis=1)[:, :, np.newaxis]  # Shift left
        img_dec_shifted_4 = np.roll(O_k, shift=1, axis=1)[:, :, np.newaxis]   # Shift right
        O_k_singleton = O_k[:, :, np.newaxis]  # Original image with an extra dimension

        # Concatenate shifts along the fourth dimension to match MATLAB's `cat(4, ...)` behavior
        img_dec_cat = np.concatenate([img_dec_shifted_1, img_dec_shifted_2, img_dec_shifted_3, img_dec_shifted_4, O_k_singleton], axis=2)

        # Add a final dimension to `img_dec_cat` for compatibility with `L`
        img_dec_cat = img_dec_cat[:, :, np.newaxis, :]
        # Calculate the regularization element
        reg = np.sum(L * img_dec_cat, axis=-1)
        reg = reg.squeeze()

        return reg

    def deconvRLTV(self, image, psf, lambda_reg):
        if self.cuda:
            print('CUDA is available and in use.')
            I = cp.asarray(image)
            O_k = cp.asarray(image)
            P = cp.asarray(psf) / cp.sum(psf)
            convolve_func = convolve
        else:
            I = np.asarray(image)
            O_k = np.asarray(image)
            P = np.asarray(psf) / np.sum(psf)
            convolve_func = np_convolve

        P_h = self.msetupLnormPrior(1, lambda_reg, 100*lambda_reg)


        for i in tqdm(range(1, self.iterations)):
            reg = self.compute_prior(O_k.get(), P_h)
            reg = cp.asarray(reg)
            ratio = I / (convolve_func(O_k, P, mode='reflect') + 1e-6)
            O_k = O_k * (convolve_func(ratio, cp.rot90(P, k=2), mode='reflect'))
            regularization_term = 1/(1-lambda_reg * reg)
            O_k = O_k * regularization_term
            O_k = cp.clip(O_k, 0, 1)

        O_k = (O_k.get() * 255).astype(np.uint8)

        if self.display:
            plt.imshow(O_k, cmap='gray')
            plt.show()

        return O_k



