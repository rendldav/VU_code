import numpy as np
import matplotlib.pyplot as plt


def psf_gauss(size=(100, 100), mean=0.0, sigma=1.0):
    # Generate x and y coordinates
    x = np.linspace(-size[0] // 2, size[0] // 2, size[0])
    y = np.linspace(-size[1] // 2, size[1] // 2, size[1])
    x, y = np.meshgrid(x, y)

    # Generate the 2D Gaussian distribution
    g = np.exp(-0.5 * ((x - mean) ** 2 + (y - mean) ** 2) / sigma ** 2)
    g /= np.sum(g)

    # Display the distribution
    plt.imshow(g, cmap='gray')
    plt.colorbar()
    plt.title(f'Gaussian PSF with mean = {mean} and sigma = {sigma}')
    plt.show()

    # Save the distribution to a file
    filename = f'Gauss_psf_mean_{mean}_sigma_{sigma}.png'
    plt.imsave(filename, g, cmap='gray')

    return g


def psf_motion(size=(100, 100), length=10, angle=45):
    # Convert the angle to radians
    angle = np.deg2rad(angle)

    # Create an array filled with zeros
    psf = np.zeros(size)

    # Calculate the start and end points of the motion path
    x0 = size[0] // 2 - np.cos(angle) * length // 2
    y0 = size[1] // 2 - np.sin(angle) * length // 2
    x1 = size[0] // 2 + np.cos(angle) * length // 2
    y1 = size[1] // 2 + np.sin(angle) * length // 2

    # Generate the x and y coordinates of the motion path
    x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)

    # Round the coordinates to the nearest integer values
    x, y = np.round(x).astype(int), np.round(y).astype(int)

    # Validate the coordinates
    within_bounds = ((x >= 0) & (x < size[0]) & (y >= 0) & (y < size[1]))
    x, y = x[within_bounds], y[within_bounds]

    # Generate the point spread function
    psf[x, y] = 1

    # Normalize the function
    psf /= np.sum(psf)

    plt.imshow(psf, cmap='gray')
    plt.colorbar()
    plt.title(f'Linear PSF with angle = {angle}')
    plt.show()

    # Save the distribution to a file
    filename = f'Linear psf.png'
    plt.imsave(filename, psf, cmap='gray')

    return psf
# Example call:
psf_gauss(size=(200, 200), mean=0, sigma=10)
psf_motion(size=(200, 200), length=80, angle=45)
