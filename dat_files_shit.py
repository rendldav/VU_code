import numpy as np
import matplotlib.pyplot as plt
from scipy.special import voigt_profile
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['figure.dpi'] = 500


def plot_voigt_1D(x_values, params_list):
    plt.figure(figsize=(10, 8))
    # Gaussian curve
    plt.plot(x_values, norm.pdf(x_values, 0, 1), color='red', label='Gaussian')
    # Voigt profiles
    colors = ['blue', 'green', 'purple', 'orange']
    for i, params in enumerate(params_list):
        gamma, sigma = params
        voigt_1D = voigt_profile(x_values, sigma, gamma)
        plt.plot(x_values, voigt_1D, color=colors[i], label=f'Voigt (gamma={gamma}, sigma={sigma})')
    plt.title("1D Voigt Profile")
    plt.xlabel('x values')
    plt.ylabel('Voigt / Gaussian profiles')
    plt.legend()
    plt.show()


def plot_voigt_2D(x_values, y_values, gamma, sigma):
    X, Y = np.meshgrid(x_values, y_values)
    Z = voigt_profile(X, sigma, gamma) * voigt_profile(Y, sigma, gamma)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title(f"2D Voigt Profile, sigma={sigma}, gamma={gamma}")
    ax.set_xlabel('x values')
    ax.set_ylabel('y values')
    ax.set_zlabel('Voigt Profile')
    plt.show()


x_values = np.linspace(-10, 10, 1000)
y_values = np.linspace(-10, 10, 1000)

# List of (gamma, sigma) pairs for the Voigt plots
params_list = [(0.5, 0.2), (1, 1), (1, 2)]

plot_voigt_1D(x_values, params_list)
plot_voigt_2D(x_values, y_values, 0.5, 1)
