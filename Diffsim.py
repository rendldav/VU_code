import numpy as np
import matplotlib.pyplot as plt
from scipy.special import voigt_profile

plt.rcParams['figure.dpi'] = 500
class DiffractionPatternSimulator:
    def __init__(self, image_size=256):
        self.image_size = image_size
        self.image = np.zeros((self.image_size, self.image_size))
        self.x, self.y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
        self.psf_sigma = 1
        self.psf_gamma = 1
        self.central_peak_amplitude = 1.0
        self.num_rings = 2
        self.ring_radii = np.linspace(20, 50, self.num_rings)
        self.min_peaks_per_ring = 1
        self.max_peaks_per_ring = 6
        self.peak_amplitude_range = (0.1, 1)

    def add_poisson_noise(self):
        """
        Add Poisson noise to the image. The noise level depends on the image values itself.
        """
        noise = np.random.poisson(self.image).astype(np.float64)
        self.image += noise

    def add_gaussian_noise(self, mean=0, std=1):
        """
        Add Gaussian noise to the image.
        :param mean: Mean value of the Gaussian noise.
        :param std: Standard deviation of the Gaussian noise.
        """
        noise = np.random.normal(mean, std, self.image.shape)
        self.image += noise

    def add_gaussian_background(self, amplitude=0.2, sigma=30):
        """
        Add a Gaussian hill as background to the image.
        :param amplitude: The maximum amplitude of the Gaussian background.
        :param sigma: The standard deviation of the Gaussian background (controls the width of the hill).
        """
        X, Y = np.meshgrid(np.linspace(0, self.image_size, self.image_size),
                           np.linspace(0, self.image_size, self.image_size))
        X0 = Y0 = self.image_size / 2
        gaussian_hill = amplitude * np.exp(-(((X - X0) ** 2 + (Y - Y0) ** 2) / (2.0 * sigma ** 2)))
        self.image += gaussian_hill

    def voigt_peak(self, x0, y0, amplitude):
        """
        Generate a Voigt peak.
        """
        r = np.sqrt((self.x - x0) ** 2 + (self.y - y0) ** 2)
        profile = voigt_profile(r, self.psf_sigma, self.psf_gamma)
        return amplitude * profile / np.max(profile)

    def visualize_voigt_profile(self):
        """
        Visualize the Voigt distribution used in the simulation in 3D.
        """
        # Create a grid and calculate Voigt profile
        x = np.linspace(-10, 10, 200)
        y = np.linspace(-10, 10, 200)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Z = voigt_profile(R, self.psf_sigma, self.psf_gamma)

        # Plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Voigt Profile')

        plt.title("3D Visualization of the Voigt Distribution")
        plt.show()


    def add_central_peak(self):
        """
        Add a large central Voigt peak.
        """
        self.image += self.voigt_peak(self.image_size // 2, self.image_size // 2, self.central_peak_amplitude)

    def add_rings(self):
        """
        Add random Voigt peaks on rings.
        """
        #np.random.seed(42)
        for radius in self.ring_radii:
            num_peaks = np.random.randint(self.min_peaks_per_ring, self.max_peaks_per_ring + 1)
            angles = np.random.uniform(0, 2 * np.pi, num_peaks)
            for angle in angles:
                peak_x = int(self.image_size // 2 + radius * np.cos(angle))
                peak_y = int(self.image_size // 2 + radius * np.sin(angle))
                amplitude = np.random.uniform(*self.peak_amplitude_range)
                self.image += self.voigt_peak(peak_x, peak_y, amplitude)

    def generate_pattern(self):
        """
        Generate the full diffraction pattern.
        """
        self.add_central_peak()
        self.add_rings()
        plt.imshow(self.image, vmax=1)
        plt.show()


    def display_image(self):
        """
        Display the generated diffraction pattern.
        """
        plt.imshow(self.image, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar()
        plt.title("Synthetic Diffraction Pattern with Voigt Peaks")
        plt.show()

    def generate_peak_mask(self, exclude_central=True, threshold=0.01, exclusion_radius=5):
        """
        Generate a binary mask where smaller peaks are located, excluding the central peak.
        :param exclude_central: Bool, whether to exclude the central peak from the mask.
        :param threshold: The intensity threshold above which a pixel is considered part of a peak.
        :param exclusion_radius: Radius around the central peak to exclude from the mask.
        """
        mask = self.image > threshold
        if exclude_central:
            center = self.image_size // 2
            for i in range(self.image_size):
                for j in range(self.image_size):
                    if np.sqrt((i - center) ** 2 + (j - center) ** 2) < exclusion_radius:
                        mask[i, j] = False
        return mask

    def display_mask(self, mask):
        """
        Display the binary mask.
        """
        plt.imshow(mask, cmap='gray')
        plt.colorbar()
        plt.title("Binary Mask of Smaller Peaks")
        plt.show()


simulator = DiffractionPatternSimulator()
simulator.central_peak_amplitude = 6.0
simulator.generate_pattern()
mask = simulator.generate_peak_mask(threshold=0.1, exclusion_radius=12)
simulator.add_gaussian_background()
simulator.add_gaussian_noise(mean=0, std=0.25)


simulator.display_image()
simulator.display_mask(mask)