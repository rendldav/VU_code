import cv2
import matplotlib.pyplot as plt
import dbase
import numpy as np
from PSF_fit import fit_and_sample_voigt_2d, fit_and_sample_gaussian_2d, smooth_psf, average_datafiles
from Deconv_class import RichardsonLucy
import stemdiff as sd
import sum
import sum_Mirek
import summ
import summ_Mirek
import ediff
import tkinter as tk
from tkinter import filedialog

class StemDiffAnalysis:
    def __init__(self, sample_name, cuda=True):
        self.cuda = cuda
        self.sample_name = sample_name
        self.data_folder = self.select_folder()
        self.df_path = self.select_folder()
        self.psf_path = self.select_file()
        self.regularization_type = ['None', 'TM', 'TV']
        self.num_of_iteration = [10, 30, 50, 70, 90]

        self.psf = np.load(self.psf_path)
        self.df = dbase.read_database(self.df_path)
        self.filenames = self.df['DatafileName'].tolist()
        self.SDATA = sd.gvars.SourceData(
            detector=sd.detectors.TimePix(),
            data_dir=self.data_folder,
            filenames=r'*.dat')
        self.DIFFIMAGES = sd.gvars.DiffImages()

    def select_folder(self):
        root = tk.Tk()
        root.withdraw()
        folder_selected = filedialog.askdirectory()

        return folder_selected

    def select_file(self):
        root = tk.Tk()
        root.withdraw()
        file_selected = filedialog.askopenfilename()

        return file_selected

    def iterations_test_NR (self):
        sum_data_Mirek = summ_Mirek.sum_datafiles(self.SDATA, self.DIFFIMAGES, self.df, deconv=0, psf=self.psf, iterate=10)
        sd.io.Arrays.save_as_image(sum_data_Mirek, 'FeO_sum_wo_deconv.png', icut=300)
        sd.io.Arrays.show(sum_data_Mirek, icut=300, cmap='viridis')
        ediff.io.plot_radial_distributions(data_to_plot=[[sum_data_Mirek, 'k-', 'No deconvolution']], xlimit=250,
                                           ylimit=180, output_file='No deconv radial plot.png')

        for num_of_iteration in self.num_of_iteration:
            sum_data_David = sum_data_David_10iter = sum.sum_datafiles(self.SDATA, self.DIFFIMAGES, self.df, deconv=1, psf=self.psf, iterate=num_of_iteration, regularization=None)
            sd.io.Arrays.show(sum_data_David_10iter, icut=300, cmap='viridis')
            sd.io.Arrays.save_as_image(sum_data_David, self.sample_name+str(num_of_iteration)+'_iter_RLNR.png')


