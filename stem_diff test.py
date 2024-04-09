import stemdiff as sd
import matplotlib.pyplot as plt
import stemdiff.dbase
import psf
import tkinter as tk
from tkinter import filedialog
import os


class StemdiffWrapper:

    def __init__(self, debug=False):
        data_folder = self.select_folder()
        self.SDATA = sd.gvars.SourceData(
            detector=sd.detectors.TimePix(),
            data_dir=data_folder,
            filenames=r'*.dat')
        self.DIFFIMAGES = sd.gvars.DiffImages()
        self.SDIR = os.path.join(data_folder, 'results')
        self.debug = debug
        self.databases_calc()

    def select_folder(self):
        root = tk.Tk()
        root.withdraw()
        folder_selected = filedialog.askdirectory()

        return folder_selected

    def databases_calc(self):
        df1 = stemdiff.dbase.calc_database(self.SDATA, self.DIFFIMAGES)
        stemdiff.dbase.save_database(df1, output_file=self.SDIR + 'dbase_all.zip')
        df1 = sd.dbase.read_database(self.SDIR+'dbase_all.zip')
        sd.io.set_plot_parameters(size=(14,10), fontsize=11)
        df2 = df1.sort_values(by=['Peaks','S'], ascending=[False, False])[0:200]
        sd.dbase.save_database(df2, output_file=self.SDIR + 'dbase_sum.zip')
        df3 = df1.sort_values(by=['Peaks','S'], ascending=[False, False])[-20:]
        sd.dbase.save_database(df3, output_file=self.SDIR + 'dbase_for_psf.zip')
        psf_guess = psf.PSFtype1.get_psf(self.SDATA, self.DIFFIMAGES, df3)
        psf.save_psf_to_disk(psf_guess, self.SDIR + 'psf.npy')
        if self.debug == 1:
            df1.head(5)
            df2.head(5)
            sd.io.Datafiles.show_from_database(self.SDATA, df2, interactive=False, max_files=3, icut=300, cmap='plasma')
            sd.io.Datafiles.show_from_database(self.SDATA, df3, interactive=False, max_files=3, icut=300, cmap='plasma')
            psf.plot_psf(psf_guess, plt_type='3D', plt_size=30)


wrap = StemdiffWrapper(debug=True)
