import stemdiff as sd
import matplotlib.pyplot as plt
import stemdiff.dbase
import psf_function
import tkinter as tk
from tkinter import filedialog
import os


class StemdiffWrapper:

    def __init__(self, sample_name, num_of_files =200, debug=False):
        data_folder = self.select_folder()
        self.num_of_files = num_of_files
        self.SDATA = sd.gvars.SourceData(
            detector=sd.detectors.TimePix(),
            data_dir=data_folder,
            filenames=r'??\*.dat')
        self.DIFFIMAGES = sd.gvars.DiffImages()
        self.SDIR = os.path.join(data_folder, 'results')
        self.debug = debug
        self.sample_name = sample_name
        self.databases_calc()

    def select_folder(self):
        root = tk.Tk()
        root.withdraw()
        folder_selected = filedialog.askdirectory()

        return folder_selected

    def databases_calc(self, analysis=True, optimal_files_selection=True, plot_examples=True):
        df1 = stemdiff.dbase.calc_database(self.SDATA, self.DIFFIMAGES)
        if self.debug:
            print(df1.head(5))
        stemdiff.dbase.save_database(df1, output_file=self.SDIR + 'dbase_all.zip')
        df1 = sd.dbase.read_database(self.SDIR+'dbase_all.zip')
        sd.io.set_plot_parameters(size=(14,10), fontsize=11)
        if analysis:
            sd.io.set_plot_parameters(size=(18, 10), fontsize=11)
            plot = df1.plot.line(y=['MaxInt'], color='green')
            plot.set_xlabel('Datafiles')
            plot.set_ylabel('Primary beam intensity')
            plot.grid()
            plt.savefig('Primary_beam_intensity '+self.sample_name+'.png', format='png')

            plot = df1.plot.line(y=['Xcenter', 'Ycenter'])
            plot.set_xlabel('Datafiles')
            plot.set_ylabel('XY-position of primary beam')
            plot.grid()
            plt.savefig('XY_position_primary_beam '+self.sample_name+'.png', format='png')

            plot = df1.plot.scatter(x='Peaks', y='S', color='red', marker='x')
            plot.set_xlabel('Number of peaks')
            plot.set_ylabel('Shannon entropy')
            plot.grid()
            plt.savefig('Number_of_peaks_entropy '+self.sample_name+'.png', format='png')

        if optimal_files_selection:
            df1b = df1[(df1.Peaks > 0)]
            df1b = df1b[(df1.MaxInt > 1000)]
            # Exclude files with XY-center too shifted
            df1b = df1b[(500 < df1.Xcenter) & (df1.Xcenter < 520)]
            df1b = df1b[(530 < df1.Ycenter) & (df1.Ycenter < 550)]

        if plot_examples:
            fig, ax = plt.subplots(nrows=1, ncols=3)
            ax[0] = df1.plot.scatter(x='Peaks', y='S', color='red', marker='x', ax=ax[0])
            ax[0].set_xlabel('Number of peaks')
            ax[0].set_ylabel('Shannon entropy')
            ax[1] = df1.plot.scatter(x='Peaks', y='MaxInt', color='orange', marker='x', ax=ax[1])
            ax[1].set_xlabel('Number of peaks')
            ax[1].set_ylabel('Maximum intensity')
            ax[2] = df1.plot.scatter(x='S', y='MaxInt', color='orange', marker='x', ax=ax[2])
            ax[2].set_xlabel('Shannon entropy')
            ax[2].set_ylabel('Maximum intensity')
            for i in range(3): ax[i].grid()
            fig.tight_layout()
            plt.savefig('NMS_plot'+self.sample_name+'.png', format='png')

        if optimal_files_selection:
            df2 = df1b.sort_values(by=['Peaks','S'], ascending=[False, False])[0:self.num_of_files]
        else:
            df2 = df1.sort_values(by=['Peaks', 'S'], ascending=[False, False])[0:self.num_of_files]

        sd.dbase.save_database(df2, output_file=self.SDIR + 'dbase_sum.zip')
        df3 = df1[(df1.Peaks==1)&(df1.MaxInt>9000)]
        df3 = df3.sort_values(by='S', ascending=False)
        df3 = df3[-20:]
        sd.dbase.save_database(df3, output_file=self.SDIR + 'dbase_for_psf.zip')
        psf_guess = psf_function.PSFtype1.get_psf(self.SDATA, self.DIFFIMAGES, df3)
        psf_function.save_psf_to_disk(psf_guess, self.SDIR + 'psf.npy')
        if self.debug == 1:
            df1.head(5)
            df2.head(5)
            sd.io.Datafiles.show_from_database(self.SDATA, df2, interactive=False, max_files=3, icut=300, cmap='plasma')
            sd.io.Datafiles.show_from_database(self.SDATA, df3, interactive=False, max_files=3, icut=300, cmap='plasma')
            psf_function.plot_psf(psf_guess, plt_type='3D', plt_size=30)


wrap = StemdiffWrapper(debug=True, sample_name='LaF3', num_of_files=1000)
