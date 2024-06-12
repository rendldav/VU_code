import numpy as np
import stemdiff.dbase
import stemdiff as sd
import idiff
import psf
import bcorr
from skimage import restoration
import tqdm
import sys
from Deconv_class import RichardsonLucy


def sum_datafiles(SDATA, DIFFIMAGES, df, deconv=0, psf=None, iterate=10, regularization=None, lambda_reg=0.05):
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize
    datafiles = [datafile[1] for datafile in df.iterrows()]
    sum_arr = np.zeros((img_size * R, img_size * R), dtype=np.float32)
    total_tasks = len(datafiles)
    sys.stderr = sys.stdout

    with tqdm.tqdm(total=total_tasks, desc="Processing ") as pbar:
        try:
            # Process each image in the database
            for index, datafile in df.iterrows():
                # Deconv0 => sum datafiles without deconvolution
                if deconv == 0:
                    sum_arr += dfile_without_deconvolution(
                        SDATA, DIFFIMAGES, datafile)
                # Deconv1 => sum datafiles with DeconvType1
                elif deconv == 1:
                    sum_arr += dfile_with_deconvolution_type1(
                        SDATA, DIFFIMAGES, datafile, psf, iterate, regularization=regularization, lambda_reg=lambda_reg)
                # Deconv2 => sum datafiles with DeconvType2
                elif deconv == 2:
                    sum_arr += dfile_with_deconvolution_type2(
                        SDATA, DIFFIMAGES, datafile, iterate)
                # Update the progress bar for each processed image
                pbar.update(1)
        except Exception as e:
            print(f"Error processing a task: {str(e)}")

    # (4) Move to the next line after the progress bar is complete
    print('')

    # (5) Post-process the summation and return the result
    return sum_postprocess(sum_arr, len(df))


def sum_postprocess(sum_of_arrays, n):
    """
    Normalize and convert the summed array to 16-bit unsigned integers.

    Parameters
    ----------
    sum_of_arrays : np.array
        Sum of the arrays -
        usually from stemdiff.sum.sum_datafiles function.
    n : int
        Number of summed arrays -
        usually from stemdiff.sum.sum_datafiles function.

    Returns
    -------
    arr : np.array
        Array representing final summation.
        The array is normalized and converted to unsigned 16bit integers.
    """
    arr = np.round(sum_of_arrays/n).astype(np.uint16)
    return(arr)


def dfile_without_deconvolution(SDATA, DIFFIMAGES, datafile):
    """
    Prepare datafile for summation without deconvolution (deconv=0).

    Parameters
    ----------
    SDATA : stemdiff.gvars.SourceData object
        The object describing source data (detector, data_dir, filenames).
    DIFFIMAGES : stemdiff.gvars.DiffImages object
        The bject describing the diffraction images/patterns.
    datafile : one row from the prepared database of datafiles
        The database of datafiles is created
        in stemdiff.dbase.calc_database function.
        Each row of the database contains
        [filename, xc, yc, MaxInt, NumPeaks, S].

    Returns
    -------
    arr : 2D numpy array
        The datafile in the form of the array,
        which is ready for summation (with DeconvType0 => see Notes below).

    Notes
    -----
    * The parameters are transferred from the `sum_datafiles` function.
    * DeconvType0 = no deconvolution,
      just summation of the prepared datafiles (upscaled, centered...).
    """

    # (0) Prepare variables
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize

    # (1) Read datafile
    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)

    # (2) Rescale/upscale datafile and THEN remove border region
    # (a) upscale datafile
    arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
    # (b) get the accurate center of the upscaled datafile
    # (the center coordinates for each datafile are saved in the database
    # (note: our datafile is one row from the database => we know the coords!
    xc, yc = (round(datafile.Xcenter), round(datafile.Ycenter))
    # (c) finally, the borders can be removed with respect to the center
    arr = stemdiff.io.Arrays.remove_edges(arr, img_size * R, xc, yc)
    # (Important technical notes:
    # (* This 3-step procedure is necessary to center the images precisely.
    # (  The accurate centers from upscaled images are saved in database.
    # (  The centers from original/non-upscaled datafiles => wrong results.
    # (* Some border region should ALWAYS be cut, for two reasons:
    # (  (i) weak/zero diffractions at edges and (ii) detector edge artifacts

    # (3) Return the datafile as an array that is ready for summation
    return (arr)


def dfile_with_deconvolution_type1(SDATA, DIFFIMAGES, datafile, psf, iterate, regularization=None, lambda_reg=0.05):
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize

    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)
    arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
    xc, yc = (round(datafile.Xcenter), round(datafile.Ycenter))
    arr = stemdiff.io.Arrays.remove_edges(arr, img_size * R, xc, yc)

    norm_const = np.max(arr)
    arr_norm = arr / np.max(arr)
    psf_norm = psf / np.max(psf)
    RL = RichardsonLucy(iterations=iterate, cuda=True, timer=False, turn_off_progress_bar=True)
    arr_deconv = RL.deconvRL(arr_norm, psf_norm)
    #arr_deconv = restoration.richardson_lucy(
    #    arr_norm, psf_norm, num_iter=iterate)

    # (c) restore original range of intensities = re-normalize
    arr = arr_deconv * norm_const


    return arr


def dfile_with_deconvolution_type2(SDATA, DIFFIMAGES, datafile, iterate):
    R = SDATA.detector.upscale
    img_size = DIFFIMAGES.imgsize
    psf_size = DIFFIMAGES.psfsize
    RL = RichardsonLucy(iterations=iterate, cuda=True)
    

    datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
    arr = stemdiff.io.Datafiles.read(SDATA, datafile_name) 
    arr = stemdiff.io.Arrays.rescale(arr, R, order=3)
    xc,yc = (round(datafile.Xcenter),round(datafile.Ycenter))
    arr = stemdiff.io.Arrays.remove_edges(arr,img_size*R,xc,yc)
    arr = bcorr.rolling_ball(arr, radius=20)
    PSF = psf.PSFtype2.get_psf(arr, psf_size, circular=True)


    arr_deconv = RL.deconvRL(arr, PSF)

    return(arr_deconv)
