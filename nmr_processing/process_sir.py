"""
Utilities for reading and processing SIR-specific NMR intensity files.

This module includes helpers for reading T1 intensity files and variable delay lists
used in SIR/T1 relaxation experiments.

TODO: Reorganize between process_sir.py and sirtools
TODO: Add plotting back into sir functions
TODO: Add better documentation for matrix math
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Model
from lmfit.models import PseudoVoigtModel
from nmrglue.fileio import bruker

from nmr_processing.plotting import plot_1d
from nmr_processing.processing import (
    get_1d_data,
    get_data_from_folder,
    get_ints_from_topspin,
    get_pseudo2d_data,
    pick_peaks_pseudo2d,
)


def load_vdlist(
    exp_path,
    delay_offset=True,
):
    """
    Load a Bruker variable delay list (VDLIST) and convert values to seconds.

    Parameters
    ----------
    exp_path : str
        Path to the experiment directory containing the `vdlist` file.
    delay_offset : bool, default: True
        If True, include half of the p1 and p2 pulse lengths in delay values.

    Returns
    -------
    np.ndarray
        Array of delay times in seconds.
    """

    vdlist_path = os.path.join(exp_path, "vdlist")

    str_arr = np.loadtxt(vdlist_path, dtype=str)
    delays = []
    for val in str_arr:
        # if val.isnumeric():
        #     delays.append(float(val))
        # else:
        try:
            delays.append(float(val))
        except ValueError as exc:
            num = float(val.rstrip("pnum"))
            c = val[-1]
            if c == "p":
                num *= 1e-12
            elif c == "n":
                num *= 1e-9
            elif c == "u":
                num *= 1e-6
            elif c == "m":
                num *= 1e-3
            else:
                raise ValueError(f"Unexpected value in vdlist: {val}") from exc
            delays.append(num)
    delays = np.array(delays)

    if delay_offset:
        pulse_lengths = bruker.read_acqus_file(exp_path)["acqus"]["P"]
        p1_and_p2 = float(pulse_lengths[1]) + float(pulse_lengths[2])
        delays += (p1_and_p2 * 1e-6) / 2
    return delays


def process_sir(
    exp_path,
    *,
    proc_num=1,
    peak_pos=None,
    regions=None,
    delay_offset=True,
):
    """
    Read the delay and intensity data from a Selective Inversion Recovery experiment.

    This function has three methods for determining peak intensities, listed in order of
    decreasing priority:
    1. Integrated intensities over the provided `regions`.
    2. Intensities extracted at the provided `peak_pos` values.
    3. Intensities extracted at the automatically-found positions of peaks.

    This function should also theoretically work for T1 measurements or other pseudo-2D
    experiments that use vdlists, but this is not tested.

    Parameters
    ----------
    exp_path : str
        Path to the experiment directory.
    proc_num : int, default: 1
        Processing number containing the processed data.
    peak_pos : array-like, optional
        Peak positions in ppm used to extract intensities.
    regions : list of tuple, optional
        List of ppm ranges to integrate across to determine peak intensities.
    delay_offset : bool, default: True
        If True, include half of the p1 and p2 pulse lengths in delay values.

    Returns
    -------
    delays : np.ndarray
        Experimental delays in seconds, with pulse length offset if `delay_offset`
        is True.
    intensities : np.ndarray
        Normalized intensity values for the selected regions/peaks.
    """

    # Load vdlist and interpret suffixes
    vdlist = os.path.join(exp_path, "vdlist")
    delays = load_vdlist(vdlist, delay_offset=delay_offset)

    bundle = get_pseudo2d_data(exp_path, proc_num=proc_num)
    bundle.update(
        pick_peaks_pseudo2d(bundle, peak_pos=peak_pos, regions=regions, normalize=True)
    )
    intensities = bundle["peak_ints_norm"]

    return delays, intensities


def make_cifit_files(
    filename,
    delays,
    intensities,
    *,
    title=None,
    peak_names=None,
    r1_guesses=None,
    k_guesses=None,
    initial_mag_guesses=None,
    final_mag_guesses=None,
    matrix=None,
    specify_vary=False,
):
    """
    Create CIFIT (`.dat` and `.mch`) input files from delay/intensity data.

    Parameters
    ----------
    filename : str
        Base filename for output files (without extension).
    delays : array-like
        Delay times in seconds.
    intensities : array-like
        Intensity values for each site at each time point. This must have n_time_points
        rows and n_sites columns.
    title : str, optional
        Title written to the header of output files.
    peak_names : list of str, optional
        Peak names to include in the `.dat` comment header.
    r1_guesses : list of float, optional
        Initial guess of R1 relaxation rates in Hz for each site. The default guesses
        are values measured for LPSC and LZC.
    k_guesses : list of float, optional
        Initial guess of the rate constant in Hz for each exchange process. The default
        guess for each process is 1 Hz, which is not likely to be a great guess.
    initial_mag_guesses : list of float, optional
        Initial guess of the initial magnetization for each site. Defaults to the first
        intensity values.
    final_mag_guesses : list of float, optional
        Initial guess of the final magnetization for each site. Defaults to the last
        intensity values.
    matrix : array-like, optional
        2D matrix describing off-diagonal contributions to exchange matrix. By default,
        this assumes all sites participate in exchange and have equal populations.
    specify_vary : bool, default: False
        If True, write in the .mch file whether parameters should be varied. Otherwise,
        write the file as Bain originally defined in his cifman.pdf.
    """

    make_dat_file(
        filename,
        delays=delays,
        intensities=intensities,
        title=title,
        peak_names=peak_names,
    )

    intensities = np.array(intensities)

    if not initial_mag_guesses:
        initial_mag_guesses = list(intensities[0])
    if not final_mag_guesses:
        final_mag_guesses = list(intensities[-1])

    make_mch_file(
        filename,
        title=title,
        n_sites=intensities.shape[1],
        initial_mag_guesses=initial_mag_guesses,
        final_mag_guesses=final_mag_guesses,
        r1_guesses=r1_guesses,
        specify_vary=specify_vary,
        k_guesses=k_guesses,
        matrix=matrix,
    )


def make_mch_file(
    filename,
    *,
    n_sites=2,
    n_procs=1,
    r1_guesses=None,
    k_guesses=None,
    initial_mag_guesses=None,
    final_mag_guesses=None,
    matrix=None,
    title="TEST",
    specify_vary=False,
):
    """
    Write a CIFIT mechanism (.mch) file describing the relaxation mechanism.

    specify_vary hard codes varying rate and M0 but keeping M_inf and T1s constant
    (for use with cifit2.1 aka cifit2 aka cifit_tp2)

    Parameters
    ----------
    filename : str
        Base filename for the output file (without extension).
    n_sites : int, default: 2
        Number of sites being analyzed.
    n_procs : int, default: 1
        Number of exchange processes.
    r1_guesses : list of float, optional
        Initial guess of R1 relaxation rates in Hz for each site. The default guesses
        are values measured for LPSC and LZC.
    k_guesses : list of float, optional
        Initial guess of the rate constant in Hz for each exchange process. The default
        guess for each process is 1 Hz, which is not likely to be a great guess.
    initial_mag_guesses : list of float, optional
        Initial guess of the initial magnetization for each site. Defaults to 1 for all
        sites, which is a bad guess. Please provide a better guess.
    final_mag_guesses : list of float, optional
        Initial guess of the final magnetization for each site. Defaults to 1 to match
        normalized intensities.
    matrix : array-like, optional
        2D matrix describing off-diagonal contributions to exchange matrix. By default,
        this assumes all sites participate in exchange and have equal populations.
    title : str, default: "TEST"
        Title written to the header of the .mch file.
    specify_vary : bool, default: False
        If True, write in the .mch file whether parameters should be varied. Otherwise,
        write the file as Bain originally defined in his cifman.pdf.
    """

    if matrix:
        matrix = np.array(matrix)
        if matrix.shape != (n_sites, n_sites):
            raise ValueError(
                "matrix must be a square array of shape `n_sites` by `n_sites`!"
            )
    else:
        if n_procs == 1:
            matrix = np.ones((n_sites, n_sites)) - np.diag(np.ones(n_sites))
        else:
            raise NotImplementedError(
                "This function can't currently handle "
                "the off-diagonals for more than one "
                "process."
            )

    mch_lines = [title]

    mch_lines.extend(["", f"{n_sites} {n_procs}"])

    if not r1_guesses:
        r1_guesses = [1 / 0.462, 1 / 2.141]  # Example reciprocal of LPSC and LZC T1s
    elif len(r1_guesses) != n_sites:
        raise ValueError("`r1_guesses` must be of length `n_sites`!")

    mch_lines.extend(["", " ".join(map(str, r1_guesses))])
    if specify_vary:
        mch_lines.append(" ".join(map(str, np.ones(n_sites, dtype=np.int8))))

    if final_mag_guesses:
        if len(final_mag_guesses) != n_sites:
            raise ValueError("`final_mag_guesses` must be of length `n_sites`!")
    else:
        final_mag_guesses = np.ones(n_sites)  # Final intensities are normalized, so 1
    mch_lines.extend(["", " ".join(map(str, final_mag_guesses))])
    if specify_vary:
        mch_lines.append(" ".join(map(str, np.ones(n_sites, dtype=np.int8))))

    if initial_mag_guesses:
        if len(initial_mag_guesses) != n_sites:
            raise ValueError("`initial_mag_guesses` must be of length `n_sites`!")
    else:
        # FIXME: This is a bad guess, it should be more like 1, -1
        initial_mag_guesses = np.ones(n_sites)
    mch_lines.extend(["", " ".join(map(str, initial_mag_guesses))])
    if specify_vary:
        mch_lines.append(" ".join(map(str, np.ones(n_sites, dtype=np.int8))))

    if k_guesses:
        if len(k_guesses) != n_sites:
            raise ValueError("`k_guesses` must be of length `n_sites`!")
    else:
        k_guesses = np.ones(n_procs)  # FIXME: This is a bad guess

    for i in range(n_procs):
        if specify_vary:
            # rate guess for the mechanism, with variation specified
            mch_lines.extend(["", str(k_guesses[i]) + " 1"])
        else:
            # rate guess for the mechanism
            mch_lines.extend(["", str(k_guesses[i])])
        mch_lines.append(str(n_sites * (n_sites - 1)))  # number of off-diagonals

        for j in range(n_sites):
            for k in range(n_sites):
                if matrix[j][k] != 0:
                    mch_lines.append(f"{j} {k} {matrix[j][k]}")

    with open(filename + ".mch", "w", encoding="utf-8") as file:
        file.writelines(s + "\n" for s in mch_lines)


def make_dat_file(filename, delays, intensities, *, title="TEST", peak_names=None):
    """
    Write a CIFIT `.dat` file containing delay and intensity data.

    Parameters
    ----------
    filename : str
        Basename for the output file (without extension).
    delays : array-like
        Delay times in seconds.
    intensities : array-like
        Intensities corresponding to each delay.
    title : str, optional
        File title line, by default "TEST".
    peak_names : list of str, optional
        peak_names for each intensity column.
    """

    data_lines = [title]

    numpoints = len(delays)
    intensities = np.array(intensities)
    if intensities.shape[0] != numpoints:
        raise ValueError("`delays` and `intensities` must have the same length!")

    data_lines.extend(["", str(numpoints), ""])

    if peak_names:
        if not isinstance(peak_names, (list, np.ndarray)):
            raise TypeError(
                "`peak_names` parameter must be of type `list` or `ndarray`"
            )
        comment = "# Tmix, " + ", ".join(map(str, peak_names))
    else:
        comment = "# Tmix, intensities..."
    data_lines.append(comment)

    delays = delays.reshape(numpoints, 1)
    data = np.concatenate((delays, intensities), axis=1)

    def _format_array(arr):
        return np.array2string(arr, precision=6, suppress_small=True)

    for line in data:
        data_lines.append("\t".join(map(_format_array, line)))
    # data_str = np.array_str(data, precision=5, suppress_small=True)
    # data_lines.extend(data_str.strip('[] ').split(']\n ['))

    with open(filename + ".dat", "w", encoding="utf-8") as file:
        file.writelines(s + "\n" for s in data_lines)


def exp_to_cifit(
    exp_path,
    *,
    filename=None,
    proc_num=1,
    peak_pos=None,
    peak_names=None,
    t1_guesses=None,
    k_guesses=None,
    initial_mag_guesses=None,
    final_mag_guesses=None,
    matrix=None,
    use_t1ints=True,
    specify_vary=False,
):
    """
    Convert a T1/SIR experiment directly from Bruker data files into CIFIT input files.

    Writes CIFIT `.dat` and `.mch` files to disk.

    Parameters
    ----------
    exp_path : str
        Path to the experiment directory.
    filename : str, optional
        Base filename for CIFIT files. By default, the files will be created with
        basenames matching the experiment number (e.g., 12.mch and 12.dat)
    proc_num : int, default: 1
        Processing number containing the processed data.
    peak_pos : list of float, optional
        Peak positions in ppm used for intensity extraction if no `t1ints.txt` file
        is present.
    peak_names : list of str, optional
        Labels for each peak.
    t1_guesses : list of float, optional
        Initial guess of T1 relaxation times in seconds for each site. The default
        guesses are values measured for LPSC and LZC.
    k_guesses : list of float, optional
        Initial guess of the rate constant in Hz for each exchange process. The default
        guess for each process is 1 Hz, which is not likely to be a great guess.
    initial_mag_guesses : list of float, optional
        Initial guess of the initial magnetization for each site. Defaults to the first
        intensity values.
    final_mag_guesses : list of float, optional
        Initial guess of the final magnetization for each site. Defaults to the last
        intensity values.
    matrix : array-like, optional
        2D matrix describing off-diagonal contributions to exchange matrix. By default,
        this assumes all sites participate in exchange and have equal populations.
    use_t1ints : bool, default: True
        If True, read data from `t1ints.txt`. Otherwise, extract intensities from
        processed pseudo-2D data in Bruker files.
    specify_vary : bool, default: False
        If True, write in the .mch file whether parameters should be varied. Otherwise,
        write the file as Bain originally defined in his cifman.pdf.

    Examples
    --------
    >>> exp_path = ('/Users/tylerpennebaker/BoxSync/wp6_exsy/EXSYstudy/'
    ...             '500.TP-2024.10.31_7Li_LZC+LPSC/219')
    >>> exp_to_cifit(exp_path, 'test', peak_pos=[1.46, -0.92])
    """

    if use_t1ints:
        metadata_bundle = get_ints_from_topspin(exp_path)
        delays = metadata_bundle["times"]
        intensities = metadata_bundle["peak_ints_norm"]
        positions = metadata_bundle["peak_pos_ppm"]
    else:
        delays, intensities = process_sir(
            exp_path, proc_num=proc_num, peak_pos=peak_pos
        )
        positions = None

    exp_name = os.path.basename(os.path.dirname(exp_path))
    exp_no = int(os.path.basename(exp_path))
    title = f"Extracted from {exp_name} exp no {exp_no}"

    if peak_names and positions:
        if len(peak_names) != len(positions):
            raise ValueError(f"Wrong number of peak names, should be {len(positions)}!")
    else:
        peak_names = [f"{s:.2f} ppm" for s in positions]

    if not t1_guesses:
        t1_guesses = [0.462, 2.141]  # Example T1 values of LPSC and LZC in s
    r1_guesses = [1 / t1 for t1 in t1_guesses]

    if filename is None:
        filename = exp_path

    make_cifit_files(
        filename,
        delays,
        intensities,
        title=title,
        peak_names=peak_names,
        r1_guesses=r1_guesses,
        specify_vary=specify_vary,
        k_guesses=k_guesses,
        matrix=matrix,
        initial_mag_guesses=initial_mag_guesses,
        final_mag_guesses=final_mag_guesses,
    )


def plot_cifit_csv(
    file_path, *, site_names=None, n_data_rows=16, n_fit_rows=101, save_path=None
):
    """
    Plot CIFIT result data from the output CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CIFIT CSV file.
    site_names : list of str, optional
        Labels for each site to display in plot legend.
    n_data_rows : int, default: 16
        Number of data rows to read from the CSV file.
    n_fit_rows : int, default: 101
        Number of smooth-fit rows to read from the CSV file.
    save_path : str, optional
        Output file path for the saved figure. If not specified, saves a PDF file with
        the same basename as the CSV.
    """

    # Set n_sites from the number of columns in the CSV file
    # Format = delay, calc1, calc2, ..., data1, data2, ..., diff1, diff2, ...
    with open(file_path, "r", encoding="utf-8") as f:
        f.readline()  # skip header lines
        line = f.readline()
        n_cols = len(re.findall(r"[.\d]+", line))
        n_sites = (n_cols - 1) // 3

    cols = ["delay"]
    cols.extend(map(str, range(n_sites * 3)))
    data_df = pd.read_csv(
        file_path,
        sep=None,
        nrows=n_data_rows,
        skiprows=1,
        header=None,
        index_col=False,
        names=cols,
    )
    # print('DATA_DF')
    # print(data_df)

    cols = ["delay"]
    cols.extend(map(str, range(n_sites)))
    fit_df = pd.read_csv(
        file_path,
        sep=None,
        nrows=n_fit_rows,
        skiprows=3 + n_data_rows,
        header=None,
        index_col=False,
        names=cols,
    )

    # print('FIT_DF')
    # print(fit_df)

    if not site_names:
        site_names = [f"Site {i+1}" for i in range(n_sites)]

    _, ax = plt.subplots()
    for i in range(n_sites):
        pts = ax.plot(
            data_df["delay"],
            data_df[str(n_sites + i)],
            ".",
            label=site_names[i] + " Data",
        )
        ax.plot(
            fit_df["delay"],
            fit_df[str(i)],
            label=site_names[i] + " Calc",
            color=pts[0].get_color(),
        )

    outpath = file_path.replace(".csv", ".out")
    if os.path.exists(outpath):
        with open(outpath, "r", encoding="utf-8") as f:
            outtext = f.read()
        outtext = outtext[outtext.find("Final") :]
        match = re.search(r"No. \d+=\s*([\.\d]+)\nChi", outtext)
        # print(match)
        if match:
            rate = match.group(1)
            k = float(rate)
            # print(k)
            ax.text(
                0.9,
                0.5,
                f"Rate = {k} Hz",
                horizontalalignment="right",
                transform=ax.transAxes,
            )
        else:
            print("Rate not found in .out file")
            return

    ax.legend()

    plt.tight_layout()
    if not save_path:
        save_path = file_path.replace(".csv", ".pdf")
    plt.savefig(save_path)
    plt.show()


########################
# Functions from leonmr:
########################
def get_1d_exsy_data(dir_path, exp_nums):
    """
    Extract 1D EXSY peak intensities from a directory of experiments.

    Parameters
    ----------
    dir_path : str
        Directory containing numbered experiment subfolders.
    exp_nums : list of int
        Experiment numbers to include in the analysis.

    Returns
    -------
    d15_vals : np.ndarray
        D15 delay values in seconds.
    peak_ints_norm : np.ndarray
        Normalized peak intensities off all experiments.

    Notes
    -----
    This function usually fails because the x values are changed between experiments.
    Not sure if this function is even useful though. Looks like lots of overlap with
    other functions. Maybe just delete.
    """

    d15_vals = []
    for exp_num in exp_nums:
        exp_path = os.path.join(dir_path, str(exp_num))
        metadata = bruker.read_acqus_file(exp_path)
        d15_vals.append(metadata["acqus"]["D"][15])
    d15_vals = np.array(d15_vals)

    data_bundle = get_data_from_folder(dir_path, exp_nums)
    if "x_vals_ppm" not in data_bundle:
        raise ValueError("x_vals_ppm don't match for all experiments! Can not proceed.")
    x_vals_ppm = data_bundle["x_vals_ppm"]
    # CHECK: order might be messed up in dict?
    y_data = [exp_bundle["y_data"] for _, exp_bundle in data_bundle.items()]

    proc_bundle = pick_peaks_pseudo2d(x_vals_ppm=x_vals_ppm, y_data=y_data)
    peak_ints_norm = proc_bundle["peak_ints_norm"]

    return d15_vals, peak_ints_norm


def analyze_lpsc_1d_exsys(dir_path, exp_nums, *, plot=False):
    """
    Analyze LPSC 1D EXSY experiments by fitting two Pseudo-Voigt peaks.

    Parameters
    ----------
    dir_path : str
        Directory containing numbered experiment subfolders.
    exp_nums : list of int
        Experiment numbers to include in the analysis.
    plot : bool, default: False
        If True, plot the initial fit on the first experiment.

    Returns
    -------
    d15_vals : np.ndarray
        D15 delay values in seconds.
    [p1_ints, p2_ints] : list of lists
        Relative intensities of the two LPSC peaks across all experiments.
    """

    first_path = os.path.join(dir_path, str(exp_nums[0]))

    if plot:
        bundle = plot_1d(first_path, f1p=3, f2p=-3.2)
        ax = bundle["ax"]
    else:
        bundle = get_1d_data(first_path)
        _, ax = plt.subplots()

    x_vals_ppm = bundle["x_vals_ppm"]
    y_data = bundle["y_data"]

    amplitude = -np.trapz(y_data, x=x_vals_ppm)

    lpsc_peak1 = PseudoVoigtModel(prefix="p1_")
    init_pars = lpsc_peak1.make_params(
        center={"value": 1.2, "min": -10, "max": 10},
        amplitude={"value": 0.6 * amplitude, "min": 0},
        sigma={"value": 0.1, "min": 0.001, "max": 3},
    )
    lpsc_peak2 = PseudoVoigtModel(prefix="p2_")
    init_pars.update(
        lpsc_peak2.make_params(
            center={"value": 1, "min": -10, "max": 10},
            amplitude={"value": 0.4 * amplitude, "min": 0},
            sigma={"value": 0.1, "min": 0.001, "max": 3},
        )
    )

    lpsc_model = lpsc_peak1 + lpsc_peak2
    first_fit = lpsc_model.fit(y_data, init_pars, x=x_vals_ppm)
    first_fit.plot_fit(ax=ax, numpoints=100, fitfmt="r-")
    lpsc_fits = [first_fit]
    lpsc_pars = lpsc_model.make_params(**first_fit.best_values)

    lpsc_pars["p1_center"].set(
        max=first_fit.best_values["p1_center"] + 0.2,
        min=first_fit.best_values["p1_center"] - 0.2,
    )
    lpsc_pars["p2_center"].set(
        max=first_fit.best_values["p2_center"] + 0.2,
        min=first_fit.best_values["p2_center"] - 0.2,
    )

    d15_vals = []
    for exp_num in exp_nums:
        exp_path = os.path.join(dir_path, str(exp_num))
        metadata = bruker.read_acqus_file(exp_path)
        d15_vals.append(metadata["acqus"]["D"][15])
    d15_vals = np.array(d15_vals)

    bundle = get_data_from_folder(dir_path, exp_nums[1:])

    x_vals_ppm = bundle["x_vals_ppm"]
    spectra = bundle["y_data"]

    for y_data in spectra:
        new_fit = lpsc_model.fit(y_data, lpsc_pars, x=x_vals_ppm)
        new_fit.plot_fit(ax=ax, numpoints=100, fitfmt="r-")
        lpsc_fits.append(new_fit)

    p1_ints = [
        fit.best_values["p1_amplitude"] / first_fit.best_values["p1_amplitude"]
        for fit in lpsc_fits
    ]
    p2_ints = [
        fit.best_values["p2_amplitude"] / first_fit.best_values["p2_amplitude"]
        for fit in lpsc_fits
    ]

    return d15_vals, [p1_ints, p2_ints]


def fit_1d_exsys(
    mixing_times, intensities, *, fixed_t1=None, plot=True, save_path=None
):
    """
    Fit 1D EXSY exchange data to a simple kinetic decay model.

    Parameters
    ----------
    mixing_times : array-like
        Mixing times in seconds.
    intensities : array-like
        Normalized peak intensities corresponding to mixing times.
    fixed_t1 : float, optional
        If provided, set as the T1 parameter and fix the value during fitting.
    plot : bool, default: True
        If True, generate and return a plot of the fit.
    save_path : str, optional
        File path to save the fit plot, if plotting is enabled.

    Returns
    -------
    lmfit.model.ModelResult or tuple
        If plot is False, returns the fit result. If plot is True, returns
        (fit_result, fig, ax), where fit_result is of type `lmfit.model.ModelResult`.
    """

    default_k = 8
    default_t1 = 0.2

    def model_1d_exsy(x, k, t1):
        return np.multiply(
            np.exp(np.multiply(-1 / t1, x)),
            np.divide(1 + np.exp(np.multiply(2 * k, x)), 2),
        )

    exsy_model = Model(model_1d_exsy)
    params = exsy_model.make_params(k=default_k, t1=default_t1)
    if fixed_t1:
        params["t1"].set(value=fixed_t1, vary=False)
    fit_result = exsy_model.fit(intensities, params, x=mixing_times)
    print(fit_result.fit_report())

    # def model_1d_exsy_fixedt1(t1_fixed):
    #     def wrapped(x, k, t1=t1_fixed):
    #         return model_1d_exsy(x, k, t1)
    #     return wrapped

    # if fixed_t1 is None:
    #     model = model_1d_exsy
    #     popt, pconv = curve_fit(
    #         model_1d_exsy, mixing_times, intensities, p0=[default_k, default_t1]
    #     )
    # else:
    #     model = model_1d_exsy_fixedt1(fixed_t1)
    #     popt, pconv = curve_fit(
    #         model_1d_exsy, mixing_times, intensities, p0=[default_k]
    #     )

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(mixing_times, intensities)

        xfit = np.linspace(min(mixing_times), max(mixing_times), 100)
        ax.plot(xfit, fit_result.eval(x=xfit), "r-")
        # fit_result.plot_fit(ax=ax, numpoints=100, fitfmt='r-')

        ax.set_xlabel("Mixing Time (s)", fontname="Arial", fontsize=16)
        ax.set_ylabel("Normalized Peak Intensity", fontname="Arial", fontsize=16)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)

        return fit_result, fig, ax

    return fit_result
