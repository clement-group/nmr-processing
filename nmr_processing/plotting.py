"""
Functions for plotting data from raw Bruker NMR files.

All functions return a dictionary containing data and/or figures with appropriate keys.

TODO: add a plot_t1_relaxation function
TODO: Change f1p/f2p to xmax and update the way it's checked
TODO: Allow kwarg input via bundle
TODO: Add overlay function for data not in same folder
TODO: Document bundles
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nmr_processing.processing import (
    get_1d_data,
    get_2d_data,
    get_data_from_folder,
    get_diff_params,
    get_pseudo2d_data,
    pick_peaks_pseudo2d,
    read_t1ints,
)
from nmr_processing.utils import find_gamma, nucleus_label

# Plot parameters
FONT_SIZE = 28
params = {
    "legend.fontsize": "large",
    # 'figure.figsize': (15,18),
    "font.family": "Arial",
    "font.weight": "normal",
    "figure.titlesize": FONT_SIZE + 2,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE * 0.8,
    "xtick.major.width": 2,
    "xtick.major.size": 10,
    "xtick.minor.visible": True,
    "xtick.minor.width": 1.5,
    "xtick.minor.size": 5,
    "ytick.labelsize": FONT_SIZE * 0.8,
    "ytick.major.width": 2,
    "ytick.major.size": 10,
    "ytick.minor.visible": True,
    "ytick.minor.width": 1.5,
    "ytick.minor.size": 5,
    "axes.labelweight": "normal",
    "axes.linewidth": 2,
    "axes.titlepad": 25,
}
plt.rcParams.update(params)


def plot_1d(
    arg, *, proc_num=1, f1p=0, f2p=0, fig_width=8, fig_height=8, save_path=None
):
    """
    Plot a 1D NMR spectrum.

    Parameters
    ----------
    arg : dict or str
        Either a data bundle or the path to a Bruker experiment folder.
    proc_num : int, default: 1
        Processing number containing the 1D dataset, used only if a path is provided.
    f1p : float, optional
        Left x-axis limit in ppm.
    f2p : float, optional
        Right x-axis limit in ppm.
    fig_width : float, default: 8
        Figure width in inches.
    fig_height : float, default: 8
        Figure height in inches.
    save_path : str, optional
        Path to save the plot figure. If not specified, the figure is not saved.

    Returns
    -------
    dict
        Data bundle updated with `fig` and `ax` objects.

    CHECK: Are f1p/f2p the correct variable names? F1!=F2...
    """

    if isinstance(arg, dict):
        bundle = arg
    elif isinstance(arg, str):
        exp_path = arg
        bundle = get_1d_data(exp_path, proc_num=proc_num)
    else:
        raise TypeError(
            "The first argument of this function must be a string of the path to the "
            "experiment folder containing the 1D NMR data or a dictionary containing "
            "the data to be plot."
        )

    y_data = bundle["y_data"]
    x_vals_ppm = bundle["x_vals_ppm"]
    x_label_text = nucleus_label(bundle["nucleus"])

    if f1p or f2p:
        x_low = np.abs(x_vals_ppm - f1p).argmin()
        x_high = np.abs(x_vals_ppm - f2p).argmin()
        if x_low > x_high:
            x_low, x_high = x_high, x_low
        x_vals_ppm = x_vals_ppm[x_low:x_high]
        y_data = y_data[x_low:x_high]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(x_vals_ppm, y_data, "k", linewidth=3)

    if f1p + f2p != 0:
        if f2p > f1p:
            ax.set_xlim(f1p, f2p)
        else:
            ax.set_xlim(f2p, f1p)

    # ax.spines[['top','right','left']].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.invert_xaxis()
    ax.set_xlabel(x_label_text)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    bundle["fig"] = fig
    bundle["ax"] = ax

    return bundle


def plot_folder(
    arg,
    *,
    exp_nums=None,
    mass=1,
    normalize=False,
    f1p=0,
    f2p=0,
    fig_width=8,
    fig_height=8,
    stacking_factor=0,
    save_path=None,
):
    """
    Plot stacked 1D NMR spectra.

    Parameters
    ----------
    arg : dict or str
        Either a bundle of multiple experiments' data or the path to a directory
        containing experiment folders.
    exp_nums : list of int, optional
        Experiment numbers to plot (e.g., [1, 5, 6, 10]). If not provided, all numeric
        subfolders are loaded.
    normalize : bool, default: False
        If True, normalize all spectra by their maximum intensity.
    mass : float or sequence, optional
        Mass normalization factor for each experiment. Not used if `normalize` is True.
        Does not normalize by default.
    f1p : float, optional
        Left x-axis limit in ppm.
    f2p : float, optional
        Right x-axis limit in ppm.
    fig_width : float, default: 8
        Figure width in inches.
    fig_height : float, default: 8
        Figure height in inches.
    stacking_factor : float, default: 0
        Vertical offset factor between stacked spectra. A value of 0 will overlay
        spectra (default behavior). A value of 1 will line up spectra baselines with the
        previous spectrum's maximum.
    save_path : str, optional
        Path to save the plot figure. If not specified, the figure is not saved.

    Returns
    -------
    dict
        Data bundle updated with `fig` and `ax` objects.
    """

    # Get bundle of experiment bundles
    if isinstance(arg, dict):
        bundle = arg
    elif isinstance(arg, str):
        dir_path = arg
        bundle = get_data_from_folder(
            dir_path=dir_path, exp_nums=exp_nums, normalize=normalize, mass=mass
        )
    else:
        raise TypeError(
            "The first argument of this function must be a string of the path to a "
            "folder containing experiment folders or a dictionary containing the data "
            "bundles to be plot."
        )

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    y_offset = 0

    for exp_num, exp_bundle in bundle.items():
        x_data = exp_bundle["x_vals_ppm"]
        y_data = exp_bundle["y_data"]

        ax.plot(x_data, y_data + y_offset, label=f"exp {exp_num}")
        y_offset = y_offset + max(y_data) * stacking_factor

    if f1p + f2p != 0:
        if f2p > f1p:
            ax.set_xlim(f1p, f2p)
        else:
            ax.set_xlim(f2p, f1p)

    # ax.spines[['top','right','left']].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.invert_xaxis()

    # If all nuclei are the same, use specific label, otherwise just use 'shift / ppm'
    if len({exp_bundle["nucleus"] for exp_bundle in bundle}) == 1:
        ax.set_xlabel(nucleus_label(bundle))
    else:
        ax.set_xlabel("shift / ppm")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    bundle["fig"] = fig
    bundle["ax"] = ax

    return bundle


def add_projections(ax, *, x, y, z):
    """
    Add top and left projection axes to a 2D contour plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Main contour plot axis.
    x : array-like
        F2 axis values for the top projection.
    y : array-like
        F1 axis values for the left projection.
    z : array-like
        2D intensity matrix used to compute projection traces.
    """

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.tick_params(pad=6)

    divider = make_axes_locatable(ax)
    ax_f2 = divider.append_axes("top", 2, pad=0.1, sharex=ax)
    ax_f1 = divider.append_axes("left", 2, pad=0.1, sharey=ax)
    ax_f2.plot(x, z.sum(axis=0), "k")
    ax_f1.plot(-z.sum(axis=1), y, "k")

    # make projection axes invisible
    ax_f2.tick_params(axis="both", which="both", bottom=False, left=False)
    ax_f1.tick_params(axis="both", which="both", bottom=False, left=False)
    ax_f2.xaxis.set_tick_params(labelbottom=False)
    ax_f1.yaxis.set_tick_params(labelleft=False)
    ax_f2.spines["left"].set_visible(False)
    ax_f2.spines["right"].set_visible(False)
    ax_f2.spines["bottom"].set_visible(False)
    ax_f2.spines["top"].set_visible(False)
    ax_f1.spines["left"].set_visible(False)
    ax_f1.spines["right"].set_visible(False)
    ax_f1.spines["bottom"].set_visible(False)
    ax_f1.spines["top"].set_visible(False)
    ax_f2.yaxis.set_ticklabels([])
    ax_f1.xaxis.set_ticklabels([])


def plot_2d(  # pylint: disable=too-many-statements
    arg,
    *,
    proc_num=1,
    f1l=0,
    f1r=0,
    f2l=0,
    f2r=0,
    factor=0.02,
    clevels=10,
    color=True,
    fig_height=12,
    fig_width=12,
    show_projections=True,
    save_path=None,
):
    """
    Plot countours of 2D NMR data.

    Parameters
    ----------
    arg : dict or str
        Either a data bundle or the path to a Bruker experiment directory.
    proc_num : int, default: 1
        Processing number containing the 2D dataset, used only if a path is provided.
    f1l : float, optional
        Lower limit of the F1 axis in ppm.
    f1r : float, optional
        Upper limit of the F1 axis in ppm.
    f2l : float, optional
        Left limit of the F2 axis in ppm.
    f2r : float, optional
        Right limit of the F2 axis in ppm.
    factor : float, default: 0.02
        Contour threshold as a fraction of the maximum intensity.
    clevels : int, default: 10
        Number of contour levels to show in plot.
    color : bool, default: True
        If True, plot color-filled contours. If False, plot black contour lines.
    fig_width : float, default: 8
        Figure width in inches.
    fig_height : float, default: 8
        Figure height in inches.
    show_projections : bool, default: True
        If True, show projection traces along both axes. Otherwise, only show 2D plot.
    save_path : str, optional
        Path to save the plot figure. If not specified, the figure is not saved.

    Returns
    -------
    dict
        Data bundle updated with `fig` and `ax` objects.

    CHECK: Does this work with pseudo-2D data?
    TODO: Remove duplicate cropping code from plot_2d
    """

    if isinstance(arg, dict):
        bundle = arg
    elif isinstance(arg, str):
        exp_path = arg
        bundle = get_2d_data(exp_path, proc_num=proc_num)
    else:
        raise TypeError(
            "The first argument of this function must be a string of the path to the "
            "experiment folder containing the 2D NMR data or a dictionary containing "
            "the data to be plot."
        )

    x = bundle["x_vals_ppm"]
    y = bundle["y_vals_ppm"]
    z = bundle["z_data"]

    ####################################
    # Index limits of plot
    f2l_temp = max(x)
    f2r_temp = min(x)
    f1l_temp = max(y)
    f1r_temp = min(y)

    if f2l < f2r:
        f2l, f2r = f2r, f2l
    if f1l < f1r:
        f1l, f1r = f1r, f1l

    xlow = np.argmax(x < f2l)
    xhigh = np.argmax(x < f2r)

    ylow = np.argmax(y < f1l)
    yhigh = np.argmax(y < f1r)

    if xlow > xhigh:
        xlow, xhigh = xhigh, xlow

    if ylow > yhigh:
        ylow, yhigh = yhigh, ylow

    if f2l == 0:
        xlow = np.argmax(x == f2l_temp)
    if f2r == 0:
        xhigh = np.argmax(x == f2r_temp)
    if f1l == 0:
        ylow = np.argmax(y == f1l_temp)
    if f1r == 0:
        yhigh = np.argmax(y == f1r_temp)

    x = x[xlow:xhigh]
    y = y[ylow:yhigh]
    z = z[ylow:yhigh, xlow:xhigh]

    threshmin = factor * np.amax(z)
    threshmax = np.amax(z)
    cc2 = np.linspace(1, clevels, clevels)
    thresvec = [
        threshmin * ((threshmax / threshmin) ** (1 / clevels)) ** (1.25 * i)
        for i in cc2
    ]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if color:
        z = np.ma.masked_where(z <= 0, z)
        ax.contour(
            x,
            y,
            z,
            levels=thresvec,
            cmap=colormaps["seismic"],
            norm=LogNorm(),
        )
    else:
        ax.contour(x, y, z, colors="black", levels=thresvec)

    ax.set_xlabel(nucleus_label(bundle["x_nucleus"]))
    ax.set_ylabel(nucleus_label(bundle["y_nucleus"]))

    if f1l + f1r != 0:
        if f1l > f1r:
            ax.set_ylim(f1r, f1l)
        else:
            ax.set_ylim(f1l, f1r)

    if f2l + f2r != 0:
        if f2l > f2r:
            ax.set_xlim(f2r, f2l)
        else:
            ax.set_xlim(f2l, f2r)

    ax.invert_xaxis()
    ax.invert_yaxis()

    if show_projections:
        add_projections(ax, x=x, y=y, z=z)
        fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    bundle["fig"] = fig
    bundle["ax"] = ax

    return bundle


def plot_slice(
    arg,
    *,
    proc_num=1,
    slice_idx=0,
    f2l=1,
    f2r=-1,
    fig_width=8,
    fig_height=8,
    save_path=None,
):
    """
    Plot a single slice from a pseudo-2D experiment.

    Parameters
    ----------
    arg : dict or str
        Either a pseudo-2D data bundle or the path to a Bruker experiment directory.
    proc_num : int, default: 1
        Processing number containing the pseudo-2D dataset, used only if a path is
        provided.
    slice_idx : int, default: 0
        Zero-based index of the slice to plot.
    f2l : float, default: 1
        Left x-axis limit in ppm.
    f2r : float, default: -1
        Right x-axis limit in ppm.
    fig_width : float, default: 8
        Figure width in inches.
    fig_height : float, default: 8
        Figure height in inches.
    save_path : str, optional
        Path to save the plot figure. If not specified, the figure is not saved.

    Returns
    -------
    dict
        Data bundle updated with `fig` and `ax` objects.

    TODO: Make plot_slice work with true 2D data.
    """

    if isinstance(arg, dict):
        bundle = arg
    elif isinstance(arg, str):
        exp_path = arg
        bundle = get_pseudo2d_data(exp_path, proc_num=proc_num)
    else:
        raise TypeError(
            "The first argument of this function must be a string of the path to the "
            "experiment folder containing the pseudo-2D NMR data or a dictionary"
            "containing the data to be plot."
        )

    x_vals = bundle["x_vals_ppm"]
    y_data = bundle["y_data"]
    nucleus = bundle["nucleus"]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(x_vals, y_data[slice_idx])

    ax.invert_xaxis()
    ax.set_xlabel(nucleus_label(nucleus))
    ax.set_ylabel("Intensity / a.u.")
    ax.set_yticks([])

    if f2l or f2r:
        if f2l < f2r:
            ax.xlim(f2r, f2l)
        else:
            ax.xlim(f2l, f2r)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    bundle["fig"] = fig
    bundle["ax"] = ax

    return bundle


def sim_diffusion(
    nuclide,
    *,
    little_delta=1,
    big_delta=20,
    max_gradient=17,
    diff_coeff=None,
    fig_width=8,
    fig_height=8,
    save_path=None,
):
    """
    Simulate and plot the attenuation curve of a diffusion experiment.

    This function is meant to help estimate appropriate parameters for your experiment
    but you should still run test experiments on your spectrometer!

    Parameters
    ----------
    nuclide : str
        Nuclide label in the form '{mass number}{symbol}', for example '1H' or '27Al'.
    little_delta : float, default: 1
        Gradient pulse duration in milliseconds.
    big_delta : float, default: 20
        Diffusion interval in milliseconds.
    max_gradient : float, default: 17
        Maximum gradient strength in G/cm. CHECK: gradient units
    diff_coeff : float or array-like, optional
        Diffusion coefficient(s) in m^2/s. If not specified, a default log-spaced range
        is used.
    fig_width : float, default: 8
        Figure width in inches.
    fig_height : float, default: 8
        Figure height in inches.
    save_path : str, optional
        Path to save the plot figure. If not specified, the figure is not saved.

    Returns
    -------
    dict
        Bundle containing the generated figure, axes, and simulation data.
    """

    # convert ms to s
    little_delta = little_delta / 1000
    big_delta = big_delta / 1000

    if diff_coeff is None:
        diff_coeff = np.logspace(-8, -15, 8)

    multiple = False
    if isinstance(diff_coeff, Sequence):
        multiple = len(diff_coeff) > 1
        if not multiple:
            diff_coeff = diff_coeff[0]

    gamma = find_gamma(nuclide)  # [MHz/T]

    # CHECK: Is this a sensible range?
    gradient_vals = np.arange(0, max_gradient * 1.01, max_gradient / 99.0)
    exponent_coefficient_list = [  # 2pi factor is because gamma is rads/T/s
        (2 * np.pi * gamma * little_delta * gradient) ** 2
        * (big_delta - (little_delta / 3))
        for gradient in gradient_vals
    ]

    # TODO: Refactor sim_diffusion so its not so big and requiring a huge if
    #       Start by making diff_coeff into a list even if len == 1
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if multiple:
        intensity_data = np.zeros(shape=(len(diff_coeff), len(gradient_vals)))
        cnt = 0
        for j in diff_coeff:
            intensity_data[cnt] = np.exp(np.multiply(-j, exponent_coefficient_list))
            cnt += 1

        colmap = colormaps["seismic"](np.linspace(0, 1, len(diff_coeff)))
        for k, c in zip(range(len(diff_coeff)), colmap):
            ax.plot(
                gradient_vals,
                intensity_data[k, :],
                color=c,
                linewidth=2,
                label=str(diff_coeff[k]) + r" $\mathregular{m^2 s^{–1}}$",
            )
    else:
        intensity_data = np.exp(np.multiply(-diff_coeff, exponent_coefficient_list))

        ax.plot(
            gradient_vals,
            intensity_data,
            linewidth=2,
            color="r",
            label=str(diff_coeff) + r" $\mathregular{m^2 s^{–1}}$",
        )
    ax.set_xlim(0, max_gradient * 1.25)
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlabel(r"Gradient Strength, g / $\mathregular{T m^{–1}}$")
    ax.set_ylabel(r"Intensity, $\mathregular{I/I_0}$")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    bundle = {
        "fig": fig,
        "ax": ax,
        "nuclide": nuclide,
        "gamma": gamma,
        "diff_coeff": diff_coeff,
        "little_delta": little_delta,
        "big_delta": big_delta,
        "gradient_vals": gradient_vals,
        "intensity_data": intensity_data,
    }

    return bundle


def plot_t2_relaxation(
    arg,
    *,
    proc_num=1,
    use_t1ints=True,
    regions=None,
    peak_pos=None,
    prominence=None,
    fig_width=8,
    fig_height=8,
    save_path=None,
):
    """
    Plot data of a T2 relaxation experiment.

    This function currently gets intensity data from either `t1ints.txt` or picking
    intensities from processed data, and gets echo delay times from the acqus
    parameters L1, L2, and CNST31. Future versions of this function may allow
    different methods of calculating echo delay times.

    Parameters
    ----------
    arg : dict or str
        Either a data bundle or the path to a Bruker experiment folder.
    proc_num : int, default: 1
        Processing number containing the dataset.
    use_t1ints : bool, default: True
        If True, read data from `t1ints.txt`. Otherwise, extract intensities from
        processed pseudo-2D data in Bruker files, using `pick_peaks_pseudo2d`.
    regions : list of tuple, optional
        List of ppm ranges to integrate across to determine peak intensities.
    peak_pos : array-like, optional
        Position(s) in ppm of peaks to extract. If None, peaks are automatically
        detected automatically using `scipy.signal.find_peaks`.
    prominence : number or ndarray or sequence, default: [0.5, 1]
        Prominence range passed to `scipy.signal.find_peaks` when peaks are
        auto-detected. Not used if `peak_pos` is provided.
    fig_width : float, default: 8
        Figure width in inches.
    fig_height : float, default: 8
        Figure height in inches.
    save_path : str, optional
        Path to save the plot figure. If not specified, the figure is not saved.

    Returns
    -------
    dict
        Data bundle updated with `fig` and `ax` objects.
    """

    if isinstance(arg, dict):
        bundle = arg
    elif isinstance(arg, str):
        exp_path = arg
        bundle = get_pseudo2d_data(exp_path, proc_num=proc_num)
    else:
        raise TypeError(
            "The first argument of this function must be a string of the path to the "
            "experiment folder containing the 1D NMR data or a dictionary containing "
            "the data to be plot."
        )

    if use_t1ints:
        t1ints_bundle = read_t1ints(exp_path, proc_num=proc_num, delay_offset=False)
        intensities = bundle["intensities"] = t1ints_bundle["intensities"]
    else:
        bundle.update(
            pick_peaks_pseudo2d(
                bundle,
                regions=regions,
                prominence=prominence,
                peak_pos=peak_pos,
                normalize=True,
            )
        )
        intensities = bundle["peak_ints_norm"]

    l1 = bundle["acqus"]["L"][1]
    l2 = bundle["acqus"]["L"][2]
    cnst31 = bundle["acqus"]["L"][31]

    echo_delays = np.arange(
        (2 * l1 / cnst31),
        (2 * ((l1) + (l2 * (len(intensities[:, 0])))) / cnst31),
        2 * l2 / cnst31,
    )
    echo_delays *= 1000  # unit = ms

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(echo_delays, intensities)
    ax.set_xlabel("Echo delay / ms")
    ax.set_ylabel("Normalized Intensity")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    bundle.update({"fig": fig, "ax": ax, "echo_delays_ms": echo_delays})

    return bundle


def plot_diffusion(
    exp_path,
    *,
    proc_num=1,
    regions=None,
    peak_pos=None,
    prominence=None,
    fig_width=8,
    fig_height=8,
    save_path=None,
):
    """
    Plot diffusion attenuation data against gradient strength.

    Parameters
    ----------
    exp_path : str
        Path to the experiment directory containing diff.xml metadata.
    proc_num : int, default: 1
        Processing number containing the pseudo-2D dataset.
    regions : list of tuple, optional
        List of ppm ranges to integrate across to determine peak intensities.
    peak_pos : array-like, optional
        Position(s) in ppm of peaks to extract. If None, peaks are automatically
        detected automatically using `scipy.signal.find_peaks`.
    prominence : number or ndarray or sequence, default: [0.5, 1]
        Prominence range passed to `scipy.signal.find_peaks` when peaks are
        auto-detected. Not used if `peak_pos` is provided.
    fig_width : float, default: 8
        Figure width in inches.
    fig_height : float, default: 8
        Figure height in inches.
    save_path : str, optional
        Path to save the plot figure. If not specified, the figure is not saved.

    Returns
    -------
    dict
        Bundle containing the generated figure, axes, and diffusion data.
    """

    bundle = get_pseudo2d_data(exp_path, proc_num=proc_num)
    bundle.update(get_diff_params(exp_path))
    bundle.update(
        pick_peaks_pseudo2d(
            bundle,
            regions=regions,
            prominence=prominence,
            peak_pos=peak_pos,
            normalize=True,
        )
    )

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(
        bundle["gradient_list"], bundle["peak_ints_norm"], "o"
    )  # , c='red', mfc='blue', mec='blue')
    ax.set_xlabel(r"Gradient Strength / G cm$\mathregular{^{-1}}$")
    ax.set_ylabel("Normalized Intensity")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    bundle["fig"] = fig
    bundle["ax"] = ax

    return bundle
