"""
Functions for reading data from raw Bruker NMR files.

All functions return a dictionary containing data with appropriate keys.
"""

import os
from xml.etree import ElementTree

import numpy as np
from nmrglue.fileio import bruker
from scipy import signal


def get_1d_data(exp_path, *, proc_num=1, include_md=False):
    """
    Read 1D NMR data from raw Bruker experiment files.

    Parameters
    ----------
    exp_path : str
        Path to the Bruker experiment directory.
    proc_num : int, default: 1
        Processing number containing the 1D dataset.
    include_md : bool, default: False
        If True, include raw Bruker metadata in the returned bundle.

    Returns
    -------
    dict
        Bundle containing x/y axis values, nucleus label, and intensity data.
    """

    pdata_path = os.path.join(exp_path, "pdata", str(proc_num))
    metadata, data = bruker.read_pdata(pdata_path)

    # Determine x axis values
    offset_freq = metadata["acqus"]["O1"]
    basic_field = metadata["acqus"]["BF1"]
    nucleus = metadata["acqus"]["NUC1"]

    pdata_size = metadata["procs"]["SI"]
    spectral_ref_freq = metadata["procs"]["SF"]
    spectral_width_hz = metadata["procs"]["SW_p"]

    spectral_ref = (spectral_ref_freq - basic_field) * 1e6
    true_center = offset_freq - spectral_ref
    x_min = true_center - spectral_width_hz / 2
    x_max = true_center + spectral_width_hz / 2
    x_vals_hz = np.linspace(x_max, x_min, num=int(pdata_size))
    x_vals_ppm = x_vals_hz / spectral_ref_freq

    bundle = {
        "data_type": "1d",
        "nucleus": nucleus,
        "x_vals_hz": x_vals_hz,
        "x_vals_ppm": x_vals_ppm,
        "y_data": data,
    }

    if include_md:
        bundle.update(metadata)

    return bundle


def get_data_from_folder(dir_path, exp_nums=None, *, normalize=False, mass=1):
    """
    Load a set of 1D Bruker experiments from a directory.

    Parameters
    ----------
    dir_path : str
        Directory containing numbered experiment subfolders.
    exp_nums : list of int, optional
        List of experiment numbers to load. If None, all numeric subfolders are loaded.
    normalize : bool, default: False
        If True, normalize all spectra by their maximum intensity.
    mass : float or sequence, optional
        Mass normalization factor for each experiment. Not used if `normalize` is True.

    Returns
    -------
    dict
        Bundle containing experiment data dictionaries indexed by experiment name.
    """

    # If exp_nums is not provided, use all experiment-numbered folders in dir_path
    if exp_nums is None:
        exp_nums = [
            item
            for item in os.listdir(dir_path)
            if os.path.isdir(os.path.join(dir_path, item)) and item.isnumeric()
        ]

    x_vals_list = []
    bundle = {}
    for exp_num in exp_nums:
        exp_path = os.path.join(dir_path, str(exp_num))
        exp_bundle = get_1d_data(exp_path)

        if bundle["data_type"] == "1d":
            raise ValueError(f"Experiment {exp_num} is not a 1d experiment!")

        if normalize:
            # y_data -= min(y_data)
            exp_bundle["y_data"] /= max(exp_bundle["y_data"])
        else:
            exp_bundle["y_data"] = exp_bundle["y_data"] / mass

        x_vals_list.append(exp_bundle["x_vals_ppm"])

        bundle[f"exp_{exp_num}"] = exp_bundle

    # If all exps have the same set of x vals, add a key to the bundle with that set
    first_x_vals = x_vals_list[0]
    if all((x_vals_ppm == first_x_vals).all() for x_vals_ppm in x_vals_list):
        bundle["x_vals_ppm"] = first_x_vals

    return bundle


def get_2d_data(exp_path, *, proc_num=1):
    """
    Read 2D NMR data from raw Bruker files.

    Parameters
    ----------
    exp_path : str
        Path to the experiment directory.
    proc_num : int, default: 1
        Processing number containing the 2D dataset.

    Returns
    -------
    dict
        Bundle containing axis values and the 2D intensity array.
    """

    pdata_path = os.path.join(exp_path, "pdata", str(proc_num))
    metadata, data = bruker.read_pdata(pdata_path)

    bundle = {"data_type": "2d", "z_data": data}

    # Determine values for x and y axes
    for dim in ["x", "y"]:
        if dim == "x":
            acqus = "acqus"
            procs = "procs"
        else:
            acqus = "acqu2s"
            procs = "proc2s"

        offset_freq = metadata[acqus]["O1"]
        basic_field = metadata[acqus]["BF1"]
        nucleus = metadata[acqus]["NUC1"]

        pdata_size = metadata[procs]["SI"]
        spectral_ref_freq = metadata[procs]["SF"]
        spectral_width_hz = metadata[procs]["SW_p"]

        spectral_ref = (spectral_ref_freq - basic_field) * 1e6
        center = offset_freq - spectral_ref
        min_hz = center - spectral_width_hz / 2
        max_hz = center + spectral_width_hz / 2
        vals_hz = np.linspace(max_hz, min_hz, num=int(pdata_size))

        bundle[f"{dim}_nucleus"] = nucleus
        bundle[f"{dim}_vals_hz"] = vals_hz
        bundle[f"{dim}_vals_ppm"] = vals_hz / spectral_ref_freq

    # TODO: Add ability to crop data 2D data using f1l,f1r or xmin, xmax

    return bundle


def get_diff_params(exp_path):
    """
    Extract diffusion experiment parameters from a diff.xml file.

    Parameters
    ----------
    exp_path : str
        Path to the experiment directory containing the diff.xml.

    Returns
    -------
    dict
        Dictionary containing:
        - little_delta: gradient pulse length in seconds.
        - big_delta: diffusion time in seconds.
        - diff_coeff_estimate: estimated diffusion coefficient in m^2/s.
        - gradient_list: list of gradient field strengths in G/cm.
    """

    xml_path = os.path.join(exp_path, "diff.xml")
    tree = ElementTree.parse(xml_path)
    root = tree.getroot()

    little_delta = float(root.find(".//delta").text)  # [ms]
    little_delta = little_delta / 1000  # [s]

    big_delta = float(root.find(".//DELTA").text)  # [ms]
    big_delta = big_delta / 1000  # [s]

    diff_coeff_estimate = float(root.find(".//exDiffCoff").text)  # [m2/s]

    x_values_element = root.find(".//xValues/List")
    if x_values_element:
        x_values_list = x_values_element.text.split()
        x_values = [float(value) for value in x_values_list]
        gradient_list = x_values[1::4]  # [G/cm]
    else:
        x_values_list = root.findall(".//X")
        gradient_list = [float(i.attrib["g"]) for i in x_values_list]

    # Convert from [G/cm] to [T/m]?
    gradient_list = [x / 100 for x in gradient_list]  # [T/m]

    bundle = {
        "little_delta": little_delta,
        "big_delta": big_delta,
        "diff_coeff_estimate": diff_coeff_estimate,
        "gradient_list": gradient_list,
    }

    return bundle


def get_pseudo2d_data(exp_path, *, proc_num=1):
    """
    Read pseudo-2D NMR data from raw Bruker files.

    Parameters
    ----------
    exp_path : str
        Path to the experiment directory.
    proc_num : int, default: 1
        Processing number containing the pseudo-2D dataset.

    Returns
    -------
    dict
        Bundle containing x axis values, intensity data, and metadata.
    """

    pdata_path = os.path.join(exp_path, "pdata", str(proc_num))
    metadata, data = bruker.read_pdata(pdata_path)

    offset_freq = metadata["acqus"]["O1"]
    basic_field = metadata["acqus"]["BF1"]
    nucleus = metadata["acqus"]["NUC1"]

    pdata_size = metadata["procs"]["SI"]
    spectral_ref_freq = metadata["procs"]["SF"]
    spectral_width_hz = metadata["procs"]["SW_p"]

    spectral_ref = (spectral_ref_freq - basic_field) * 1e6
    center = offset_freq - spectral_ref
    x_min_hz = center - spectral_width_hz / 2
    x_max_hz = center + spectral_width_hz / 2
    x_vals_hz = np.linspace(x_max_hz, x_min_hz, num=int(pdata_size))
    x_vals_ppm = x_vals_hz / spectral_ref_freq

    bundle = {
        "data_type": "pseudo2d",
        "nucleus": nucleus,
        "x_vals_hz": x_vals_hz,
        "x_vals_ppm": x_vals_ppm,
        "y_data": data,
        "acqus": metadata["acqus"],
        "acqu2s": metadata["acqu2s"],
    }

    return bundle


def pick_peaks_pseudo2d(
    bundle=None,
    *,
    x_vals_ppm=None,
    y_data=None,
    peak_pos=None,
    regions=None,
    prominence=None,
    normalize=True,
):
    """
    Extract peak intensities from each slice of a pseudo-2D dataset. Peak
    intensities are normalized by the slice with the largest total intensity.

    This function has three methods for determining peak intensities, listed in order of
    decreasing priority:
    1. Integrated intensities over the provided `regions`.
    2. Intensities extracted at the provided `peak_pos` values.
    3. Intensities extracted at the automatically-found positions of peaks.

    Parameters
    ----------
    bundle : dict, optional
        Data bundle containing keys 'x_vals_ppm' and 'y_data' for the pseudo-2D dataset.
        If provided, these values will be used instead of the corresponding parameters.
    x_vals_ppm : array-like
        X-axis values in ppm.
    y_data : array-like
        2D intensity data where each row is a slice/1D spectrum.
    peak_pos : array-like, optional
        Position(s) in ppm of peaks to extract. If None, peaks are automatically
        detected automatically using `scipy.signal.find_peaks`.
    regions : list of tuple, optional
        List of ppm ranges to integrate across to determine peak intensities.
    prominence : number or ndarray or sequence, default: [0.5, 1]
        Prominence range passed to `scipy.signal.find_peaks` when peaks are
        auto-detected. Not used if `peak_pos` is provided.
    normalize : bool, default: True
        If True, normalize each spectrum by the maximum intensity of all slices.
        Otherwise, return raw intensities.

    Returns
    -------
    dict
        Bundle containing peak indices, ppm positions, raw intensities, and normalized
        intensities.

    TODO: allow no xdata input to pick_peaks_pseudo2d
    """

    # Unpack or create bundle depending on args
    if bundle:
        x_vals_ppm = bundle["x_vals_ppm"]
        y_data = bundle["y_data"]
    else:
        bundle = {"x_vals_ppm": np.array(x_vals_ppm), "y_data": np.array(y_data)}

    # If regions are provided, integrate across those regions to get peak intensities.
    # Otherwise, find intensities at peak positions.
    if regions:
        bundle["peak_pick_method"] = "integrate_regions"
        ints = []
        for x_min, x_max in regions:
            if x_min > x_max:
                x_min, x_max = x_max, x_min
            idx_filter = (x_vals_ppm >= x_min) & (x_vals_ppm <= x_max)
            ints.append(np.trapz(x_vals_ppm[idx_filter], y_data[idx_filter]))

        bundle.update(
            {
                "peak_pick_method": "integrate_regions",
                "integration_regions": regions,
                "peak_ints": np.concatenate(ints, axis=1),
            }
        )
    else:
        if peak_pos:
            bundle["peak_pick_method"] = "peak_positions"
            # Find the indices of the closest x_vals_ppm to each peak_pos
            peak_idx = [np.abs(x_vals_ppm - ppm).argmin() for ppm in peak_pos]
        else:
            bundle["peak_pick_method"] = "find_peaks"
            # Find peaks using find_peaks if neither peak_pos nor regions provided
            if prominence is None:
                prominence = [0.5, 1]
            # Choose the slice with the highest total intensity for peak picking
            best_slice = max(y_data, key=np.sum)
            best_slice = best_slice - min(best_slice)
            best_slice = best_slice / max(best_slice)

            peak_idx = signal.find_peaks(best_slice, prominence=prominence)[0]
        # ppm values of picked peaks, might be slightly different from input peak_pos
        peak_positions = x_vals_ppm[peak_idx]
        # Get intensities of picked peaks in all slices
        peak_ints = np.array([[y_slice[i] for i in peak_idx] for y_slice in y_data])

        bundle.update(
            {
                "peak_idx": peak_idx,
                "peak_pos_ppm": peak_positions,
                "peak_ints": peak_ints,
            }
        )

    if normalize:
        # Normalize by max intensity of each peak, and add to bundle
        bundle["peak_ints_norm"] = peak_ints / peak_ints.max(axis=0)

    return bundle
