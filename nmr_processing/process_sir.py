import os
import numpy as np
from nmr_processing.leonmr import xf2, xf2_peak_pick


def load_vdlist(path):
    str_arr = np.loadtxt(path, dtype=str)
    delays = []
    for val in str_arr:
        if val.isnumeric():
            delays.append(float(val))
        else:
            num = float(val.rstrip('pnum'))
            c = val[-1]
            if c == 'p':
                num *= 1e-12
            elif c == 'n':
                num *= 1e-9
            elif c == 'u':
                num *= 1e-6
            elif c == 'm':
                num *= 1e-3
            else:
                raise ValueError(f'Unexpected value in vdlist: {val}')
            delays.append(num)
    return np.array(delays)


def process_sir(exp_path, procno=1, peak_pos=[], f2l=2, f2r=-2):
    """
    Process a Selective Inversion Recovery experiment.
    Pass in the path to the experiment and this returns the data.
    """

    # Load vdlist and interpret suffixes
    vdlist = os.path.join(exp_path, "vdlist")
    delays = load_vdlist(vdlist)

    xAxppm, real_spectrum, params = xf2(exp_path, f2l=f2l, f2r=f2r)
    if peak_pos == []:
        intensities, positions = xf2_peak_pick(xAxppm, real_spectrum,
                                               prominence=[0.9, 1])
    else:
        ints = []
        for peak in peak_pos:
            ints.append(xf2_peak_pick(xAxppm, real_spectrum, peak_pos=peak))
        intensities = np.concatenate(ints, axis=1)
        positions = peak_pos

    return delays, intensities, positions


def make_cifit_files(filename, delays, intensities, title=None, names=[]):
    data_lines = [title] if title else ['TEST']

    numpoints = len(delays)
    intensities = np.array(intensities)
    assert intensities.shape[0] == numpoints

    data_lines.extend(['', str(numpoints), ''])

    if list(names):
        comment = '# Tmix, ' + ', '.join(map(str, names))
    else:
        comment = '# Tmix, unknown intensities...'
    data_lines.append(comment)

    delays = delays.reshape(numpoints, 1)
    data = np.concatenate((delays, intensities), axis=1)

    def format_array(arr):
        return np.array2string(arr, precision=5, suppress_small=True)

    for line in data:
        data_lines.append('\t'.join(map(format_array, line)))
    # data_str = np.array_str(data, precision=5, suppress_small=True)
    # data_lines.extend(data_str.strip('[] ').split(']\n ['))

    with open(filename+'.dat', 'w') as file:
        file.writelines(s + '\n' for s in data_lines)


def exp_to_cifit(exp_path, outfile, procno=1, peak_pos=[]):
    delays, ints, positions = process_sir(exp_path, procno=procno,
                                          peak_pos=peak_pos)

    exp_name = os.path.dirname(exp_path)
    exp_no = int(os.path.basename(exp_path))
    title = f"Extracted from {exp_name} exp no {exp_no}"
    names = [f'{s:.2f} ppm' for s in positions]

    make_cifit_files(outfile, delays, ints, title=title, names=names)


# exp_path = '/Users/tylerpennebaker/BoxSync/wp6_exsy/EXSYstudy/500.TP-2024.10.31_7Li_LZC+LPSC/219'
# exp_to_cifit(exp_path, 'test', peak_pos=[1.46, -0.92])
