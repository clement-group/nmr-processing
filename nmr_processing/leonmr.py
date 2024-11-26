# NMR Import/Plotting Functions
#
# NMR1D(datapath, procno=1, showplot=True, f1p=0, f2p=0,plwidth=15,plheight=12,normalise=False)
# NUC_label(NUC)
# stackplot(datadir,Expt_no, nuc, f1p=0, f2p=0, plwidth=15,plheight=18, normalise=False)
# NMR2D(datapath, procno=1, mass=1, f1l=0, f1r=0, f2l=0, f2r=0, factor = 0.02, clevels = 6, frame=False, homonuclear=False, plheight =18, plwidth = 18)

import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scipy.optimize
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from lmfit import Model
from lmfit.models import PseudoVoigtModel
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.colors import LogNorm
# from sklearn.metrics import r2_score

# Plot parameters
fsize = 28
params = {'legend.fontsize': 'large',
        # 'figure.figsize': (15,18),
        'font.family': 'Arial',
        'font.weight': 'bold',
        'axes.labelsize': fsize,
        'axes.titlesize': fsize,
        'xtick.labelsize': fsize*0.8,
        'xtick.major.width': 2,
        'xtick.major.size': 10,
        'xtick.minor.visible': True,
        'xtick.minor.width': 1.5,
        'xtick.minor.size': 5,
        'ytick.labelsize': fsize*0.8,
        'ytick.major.width': 2,
        'ytick.major.size': 10,
        'ytick.minor.visible': True,
        'ytick.minor.width': 1.5,
        'ytick.minor.size': 5,
        'axes.labelweight': 'bold',
        'axes.linewidth': 2,
        'axes.titlepad': 25}
plt.rcParams.update(params)

def NMR1D(datapath, procno=1, mass=1, showplot=True, f1p=0, f2p=0, plwidth=15, plheight=12, normalise=False):
    """
    x, y = NMR1D(datapath, procno=1, mass=1, showplot=True, f1p=0, f2p=0, plwidth=15,plheight=12,normalise=False)

    Function to plot stacked 1D NMR data from raw Bruker files.

    datapath: top-level experiment folder containing the 2D NMR data
    procno: process number of data to be plotted (default = 1)
    mass: mass of sample for comparison to others (default = 1)
    normalise: switch to normalise data or not (default = False). British spelling, I'm foreign.
    showplot: switch for showing plot (True) or just load x, y data (False) (default = True)
    f1p/f2p: left and right limits of x-axis, order agnostic
    plheight/plwidth: plot height/width in inches (default = 15x18)
    """

    # datapath = "/Users/leo/Dropbox (City College)/NMR Data/ssNMR Leo/INDQ_5th_Discharge_2020-11-29/320/"
    # procno = 1
    real_spectrum_path = os.path.join(datapath,"pdata",str(procno),"1r")
    procs = os.path.join(datapath,"pdata",str(procno),"procs")
    acqus = os.path.join(datapath,"acqus")

    # Bruker file format information
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Bruker binary files (ser/fid) store data as an array of numbers whose
    # endianness is determined by the parameter BYTORDA (1 = big endian, 0 = little
    # endian), and whose data type is determined by the parameter DTYPA (0 = int32,
    # 2 = float64). Typically the direct dimension is digitally filtered. The exact
    # method of removing this filter is unknown but an approximation is available.
    # Bruker JCAMP-DX files (acqus, etc) are text file which are described by the
    # `JCAMP-DX standard <http://www.jcamp-dx.org/>`_.  Bruker parameters are
    # prefixed with a '$'.

    real_spectrum = np.fromfile(real_spectrum_path, dtype='int32', count=-1)
    # print(np.size(real_spectrum))
    # x = pd.DataFrame(range(np.size(real_spectrum)))

    # Get aqcus
    O1str = '##$O1= '
    OBSstr = '##$BF1= '
    NUCstr = '##$NUC1= <'

    O1 = float("NaN")
    OBS = float("NaN")
    NUC = ""

    with open(acqus,"rb") as input:
        for line in input:
    #         print(line.decode())
            if O1str in line.decode():
                linestr = line.decode()
                O1 = float(linestr[len(O1str):len(linestr)-1])
            if OBSstr in line.decode():
                linestr = line.decode()
                OBS = float(linestr[len(OBSstr):len(linestr)-1])
            if NUCstr in line.decode():
                linestr = line.decode()
                NUC = str(linestr[len(NUCstr):len(linestr)-2])
            if ~np.isnan(O1) and ~np.isnan(OBS) and not len(NUC)==0:
                break

    # Get procs

    SWstr = '##$SW_p= '
    SIstr = '##$SI= '
    SFstr = '##$SF= '
    NCstr = '##$NC_proc= '

    SW = float("NaN")
    SI = float("NaN")
    SF = float("NaN")
    NC_proc = float("NaN")

    with open(procs,"rb") as input:
        for line in input:
            if SWstr in line.decode():
                linestr = line.decode()
                SW = float(linestr[len(SWstr):len(linestr)-1])
            if SIstr in line.decode():
                linestr = line.decode()
                SI = float(linestr[len(SIstr):len(linestr)-1])
            if SFstr in line.decode():
                linestr = line.decode()
                SF = float(linestr[len(SFstr):len(linestr)-1])
            if NCstr in line.decode():
                linestr = line.decode()
                NC_proc = float(linestr[len(NCstr):len(linestr)-1])
            if ~np.isnan(SW) and ~np.isnan(SI) and ~np.isnan(NC_proc) and ~np.isnan(SF):
                break

    # Determine x axis values
    SR = (SF-OBS)*1000000
    true_centre = O1-SR
    xmin = true_centre-SW/2
    xmax = true_centre+SW/2
    xAxHz = np.linspace(xmax,xmin,num=int(SI))
    xAxppm = xAxHz/SF
    real_spectrum = real_spectrum*2**NC_proc
    labeltext = NUC_label(NUC)

    if f1p or f2p:
        xlow = np.abs(xAxppm-f1p).argmin()
        xhigh = np.abs(xAxppm-f2p).argmin()
        if xlow > xhigh:
            xlow, xhigh = xhigh, xlow
        xAxppm = xAxppm[xlow:xhigh]
        real_spectrum = real_spectrum[xlow:xhigh]

    if normalise:
        real_spectrum -= min(real_spectrum)
        real_spectrum /= max(real_spectrum)
    else:
        real_spectrum = real_spectrum/mass
    fig,ax = 0,0
    if showplot:
        fig, ax = plt.subplots()
        # plt.figure(figsize=(10,6))
        plt.plot(xAxppm,real_spectrum,'k',linewidth=3)
        plt.xlabel(labeltext)
        if f1p + f2p != 0:
            if f2p>f1p:
                plt.xlim(f1p,f2p)
            else:
                plt.xlim(f2p,f1p)

        # ax.spines[['top','right','left']].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.yticks([])
        ax.invert_xaxis()
        fig.set_figheight(plheight)
        fig.set_figwidth(plwidth)

    return xAxppm, real_spectrum, fig, ax

############################################################################################################################
############################################################################################################################

def NUC_label(NUC):
    # Determine axis label values
    NUCalpha = NUC.strip('0123456789')
    NUCnum = NUC[:len(NUC)-len(NUCalpha)]
    spinhalflist = ['1H','3H','3He','13C','15N','19F','29Si','31P','57Fe','77Se','89Y',
    '103Rh','107Ag','109Ag','111Cd','113Cd','115Sn','117Sn','119Sn','123Te','125Te',
    '129Xe','169Tm','171Yb','183W','187Os','195Pt','199Hg','203Tl','205Tl','207Pb']
    if NUC in spinhalflist:
        labeltext = '$\mathregular{^{'+NUCnum+'}}$'+NUCalpha+" chemical shift / ppm"
    else:
        labeltext = '$\mathregular{^{'+NUCnum+'}}$'+NUCalpha+" shift / ppm"
    return labeltext

############################################################################################################################
############################################################################################################################

def stackplotfolder(datadir,Expt_no, nuc, f1p=0, f2p=0, plwidth=15,plheight=18, normalise=False):
    """
    stackplot(datadir,Expt_no, nuc, f1p=0, f2p=0, plwidth=15,plheight=18, normalise=False)

    Function to plot stacked 1D NMR data from raw Bruker files.

    datadir: top-level data directory containing all 1D NMR experiment folders.
    Expt_no: list of experiment numbers (e.g., [1, 5, 6, 10])
    nuc: nucleus used, format '##LETTER', e.g., '1H', '13C'
    f1p/f2p: left and right limits of x-axis, order agnostic
    plheight/plwidth: plot height/width in inches (default = 15x18)
    """

    datapath = [os.path.join(datadir,str(num)) for num in Expt_no]
    nuclabel = NUC_label(nuc)

    fig, ax = plt.subplots()

    # cnt=0
    yOffset = 0
    for num in Expt_no:
        datapath = os.path.join(datadir,str(num))
        x,y, figx, axx = NMR1D(datapath,showplot=False,procno=1,normalise=normalise)
        plt.plot(x,y+yOffset,'k',linewidth=2)
        yOffset = yOffset + max(y)*1.25

    if f1p + f2p != 0:
        if f2p>f1p:
            plt.xlim(f1p,f2p)
        else:
            plt.xlim(f2p,f1p)

    plt.xlabel(nuclabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.yticks([])
    ax.invert_xaxis()
    fig.set_figheight(plheight)
    fig.set_figwidth(plwidth)
    # ax.spines[['top','right','left']].set_visible(False)

    return fig, ax


def overlayplotfolder(datadir, Expt_no, nuc, f1p=0, f2p=0, plwidth=15, plheight=18, normalise=False):
    """
    stackplot(datadir,Expt_no, nuc, f1p=0, f2p=0, plwidth=15,plheight=18, normalise=False)

    Function to plot stacked 1D NMR data from raw Bruker files.

    datadir: top-level data directory containing all 1D NMR experiment folders.
    Expt_no: list of experiment numbers (e.g., [1, 5, 6, 10])
    nuc: nucleus used, format '##LETTER', e.g., '1H', '13C'
    f1p/f2p: left and right limits of x-axis, order agnostic
    plheight/plwidth: plot height/width in inches (default = 15x18)
    """

    datapath = [os.path.join(datadir, str(num)) for num in Expt_no]
    nuclabel = NUC_label(nuc)

    fig, ax = plt.subplots()

    real_spectrum = []

    # cnt=0
    yOffset = 0
    for num in Expt_no:
        datapath = os.path.join(datadir,str(num))
        x, y, figx, axx = NMR1D(datapath, showplot=False, procno=1, normalise=normalise)
        plt.plot(x,y+yOffset,linewidth=2)
        yOffset = 0
        real_spectrum.append(y)
    ppm = x
    real_spectrum = np.array(real_spectrum)

    if f2p > f1p:
        plt.xlim(f1p, f2p)
    else:
        plt.xlim(f2p, f1p)

    plt.xlabel(nuclabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.yticks([])
    ax.invert_xaxis()
    fig.set_figheight(plheight)
    fig.set_figwidth(plwidth)
    # ax.spines[['top','right','left']].set_visible(False)

    return ppm, real_spectrum, fig, ax


def readfolder(datadir, Expt_no, normalise=False):
    """
    Function to read in stacked 1D NMR data from raw Bruker files.

    datadir: top-level data directory containing all 1D NMR experiment folders.
    Expt_no: list of experiment numbers (e.g., [1, 5, 6, 10])
    """

    datapath = [os.path.join(datadir, str(num)) for num in Expt_no]
    real_spectrum = []

    for num in Expt_no:
        datapath = os.path.join(datadir, str(num))
        x, y, figx, axx = NMR1D(datapath, showplot=False, procno=1, normalise=normalise)
        real_spectrum.append(y)
    ppm = x
    real_spectrum = np.array(real_spectrum)

    return ppm, real_spectrum


def stackplot(datapath, nuc, f1p=0, f2p=0, plwidth=15,plheight=18, normalise=False):
    """
    stackplot(datadir,Expt_no, nuc, f1p=0, f2p=0, plwidth=15,plheight=18, normalise=False)

    Function to plot stacked 1D NMR data from raw Bruker files.

    datadir: top-level data directory containing all 1D NMR experiment folders.
    Expt_no: list of experiment numbers (e.g., [1, 5, 6, 10])
    nuc: nucleus used, format '##LETTER', e.g., '1H', '13C'
    f1p/f2p: left and right limits of x-axis, order agnostic
    plheight/plwidth: plot height/width in inches (default = 15x18)
    """

    # datapath = [os.path.join(datadir,str(num)) for num in Expt_no]
    nuclabel = NUC_label(nuc)

    fig, ax = plt.subplots()

    # cnt=0
    yOffset = 0
    for expt in datapath:
        x,y, figx, axx = NMR1D(expt,showplot=False,procno=1,normalise=normalise)
        plt.plot(x,y+yOffset,'k',linewidth=2)
        yOffset = yOffset + max(y)*1.25

    if f1p + f2p != 0:
        if f2p>f1p:
            plt.xlim(f1p,f2p)
        else:
            plt.xlim(f2p,f1p)

    plt.xlabel(nuclabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.yticks([])
    ax.invert_xaxis()
    fig.set_figheight(plheight)
    fig.set_figwidth(plwidth)
    # ax.spines[['top','right','left']].set_visible(False)

    return fig, ax

##################################
########## 2D NMR Plot ###########
##################################
def get_2D_data(datapath, procno=1, f1l=0, f1r=0, f2l=0, f2r=0, homonuclear=False):
    """
    xppm, yppm, 2d_intensities = get_2D_data(datapath, procno=1, f1l=0, f1r=0, f2l=0, f2r=0, homonuclear=False)

    Function to get 2D NMR data from raw Bruker files. 1D data are projections along each axis by summation over each dimension.
    May update for optional external projections.

    datapath: top-level experiment folder containing the 2D NMR data
    procno: process number of data to be plotted (default = 1)
    f1l/f1r: left and right limits of F1 (vertical) dimension
    f2l/f2r: left and right limits of F2 (horizontal) dimension
    homonuclear: True/False, based on whether experiment is homo/heteronuclear (default = False)
    """

    real_spectrum_path = os.path.join(datapath,"pdata",str(procno),"2rr")
    procs = os.path.join(datapath,"pdata",str(procno),"procs")
    acqus = os.path.join(datapath,"acqus")
    proc2s = os.path.join(datapath,"pdata",str(procno),"proc2s")
    acqu2s = os.path.join(datapath,"acqu2s")

    ########################################################################

    # Bruker file format information
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Bruker binary files (ser/fid) store data as an array of numbers whose
    # endianness is determined by the parameter BYTORDA (1 = big endian, 0 = little
    # endian), and whose data type is determined by the parameter DTYPA (0 = int32,
    # 2 = float64). Typically the direct dimension is digitally filtered. The exact
    # method of removing this filter is unknown but an approximation is available.
    # Bruker JCAMP-DX files (acqus, etc) are text file which are described by the
    # `JCAMP-DX standard <http://www.jcamp-dx.org/>`_.  Bruker parameters are
    # prefixed with a '$'.

    # real_spectrum = np.fromfile(real_spectrum_path, dtype='int32', count=-1)
    # print(np.shape(real_spectrum))
    ####################################

    # Get aqcus
    O1str = '##$O1= '
    OBSstr = '##$BF1= '
    NUCstr = '##$NUC1= <'
    SWstr = "##$SW_h= "

    O1 = float("NaN")
    OBS = float("NaN")
    NUC = ""
    SW = float("NaN")

    with open(acqus,"rb") as input:
        for line in input:
            if SWstr in line.decode():
                linestr = line.decode()
                SW = float(linestr[len(SWstr):len(linestr)-1])
            if O1str in line.decode():
                linestr = line.decode()
                O1 = float(linestr[len(O1str):len(linestr)-1])
            if OBSstr in line.decode():
                linestr = line.decode()
                OBS = float(linestr[len(OBSstr):len(linestr)-1])
            if NUCstr in line.decode():
                linestr = line.decode()
                NUC = str(linestr[len(NUCstr):len(linestr)-2])
            if ~np.isnan(SW) and ~np.isnan(O1) and ~np.isnan(OBS) and not len(NUC)==0:
                break

    ####################################

    # Get procs

    # SWstr = '##$SW_p= '
    SIstr = '##$SI= '
    SFstr = '##$SF= '
    NCstr = '##$NC_proc= '
    XDIM_F2str = '##$XDIM= '

    SI = float("NaN")
    SF = float("NaN")
    NC_proc = float("NaN")
    XDIM_F2 = int(0)

    with open(procs,"rb") as input:
        for line in input:
            if SIstr in line.decode():
                linestr = line.decode()
                SI = float(linestr[len(SIstr):len(linestr)-1])
            if SFstr in line.decode():
                linestr = line.decode()
                SF = float(linestr[len(SFstr):len(linestr)-1])
            if NCstr in line.decode():
                linestr = line.decode()
                NC_proc = float(linestr[len(NCstr):len(linestr)-1])
            if XDIM_F2str in line.decode():
                linestr = line.decode()
                XDIM_F2 = int(linestr[len(XDIM_F2str):len(linestr)-1])
            if ~np.isnan(SI) and ~np.isnan(NC_proc) and ~np.isnan(SF) and XDIM_F2!=int(0):
                break

    ####################################

    # Get aqcu2s for indirect dimension
    O1str_2 = '##$O1= '
    OBSstr_2 = '##$BF1= '
    NUCstr_2 = '##$NUC1= <'
    SWstr_2 = '##$SW_h= '

    O1_2 = float("NaN")
    OBS_2 = float("NaN")
    NUC_2 = ""
    SW_2 = float("NaN")

    with open(acqu2s,"rb") as input:
        for line in input:
            if SWstr_2 in line.decode():
                linestr = line.decode()
                SW_2 = float(linestr[len(SWstr_2):len(linestr)-1])
            if O1str_2 in line.decode():
                linestr = line.decode()
                O1_2 = float(linestr[len(O1str_2):len(linestr)-1])
            if OBSstr_2 in line.decode():
                linestr = line.decode()
                OBS_2 = float(linestr[len(OBSstr_2):len(linestr)-1])
            if NUCstr_2 in line.decode():
                linestr = line.decode()
                NUC_2 = str(linestr[len(NUCstr_2):len(linestr)-2])
            if ~np.isnan(SW_2) and ~np.isnan(O1_2) and ~np.isnan(OBS_2) and not len(NUC_2)==0:
                break

    ####################################

    # Get proc2s for indirect dimension

    SIstr_2 = '##$SI= '
    SFstr_2 = '##$SF= '
    NCstr_2 = '##$NC_proc= '
    XDIM_F1str = '##$XDIM= '

    XDIM_F1 = int(0)
    SI_2 = float("NaN")
    SF_2 = float("NaN")
    NC_proc_2 = float("NaN")

    with open(proc2s,"rb") as input:
        for line in input:
            if SIstr_2 in line.decode():
                linestr = line.decode()
                SI_2 = float(linestr[len(SIstr_2):len(linestr)-1])
            if SFstr_2 in line.decode():
                linestr = line.decode()
                SF_2 = float(linestr[len(SFstr_2):len(linestr)-1])
            if NCstr_2 in line.decode():
                linestr = line.decode()
                NC_proc_2 = float(linestr[len(NCstr_2):len(linestr)-1])
            if XDIM_F1str in line.decode():
                linestr = line.decode()
                XDIM_F1 = int(linestr[len(XDIM_F1str):len(linestr)-1])
            if ~np.isnan(SI_2) and ~np.isnan(NC_proc_2) and ~np.isnan(SF_2) and XDIM_F1!=int(0):
                break

    # expt_parameters = {'sw2': SW_2,'sf2': SF_2, 'o2':O1_2,
    # 'BF2':OBS_2,
    # 'nuc2': NUC_2,
    # 'sw':SW,'sf1':SF,'o1':O1,'BF1':OBS,'nuc1':NUC
    #                    }

    ####################################

    # Determine x axis values
    SR = (SF-OBS)*1000000
    true_centre = O1-SR
    xmin = true_centre-SW/2
    xmax = true_centre+SW/2
    xAxHz = np.linspace(xmax,xmin,num=int(SI))
    xAxppm = xAxHz/SF

    ####################################
    if homonuclear:
        NUC_2 = NUC
        SR_2 = SR
        O1_2 = O1
    else:
    # Determine y axis values
        SR_2 = (SF_2-OBS_2)*1000000

    true_centre_2 = O1_2-SR_2
    ymin = true_centre_2-SW_2/2
    ymax = true_centre_2+SW_2/2
    yAxHz = np.linspace(ymax,ymin,num=int(SI_2))
    yAxppm = yAxHz/SF_2
    # print(true_centre,true_centre_2)
    ####################################
    # Index limits of plot
    f2l_temp = max(xAxppm)
    f2r_temp = min(xAxppm)
    f1l_temp = max(yAxppm)
    f1r_temp = min(yAxppm)

    if f2l<f2r:
        f2l, f2r = f2r,f2l
    if f1l<f1r:
        f1l, f1r = f1r,f1l

    xlow = np.argmax(xAxppm<f2l)
    xhigh = np.argmax(xAxppm<f2r)
    # print(xlow,xhigh)
    ylow = np.argmax(yAxppm<f1l)
    yhigh = np.argmax(yAxppm<f1r)
    # print(ylow,yhigh)

    if xlow > xhigh:
        xlow, xhigh = xhigh, xlow
    # print(xlow,xhigh)

    if ylow > yhigh:
        ylow, yhigh = yhigh, ylow
    # print(ylow,yhigh)

    if f2l == 0:
        xlow = np.argmax(xAxppm==f2l_temp)
    if f2r == 0:
        xhigh = np.argmax(xAxppm==f2r_temp)
    if f1l == 0:
        ylow = np.argmax(yAxppm==f1l_temp)
    if f1r == 0:
        yhigh = np.argmax(yAxppm==f1r_temp)

    xAxppm = xAxppm[xlow:xhigh]
    yAxppm = yAxppm[ylow:yhigh]


    # Reshape 2d data to match up dimensions
    # real_spectrum = np.fromfile(real_spectrum_path, np.int32, count=-1)
    real_spectrum = np.fromfile(real_spectrum_path, dtype='<i4', count=-1)
    if not bool(real_spectrum.any()):
        print(real_spectrum)
        print("Error: Spectrum not read.")
    real_spectrum_2d = real_spectrum.reshape([int(SI_2),int(SI)])

    if XDIM_F1 == 1:
        real_spectrum_2d = real_spectrum.reshape([int(SI_2),int(SI)])
    else:
        # to shape the column matrix according to Bruker's format, matrices are broken into (XDIM_F1,XDIM_F2) submatrices, so reshaping where XDIM_F1!=1 requires this procedure.
        column_matrix = real_spectrum
        submatrix_rows = int(SI_2 // XDIM_F1)
        submatrix_cols = int(SI // XDIM_F2)
        submatrix_number = submatrix_cols*submatrix_rows

        blocks = np.array(np.array_split(column_matrix,submatrix_number))  # Split into submatrices
        blocks = np.reshape(blocks,(submatrix_rows,submatrix_cols,-1)) # Reshape these submatrices so each has its own 1D array
        real_spectrum_2d = np.vstack([np.hstack([np.reshape(c, (XDIM_F1, XDIM_F2)) for c in b]) for b in blocks])  # Concatenate submatrices in the correct orientation

    real_spectrum_2d = real_spectrum_2d[ylow:yhigh,xlow:xhigh]

    return xAxppm, yAxppm, real_spectrum_2d

def NMR2D(datapath, procno=1, f1l=0, f1r=0, f2l=0, f2r=0, factor=0.02, clevels=6, colour=True, homonuclear=False, plheight=12, plwidth=12):
    """
    fig, ax, 2d_spectrum = NMR2D(datapath, procno=1, f1l=0, f1r=0, f2l=0, f2r=0, factor = 0.02, clevels = 6, homonuclear=False, plheight =12, plwidth = 12)

    Function to plot 2D NMR data from raw Bruker files. 1D data are projections along each axis by summation over each dimension.
    May update for optional external projections.

    datapath: top-level experiment folder containing the 2D NMR data
    procno: process number of data to be plotted (default = 1)
    f1l/f1r: left and right limits of F1 (vertical) dimension
    f2l/f2r: left and right limits of F2 (horizontal) dimension
    factor: minimum value for the contours (factor*max value) (default = 0.02, 2% of max signal)
    clevels: number of contour levels for plot (default = 6)
    colour: log-scaled colour contours or just black lines (default = True, colour on)
    homonuclear: True/False, based on whether experiment is homo/heteronuclear (default = False)
    plheight/plwidth: plot height/width in inches (default = 12x12)
    """

    real_spectrum_path = os.path.join(datapath,"pdata",str(procno),"2rr")
    procs = os.path.join(datapath,"pdata",str(procno),"procs")
    acqus = os.path.join(datapath,"acqus")
    proc2s = os.path.join(datapath,"pdata",str(procno),"proc2s")
    acqu2s = os.path.join(datapath,"acqu2s")

    ########################################################################

    # Bruker file format information
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Bruker binary files (ser/fid) store data as an array of numbers whose
    # endianness is determined by the parameter BYTORDA (1 = big endian, 0 = little
    # endian), and whose data type is determined by the parameter DTYPA (0 = int32,
    # 2 = float64). Typically the direct dimension is digitally filtered. The exact
    # method of removing this filter is unknown but an approximation is available.
    # Bruker JCAMP-DX files (acqus, etc) are text file which are described by the
    # `JCAMP-DX standard <http://www.jcamp-dx.org/>`_.  Bruker parameters are
    # prefixed with a '$'.

    # real_spectrum = np.fromfile(real_spectrum_path, dtype='int32', count=-1)
    # print(np.shape(real_spectrum))
    ####################################

    # Get aqcus
    O1str = '##$O1= '
    OBSstr = '##$BF1= '
    NUCstr = '##$NUC1= <'
    SWstr = "##$SW_h= "

    O1 = float("NaN")
    OBS = float("NaN")
    NUC = ""
    SW = float("NaN")

    with open(acqus,"rb") as input:
        for line in input:
            if SWstr in line.decode():
                linestr = line.decode()
                SW = float(linestr[len(SWstr):len(linestr)-1])
            if O1str in line.decode():
                linestr = line.decode()
                O1 = float(linestr[len(O1str):len(linestr)-1])
            if OBSstr in line.decode():
                linestr = line.decode()
                OBS = float(linestr[len(OBSstr):len(linestr)-1])
            if NUCstr in line.decode():
                linestr = line.decode()
                NUC = str(linestr[len(NUCstr):len(linestr)-2])
            if ~np.isnan(SW) and ~np.isnan(O1) and ~np.isnan(OBS) and not len(NUC)==0:
                break

    ####################################

    # Get procs

    # SWstr = '##$SW_p= '
    SIstr = '##$SI= '
    SFstr = '##$SF= '
    NCstr = '##$NC_proc= '
    XDIM_F2str = '##$XDIM= '

    SI = float("NaN")
    SF = float("NaN")
    NC_proc = float("NaN")
    XDIM_F2 = int(0)

    with open(procs,"rb") as input:
        for line in input:
            if SIstr in line.decode():
                linestr = line.decode()
                SI = float(linestr[len(SIstr):len(linestr)-1])
            if SFstr in line.decode():
                linestr = line.decode()
                SF = float(linestr[len(SFstr):len(linestr)-1])
            if NCstr in line.decode():
                linestr = line.decode()
                NC_proc = float(linestr[len(NCstr):len(linestr)-1])
            if XDIM_F2str in line.decode():
                linestr = line.decode()
                XDIM_F2 = int(linestr[len(XDIM_F2str):len(linestr)-1])
            if ~np.isnan(SI) and ~np.isnan(NC_proc) and ~np.isnan(SF) and XDIM_F2!=int(0):
                break

    ####################################

    # Get aqcu2s for indirect dimension
    O1str_2 = '##$O1= '
    OBSstr_2 = '##$BF1= '
    NUCstr_2 = '##$NUC1= <'
    SWstr_2 = '##$SW_h= '

    O1_2 = float("NaN")
    OBS_2 = float("NaN")
    NUC_2 = ""
    SW_2 = float("NaN")

    with open(acqu2s,"rb") as input:
        for line in input:
            if SWstr_2 in line.decode():
                linestr = line.decode()
                SW_2 = float(linestr[len(SWstr_2):len(linestr)-1])
            if O1str_2 in line.decode():
                linestr = line.decode()
                O1_2 = float(linestr[len(O1str_2):len(linestr)-1])
            if OBSstr_2 in line.decode():
                linestr = line.decode()
                OBS_2 = float(linestr[len(OBSstr_2):len(linestr)-1])
            if NUCstr_2 in line.decode():
                linestr = line.decode()
                NUC_2 = str(linestr[len(NUCstr_2):len(linestr)-2])
            if ~np.isnan(SW_2) and ~np.isnan(O1_2) and ~np.isnan(OBS_2) and not len(NUC_2)==0:
                break

    ####################################

    # Get proc2s for indirect dimension

    SIstr_2 = '##$SI= '
    SFstr_2 = '##$SF= '
    NCstr_2 = '##$NC_proc= '
    XDIM_F1str = '##$XDIM= '

    XDIM_F1 = int(0)
    SI_2 = float("NaN")
    SF_2 = float("NaN")
    NC_proc_2 = float("NaN")

    with open(proc2s,"rb") as input:
        for line in input:
            if SIstr_2 in line.decode():
                linestr = line.decode()
                SI_2 = float(linestr[len(SIstr_2):len(linestr)-1])
            if SFstr_2 in line.decode():
                linestr = line.decode()
                SF_2 = float(linestr[len(SFstr_2):len(linestr)-1])
            if NCstr_2 in line.decode():
                linestr = line.decode()
                NC_proc_2 = float(linestr[len(NCstr_2):len(linestr)-1])
            if XDIM_F1str in line.decode():
                linestr = line.decode()
                XDIM_F1 = int(linestr[len(XDIM_F1str):len(linestr)-1])
            if ~np.isnan(SI_2) and ~np.isnan(NC_proc_2) and ~np.isnan(SF_2) and XDIM_F1!=int(0):
                break

    # expt_parameters = {'sw2': SW_2,'sf2': SF_2, 'o2':O1_2,
    # 'BF2':OBS_2,
    # 'nuc2': NUC_2,
    # 'sw':SW,'sf1':SF,'o1':O1,'BF1':OBS,'nuc1':NUC
    #                    }

    ####################################

    # Determine x axis values
    SR = (SF-OBS)*1000000
    true_centre = O1-SR
    xmin = true_centre-SW/2
    xmax = true_centre+SW/2
    xAxHz = np.linspace(xmax,xmin,num=int(SI))
    xAxppm = xAxHz/SF

    ####################################
    if homonuclear:
        NUC_2 = NUC
        SR_2 = SR
        O1_2 = O1
    else:
    # Determine y axis values
        SR_2 = (SF_2-OBS_2)*1000000

    true_centre_2 = O1_2-SR_2
    ymin = true_centre_2-SW_2/2
    ymax = true_centre_2+SW_2/2
    yAxHz = np.linspace(ymax,ymin,num=int(SI_2))
    yAxppm = yAxHz/SF_2
    # print(true_centre,true_centre_2)
    ####################################
    # Index limits of plot
    f2l_temp = max(xAxppm)
    f2r_temp = min(xAxppm)
    f1l_temp = max(yAxppm)
    f1r_temp = min(yAxppm)

    if f2l<f2r:
        f2l, f2r = f2r,f2l
    if f1l<f1r:
        f1l, f1r = f1r,f1l

    xlow = np.argmax(xAxppm<f2l)
    xhigh = np.argmax(xAxppm<f2r)
    # print(xlow,xhigh)
    ylow = np.argmax(yAxppm<f1l)
    yhigh = np.argmax(yAxppm<f1r)
    # print(ylow,yhigh)

    if xlow > xhigh:
        xlow, xhigh = xhigh, xlow
    # print(xlow,xhigh)

    if ylow > yhigh:
        ylow, yhigh = yhigh, ylow
    # print(ylow,yhigh)

    if f2l == 0:
        xlow = np.argmax(xAxppm==f2l_temp)
    if f2r == 0:
        xhigh = np.argmax(xAxppm==f2r_temp)
    if f1l == 0:
        ylow = np.argmax(yAxppm==f1l_temp)
    if f1r == 0:
        yhigh = np.argmax(yAxppm==f1r_temp)

    xAxppm = xAxppm[xlow:xhigh]
    yAxppm = yAxppm[ylow:yhigh]


    # Reshape 2d data to match up dimensions
    # real_spectrum = np.fromfile(real_spectrum_path, np.int32, count=-1)
    real_spectrum = np.fromfile(real_spectrum_path, dtype='<i4', count=-1)
    if not bool(real_spectrum.any()):
        print(real_spectrum)
        print("Error: Spectrum not read.")
    real_spectrum_2d = real_spectrum.reshape([int(SI_2),int(SI)])

    if XDIM_F1 == 1:
        real_spectrum_2d = real_spectrum.reshape([int(SI_2),int(SI)])
    else:
        # to shape the column matrix according to Bruker's format, matrices are broken into (XDIM_F1,XDIM_F2) submatrices, so reshaping where XDIM_F1!=1 requires this procedure.
        column_matrix = real_spectrum
        submatrix_rows = int(SI_2 // XDIM_F1)
        submatrix_cols = int(SI // XDIM_F2)
        submatrix_number = submatrix_cols*submatrix_rows

        blocks = np.array(np.array_split(column_matrix,submatrix_number))  # Split into submatrices
        blocks = np.reshape(blocks,(submatrix_rows,submatrix_cols,-1)) # Reshape these submatrices so each has its own 1D array
        real_spectrum_2d = np.vstack([np.hstack([np.reshape(c, (XDIM_F1, XDIM_F2)) for c in b]) for b in blocks])  # Concatenate submatrices in the correct orientation

    real_spectrum_2d = real_spectrum_2d[ylow:yhigh,xlow:xhigh]

    Threshmin = factor*np.amax(real_spectrum_2d)
    Threshmax = np.amax(real_spectrum_2d)
    cc2 = np.linspace(1,clevels,clevels)
    thresvec = [Threshmin*((Threshmax/Threshmin)**(1/clevels))**(1.25*i) for i in cc2]

    ####################################

    labeltextx = NUC_label(NUC)
    labeltexty = NUC_label(NUC_2)
    fig, ax = plt.subplots()

    if colour:
        real_spectrum_2d = np.ma.masked_where(real_spectrum_2d <= 0, real_spectrum_2d)
        cs=ax.contour(xAxppm,yAxppm,real_spectrum_2d,levels=thresvec,cmap=cm.seismic, norm=LogNorm())
    else:
        cs=ax.contour(xAxppm,yAxppm,real_spectrum_2d,colors='black',levels=thresvec)
    ax.tick_params(pad=6)
    divider = make_axes_locatable(ax)
    plt.xlabel(labeltextx)
    plt.ylabel(labeltexty)
    if f1l + f1r != 0:
        if f1l>f1r:
            plt.ylim(f1r,f1l)
        else:
            plt.ylim(f1l,f1r)

    if f2l + f2r != 0:
        if f2l>f2r:
            plt.xlim(f2r,f2l)
        else:
            plt.xlim(f2l,f2r)

    # # ax.spines[['top','right','left']].set_visible(frame)
    # # plt.yticks([])

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    # # below height and pad are in inches
    ax_f2 = divider.append_axes("top", 2, pad=0.1, sharex=ax)
    ax_f1 = divider.append_axes("left", 2, pad=0.1, sharey=ax)

    ax_f2.plot(xAxppm,real_spectrum_2d.sum(axis=0),'k')
    f1 = real_spectrum_2d.sum(axis=1)
    ax_f1.plot(-f1,yAxppm,'k')

    # make some labels invisible
    frame=False
    ax_f2.tick_params(axis='both',which="both",bottom=False,left=False)
    ax_f1.tick_params(axis='both',which="both",bottom=False,left=False)
    ax_f2.xaxis.set_tick_params(labelbottom=False)
    ax_f1.yaxis.set_tick_params(labelleft=False)
    ax_f2.spines['left'].set_visible(frame)
    ax_f2.spines['right'].set_visible(frame)
    ax_f2.spines['bottom'].set_visible(frame)
    ax_f2.spines['top'].set_visible(frame)
    ax_f1.spines['left'].set_visible(frame)
    ax_f1.spines['right'].set_visible(frame)
    ax_f1.spines['bottom'].set_visible(frame)
    ax_f1.spines['top'].set_visible(frame)
    ax_f2.yaxis.set_ticklabels([])
    ax_f1.xaxis.set_ticklabels([])
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    fig.set_figheight(plheight)
    fig.set_figwidth(plwidth)


    # ax_f2.xaxis.set_tick_params(labelleft=False)
    # ax_f1.yaxis.set_tick_params(labelbottom=False)
    # ax_f1.spines[['top','right','left','bottom']].set_visible(frame)
    # ax_f1.yaxis.yticks([])

    plt.tight_layout()
    # plt.show()
    ####################################
    return fig, ax, real_spectrum_2d


def find_gamma(isotope):
    gammalist = [
        ['Name', 'Nuclide', 'Spin', 'Magnetic Moment', 'Gyromagnetic Ratio (MHz/T)', 'Quadrupole Moment (fm^2)'],
        ['Hydrogen', '1H', '0.5', '4.83735', '42.57746460132430', '---'],
        ['Deuterium', '2H', '1', '1.21260', '6.53590463949470', '0.28600'],
        ['Helium', '3He', '0.5', '-3.68515', '-32.43603205003720', '---'],
        ['Tritium', '3H', '0.5', '5.15971', '45.41483118028370', '---'],
        ['Lithium', '6Li', '1', '1.16256', '6.26620067293118', '-0.08080'],
        ['Lithium', '7Li', '1.5', '4.20408', '16.54845000000000', '-4.01000'],
        ['Beryllium', '9Be', '1.5', '-1.52014', '-5.98370064894306', '5.28800'],
        ['Boron', '10B', '3', '2.07921', '4.57519531807410', '8.45900'],
        ['Boron', '11B', '1.5', '3.47103', '13.66297000000000', '4.05900'],
        ['Carbon', '13C', '0.5', '1.21661', '10.70839020506340', '---'],
        ['Nitrogen', '14N', '1', '0.57100', '3.07770645852245', '2.04400'],
        ['Nitrogen', '15N', '0.5', '-0.49050', '-4.31726881729937', '---'],
        ['Oxygen', '17O', '2.5', '-2.24077', '-5.77426865932844', '-2.55800'],
        ['Fluorine', '19F', '0.5', '4.55333', '40.07757016369700', '---'],
        ['Neon', '21Ne', '1.5', '-0.85438', '-3.36307127148622', '10.15500'],
        ['Sodium', '23Na', '1.5', '2.86298', '11.26952278792250', '10.40000'],
        ['Magnesium', '25Mg', '2.5', '-1.01220', '-2.60834261585015', '19.94000'],
        ['Aluminum', '27Al', '2.5', '4.30869', '11.10307854843700', '14.66000'],
        ['Silicon', '29Si', '0.5', '-0.96179', '-8.46545000000000', '---'],
        ['Phosphorus', '31P', '0.5', '1.95999', '17.25144000000000', '---'],
        ['Sulfur', '33S', '1.5', '0.83117', '3.27171633415147', '-6.78000'],
        ['Chlorine', '35Cl', '1.5', '1.06103', '4.17654000000000', '-8.16500'],
        ['Chlorine', '37Cl', '1.5', '0.88320', '3.47653283041643', '-6.43500'],
        ['Potassium', '39K', '1.5', '0.50543', '1.98953228161455', '5.85000'],
        ['Potassium', '40K', '4', '-1.45132', '-2.47372936498302', '-7.30000'],
        ['Potassium', '41K', '1.5', '0.27740', '1.09191431807057', '7.11000'],
        ['Calcium', '43Ca', '3.5', '-1.49407', '-2.86967503240704', '-4.08000'],
        ['Scandium', '45Sc', '3.5', '5.39335', '10.35908000000000', '-22.00000'],
        ['Titanium', '47Ti', '2.5', '-0.93294', '-2.40404000000000', '30.20000'],
        ['Titanium', '49Ti', '3.5', '-1.25201', '-2.40475161264699', '24.70000'],
        ['Vanadium', '50V', '6', '3.61376', '4.25047148768370', '21.00000'],
        ['Vanadium', '51V', '3.5', '5.83808', '11.21327743103380', '-5.20000'],
        ['Chromium', '53Cr', '1.5', '-0.61263', '-2.41152000000000', '-15.00000'],
        ['Manganese', '55Mn', '2.5', '4.10424', '10.57624385581420', '33.00000'],
        ['Iron', '57Fe', '0.5', '0.15696', '1.38156039900351', '---'],
        ['Cobalt', '59Co', '3.5', '5.24700', '10.07769000000000', '42.00000'],
        ['Nickel', '61Ni', '1.5', '-0.96827', '-3.81144000000000', '16.20000'],
        ['Copper', '63Cu', '1.5', '2.87549', '11.31876532731510', '-22.00000'],
        ['Copper', '65Cu', '1.5', '3.07465', '12.10269891500850', '-20.40000'],
        ['Zinc', '67Zn', '2.5', '1.03556', '2.66853501532750', '15.00000'],
        ['Gallium', '69Ga', '1.5', '2.60340', '10.24776396876680', '17.10000'],
        ['Gallium', '71Ga', '1.5', '3.30787', '13.02073645775120', '10.70000'],
        ['Germanium', '73Ge', '4.5', '-0.97229', '-1.48973801382307', '-19.60000'],
        ['Arsenic', '75As', '1.5', '1.85835', '7.31501583241246', '31.40000'],
        ['Selenium', '77Se', '0.5', '0.92678', '8.15731153773769', '---'],
        ['Bromine', '79Br', '1.5', '2.71935', '10.70415668357710', '31.30000'],
        ['Bromine', '81Br', '1.5', '2.93128', '11.53838323328760', '26.20000'],
        ['Krypton', '83Kr', '4.5', '-1.07311', '-1.64423000000000', '25.90000'],
        ['Rubidium', '85Rb', '2.5', '1.60131', '4.12642612503788', '27.60000'],
        ['Strontium', '87Sr', '4.5', '-1.20902', '-1.85246804462381', '33.50000'],
        ['Rubidium', '87Rb', '1.5', '3.55258', '13.98399000000000', '13.35000'],
        ['Yttrium', '89Y', '0.5', '-0.23801', '-2.09492468493000', '---'],
        ['Zirconium', '91Zr', '2.5', '-1.54246', '-3.97478329525992', '-17.60000'],
        ['Niobium', '93Nb', '4.5', '6.82170', '10.45234000000000', '-32.00000'],
        ['Molybdenium', '95Mo', '2.5', '-1.08200', '-2.78680000000000', '-2.20000'],
        ['Molybdenium', '97Mo', '2.5', '-1.10500', '-2.84569000000000', '25.50000'],
        ['Ruthenium', '99Ru', '2.5', '-0.75880', '-1.95601000000000', '7.90000'],
        ['Technetium', '99Tc', '4.5', '6.28100', '9.62251000000000', '-12.90000'],
        ['Ruthenium', '101Ru', '2.5', '-0.85050', '-2.19156000000000', '45.70000'],
        ['Rhodium', '103Rh', '0.5', '-0.15310', '-1.34772000000000', '---'],
        ['Palladium', '105Pd', '2.5', '-0.76000', '-1.95761000000000', '66.00000'],
        ['Silver', '107Ag', '0.5', '-0.19690', '-1.73307000631627', '---'],
        ['Silver', '109Ag', '0.5', '-0.22636', '-1.99239707059020', '---'],
        ['Cadmium', '111Cd', '0.5', '-1.03037', '-9.06914203769978', '---'],
        ['Indium', '113In', '4.5', '6.11240', '9.36547000000000', '79.90000'],
        ['Cadmium', '113Cd', '0.5', '-1.07786', '-9.48709883375341', '---'],
        ['Indium', '115In', '4.5', '6.12560', '9.38569000000000', '81.00000'],
        ['Tin', '115Sn', '0.5', '-1.59150', '-14.00770000000000', '---'],
        ['Tin', '117Sn', '0.5', '-1.73385', '-15.26103326770140', '---'],
        ['Tin', '119Sn', '0.5', '-1.81394', '-15.96595000000000', '---'],
        ['Antimony', '121Sb', '2.5', '3.97960', '10.25515000000000', '-36.00000'],
        ['Antimony', '123Sb', '3.5', '2.89120', '5.55323000000000', '-49.00000'],
        ['Tellurium', '123Te', '0.5', '-1.27643', '-11.23491000000000', '---'],
        ['Tellurium', '125Te', '0.5', '-1.53894', '-13.54542255864230', '---'],
        ['Iodine', '127I', '2.5', '3.32871', '8.57776706639786', '-71.00000'],
        ['Xenon', '129Xe', '0.5', '-1.34749', '-11.86039000000000', '---'],
        ['Xenon', '131Xe', '1.5', '0.89319', '3.51586001685444', '-11.40000'],
        ['Cesium', '133Cs', '3.5', '2.92774', '5.62334202679439', '-0.34300'],
        ['Barium', '135Ba', '1.5', '1.08178', '4.25819000000000', '16.00000'],
        ['Barium', '137Ba', '1.5', '1.21013', '4.76342786926888', '24.50000'],
        ['Lanthanum', '138La', '5', '4.06809', '5.66152329764214', '45.00000'],
        ['Lanthanum', '139La', '3.5', '3.15568', '6.06114544425158', '20.00000'],
        ['Praseodymium', '141Pr', '2.5', '5.05870', '13.03590000000000', '-5.89000'],
        ['Neodymium', '143Nd', '3.5', '-1.20800', '-2.31889000000000', '-63.00000'],
        ['Neodymium', '145Nd', '3.5', '-0.74400', '-1.42921000000000', '-33.00000'],
        ['Samarium', '147Sm', '3.5', '-0.92390', '-1.77458000000000', '-25.90000'],
        ['Samarium', '149Sm', '3.5', '-0.76160', '-1.46295000000000', '7.40000'],
        ['Europium', '151Eu', '2.5', '4.10780', '10.58540000000000', '90.30000'],
        ['Europium', '153Eu', '2.5', '1.81390', '4.67422000000000', '241.20000'],
        ['Gadolinium', '155Gd', '1.5', '-0.33208', '-1.30717137860235', '127.00000'],
        ['Gadolinium', '157Gd', '1.5', '-0.43540', '-1.71394000000000', '135.00000'],
        ['Terbium', '159Tb', '1.5', '2.60000', '10.23525000000000', '143.20000'],
        ['Dysprosium', '161Dy', '0.5', '-0.56830', '-1.46438000000000', '250.70000'],
        ['Dysprosium', '163Dy', '1', '0.79580', '2.05151000000000', '264.80000'],
        ['Holmium', '165Ho', '0.5', '4.73200', '9.08775000000000', '358.00000'],
        ['Erbium', '167Er', '0.5', '-0.63935', '-1.22799179441414', '356.50000'],
        ['Thulium', '169Tm', '1', '-0.40110', '-3.53006000000000', '---'],
        ['Ytterbium', '171Yb', '1.5', '0.85506', '7.52612000000000', '---'],
        ['Ytterbium', '173Yb', '1.5', '-0.80446', '-2.07299000000000', '280.00000'],
        ['Lutetium', '175Lu', '3', '2.53160', '4.86250000000000', '349.00000'],
        ['Lutetium', '176Lu', '1.5', '3.38800', '3.45112000000000', '497.00000'],
        ['Hafnium', '177Hf', '0.5', '0.89970', '1.72842000000000', '336.50000'],
        ['Hafnium', '179Hf', '1', '-0.70850', '-1.08560000000000', '379.30000'],
        ['Tantalum', '181Ta', '0.5', '2.68790', '5.16267000000000', '317.00000'],
        ['Tungsten', '183W', '2.5', '0.20401', '1.79564972994000', '---'],
        ['Rhenium', '185Re', '0.5', '3.77100', '9.71752000000000', '218.00000'],
        ['Osmium', '187Os', '1.5', '0.11198', '0.98563064707380', '---'],
        ['Rhenium', '187Re', '1.5', '3.80960', '9.81700000000000', '207.00000'],
        ['Osmium', '189Os', '2.5', '0.85197', '3.35360155237225', '85.60000'],
        ['Iridium', '191Ir', '2.5', '0.19460', '0.76585000000000', '81.60000'],
        ['Iridium', '193Ir', '0.5', '0.21130', '0.83190000000000', '75.10000'],
        ['Platinum', '195Pt', '0.5', '1.05570', '9.29226000000000', '---'],
        ['Gold', '197Au', '1.5', '0.19127', '0.75289837379052', '54.70000'],
        ['Mercury', '199Hg', '1.5', '0.87622', '7.71231431685275', '---'],
        ['Mercury', '201Hg', '1.5', '-0.72325', '-2.84691587554490', '38.60000'],
        ['Thallium', '203Tl', '1.5', '2.80983', '24.73161181836180', '---'],
        ['Thallium', '205Tl', '4', '2.83747', '24.97488014887780', '---'],
        ['Lead', '207Pb', '1.5', '1.00906', '8.88157793726598', '---'],
        ['Bismuth', '209Bi', '3.5', '4.54440', '6.96303000000000', '-51.60000'],
        ['Uranium', '235U', '3.5', '-0.43000', '-0.82761000000000', '493.60000']
        ]

    df = pd.DataFrame(gammalist)
    df.columns = df.iloc[0]
    df = df[1:]
    if df['Nuclide'].isin([isotope]).any():
        gamma = df.loc[df['Nuclide'] == isotope,"Gyromagnetic Ratio (MHz/T)"]
        gamma = float(gamma.iloc[0])*1e6
    else:
        print("Isotope string not recognised, please input a string of format e.g., '1H' or '27Al' and ensure it is an NMR active nuclide.")
        return


    return gamma

def sim_diffusion(NUC, delta = 1, DELTA = 20, maxgrad = 17, D = 0):
    """
    fig, ax = sim_diffusion(NUC, delta=1, DELTA = 20, maxgrad = 17, D = 0)

    Function to help estimate appropriate diffusion experiment parameters. Can set the maximum gradient,
    little delta, and big DELTA to understand the level of attenuation/shape of the curve for whatever nuclide.

    D = 0 is a placeholder, if left as 0, D will be a range of 1e-7 to 1e-15 stepping by order of magnitude,
    if a value for D is set, only one line will be plotted.
    """

    from matplotlib import cm

    switch = 1
    delta = delta/1000
    DELTA = DELTA/1000
    if D == 0:
        D = np.logspace(-8,-15,8)
        switch = 0
    gamma = find_gamma(NUC)  # [10^7 1/T/s]
    G = np.arange(0,maxgrad+(maxgrad/100.0),maxgrad/99.0)
    B = [(2*np.pi*gamma*delta*i)**2 * (DELTA-(delta/3)) for i in G]

    if switch == 0:
        I = np.zeros(shape=(len(D),len(G)))
        cnt=0
        for j in D:
            Inow = np.exp(np.multiply(-j,B))
            I[cnt] = Inow
            cnt+=1
    else:
        I = np.exp(np.multiply(-D,B))


    fig, ax = plt.subplots()
    if switch == 0:
        colmap=cm.seismic(np.linspace(0,1,len(D)))
        [plt.plot(G,I[k,:],color=c, linewidth=2,label=str(D[k])+" $\mathregular{m^2 s^{–1}}$") for k,c in zip(range(len(D)),colmap)]
    else:
        plt.plot(G,I,linewidth=2,color='r',label=str(D)+" $\mathregular{m^2 s^{–1}}$")
    ax.set_xlim(0,maxgrad*1.25)
    plt.legend(loc='upper right',frameon=False)
    plt.xlabel('Gradient Strength, g / $\mathregular{T m^{–1}}$')
    plt.ylabel("Intensity, $\mathregular{I/I_0}$")
    plt.show()
    return fig,ax

def xf2(datapath, procno=1, mass=1, f2l=10, f2r=0):
    """
    real_spectrum, expt_paramters = xf2(datapath, procno=1, mass=1, f2l=10, f2r=0)
    loads slices of a pseudo-2D experiment as individual 1D experiments, also loads the relevant experimental parameters for T2 or diffusion experiments.
    """
    real_spectrum_path = os.path.join(datapath,"pdata",str(procno),"2rr")
    procs = os.path.join(datapath,"pdata",str(procno),"procs")
    acqus = os.path.join(datapath,"acqus")
    proc2s = os.path.join(datapath,"pdata",str(procno),"proc2s")
    acqu2s = os.path.join(datapath,"acqu2s")

    ########################################################################

    # Bruker file format information
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Bruker binary files (ser/fid) store data as an array of numbers whose
    # endianness is determined by the parameter BYTORDA (1 = big endian, 0 = little
    # endian), and whose data type is determined by the parameter DTYPA (0 = int32,
    # 2 = float64). Typically the direct dimension is digitally filtered. The exact
    # method of removing this filter is unknown but an approximation is available.
    # Bruker JCAMP-DX files (acqus, etc) are text file which are described by the
    # `JCAMP-DX standard <http://www.jcamp-dx.org/>`_.  Bruker parameters are
    # prefixed with a '$'.

    ####################################

    # Get aqcus
    O1str = '##$O1= '
    OBSstr = '##$BF1= '
    NUCstr = '##$NUC1= <'
    Lstr = "##$L= (0..31)"
    CNSTstr = "##$CNST= (0..63)"

    O1 = float("NaN")
    OBS = float("NaN")
    NUC = ""
    L1 = float("NaN")
    L2 = float("NaN")
    CNST31 = float("NaN")

    with open(acqus,"rb") as input:
        for line in input:
    #         print(line.decode())
            if O1str in line.decode():
                linestr = line.decode()
                O1 = float(linestr[len(O1str):len(linestr)-1])
            if OBSstr in line.decode():
                linestr = line.decode()
                OBS = float(linestr[len(OBSstr):len(linestr)-1])
            if NUCstr in line.decode():
                linestr = line.decode()
                NUC = str(linestr[len(NUCstr):len(linestr)-2])
            if Lstr in line.decode():
                line = next(input)
                linestr = line.decode()
                L = (linestr.strip("\n").split(" "))
                L1 = float(L[1])
                L2 = float(L[2])
            if CNSTstr in line.decode():
                CNST = []
                line = next(input)
                while "##$CPDPRG=" not in str(line):
                    linestr = line.decode()
                    CNST.extend(linestr.strip("\n").split(" "))
                    line = next(input)
                CNST31 = float(CNST[31])
            if ~np.isnan(O1) and ~np.isnan(OBS) and ~np.isnan(L1) and ~np.isnan(CNST31) and not len(NUC)==0:
                break

    ####################################

    # Get procs

    SWstr = '##$SW_p= '
    SIstr = '##$SI= '
    SFstr = '##$SF= '
    NCstr = '##$NC_proc= '

    SW = float("NaN")
    SI = float("NaN")
    SF = float("NaN")
    NC_proc = float("NaN")

    with open(procs,"rb") as input:
        for line in input:
            if SWstr in line.decode():
                linestr = line.decode()
                SW = float(linestr[len(SWstr):len(linestr)-1])
            if SIstr in line.decode():
                linestr = line.decode()
                SI = float(linestr[len(SIstr):len(linestr)-1])
            if SFstr in line.decode():
                linestr = line.decode()
                SF = float(linestr[len(SFstr):len(linestr)-1])
            if NCstr in line.decode():
                linestr = line.decode()
                NC_proc = float(linestr[len(NCstr):len(linestr)-1])
            if ~np.isnan(SW) and ~np.isnan(SI) and ~np.isnan(NC_proc) and ~np.isnan(SF):
                break

    ####################################

    # Get aqcu2s for indirect dimension
    O1str_2 = '##$O1= '
    OBSstr_2 = '##$BF1= '
    NUCstr_2 = '##$NUC1= <'
    TDstr_2 = "##$TD= "

    O1_2 = float("NaN")
    OBS_2 = float("NaN")
    NUC_2 = ""
    TD_2 = float("NaN")

    with open(acqu2s,"rb") as input:
        for line in input:
    #         print(line.decode())
            if O1str_2 in line.decode():
                linestr = line.decode()
                O1_2 = float(linestr[len(O1str_2):len(linestr)-1])
            if OBSstr_2 in line.decode():
                linestr = line.decode()
                OBS_2 = float(linestr[len(OBSstr_2):len(linestr)-1])
            if NUCstr_2 in line.decode():
                linestr = line.decode()
                NUC_2 = str(linestr[len(NUCstr_2):len(linestr)-2])
            if TDstr_2 in line.decode():
                linestr = line.decode()
                TD_2  = float(linestr.strip(TDstr_2))
            if ~np.isnan(O1_2) and ~np.isnan(OBS_2) and ~np.isnan(TD_2) and not len(NUC_2)==0:
                break

    ####################################

    # # Get proc2s for indirect dimension

    SIstr_2 = '##$SI= '
    SI_2 = float("NaN")

    with open(proc2s,"rb") as input:
        for line in input:
            if SIstr_2 in line.decode():
                linestr = line.decode()
                SI_2 = float(linestr[len(SIstr_2):len(linestr)-1])
            if ~np.isnan(SI_2):
                break

    ####################################

    # Determine x axis values
    SR = (SF-OBS)*1000000
    true_centre = O1-SR
    xmin = true_centre-SW/2
    xmax = true_centre+SW/2
    xAxHz = np.linspace(xmax,xmin,num=int(SI))
    xAxppm = xAxHz/SF

    real_spectrum = np.fromfile(real_spectrum_path, dtype='<i4', count=-1)
    real_spectrum = real_spectrum.reshape([int(SI_2),int(SI)])

    xlow = np.abs(xAxppm-f2l).argmin()
    xhigh = np.abs(xAxppm-f2r).argmin()
    if xlow > xhigh:
        xlow, xhigh = xhigh, xlow
    xAxppm = xAxppm[xlow:xhigh]
    real_spectrum = real_spectrum[:,xlow:xhigh]

    expt_parameters = {'NUC': NUC, "L1": L1, "CNST31": CNST31}

    return xAxppm, real_spectrum, expt_parameters


def diff_params_import(datapath, NUC):
    """
    delta, DELTA, expectedD, Gradlist = diff_params_import(datapath, NUC)
    obtains delta, DELTA, a guess for D, and the gradient list from the diff.xml file
    """
    import xml.etree.ElementTree as ET

    diff_params_path = os.path.join(datapath,"diff.xml")

    tree = ET.parse(diff_params_path)
    root = tree.getroot()

    delta = float(root.find(".//delta").text) # [ms]
    delta = delta/1000 # [s]
    DELTA = float(root.find(".//DELTA").text) # [ms]
    DELTA = DELTA/1000  # [s]
    exD = float(root.find(".//exDiffCoff").text) # [m2/s]
    x_values_element = root.find(".//xValues/List")
    x_values_list = x_values_element.text.split()
    x_values = [float(value) for value in x_values_list]
    Gradlist = x_values[1::4] # [G/cm]
    Gradlist = [x/100 for x in Gradlist] # [T/m]

    gamma = find_gamma(NUC)  # [10^7 1/T/s]
    # gamma = gamma  # [1/T/s]

    return delta, DELTA, exD, Gradlist, gamma


def xf2_peak_pick(xAxppm, real_spectrum, prominence=[0.001, 1],
                  peak_pos=float("NaN"), f1p=0, f2p=0, plot=True):
    """
    peak_ints_norm = xf2_peak_pick(xAxppm, real_spectrum, prominence = [0.001, 1], peak_pos = float("NaN"))
    peak finder from the xf2 function, if peak_pos is defined, the fit for only that x value (in ppm) will be shown.
    """

    first_slice = real_spectrum[0, :]
    last_slice = real_spectrum[-1, :]
    best_slice = (first_slice if (np.sum(first_slice) > np.sum(first_slice))
                  else last_slice)

    min_best_slice = min(best_slice)
    best_slice = best_slice-min_best_slice
    max_best_slice = max(best_slice)
    best_slice = best_slice/max_best_slice
    real_spectrum = real_spectrum-min_best_slice
    real_spectrum = real_spectrum/max_best_slice

    if np.isnan(peak_pos):
        pl = find_peaks(best_slice, prominence=prominence)
        pl = pl[0] # indices of picked peaks
        peak_positions = xAxppm[pl] # ppm values of picked peaks
        cols = [str(round(items, 2)) for items in peak_positions]
    else:
        pl = [np.where(xAxppm <= peak_pos)[0][0]]  # broken!
        cols = ['peak']

    # best_slice_pl = real_spectrum[0,pl]
    # peak_slices = real_spectrum[:,pl]

    if plot:
        fig, ax = plt.subplots()

    # All Slices
    peak_ints = []
    for slices in real_spectrum:
        current_slice = slices
        peak_ints_now = [float(current_slice[i]) for i in pl]
        peak_ints.append(peak_ints_now)
        if plot:
            plt.plot(xAxppm, current_slice)

    if plot:
        ax.vlines(x=peak_pos, ymin=-0.075, ymax=0.0, color='r')
        ax.invert_xaxis()
        ax.set_xlabel("Shift / ppm")
        ax.set_ylabel("Normalized Intensity")

        if f1p != 0 and f2p != 0:
            if f2p < f1p:
                plt.xlim(f1p, f2p)
            else:
                plt.xlim(f2p, f1p)

    max_peak_ints = np.amax(peak_ints, axis=0)
    peak_ints_norm = []
    for slices2 in peak_ints:
        current_slice2 = np.divide(slices2, max_peak_ints)
        peak_ints_norm.append(current_slice2)

    # peak_intensity = pd.DataFrame(np.array(peak_ints_norm), columns=cols)
    # print(peak_intensity)

    if np.isnan(peak_pos):
        return np.array(peak_ints_norm), peak_positions
    else:
        return np.array(peak_ints_norm)


def read_1d_exsys(datapath, expnos, peak_pos=None, plot=False):
    d15s = []
    for expno in expnos:
        Dstr = "##$D= (0..63)"
        acqus = os.path.join(datapath, str(expno), "acqus")
        with open(acqus, "rb") as input:
            for line in input:
                if Dstr in line.decode():
                    line = next(input)
                    linestr = line.decode()
                    D = (linestr.strip("\n").split(" "))
                    D15 = float(D[15])
                    break
        d15s.append(D15)
    d15s = np.array(d15s)

    ppm, spectra = readfolder(datapath, expnos)
    peak_ints = xf2_peak_pick(ppm, spectra, prominence=[0.1, 1], f1p=4, f2p=-1.5, plot=plot)
    peak_ints = peak_ints.transpose()

    return d15s, peak_ints


def analyze_lpsc_1d_exsys(datapath, expnos, peak_pos=None, plot=False):
    first_path = os.path.join(datapath, str(expnos[0]))
    ppm, intensity, fig, ax = NMR1D(first_path, f1p=3, f2p=-3.2)

    amplitude = -np.trapz(intensity, x=ppm)

    lpsc_peak1 = PseudoVoigtModel(prefix='p1_')
    init_pars = lpsc_peak1.make_params(center=dict(value=1.2, min=-10, max=10),
                                       amplitude=dict(value=0.6*amplitude, min=0),
                                       sigma=dict(value=.1, min=0.001, max=3))
    lpsc_peak2 = PseudoVoigtModel(prefix='p2_')
    init_pars.update(lpsc_peak2.make_params(center=dict(value=1, min=-10, max=10),
                                            amplitude=dict(value=0.4*amplitude, min=0),
                                            sigma=dict(value=.1, min=0.001, max=3)))

    lpsc_model = lpsc_peak1 + lpsc_peak2
    first_fit = lpsc_model.fit(intensity, init_pars, x=ppm)
    first_fit.plot_fit(ax=ax, numpoints=100, fitfmt='r-')
    lpsc_fits = [first_fit]
    lpsc_pars = lpsc_model.make_params(**first_fit.best_values)

    lpsc_pars['p1_center'].set(max=first_fit.best_values['p1_center']+.2,
                               min=first_fit.best_values['p1_center']-.2)
    lpsc_pars['p2_center'].set(max=first_fit.best_values['p2_center']+.2,
                               min=first_fit.best_values['p2_center']-.2)

    d15s = []
    for expno in expnos:
        Dstr = "##$D= (0..63)"
        acqus = os.path.join(datapath, str(expno), "acqus")
        with open(acqus, "rb") as input:
            for line in input:
                if Dstr in line.decode():
                    line = next(input)
                    linestr = line.decode()
                    D = (linestr.strip("\n").split(" "))
                    D15 = float(D[15])
                    break
        d15s.append(D15)
    d15s = np.array(d15s)

    ppm, spectra = readfolder(datapath, expnos[1:])

    for intensity in spectra:
        new_fit = lpsc_model.fit(intensity, lpsc_pars, x=ppm)
        new_fit.plot_fit(ax=ax, numpoints=100, fitfmt='r-')
        lpsc_fits.append(new_fit)

    p1_ints = [fit.best_values['p1_amplitude']
               / first_fit.best_values['p1_amplitude'] for fit in lpsc_fits]
    p2_ints = [fit.best_values['p2_amplitude']
               / first_fit.best_values['p2_amplitude'] for fit in lpsc_fits]

    return d15s, [p1_ints, p2_ints]


def fit_1d_exsys(mixtimes, intensities, savename=None, fixed_t1=None, plot=True):
    DEFAULT_K = 8
    DEFAULT_T1 = 0.2

    def exsy1dfit(x, k, t1):
        return np.multiply(np.exp(np.multiply(-1/t1, x)),
                           np.divide(1+np.exp(np.multiply(2*k, x)), 2))

    exsy_model = Model(exsy1dfit)
    params = exsy_model.make_params(k=DEFAULT_K, t1=DEFAULT_T1)
    if fixed_t1 is not None:
        params['t1'].set(value=fixed_t1, vary=False)
    fit_result = exsy_model.fit(intensities, params, x=mixtimes)
    print(fit_result.fit_report())

    # def exsy1dfit_fixedt1(t1_fixed):
    #     def wrapped(x, k, t1=t1_fixed):
    #         return exsy1dfit(x, k, t1)
    #     return wrapped

    # if fixed_t1 is None:
    #     model = exsy1dfit
    #     popt, pconv = curve_fit(exsy1dfit, mixtimes, intensities, p0=[DEFAULT_K, DEFAULT_T1])
    # else:
    #     model = exsy1dfit_fixedt1(fixed_t1)
    #     popt, pconv = curve_fit(exsy1dfit, mixtimes, intensities, p0=[DEFAULT_K])

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(mixtimes, intensities)

        xfit = np.linspace(min(mixtimes), max(mixtimes), 100)
        ax.plot(xfit, fit_result.eval(x=xfit), 'r-')
        # fit_result.plot_fit(ax=ax, numpoints=100, fitfmt='r-')

        ax.set_xlabel('Mixing Time (s)', fontname='Arial', fontsize=16)
        ax.set_ylabel('Normalized Peak Intensity', fontname='Arial', fontsize=16)

        if savename is not None:
            plt.savefig(savename, bbox_inches='tight', dpi=300)

        return fit_result, fig, ax

    return fit_result


def T2_plot(peak_ints_norm, L1, CNST31):
    """
    T2 plotting function, uses data read from the xf2 function
    T2_plot(peak_ints_norm, L1, CNST31)
    """
    echo_delay = np.arange((2*L1/CNST31),(2*((L1)+(L2*(len(real_spectrum[:,0]))))/CNST31),2*L2/CNST31)
    echo_delay *= 1000 # unit [=] ms
    fig2,ax2 = plt.subplots()
    lines = plt.plot(echo_delay,peak_ints_norm)
    ax2.set_xlabel("Echo delay / ms")
    ax2.set_ylabel("Normalized Intensity")


def diff_plot(peak_ints_norm, datapath, NUC):
    """
    Diffusion plotting function, uses data read from the xf2 function
    G, grad_params = diff_plot(peak_ints_norm, datapath, NUC)
    """
    delta, DELTA, expD, G, gamma = diff_params_import(datapath, NUC)
    # print(delta,DELTA,expD,G, gamma)
    fig2,ax2 = plt.subplots()
    lines = plt.plot(G,peak_ints_norm, 'o')#, c='red', mfc='blue', mec='blue')
    ax2.set_xlabel("Gradient Strength / G cm$\mathregular{^{-1}}$")
    ax2.set_ylabel("Normalized Intensity")
    grad_params = {"delta": delta, "DELTA": DELTA, "gamma": gamma, "expD":expD}
    return G, grad_params

def T2_Fit(x, y, t0=0.5,c0=1,beta0=0.5,showall=False,fittype = "default"):

    def monoExp_t(x, t):
        result=[]
        for i in x:
            result.append(np.exp(-(1/t) * i))
        return result

    def doubleExp(x, m1, t1,t2):
        result=[]
        for i in x:
            result.append(m1*np.exp(-(1/t1) * i)+(1-m1)*np.exp(-(1/t2) * i))
            # result.append(m1*np.exp(-(1/t1) * i)+m2*np.exp(-(1/t2) * i))

        return result

    def stretchExp(x, t, beta):
        result=[]
        for i in x:
            result.append(np.exp(-(i/t)))#**beta))
        return result

    method_str = ["Mono-exponential", "Bi-exponential", "Stretched Exponential"]


    params, cv = scipy.optimize.curve_fit(monoExp_t, x, y, t0)
    mono_t = params
    monoT2 = monoExp_t(x,mono_t)
    R_sq_Mono = r2_score(y, monoT2)

    param_bounds2 = ([0,0,0],[1,1000000,1000000])
    p0bi = (0.5,t0,t0) # start with values near those we expect
    params, cv = scipy.optimize.curve_fit(doubleExp, x, y, p0bi,bounds=param_bounds2)
    m1,t1,t2 = params
    biexpT2 = doubleExp(x,m1,t1,t2)
    R_sq_Bi = r2_score(y, biexpT2)
    m2=1-m1

    p0str = (t0, beta0) # start with values near those we expect --> c is near 1, T2 is close to to 4ms, use a beta of 0.5
    params, cv = scipy.optimize.curve_fit(stretchExp, x, y, p0str,maxfev=1000)
    str_t, beta = params
    stretchT2 = stretchExp(x,str_t,beta)
    R_sq_Stretch = r2_score(y, stretchT2)

    All_R = [R_sq_Mono, R_sq_Bi, R_sq_Stretch]
    fit_types = ["Mono-exponential", "Bi-exponential", "Stretched exponential"]
    R_max = max(All_R)
    method_choice = All_R.index(R_max)
    if fittype != "default":
        method_choice = fit_types.index(fittype)
    # print(All_R, method_choice)

    if fittype == "default":
        if method_choice == 0:
            YY = monoT2

        elif method_choice == 1:
            YY = biexpT2

        elif method_choice == 2:
            YY = stretchT2
    else:
        if method_choice == 0:
            YY = monoT2
        elif method_choice == 1:
            YY = biexpT2
        elif method_choice == 2:
            YY = stretchT2

    if YY == monoT2:
        txt_disp = f'T$_2$ = {round(float(mono_t),6)}'
    elif YY == biexpT2:
        txt_disp = f'Component 1: T$_2$ = {round(t1,6)} ms, w = {round(m1,3)}\nComponent 2: T$_2$ = {round(t2,6)} ms, w = {round(m2,3)}'
    elif YY == stretchT2:
        txt_disp = f'T$_2$ = {round(str_t,6)} ms\nβ = {round(beta,3)}'#\nc = {round(c,3)}'

    fig,ax = plt.subplots()

    if showall:
        plt.plot(x, y, 'o', color='black', label="Experimental Data")
        plt.plot(x, monoT2, '--', color='teal', label=method_str[0]+" fit")
        plt.plot(x, biexpT2, '-.', color='orange', label=method_str[1]+" fit")
        plt.plot(x, stretchT2, ':', color='green', label=method_str[2]+" fit")
        plt.xlabel('Echo delay / ms')
        plt.ylabel('Normalized intensity')
        plt.legend(loc="right")
        plt.show()
    else:
        plt.plot(x, y, 'o', color='blue', label="Experimental Data")
        plt.plot(x, YY, '--', color='red', label=method_str[method_choice]+" fit")
        plt.xlabel('Echo delay / ms')
        plt.ylabel('Normalized intensity')
        plt.text(0.95,0.95,txt_disp, transform=ax.transAxes, ha="right",va="top")
        plt.legend(loc="right")
        plt.ylim(-0.05, max(y)*1.1)
        plt.show()

    display(f"R² = {R_max}")
