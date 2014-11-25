"""
Created on Thu Jul 31 16:43:42 2014

@author: torre
"""

import numpy
import quantities as pq
import matplotlib.pyplot as plt
import copy
from scipy.optimize import curve_fit


def load(filename):
    '''
    Load a .npy or .npz archive and return the internal item.

    Parameters
    ----------
    filename : str
        the file to be loaded, including the path to the file.


    Returns
    -------
    dict
        a python dictionary, with the following key:
        * 'st': a list of SpikeTrains (one per trial)
        * '__doc__': a description of the data
        * '__version__': the Matlab version used to generate the data
        * '__header__': None
        * 'sptrmat': matrix of binned spike trains
    '''
    return numpy.load(filename).item()


def _rescale(x, q):
    '''
    Rescale a list of quantity objects to the desired quantity

    arguments:
    ----------
        x: list of Quantity objects
            the list of elements to rescale

        q: a Quantity
            the quantity to which to rescale the elements of x

    returns:
    --------
        a copy of x with all elements rescaled to quantity q
    '''
    x_rescaled = copy.deepcopy(x)
    for i, xx in enumerate(x):
        x_rescaled[i] = xx.rescale(q)
    return x_rescaled


def add_raster(fig, (n_row, n_col, panel), sts, ms=4, title='',
    xlabel='', ylabel=''):
    '''
    Add a raster plot of spike trains to a figure, at the specified position.

    parameters:
    ----------
    sts: list of SpikeTrains
        the list of spike trains for which to make a raster plot

    fig: matplotlib.pyplot.figure object
        the figure to which to add the raster plot

    (n_row, n_col, panel): tuple of integers
        the number of rows and columns in the figure, and the panel id in
        which to plot the raster display.

    ms: float, optional
        the size of the spike markers in the raster plot

    returns:
    -------
    Returns the figure fig, enriched with the raster plot at the specified
    position
    '''

    sts_ms = _rescale(sts, 'ms')
    ax = fig.add_subplot(n_row, n_col, panel)
    for i, st in enumerate(sts_ms):
        ax.plot(st.magnitude, [i+1]*len(st), '.', ms=ms, color='k')

    t0 = min([st.t_start for st in sts_ms]).rescale('ms').magnitude
    T  = max([st.t_stop for st in sts_ms]).rescale('ms').magnitude
    ax.set_xlim((t0, T))
    ax.set_ylim(0, len(sts)+1)
    ax.set_xlabel(xlabel+' (ms)', size=12)
    ax.set_ylabel(ylabel, size=12)
    ax.set_title(title)
    fig.add_axes(ax)

    return fig


def raster(sts, ms=4, title='', xlabel='', ylabel=''):
    '''
    Make a raster plot of a list of spike trains.

    Arguments
    ----------
    sts: list of SpikeTrains
        the list of spike trains for which to make a raster plot

    ms: float, optional
        the size of the spike markers in the raster plot

    Returns
    -------
    None
    '''

    sts_ms = _rescale(sts, 'ms')
    t0 = min([st.t_start for st in sts_ms]).rescale('ms').magnitude
    T = max([st.t_stop for st in sts_ms]).rescale('ms').magnitude

    for i, st in enumerate(sts_ms):
        plt.plot(st.magnitude, [i + 1] * len(st), '.', ms=ms, color='k')

    plt.xlim((t0, T))
    plt.ylim(0, len(sts) + 1)
    plt.xlabel(xlabel + ' (ms)', size=12)
    plt.ylabel(ylabel, size=12)
    plt.title(title)


def add_hist(fig, (n_row, n_col, panel), left, height, width=None,
    bottom=None, title='', xlabel='', ylabel=''):
    '''
    Add an axis to the specified figure, containing a bar plot

    Arguments
    ----------
    fig: matplotlib.pyplot.figure object
        the figure to which to add the raster plot
    (n_row, n_col, panel): tuple of integers
        the number of rows and columns in the figure, and the panel id in
        which to plot the raster display.
    left: array or Quantity array
        the left ends of the histogram bins
    height: array or Quantity array
        the left ends of the histogram bins
    width: float or array (also as a Quantity). Optional, default is None
        the width of each histogram bar
    bottom: float or array (also as a Quantity). Optional, default is None
        the bottom of each histogram bar
    title: str, optional. Default is ''
        title to assign to the generated axis

    Returns
    -------
        None
    '''
    if isinstance(left, pq.Quantity):
        left_dl = left.magnitude
        if left.units == pq.dimensionless:
            x_unit = ''
        else:
            x_unit = ' (%s)' % (left.units.__str__().split(' ')[-1])
    else:
        left_dl = left
        x_unit = ''

    if isinstance(height, pq.Quantity):
        height_dl = height.magnitude
        if height.units == pq.dimensionless:
            y_unit = ''
        else:
            y_unit = ' (%s)' % (height.units.__str__().split(' ')[-1])
    else:
        height_dl = height
        y_unit = ''

    width_dl = 0 if width == None else width.rescale(left.units).magnitude
    bottom_dl = 0 if bottom == None else bottom.rescale(height.units).magnitude

    ax = fig.add_subplot(n_row, n_col, panel)
    ax.bar(left=left_dl, height=height_dl, color='.5', width=width_dl,
           bottom=bottom_dl)
    ax.set_title(title)

    x0, x1 = min(left_dl), max(left_dl + width_dl)
    y0, y1 = min(height_dl) * .9, max(height_dl + bottom_dl)
    ax.set_xlim((x0, x1))
    ax.set_ylim((y0, y1))
    ax.set_xlabel(xlabel + x_unit, size=12)
    ax.set_ylabel(ylabel + y_unit, size=12)


def hist(left, height, width=None, bottom=None, title='',
        xlabel='', ylabel=''):
    '''
    Plot Quantity objects in a bar plot

    Arguments
    ---------
    left: array or Quantity array
        the left ends of the histogram bins
    height: array or Quantity array
        the left ends of the histogram bins
    width: float or array (also as a Quantity). Optional, default is None
        the width of each histogram bar
    bottom: float or array (also as a Quantity). Optional, default is None
        the bottom of each histogram bar
    title: str, optional. Default is ''
        title to assign to the generated axis
    xlabel: str
        label of the x-axis
    ylabel: str
        label of the y-axis

    Returns
    -------
    None
    '''

    if isinstance(left, pq.Quantity):
        left_dl = left.magnitude
        if left.units == pq.dimensionless:
            x_unit = ''
        else:
            x_unit = ' (%s)' % (left.units.__str__().split(' ')[-1])
    else:
        left_dl = left
        x_unit = ''

    if isinstance(height, pq.Quantity):
        height_dl = height.magnitude
        if height.units == pq.dimensionless:
            y_unit = ''
        else:
            y_unit = ' (%s)' % (height.units.__str__().split(' ')[-1])
    else:
        height_dl = height
        y_unit = ''

    width_dl = 0 if width == None else width.rescale(left.units).magnitude
    bottom_dl = 0 if bottom == None else bottom.rescale(height.units).magnitude

    plt.bar(left=left_dl, height=height_dl, color='.5',
             width=width_dl, bottom=bottom_dl)
    plt.title(title)

    x0, x1 = min(left_dl), max(left_dl + width_dl)
    y0, y1 = min(height_dl) * .9, max(height_dl + bottom_dl)
    plt.xlim((x0, x1))
    plt.ylim((y0, y1))
    plt.xlabel(xlabel + x_unit, size=12)
    plt.ylabel(ylabel + y_unit, size=12)
    #return ax

def _expo(x, r):
    if isinstance(x, pq.Quantity) or isinstance(r, pq.Quantity):
        return r*numpy.exp(-(r*x).rescale(pq.dimensionless).magnitude)
    else:
        return r*numpy.exp(-(r*x))


def fit_to_exp(x, y):
    '''
    Fit points (x[i], y[i]) to an exponential probability density function

                f(x) = m*exp(-m*x)

    by fitting the mean parameter m.

    Arguments
    ---------
    x: array or Quantity array
        points' abscissa
    y: array or Quantity array
        points' ordinate

    Returns
    -------
    m: float or Quantity
        estimate of the distribution's mean parameter
    z: array or Quantity array
        value taken by the fit distribution in each abscissa x[i]
    '''

    x_dl = x if not isinstance(x, pq.Quantity) else x.simplified.magnitude
    y_dl = y if not isinstance(y, pq.Quantity) else y.simplified.magnitude

    r, cov = curve_fit(_expo, x_dl, y_dl)

    if isinstance(y, pq.Quantity): r = (r * pq.Hz).rescale(y.units)

    return r, _expo(x, r)

