from warnings import warn

import numpy as np
from scipy import ndimage, interpolate
from scipy.signal import medfilt


def measure_of_chaos(im, nlevels, interp='interpolate', q_val=99., overwrite=True):
    """

    :param im:
    :param nlevels:
    :param interp:
    :param q_val:
    :param overwrite: Whether the input image can be overwritten to save memory
    :return:
    """
    # don't process empty images
    if np.sum(im) == 0:
        return np.nan, [], [], []
    sum_notnull = np.sum(im > 0)
    # reject very sparse images
    if sum_notnull < 4:
        return np.nan, [], [], []

    if not overwrite:
        # don't modify original image, make a copy
        im = im.copy()

    notnull_mask = _nan_to_zero(im)
    im_q = _quantile_threshold(im, notnull_mask, q_val)

    # interpolate to clean missing values
    interp_func = None
    if not interp:
        interp_func = lambda x: x
    elif interp == 'interpolate' or interp is True:  # interpolate to replace missing data - not always present
        def interp_func(image):
            try:
                return _interpolate(image, notnull_mask)
            except:
                warn('interp bail out')
    elif interp == 'median':
        interp_func = medfilt
    else:
        raise ValueError('{}: interp option not recognised'.format(interp))
    im_clean = interp_func(im) / im_q

    # Level Sets Calculation
    return _level_sets(im_clean, nlevels, sum_notnull)


def _nan_to_zero(im):
    """
    Set all values to zero that are less than zero or nan; return the indices of all elements that are zero after
    the modification (i.e. those that have been nan or smaller than or equal to zero before calling this function). The
    returned boolean array has the same shape, False denotes that a value is zero now.

    :param im: the array which nan-values will be set to zero in-place
    :return: A boolean array of the same shape as :code:`im`
    """
    if im is None:
        raise AttributeError("im must be an array, not None")
    notnull = im > 0  # does not include nan values
    if notnull is True:
        raise TypeError("im must be an array")
    im[~notnull] = 0
    return notnull


def _quantile_threshold(im, notnull_mask, q_val):
    """
    Set all values greater than the :code:`q_val`-th percentile to the :code:`q_val`-th percentile (i.e. flatten out
    everything greater than the :code:`q_val`-th percentile). For determining the percentile, only nonzero pixels are
    taken into account, that is :code:`im[notnull_mask]`.

    :param im: the array to remove the hotspots from
    :param notnull_mask: index array for the values greater than zero
    :param q_val: percentile to use
    :return: The :code:`q_val`-th percentile
    """
    im_q = np.percentile(im[notnull_mask], q_val)
    im_rep = im > im_q
    im[im_rep] = im_q
    return im_q


def _interpolate(im, notnull_mask):
    """
    Use spline interpolation to fill in missing values.

    :param im: the entire image, including nan or zero values
    :param notnull_mask: index array for the values greater than zero
    :return: the interpolated array
    """
    im_size = im.shape
    X, Y = np.meshgrid(np.arange(0, im_size[1]), np.arange(0, im_size[0]))
    f = interpolate.interp2d(X[notnull_mask], Y[notnull_mask], im[notnull_mask])
    im = f(np.arange(0, im_size[1]), np.arange(0, im_size[0]))
    return im


def _level_sets(im_clean, nlevels, sum_notnull):
    """
    Calculate the level sets measure for given image.

    :param im_clean: 2d array with :code:`im_clean.max() == 1`
    :param int nlevels: number of levels to search for objects (positive integer)
    :param float sum_notnull: sum of all non-zero elements in the original array (positive number)
    :return:
    """
    sum_notnull = float(sum_notnull)
    if not min(nlevels, sum_notnull) > 0:
        raise ValueError("nlevels and sum_notnull must be positive")

    # calculate levels
    levels = np.linspace(0, 1, nlevels)  # np.amin(im), np.amax(im)
    # hardcoded morphology masks
    dilate_mask = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    erode_mask = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    # Go through levels and calculate number of objects
    num_objs = []
    for lev in levels:
        # Threshold at level
        bw = (im_clean > lev)
        # Morphological operations
        bw = ndimage.morphology.binary_dilation(bw, structure=dilate_mask)
        bw = ndimage.morphology.binary_erosion(bw, structure=erode_mask)
        # Record objects at this level
        num_objs.append(ndimage.label(bw)[1])  # second output is number of objects
    sum_vals = np.sum(num_objs)
    if sum_vals == nlevels * num_objs[-1]:  # all objects are in the highest level
        sum_vals = 0
    measure_value = float(sum_vals) / (sum_notnull * nlevels)
    return measure_value, im_clean, levels, num_objs
