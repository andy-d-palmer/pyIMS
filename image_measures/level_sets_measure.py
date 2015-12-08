from warnings import warn

import numpy as np
from scipy import ndimage, interpolate
from scipy.signal import medfilt


def measure_of_chaos(im, nlevels, interp='interpolate', q_val=99., overwrite=True):
    """
    Compute a measure for the spatial chaos in given image using the level sets method.

    :param im: 2d array
    :param nlevels: how many levels to use
    :type nlevels: int
    :param interp: interpolation option to use before calculating the measure. None or False means no interpolation. 'interp' or True means spline interpolation. 'median' means median filter.
    :param q_val: the percentile above which to flatten the image
    :type q_val: float
    :param overwrite: Whether the input image can be overwritten to save memory
    :type overwrite: bool
    :return: the measured value
    :rtype: float
    :raises ValueError: if nlevels <= 0 or q_val is an invalid percentile or an unknown interp value is used
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
    # TODO: remove this statement. Reasons:
    #  - the interpolation afterwards might fill in values greater than 1 leading to a non-normalized array
    #  - see function definition
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
    # TODO: divide by max(im) instead of im_q. Reason:
    # After interpolating, there can be values > im_q
    im_clean = interp_func(im) / im_q  # normalize to 1

    # Level Sets Calculation
    object_counts = _level_sets(im_clean, nlevels)
    # TODO: return the plain value instead of a tuple
    return _measure(object_counts, sum_notnull),


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


# TODO: Remove this function. Reason: this should happen outside of the algorithm if desired
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


def _dilation_and_erosion(im, dilate_mask=None, erode_mask=None):
    dilate_mask = dilate_mask or [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    erode_mask = erode_mask or [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    return ndimage.binary_erosion(ndimage.morphology.binary_dilation(im, structure=dilate_mask), structure=erode_mask)


def _level_sets(im_clean, nlevels, prep=_dilation_and_erosion):
    """
    Divide the image into level sets and count the number of objects in each of them.

    :param im_clean: 2d array with :code:`im_clean.max() == 1`
    :param int nlevels: number of levels to search for objects (positive integer)
    :param prep: callable that takes a 2d array as its only argument and returns a 2d array
    :return: sequence with the number of objects in each respective level
    """
    if nlevels <= 0:
        raise ValueError("nlevels must be positive")
    prep = prep or (lambda x: x)  # if no preprocessing should be done, use the identity function

    # TODO change the levels. Reason:
    #  - in the for loop, the > operator is used. The highest level is 1, therefore the highest level set will always
    #    be empty. The ndimage.label function then returns 1 as the number of objects in the empty image, although it
    #    should be zero.
    # Proposed solution:
    # levels = np.linspace(0, 1, nlevels + 2)[1:-1]
    # That is, create nlevels + 2 levels, then throw away the zero level and the one level
    # or:
    # levels = np.linspace(0, 1, nlevels)[1:-1]
    # That is, only use nlevels - 2 levels. This means that the output array will have a size of nlevels - 2
    levels = np.linspace(0, 1, nlevels)  # np.amin(im), np.amax(im)
    # Go through levels and calculate number of objects
    num_objs = []
    for lev in levels:
        # Threshold at level
        bw = (im_clean > lev)
        bw = prep(bw)
        # Record objects at this level
        num_objs.append(ndimage.label(bw)[1])  # second output is number of objects
    return num_objs


def _measure(num_objs, sum_notnull):
    """
    Calculate a statistic for the object counts.

    :param num_objs: number of objects found in each level, respectively
    :param float sum_notnull: sum of all non-zero elements in the original array (positive number)
    :return: the calculated value
    """
    num_objs = np.asarray(num_objs, dtype=np.int_)
    nlevels = len(num_objs)
    sum_notnull = float(sum_notnull)
    if sum_notnull <= 0:
        raise ValueError("sum_notnull must be positive")
    if min(num_objs) < 1:
        raise ValueError("must have at least one object in each level")
    if nlevels < 1:
        raise ValueError("array of object counts is empty")

    sum_vals = np.sum(num_objs)
    # TODO: delete the if statement. Reasons:
    #  - the intention to ignore images with just a couple of noise pixels should be addressed at the very beginning
    #    of the measure_of_chaos function
    #  - it is buggy: the number of objects is not necessarily monotonic, e.g. [1, 3, 2] would get a score of 0
    #    because its mean is equal to the last value just by coincidence
    if sum_vals == nlevels * num_objs[-1]:  # all objects are in the highest level
        sum_vals = 0
    # TODO: use 1 - ... Reason: bigger is better
    measure_value = float(sum_vals) / (sum_notnull * nlevels)
    return measure_value
