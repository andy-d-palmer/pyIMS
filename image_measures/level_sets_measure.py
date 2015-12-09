from warnings import warn

import numpy as np
from scipy import ndimage
from scipy.signal import medfilt

from imutils import nan_to_zero, interpolate

# try to use cv2 for faster image processing
try:
    import cv2

    cv2.connectedComponents  # relatively recent addition, so check presence
    opencv_found = True
except (ImportError, AttributeError):
    opencv_found = False


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
        return np.nan  #, [], [], []
    sum_notnull = np.sum(im > 0)
    # reject very sparse images
    if sum_notnull < 4:
        return np.nan  #, [], [], []

    if not overwrite:
        # don't modify original image, make a copy
        im = im.copy()

    notnull_mask = nan_to_zero(im)

    # interpolate to clean missing values
    interp_func = None
    if not interp:
        interp_func = lambda x: x
    elif interp == 'interpolate' or interp is True:  # interpolate to replace missing data - not always present
        def interp_func(image):
            try:
                return interpolate(image, notnull_mask)
            except:
                warn('interp bail out')
    elif interp == 'median':
        interp_func = medfilt
    else:
        raise ValueError('{}: interp option not recognised'.format(interp))
    im = interp_func(im)
    im_clean = im / np.max(im)  # normalize to 1

    # Level Sets Calculation
    object_counts = _level_sets(im_clean, nlevels)
    # TODO: return the plain value instead of a tuple
    return _measure(object_counts, sum_notnull)


def _dilation_and_erosion(im, dilate_mask=None, erode_mask=None):
    dilate_mask = dilate_mask or [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    erode_mask = erode_mask or [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    if opencv_found:
        cv2.dilate(im, dilate_mask)
        cv2.erode(im, erode_mask)
        return im
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
    count_func = (lambda im: cv2.connectedComponents(im)[0] - 1) if opencv_found else (lambda im: ndimage.label(im)[1])
    for lev in levels:
        # Threshold at level
        bw = (im_clean > lev)
        bw = prep(bw)
        # Record objects at this level
        num_objs.append(count_func(bw))
    return num_objs


def _measure(num_objs, sum_notnull):
    """
    Calculate a statistic for the object counts.

    :param num_objs: number of objects found in each level, respectively
    :param float sum_notnull: sum of all non-zero elements in the original array (positive number)
    :return: the calculated value
    """
    if np.unique(num_objs).shape[0] < 2:
        return np.nan

    num_objs = np.asarray(num_objs, dtype=np.int_)
    nlevels = len(num_objs)
    sum_notnull = float(sum_notnull)
    if sum_notnull <= 0:
        raise ValueError("sum_notnull must be positive")
    if min(num_objs) < 0:
        raise ValueError("must have at least one object in each level")
    if nlevels < 1:
        raise ValueError("array of object counts is empty")

    sum_vals = np.sum(num_objs)
    measure_value = 1 - float(sum_vals) / (sum_notnull * nlevels)
    return measure_value
