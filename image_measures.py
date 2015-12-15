import numpy as np
from scipy import ndimage

from imutils import nan_to_zero

# try to use cv2 for faster image processing
try:
    import cv2

    cv2.connectedComponents  # relatively recent addition, so check presence
    opencv_found = True
except (ImportError, AttributeError):
    opencv_found = False


def measure_of_chaos(im, nlevels, overwrite=True):
    """
    Compute a measure for the spatial chaos in given image using the level sets method.

    :param im: 2d array
    :param nlevels: how many levels to use
    :type nlevels: int
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

    notnull_mask = nan_to_zero(im)
    im_clean = im / np.max(im)  # normalize to 1

    # Level Sets Calculation
    object_counts = _level_sets(im_clean, nlevels)
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
    measure_value = 1 - float(sum_vals) / (sum_notnull * nlevels)
    return measure_value


def isotope_pattern_match(images_flat, theor_iso_intensities):
    """
    This function calculates a match between a list of isotope ion images and a theoretical intensity vector.

    :param images_flat: 2d array of pixel intensities with shape (d1, d2) where d1 is the number of images and d2 is the number of pixels per image, i.e. :code:`images_flat[i]` is the i-th flattened image
    :param theor_iso_intensities: 1d array of theoretical isotope intensities with shape d1, i.e :code:`theor_iso_intensities[i]` is the theoretical isotope intensity corresponding to the i-th image
    :return: measure value between 0 and 1, bigger is better
    :rtype: float
    """
    image_ints = []
    not_null = images_flat[0] > 0
    for ii, _ in enumerate(theor_iso_intensities):
        image_ints.append(np.sum(images_flat[ii, not_null]))
    pattern_match = 1 - np.mean(abs(theor_iso_intensities / np.linalg.norm(theor_iso_intensities) -
                                    image_ints / np.linalg.norm(image_ints)))
    if pattern_match == 1.:
        return 0
    return pattern_match


def isotope_image_correlation(images_flat, weights=None):
    """
    Function for calculating a weighted average measure of image correlation with the principle image.

    :param images_flat: numpy array of pixel intensities
    :param weights: weighting to put on each im
    :return: measure_value
    """

    if len(images_flat) < 2:
        return 0
    else:
        # slightly faster to compute all correlations and pull the elements needed
        iso_correlation = np.corrcoef(images_flat)[1:, 0]
        # when all values are the same (e.g. zeros) then correlation is undefined
        iso_correlation[np.isinf(iso_correlation)] = 0
        return np.average(iso_correlation, weights=weights)
