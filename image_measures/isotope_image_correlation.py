__author__ = 'intsco'

import numpy as np


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
