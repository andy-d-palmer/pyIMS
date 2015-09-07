__author__ = 'intsco'

import numpy as np


def isotope_image_correlation(images, weights=None):
    """
    Function for calculating a weighted average measure of image correlation with the principle image
    :param images: numpy array of pixel intensities
    :param weights: weighting to put on each im
    :param theoretical_isotope_intensity:
    :return: measure_value
    """
    if len(images) < 2:
        return 1.0
    else:
        images_flat = [img.flat[:] for img in images]
        iso_correlation = np.corrcoef(images_flat)[1:,0]  # slightly faster to compute all correlations and pull the elements needed
        iso_correlation[np.isnan(iso_correlation)] = 0 # when all values are the same (e.g. zeros) then correlation is undefined
        return np.average(iso_correlation, weights=weights)
