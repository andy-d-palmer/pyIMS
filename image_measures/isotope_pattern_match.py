__author__ = 'intsco'

import numpy as np


def isotope_pattern_match(images_flat, theor_iso_intensities):
    """
    This function calculates a match between a set of isotope ion images and a theoretical intensity vector
    :param images: numpy array of pixel intensities
    :param theor_iso_intensities:
    :return: measure_value
    """
    # test
    image_ints = []
    for ii, _ in enumerate(theor_iso_intensities):
        image_ints.append(np.sum(images_flat[ii]))
    pattern_match = 1 - np.mean(abs(theor_iso_intensities/np.linalg.norm(theor_iso_intensities) -
                                    image_ints/np.linalg.norm(image_ints)))
    return pattern_match if not np.isnan(pattern_match) and pattern_match < 1.0 else 0
