__author__ = 'intsco'

import numpy as np


def isotope_pattern_match(images, theoretical_isotope_intensities):
    """
    This function calculates a match between a set of isotope ion images and a theoretical intensity vector
    :param images: numpy array of pixel intensities
    :param theoretical_isotope_intensities:
    :return: measure_value
    """
    isotope_ints = theoretical_isotope_intensities[:len(images)]
    images_flat = [img.flat[:] for img in images]
    not_null = images_flat[0] > 0
    image_ints = []
    for ii, _ in enumerate(isotope_ints):
        image_ints.append(np.sum(images_flat[ii][not_null]))
    pattern_match = 1 - np.mean(abs(isotope_ints/np.linalg.norm(isotope_ints) - image_ints/np.linalg.norm(image_ints)))
    return pattern_match if not np.isnan(pattern_match) else 0
