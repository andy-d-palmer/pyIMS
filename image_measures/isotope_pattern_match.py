__author__ = 'intsco'

import numpy as np


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
        image_ints.append(np.sum(images_flat[ii][not_null]))
    pattern_match = 1 - np.mean(abs(theor_iso_intensities / np.linalg.norm(theor_iso_intensities) -
                                    image_ints / np.linalg.norm(image_ints)))
    if pattern_match == 1.:
        return 0
    return pattern_match
