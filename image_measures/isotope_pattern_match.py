''' Function for calculating a measure of image noise using level sets
# Inputs: images - numpy array of pixel intensities
# Outputs: measure_value
 
This function calculates a match between a set of isotope ion images and a theoretical intenisty vector

 '''
def isotope_pattern_match(images,theoretical_isotope_intensity): 
    """
    :type images: array of images (each image as a vector)
    :type theoretical_isotope_intensity: list of intensity values
    """
    import numpy as np
    not_null = images[0]>0
    image_intensities = []
    for ii in range(0, len(theoretical_isotope_intensity)):
        image_intensities.append(np.sum(images[ii][not_null]))
    isotope_pattern_match = 1-np.mean(abs(theoretical_isotope_intensity/np.linalg.norm(theoretical_isotope_intensity) - image_intensities/np.linalg.norm(image_intensities)))    
    return isotope_pattern_match

