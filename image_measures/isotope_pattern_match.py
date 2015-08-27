''' Function for calculating a measure of image noise using level sets
# Inputs: image - numpy array of pixel intensities
         nlevels - number of levels to calculate over (note that we approximating a continuious distribution with a 'large enough' number of levels)
# Outputs: measure_value
 
This function calculates a match between a set of isotope ion images and a theoretical intenisty vector

 '''
def isotope_pattern_match(images,theoretical_isotope_intensity): 
    import numpy as np
    from scipy import ndimage, misc, ndarray, interpolate 

    image_intensities = [sum(images[ii]) for ii in range(0,len(theoretical_isotope_intensity))]
    isotope_pattern_match = 1-np.mean(abs(theoretical_isotope_intensity/np.linalg.norm(theoretical_isotope_intensity) - image_intensities/np.linalg.norm(image_intensities)))    
    return isotope_pattern_match 


