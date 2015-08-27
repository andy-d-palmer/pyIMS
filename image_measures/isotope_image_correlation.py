''' Function for calculating a weighted average measure of image correlation with the principle image
# Inputs: images_as_2darray - array of image vectors (each image is one row in array)
          weights - weighting to put on each im
          mask - choice to just use pixels that are positive in the first image
# Outputs: measure_value
 
This function calculates the correlation between images 
'''
def isotope_image_correlation(images_as_2darray,weights=[],mask=False): 
    import numpy as np
    from scipy import ndimage, misc, ndarray, interpolate 
    from scipy.signal import medfilt
    if mask ==True:
        images_as_2darray = np.ma.masked_array(images_as_2darray, mask=images_as_2darray[:,0]>0)
        iso_correlation=np.ma.corrcoef(images_as_2darray)[1:,0]
    else:
        iso_correlation = np.corrcoef(images_as_2darray)[1:,0]  # slightly faster to compute all correlations and pull the elements needed
    iso_correlation[np.isnan(iso_correlation)] = 0 # when all values are the same (e.g. zeros) then correlation is undefined
    if weights==[]:
        return np.average(iso_correlation)
    else:
        return np.average(iso_correlation,weights=weights)

