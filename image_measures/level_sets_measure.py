''' Function for calculating a measure of image noise using level sets
# Inputs: image - numpy array of pixel intensities
         nlevels - number of levels to calculate over (note that we approximating a continuious distribution with a 'large enough' number of levels)
# Outputs: measure_value
 
This function calculates the number of connected regions above a threshold at an evenly spaced number of threshold
between the min and max values of the image.

There is some morphological operations to deal with orphaned pixels and heuristic parameters in the image processing.

# Usage    
img = misc.imread('/Users/palmer/Copy/ion_image.png').astype(float)
print measure_of_chaos(img,20)
 '''

# try to use cv2 for faster image processing
try:
    import cv2
    cv2.connectedComponents # relatively recent addition, so check presence
    opencv_found = True
except (ImportError, AttributeError):
    opencv_found = False


def clean_image(im_clean,q_val,interp,do_hotspot=False):
    from scipy.signal import medfilt
    import numpy as np
    from scipy import interpolate

    # Image properties
    notnull=im_clean>0 #does not include nan values
    im_clean[notnull==False]=0
    im_size=np.shape(im_clean)
    if do_hotspot:
        # hot spot removal (quantile threshold)
        im_q = np.percentile(im_clean[notnull],q_val)
        im_rep =  im_clean>im_q
        im_clean[im_rep] = im_q
    # interpolate to clean missing values
    if any([interp == '', interp==False]):
        #do nothing
        im_clean=im_clean
    elif interp == 'interpolate' or interp==True:        # interpolate to replace missing data - not always present
        try:
            # interpolate to replace missing data - not always present
            X,Y=np.meshgrid(np.arange(0,im_size[1]),np.arange(0,im_size[0]))
            f=interpolate.interp2d(X[notnull],Y[notnull],im_clean[notnull])
            im_clean=f(np.arange(0,im_size[1]),np.arange(0,im_size[0]))
        except:
            print 'interp bail out'
            im_clean = np.zeros(np.shape(im_clean)) # if interp fails, bail out
    elif interp=='median':
            im_clean = medfilt(im_clean)
    else:
        raise ValueError('{}: interp option not recognised'.format(interp))
    # scale max to 1
    im_clean /= np.max(im_clean)
    return im_clean


def measure_of_chaos(im,nlevels,interp='interpolate',q_val = 99.,clean_im=True):
    global opencv_found
    import numpy as np
    from scipy import ndimage

    ## Image Pre-Processing
    # don't process empty images
    if np.sum(im)==0:
        return np.nan,[],[],[]
    sum_notnull = np.sum(im > 0)
    #reject very sparse images
    if sum_notnull < 4:
        return np.nan,[],[],[]
    if clean_im: # should not be hidden in this funtion
        im=clean_image(im,q_val,interp)

    ## Level Sets Calculation
    # calculate levels
    levels = np.linspace(0,1,nlevels) #np.amin(im), np.amax(im)
    # hardcoded morphology masks
    dilate_mask = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    erode_mask = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.uint8)
    label_mask = np.ones((3,3))

    def count_objects_ndimage(im, lev):
        # Threshold at level
        bw = (im > lev)
        # Morphological operations
        bw=ndimage.morphology.binary_dilation(bw,structure=dilate_mask)
        bw=ndimage.morphology.binary_erosion(bw,structure=erode_mask)
        # Record objects at this level
        return ndimage.label(bw)[1] #second output is number of objects

    def count_objects_opencv(im, lev):
        bw = (im > lev).astype(np.uint8)
        cv2.dilate(bw, dilate_mask)
        cv2.erode(bw, erode_mask)
        return cv2.connectedComponents(bw)[0] - 1 # extra obj is background

    count_objects = count_objects_opencv if opencv_found else count_objects_ndimage

    # Go through levels and calculate number of objects
    num_objs = [count_objects(im, level) for level in levels]
    measure_value = 1.-float(np.sum(num_objs))/(sum_notnull*nlevels)
    return measure_value,im,levels,num_objs

def measure_of_chaos_fit(im,nlevels,interp='interpolate',q_val = 99.): 
    # this updates the scoring function from the main algorithm. 
    from scipy.optimize import curve_fit
    import numpy as np
    def func(x,a,b):
        from scipy.stats import norm
        return norm.cdf(x, loc=a, scale=b)
    
    measure_value,im,levels,num_objs = measure_of_chaos(im,nlevels,interp=interp,q_val=q_val)
    if measure_value == np.nan: #if basic algorithm failed then we're going to fail here too
        return np.nan
    cdf_curve = np.cumsum(num_objs)/float(np.sum(num_objs))
    popt, pcov = curve_fit(func, np.linspace(0,1,nlevels), cdf_curve, p0=(0.5,0.05))
    pdf_fitted = func(np.linspace(0,1,nlevels),popt[0],popt[1])
    #return 1-np.sqrt(np.sum((pdf_fitted - cdf_curve)**2))
    return 1-np.sum(np.abs((pdf_fitted - cdf_curve)))

    
