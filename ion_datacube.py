import h5py
import numpy as np
from pyMS.mass_spectrum import mass_spectrum
class ion_datacube():
    # a class for holding a datacube from an MSI dataset and producing ion images
    def __init__(self,step_size=[]): # define mandatory parameters (todo- validator to check all present)
        self.xic = [] # 'xtracted ion chromatogram (2d array of intensities(vector of time by mzs))
        self.bounding_box = [] # tuple (x,y,w,h) in co-ordinate space
        self.coords =[] # co-ordiantes of every pixel
        self.mzs = [] # centroids of each xic vector
        self.tol = [] # tolerance window around centroids in m/z
        if step_size == []:
            self.step_size = []
        else:
            self.step_size = step_size
    def add_xic(self,xic,mz,tol):
        # add an eXtracted Ion Chromatogram to the datacube 
        if len(self.coords) != len(xic):    
            raise ValueError('size of co-ordinates to not match xic (coords:{} xic:{})'.format(len(self.coords),len(xic)))
        self.xic.append(xic) # = np.concatenate((self.xic,xic),axis=1)
        self.mzs = np.concatenate((self.mzs,mz))
        self.tol = np.concatenate((self.tol,tol))

    def remove_xic(self,mz):
        #remove an xic and related info
        index_to_remove = self.mz.index(mz)
        raise NotImplementedError
    def add_coords(self,coords):
        # record spatial co-ordinates for each spectrum
        # coords is a ix3 array with x,y,z coords
        self.coords = coords
        #if len(self.xic)==0:
        #    self.xic = np.zeros((len(coords),0))
        self.calculate_bounding_box()
        self.coord_to_index()
    def calculate_bounding_box(self):
        self.bounding_box=[]
        for ii in range(0,3):
            self.bounding_box.append(np.amin(self.coords[:,ii]))
            self.bounding_box.append(np.amax(self.coords[:,ii]))
    def coord_to_index(self,transform_type='reg_grid',params=()):
        # this function maps coords onto pixel indicies (assuming a grid defined by bounding box and transform type)
        # -implement methods such as interp onto grid spaced over coords
        # -currently treats coords as grid positions, 
        if transform_type == 'reg_grid':
            # data was collected on a grid
            # - coordinates can transformed directly into pixel indiceis
            # - subtract smallest value and divide by x & y step size
            pixel_indices = np.zeros(len(self.coords))
            _coord = np.asarray(self.coords) 
            _coord = np.around(_coord,5) # correct for numerical precision   
            _coord = _coord - np.amin(_coord,axis=0)
            if self.step_size==[]: #no additional info, guess step size in xyz
                self.step_size = np.zeros((3,1))
                for ii in range(0,3):
                    self.step_size[ii] = np.mean(np.diff(np.unique(_coord[:,ii])))

            # coordinate to pixels
            _coord /= np.reshape(self.step_size, (3,))
            _coord_max = np.amax(_coord,axis=0)
            self.nColumns = _coord_max[1]+1
            self.nRows = _coord_max[0]+1            
            pixel_indices = _coord[:,0] * self.nColumns + _coord[:,1]
            pixel_indices = pixel_indices.astype(np.int32)
        else:
            print 'transform type not recognised'
            raise ValueError
        self.pixel_indices = pixel_indices
    def xic_to_image(self,xic_index):
        xic = self.xic[xic_index]
        # turn xic into an image
        img = -1+np.zeros(self.nRows*self.nColumns)
        img[self.pixel_indices] = xic
        img=np.reshape(img,(self.nRows,self.nColumns))
        return img
