import os
import h5py
import numpy as np
import bisect
import sys
import matplotlib.pyplot as plt

# import our MS libraries
sys.path.append('/Users/palmer/Documents/python_codebase/')
from pyMS.mass_spectrum import mass_spectrum
from pyIMS.ion_datacube import ion_datacube
class inMemoryIMS_hdf5():
    def __init__(self,filename,min_mz=0.,max_mz=np.inf,min_int=0.,index_range=[]):
        file_size = os.path.getsize(filename)
        self.load_file(filename,min_mz,max_mz,index_range=index_range)
        
    def load_file(self,filename,min_mz=0,max_mz=np.inf,min_int=0,index_range=[]):
        # parse file to get required parameters
        # can use thin hdf5 wrapper for getting data from file
        self.file_dir, self.filename = file_type=os.path.splitext(filename)
        self.file_type = file_type
        self.hdf = h5py.File(filename,'r')   #Readonly, file must exist
        if index_range == []:
            self.index_list = map(int,self.hdf['/spectral_data'].keys())
        else: 
            self.index_list = index_range
        self.coords = self.get_coords()
        # load data into memory
        self.mz_list = []
        self.count_list = []
        self.idx_list = []
        for ii in self.index_list:
            # load spectrum, keep values gt0 (shouldn't be here anyway)
            this_spectrum = self.get_spectrum(ii)
            mzs,counts = this_spectrum.get_spectrum(source='centroids')
            if len(mzs) != len(counts):
                raise TypeError('length of mzs ({}) not equal to counts ({})'.format(len(mzs),len(counts)))
            # Enforce data limits
            valid = np.where((mzs>min_mz) & (mzs<max_mz) & (counts > min_int))
            counts=counts[valid]
            mzs=mzs[valid]

            # append ever-growing lists (should probably be preallocated or piped to disk and re-loaded)
            self.mz_list.append(mzs)
            self.count_list.append(counts)
            self.idx_list.append(np.ones(len(mzs), dtype=int) * ii)
            
        print 'loaded spectra'
        self.mz_list = np.concatenate(self.mz_list)
        self.count_list = np.concatenate(self.count_list)
        self.idx_list = np.concatenate(self.idx_list)
        # sort by mz for fast image formation
        mz_order = np.argsort(self.mz_list)
        self.mz_list = self.mz_list[mz_order]
        self.count_list = self.count_list[mz_order]
        self.idx_list = self.idx_list[mz_order]
        self.mz_min=self.mz_list[0]
        self.mz_max=self.mz_list[-1]
        self.histogram_mz_axis={}
        print 'file loaded'
    def get_coords(self):
        coords = np.zeros((len(self.index_list),3))
        for k in self.index_list:
            coords[k,:] = self.hdf['/spectral_data/'+str(k)+'/coordinates/']
        return coords
    def get_spectrum(self,index):
        this_spectrum = mass_spectrum()
        tmp_str='/spectral_data/%d' %(index) 
        try:
            this_spectrum.add_spectrum(self.hdf[tmp_str+'/mzs/'],self.hdf[tmp_str+'/intensities/'])
            got_spectrum=True
        except KeyError:
            got_spectrum=False
        try:
            this_spectrum.add_centroids(self.hdf[tmp_str+'/centroid_mzs/'],self.hdf[tmp_str+'/centroid_intensities/'])
            got_centroids=True
        except KeyError:
            got_centroids=False
        if not any([got_spectrum,got_centroids]):
            raise ValueError('No spectral data found in index {}'.format(index))
        return this_spectrum
    def get_ion_image(self,mzs,tols,tol_type='ppm'):
        if tol_type=='ppm':
            tols = tols*mzs/1e6 # to m/z
        data_out = ion_datacube()
        data_out.add_coords(self.coords)
        for mz,tol in zip(mzs,tols):
            mz_upper = mz + tol
            mz_lower = mz - tol
            idx_left = bisect.bisect_left(self.mz_list,mz_lower)
            idx_right = bisect.bisect_right(self.mz_list,mz_upper)
            # slice list for code clarity
            count_vect = np.concatenate((np.asarray([0]),self.count_list[idx_left:idx_right],np.asarray([0])))
            idx_vect = np.concatenate((np.asarray([0]),self.idx_list[idx_left:idx_right],np.asarray([max(self.index_list)])))
            # bin vectors
            ion_vect=np.bincount(idx_vect,count_vect)
            data_out.add_xic(ion_vect,[mz],[tol])
        return data_out
        # Form histogram axis
    def generate_histogram_axis(self,ppm=1.):
        mz_current = self.mz_min
        ppm_mult=ppm*1e-6
        mz_list = []
        while mz_current<self.mz_max:
            mz_list.append(mz_current)
            mz_current=mz_current+mz_current*ppm_mult
        self.histogram_mz_axis[ppm] = mz_list
    def get_histogram_axis(self,ppm=1.):
        try:
            mz_axis = self.histogram_mz_list[ppm]
        except:
            self.generate_histogram_axis(ppm=ppm)
        return self.histogram_mz_axis[ppm]
    def generate_summary_spectrum(self,summary_type='mean',ppm=1.):
        hist_axis = self.get_histogram_axis(ppm=ppm)
        # calcualte mean along some m/z axis
        mean_spec=np.zeros(np.shape(hist_axis))
        for ii in range(0,len(hist_axis)-1):   
            mz_upper = hist_axis[ii+1]
            mz_lower = hist_axis[ii]
            idx_left = bisect.bisect_left(self.mz_list,mz_lower)
            idx_right = bisect.bisect_right(self.mz_list,mz_upper)
            # slice list for code clarity
            count_vect = self.count_list[idx_left:idx_right]
            if summary_type=='mean':
                count_vect = self.count_list[idx_left:idx_right]
                mean_spec[ii]=np.sum(count_vect)
            elif summary_type=='freq':
                idx_vect = self.idx_list[idx_left:idx_right]
                mean_spec[ii]=float(len(np.unique(idx_vect)))
        if summary_type=='mean':
                mean_spec=mean_spec/len(self.index_list)
        elif summary_type=='freq':
                mean_spec=mean_spec/len(self.index_list)
        return hist_axis,mean_spec
