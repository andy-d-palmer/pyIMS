import h5py
import numpy as np
from pyMS.mass_spectrum import mass_spectrum
from pyIMS.ion_datacube import ion_datacube

class IMSdataset():
    def __init__(self,filepath):
        self.f = h5py.File(filepath)
        self.spectra_group='InitialMeasurement'
	self.file_version = int(self.f['Version'][0])
	if self.file_version == 4:
		self = self.parse_4()
	elif self.file_version == 14:
		self=self.parse_14()
	else:
		raise ValueError('unsupported file version'+self.file_version)
	self.consistent_mz=True

    def parse_14(self):
	self.spot_index=self.f['Spots'].keys()
        self.coords=np.asarray(self.f['/Registrations/0/Coordinates']).T
	return self	
    def parse_4(self):
	self.spot_index=self.f['Spots'].keys()
	self.coords=np.asarray(self.f['Coordinates']).T
	return self
    def get_mzs(self,index):
        mzs=self.f['Spots/{}/{}/SamplePositions/SamplePositions'.format(index,self.spectra_group)]
        return np.asarray(mzs)
    def get_intensities(self,index):
        intensities=self.f['Spots/{}/{}/Intensities'.format(index,self.spectra_group)]
        return np.asarray(intensities)
    
    def get_mean_spectrum(self):
        for region in self.f['/Regions']:
            if 'name' in self.f['/Regions/'+region].attrs.keys():
                if self.f['/Regions/'+region].attrs['name'] == self.spectra_group+':MeanSpectrum':
                    mean_spectrum_mzs=self.f['/Regions/'+region+'/SamplePositions/SamplePositions']
                    mean_spectrum_mzs=np.asarray(mean_spectrum_mzs)
                    mean_spectrum_intensities=self.f['/Regions/'+region+'/Intensities']
                    mean_spectrum_intensities=np.asarray(mean_spectrum_intensities)
        return mean_spectrum_mzs,mean_spectrum_intensities

    def get_ion_image(self,mz_list,tol_list,tol_type='mz',return_method='sum'):
        # todo - use tol_type and ensure tol is a vector
        # define mz ranges once
        mz_list_upper = np.zeros(np.shape(mz_list))
        mz_list_lower = np.zeros(np.shape(mz_list))
        for mm in range(0,len(mz_list)):
            mz_list_upper[mm] = mz_list[mm]+tol_list[mm]
            mz_list_lower[mm] = mz_list[mm]-tol_list[mm]
        # sum intensities
        # todo - implement alternative return_method (e.g. max, median sum)
        xic_array = np.zeros((len(self.index_list),len(mz_list)))
        if self.consistent_mz==True:
            print "consistent"
            this_spectrum = self.get_spectrum(0)
	    # precalculate which mzs should be made
            mz_index = np.zeros((len(this_spectrum.mzs),len(mz_list)),dtype=bool)
            for mm in range(0,len(mz_list)):
                mz_index[:,mm] = (this_spectrum.mzs<mz_list_upper[mm]) & (this_spectrum.mzs>mz_list_lower[mm])
            for ii in self.index_list:
                this_spectrum = self.get_spectrum(ii)
                for mm in range(0,len(mz_list)):
                   xic_array[ii,mm] = sum(this_spectrum.intensities[mz_index[:,mm]])
        else:
            print "inconsistent"
            for ii in self.index_list:
                this_spectrum = self.get_spectrum(ii)
                for mm in range(0,len(mz_list)):
                    mz_index = (this_spectrum.mzs<mz_list_upper[mm]) & (this_spectrum.mzs>mz_list_lower[mm])            
                    xic_array[ii,mm] = sum(this_spectrum.intensities[mz_index])
        data_out = ion_datacube()
        data_out.add_coords(self.coords)
        data_out.add_xic(xic_array,mz_list,tol_list)
        return data_out

    def load_to_memory(self):
        mzs=self.get_mzs(0)
        tol=mzs[1:]-mzs[0:-1]
        tol=np.append(tol,tol[-1])
        xic_array=np.zeros((len(self.spot_index),len(mzs)))
        for ii in self.spot_index:
            xic_array[int(ii),:]=self.get_intensities(ii)
        data_out = ion_datacube()
        data_out.add_coords(self.coords)
        data_out.add_xic(xic_array,mzs,tol)
        return data_out
