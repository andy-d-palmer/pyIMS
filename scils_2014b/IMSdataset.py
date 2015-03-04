class IMS_dataset():
    def __init__(self,filepath):
        self.f = h5py.File(filepath)
        self.spectra_group='InitialMeasurement'
        self.spot_index=self.f['Spots'].keys()
        self.coords=self.f['/Registrations/0/Coordinates']
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
    def xic_to_image(self,im_vect):
        im=np.zeros((max(self.coords[:,1]-min))
        for xx,yy in self.coords:
    def load_to_memory(self):
        mzs=self.get_mzs(0)
        data_array=np.zeros((len(self.spot_index),len(mzs)))
        for ii in self.spot_index:
            data_array[int(ii),:]=self.get_intensities(ii)
        return mzs,data_array
