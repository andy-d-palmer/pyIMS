import h5py
import sys
import numpy as np

class slFile():
    def __init__(self,input_filename):
        self.load_file(input_filename)

    def load_file(self,input_filename):
        # get a handle on the file
        self.sl = h5py.File(input_filename,'r')     #Readonly, file must exist
        ### get root groups from input data
        self.initialMeasurement = self.sl['SpectraGroups']['InitialMeasurement']
        self.file_version = self.sl['Version'][0]
        # some hard-coding to deal with different file versions
        if self.file_version in [16,17,]:
            self.coords = np.asarray(self.sl['Registrations']['0']['Coordinates'])
        else:
            raise ValueError('File version {} out of range'.format(self.file_version))
        self.Mzs = np.asarray(self.initialMeasurement['SamplePositions']['SamplePositions']) # we don't write this but will use it for peak detection
        self.spectra = self.initialMeasurement['spectra']


    def get_spectrum(self,index):
        intensities = np.asarray(self.spectra[index,:])
        return self.Mzs, intensities


def centroid_imzml(input_filename, output_filename, step=[], apodization=False, w_size=10, min_intensity=1e-5):
    # write a file to imzml format (centroided)
    """
    :type min_intensity: float
    """
    from pyimzml.ImzMLWriter import ImzMLWriter
    from pyMS.centroid_detection import gradient
    sl = slFile(input_filename)
    mz_dtype = sl.Mzs.dtype
    int_dtype = sl.get_spectrum(0)[1].dtype
    # Convert coords to index -> kinda hacky
    coords = np.asarray(sl.coords).T.round(5)
    coords -= np.amin(coords, axis=0)
    if step==[]: #have a guesss
        step = np.array([np.mean(np.diff(np.unique(coords[:, i]))) for i in range(3)])
        step[np.isnan(step)] = 1
    coords /= np.reshape(step, (3,))
    coords = coords.round().astype(int)
    ncol, nrow, _ = np.amax(coords, axis=0) + 1
    print 'dim: {} x {}'.format(nrow,ncol)
    n_total = np.shape(sl.spectra)[0]
    with ImzMLWriter(output_filename, mz_dtype=mz_dtype, intensity_dtype=int_dtype) as imzml:
        done = 0
        for key in range(0,n_total):
            mzs,intensities = sl.get_spectrum(key)
            if apodization:
                import scipy.signal as signal
                #todo - add to processing list in imzml
                win = signal.hann(w_size)
                intensities = signal.fftconvolve(intensities, win, mode='same') / sum(win)
            mzs_c, intensities_c, _ = gradient(mzs, intensities, min_intensity=min_intensity)
            pos = coords[key]
            pos = (nrow - 1 - pos[1], pos[0], pos[2])
            imzml.addSpectrum(mzs_c, intensities_c, pos)
            done += 1
            if done % 1000 == 0:
                print "[%s] progress: %.1f%%" % (input_filename, float(done) * 100.0 / n_total)
        print "finished!"


def centroid_IMS(input_filename, output_filename,instrumentInfo={},sharedDataInfo={}):
    from pyMS.centroid_detection import gradient
    # write out a IMS_centroid.hdf5 file
    sl = slFile(input_filename)
    n_total = np.shape(sl.spectra)[0]
    with h5py.File(output_filename,'w') as f_out:
        ### make root groups for output data
        spectral_data = f_out.create_group('spectral_data')
        spatial_data = f_out.create_group('spatial_data')
        shared_data = f_out.create_group('shared_data')

        ### populate common variables - can hardcode as I know what these are for h5 data
        # parameters
        instrument_parameters_1 = shared_data.create_group('instrument_parameters/001')
        if instrumentInfo != {}:
            for tag in instrumentInfo:
                instrument_parameters_1.attrs[tag] = instrumentInfo[tag]
        # ROIs
            #todo - determine and propagate all ROIs
        roi_1 = shared_data.create_group('regions_of_interest/001')
        roi_1.attrs['name'] = 'root region'
        roi_1.attrs['parent'] = ''
        # Sample
        sample_1 = shared_data.create_group('samples/001')
        if sharedDataInfo != {}:
            for tag in sharedDataInfo:
                sample_1.attrs[tag] = sharedDataInfo[tag]

        done = 0
        for key in range(0,n_total):
            mzs,intensities = sl.get_spectrum(key)
            mzs_c, intensities_c, _ = gradient(mzs, intensities)
            this_spectrum = spectral_data.create_group(str(key))
            _ = this_spectrum.create_dataset('centroid_mzs',data=np.float32(mzs_c),compression="gzip",compression_opts=9)
            # intensities
            _ = this_spectrum.create_dataset('centroid_intensities',data=np.float32(intensities_c),compression="gzip",compression_opts=9)
            # coordinates
            _ = this_spectrum.create_dataset('coordinates',data=(sl.coords[0, key],sl.coords[1, key],sl.coords[2, key]))
            ## link to shared parameters
            # ROI
            this_spectrum['ROIs/001'] = h5py.SoftLink('/shared_data/regions_of_interest/001')
            # Sample
            this_spectrum['samples/001'] = h5py.SoftLink('/shared_data/samples/001')
            # Instrument config
            this_spectrum['instrument_parameters'] = h5py.SoftLink('/shared_data/instrument_parameters/001')
            done += 1
            if done % 1000 == 0:
                print "[%s] progress: %.1f%%" % (input_filename, float(done) * 100.0 / n_total)
        print "finished!"

if __name__ == '__main__':
    centroid_imzml(sys.argv[1], sys.argv[1][:-3] + ".imzML")
