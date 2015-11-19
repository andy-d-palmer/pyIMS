
def h5(filename_in, filename_out,info,smoothMethod="nosmooth"):
    import h5py
    import numpy as np
    import datetime
    import scipy.signal as signal
    from pyMS import centroid_detection
    import sys
    #from IPython.display import display, clear_output
    def nosmooth(mzs,intensities):
        return mzs,intensities

    def sg_smooth(mzs,intensities,n_smooth=1):
        for n in range(0,n_smooth):
            intensities = signal.savgol_filter(intensities,5,2)
        intensities[intensities<0]=0
        return mzs,intensities

    def apodization(mzs,intensities,w_size=10):
        win = signal.hann(w_size)
        win = signal.hann(w_size)
        intensities = signal.fftconvolve(intensities, win, mode='same') / sum(win)
        intensities[intensities<1e5]=0
        return mzs,intensities
    ### Open files
    f_in = h5py.File(filename_in, 'r')  # Readonly, file must exist
    f_out = h5py.File(filename_out, 'w')  # create file, truncate if exists
    print filename_in
    print filename_out
    ### get root groups from input data
    root_group_names = f_in.keys()
    spots = f_in['Spots']
    file_version = f_in['Version'][0]
    # some hard-coding to deal with different file versions
    if file_version > 5:
        coords = f_in['Registrations']['0']['Coordinates']
    else:
        coords = f_in['Coordinates']
    spectraGroup = 'InitialMeasurement'
    Mzs = np.asarray(f_in['/SamplePositions/GlobalMassAxis/']['SamplePositions']) # we don't write this but will use it for peak detection

    ### make root groups for output data
    spectral_data = f_out.create_group('spectral_data')
    spatial_data = f_out.create_group('spatial_data')
    shared_data = f_out.create_group('shared_data')

    ### populate common variables - can hardcode as I know what these are for h5 data
    # parameters
    instrument_parameters_1 = shared_data.create_group('instrument_parameters/001')
    instrument_parameters_1.attrs['instrument name'] = 'Bruker Solarix 7T'
    instrument_parameters_1.attrs['mass range'] = [Mzs[0],Mzs[-1]]
    instrument_parameters_1.attrs['analyser type'] = 'FTICR'
    instrument_parameters_1.attrs['smothing during convertion'] = smoothMethod
    instrument_parameters_1.attrs['data conversion'] = 'h5->hdf5:'+str(datetime.datetime.now())
    # ROIs
        #todo - determine and propagate all ROIs
    sample_1 = shared_data.create_group('samples/001')
    sample_1.attrs['name'] = info["sample_name"]
    sample_1.attrs['source'] = info["sample_source"]
    sample_1.attrs['preparation'] = info["sample_preparation"]
    sample_1.attrs['MALDI matrix'] = info["maldi_matrix"]
    sample_1.attrs['MALDI matrix application'] = info["matrix_application"]
    ### write spectra
    n = 0
    for key in spots.keys():
        spot = spots[key]
        ## make new spectrum
        #mzs,intensities = nosmooth(Mzs,np.asarray(spot[spectraGroup]['Intensities']))
        if smoothMethod == 'nosmooth':
            mzs,intensities = mzs,intensities = nosmooth(Mzs,np.asarray(spot[spectraGroup]['Intensities']))
        elif smoothMethod == 'nosmooth':
            mzs,intensities = sg_smooth(Mzs,np.asarray(spot[spectraGroup]['Intensities']))
        elif smoothMethod == 'apodization':
            mzs,intensities = apodization(Mzs,np.asarray(spot[spectraGroup]['Intensities']))
        else:
            raise ValueError('smooth method not one of: [nosmooth,nosmooth,apodization]')
        mzs_list, intensity_list, indices_list = centroid_detection.gradient(mzs,intensities, max_output=-1, weighted_bins=3)

        # add intensities
        this_spectrum = spectral_data.create_group(key)
        this_intensities = this_spectrum.create_dataset('centroid_intensities', data=np.float32(intensity_list),
                                                    compression="gzip", compression_opts=9)
        # add coordinates
        key_dbl = float(key)
        this_coordiantes = this_spectrum.create_dataset('coordinates',
                                                    data=(coords[0, key_dbl], coords[1, key_dbl], coords[2, key_dbl]))
        ## link to shared parameters
        # mzs
        this_mzs = this_spectrum.create_dataset('centroid_mzs', data=np.float32(mzs_list), compression="gzip",
                                            compression_opts=9)
        # ROI
        this_spectrum['ROIs/001'] = h5py.SoftLink('/shared_data/regions_of_interest/001')
        # Sample
        this_spectrum['samples/001'] = h5py.SoftLink('/shared_data/samples/001')
        # Instrument config
        this_spectrum['instrument_parameters'] = h5py.SoftLink('/shared_data/instrument_parameters/001')
        n += 1
        if np.mod(n, 10) == 0:
            #clear_output(wait=True)
            print('{:3.2f}\% complete\r'.format(100.*n/np.shape(spots.keys())[0], end="\r")),
            sys.stdout.flush()

    f_in.close()
    f_out.close()
    print 'fin'

if __name__ == '__main__':
    h5(sys.argv[1], sys.argv[1][:-3] + ".hdf5")