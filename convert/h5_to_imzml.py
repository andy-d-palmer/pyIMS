import h5py
import sys
import numpy as np

from pyImzML.pyimzml.imzmlwriter import ImzMLWriter

def centroidh5(input_filename, output_filename):
    from pyMS.centroid_detection import gradient
    h5 = h5py.File(input_filename)
    g = h5['Spots/0/InitialMeasurement/']
    mz_dtype = g['SamplePositions/SamplePositions'][:].dtype
    int_dtype = g['Intensities'][:].dtype
    with ImzMLWriter(output_filename, mz_dtype=mz_dtype, intensity_dtype=int_dtype) as imzml:
        coords = np.asarray(h5['Registrations/0/Coordinates']).T.round(5)
        coords -= np.amin(coords, axis=0)
        step = np.array([np.mean(np.diff(np.unique(coords[:, i]))) for i in range(3)])
        step[np.isnan(step)] = 1
        coords /= np.reshape(step, (3,))
        coords = coords.round().astype(int)
        ncol, nrow, _ = np.amax(coords, axis=0) + 1
        print 'dim: {} x {}'.format(nrow,ncol)
        n_total = len(h5['Spots'].keys())
        done = 0
        keys = map(str, sorted(map(int, h5['Spots'].keys())))
        for index, pos in zip(keys, coords):
            g = h5['Spots/' + index + '/InitialMeasurement/']
            mzs = g['SamplePositions/SamplePositions'][:]
            intensities = g['Intensities'][:]
            mzs_c, intensities_c, _ = gradient(mzs, intensities)
            pos = (nrow - 1 - pos[1], pos[0], pos[2])
            imzml.addSpectrum(mzs_c, intensities_c, pos)
            done += 1
            if done % 1000 == 0:
                print "[%s] progress: %.1f%%" % (input_filename, float(done) * 100.0 / n_total)
        print "finished!"

def hdf5_centroids_IMS(input_filename, output_filename):
    # Convert hdf5 to imzML
    imsDataset = inMemoryIMS(input_filename,cache_spectra=False,do_summary=False)
    coords = imsDataset.coords
    coords -= np.amin(coords, axis=0)
    step = []
    for i in range(3):
        u = list(np.diff(np.unique(coords[:, i])))
        if u == []:
            step.append(1)
        else:
            step.append(np.median(u))
    step = np.asarray(step)
    coords /= np.reshape(step, (3,))
    coords = coords.round().astype(int)
    ncol, nrow, _ = np.amax(coords, axis=0) + 1
    print 'dim: {} x {}'.format(nrow,ncol)
    mz_dtype = imsDataset.get_spectrum(0).get_spectrum(source="centroids")[0].dtype
    int_dtype = imsDataset.get_spectrum(0).get_spectrum(source="centroids")[1].dtype
    n_total = len(imsDataset.index_list)
    with ImzMLWriter(output_filename, mz_dtype=mz_dtype, intensity_dtype=int_dtype) as imzml:
        done = 0
        for index in imsDataset.index_list:
            this_spectrum = imsDataset.get_spectrum(index)
            mzs,intensities = this_spectrum.get_spectrum(source='centroids')
            pos = coords[index]
            pos = (nrow - 1 - pos[0], pos[1], pos[2])
            imzml.addSpectrum(mzs, intensities, pos)
            done += 1
            if done % 1000 == 0:
                print "[%s] progress: %.1f%%" % (input_filename, float(done) * 100.0 / n_total)
        print "finished!"

if __name__ == '__main__':
    centroidh5(sys.argv[1], sys.argv[1][:-3] + ".imzML")
