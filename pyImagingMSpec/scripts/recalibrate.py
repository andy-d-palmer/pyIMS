import argparse
import numpy as np
from pyimzml.ImzMLWriter import ImzMLWriter
from pyImagingMSpec.inMemoryIMS import inMemoryIMS

def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or abs(value - array[idx-1]) < abs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def calculate_mass_deviation(input_filename, known_peaks):
    ims_dataset = inMemoryIMS(input_filename)
    delta = np.zeros([len(ims_dataset.coords), len(known_peaks)])
    for ii in range(len(ims_dataset.coords)):
        spec = ims_dataset.get_spectrum(ii).get_spectrum(source='centroids')
        for jj, peak in enumerate(known_peaks):
            nearest_peak = find_nearest(spec[0], peak)
            delta[ii,jj] = 1e6*(nearest_peak-peak)/peak
    return delta

def poly_from_deltas(known_mzs, delta, max_ppm=100, polyorder=3):
    f = lambda x: np.median(x[np.abs(x) < max_ppm])
    median_delta = [f(delta[:, ii]) for ii in range(len(known_mzs))]
    z = np.polyfit(known_mzs, median_delta, polyorder)
    p = np.poly1d(z)
    return p

def do_recalibration(input_filename, output_filename, p):
    ims_dataset = inMemoryIMS(input_filename)
    with ImzMLWriter(output_filename) as file_out:
        for ii in range(len(ims_dataset.coords)):
            spec = ims_dataset.get_spectrum(ii).get_spectrum(source='centroids')
            mzs = spec[0]
            mzs_recal = [m-(1e-6)*m*p(m) for m in mzs]
            file_out.addSpectrum(mzs_recal, spec[1], ims_dataset.coords[ii,[1,0,2]])

def recalibrate_dataset(input_filename, output_filename, known_peaks):
    deltas = calculate_mass_deviation(input_filename, known_peaks)
    p = poly_from_deltas(known_peaks, deltas)
    do_recalibration(input_filename, output_filename, p)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="recalibrate centroided imaging MS file")
    parser.add_argument('input', type=str, help="input imaging MS file")
    parser.add_argument('output', type=str, help="output filename")
    parser.add_argument('known_peaks', metavar='N', type=float, nargs='+',
                        help='an integer for the accumulator')
    args = parser.parse_args()

    recalibrate_dataset(args.input, args.output, args.known_peaks)
