#!/usr/bin/env python

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Langendorff:
    """
    Class to analyze atf generated files from a PClamp isolated heart experiment.
    Usage fo the class to analyze a file: data = Langendorff("filename.atf")
    """

    #===========================================================
    # # General functions
    #===========================================================
    def _datacheck_peakdetect(self, x_axis, y_axis):
        if x_axis is None:
            x_axis = range(len(y_axis))

        if len(y_axis) != len(x_axis):
            raise (ValueError,
                    'Input vectors y_axis and x_axis must have same length')

        #needs to be a numpy array
        y_axis = np.array(y_axis)
        x_axis = np.array(x_axis)
        return x_axis, y_axis

    def _peakdetect(self, y_axis, x_axis = None, lookahead = 300, delta=5):
        """
        Converted from/based on a MATLAB script at:
        http://billauer.co.il/peakdet.html

        function for detecting local maximas and minmias in a signal.
        Discovers peaks by searching for values which are surrounded by lower
        or larger values for maximas and minimas respectively

        keyword arguments:
        y_axis -- A list containg the signal over which to find peaks
        x_axis -- (optional) A x-axis whose values correspond to the y_axis list
            and is used in the return to specify the postion of the peaks. If
            omitted an index of the y_axis is used. (default: None)
        lookahead -- (optional) distance to look ahead from a peak candidate to
            determine if it is the actual peak (default: 200)
            '(sample / period) / f' where '4 >= f >= 1.25' might be a good value
        delta -- (optional) this specifies a minimum difference between a peak and
            the following points, before a peak may be considered a peak. Useful
            to hinder the function from picking up false peaks towards to end of
            the signal. To work well delta should be set to delta >= RMSnoise * 5.
            (default: 0)
                delta function causes a 20% decrease in speed, when omitted
                Correctly used it can double the speed of the function

        return -- two lists [max_peaks, min_peaks] containing the positive and
            negative peaks respectively. Each cell of the lists contains a tupple
            of: (position, peak_value)
            to get the average peak value do: np.mean(max_peaks, 0)[1] on the
            results to unpack one of the lists into x, y coordinates do:
            x, y = zip(*tab)
        """
        max_peaks = []
        min_peaks = []
        dump = []   #Used to pop the first hit which almost always is false

        # check input data
        x_axis, y_axis = self._datacheck_peakdetect(x_axis, y_axis)
        # store data length for later use
        length = len(y_axis)


        #perform some checks
        if lookahead < 1:
            raise ValueError("Lookahead must be '1' or above in value")
        if not (np.isscalar(delta) and delta >= 0):
            raise ValueError("delta must be a positive number")

        #maxima and minima candidates are temporarily stored in
        #mx and mn respectively
        mn, mx = np.Inf, -np.Inf

        #Only detect peak if there is 'lookahead' amount of points after it
        for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
                                            y_axis[:-lookahead])):
            if y > mx:
                mx = y
                mxpos = x
            if y < mn:
                mn = y
                mnpos = x

            ####look for max####
            if y < mx-delta and mx != np.Inf:
                #Maxima peak candidate found
                #look ahead in signal to ensure that this is a peak and not jitter
                if y_axis[index:index+lookahead].max() < mx:
                    max_peaks.append([mxpos, mx])
                    dump.append(True)
                    #set algorithm to only find minima now
                    mx = np.Inf
                    mn = np.Inf
                    if index+lookahead >= length:
                        #end is within lookahead no more peaks can be found
                        break
                    continue
                #else:  #slows shit down this does
                #    mx = ahead
                #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]

            ####look for min####
            if y > mn+delta and mn != -np.Inf:
                #Minima peak candidate found
                #look ahead in signal to ensure that this is a peak and not jitter
                if y_axis[index:index+lookahead].min() > mn:
                    min_peaks.append([mnpos, mn])
                    dump.append(False)
                    #set algorithm to only find maxima now
                    mn = -np.Inf
                    mx = -np.Inf
                    if index+lookahead >= length:
                        #end is within lookahead no more peaks can be found
                        break
                #else:  #slows shit down this does
                #    mn = ahead
                #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]


        #Remove the false hit on the first value of the y_axis
        try:
            if dump[0]:
                max_peaks.pop(0)
            else:
                min_peaks.pop(0)
            del dump
        except IndexError:
            #no peaks were found, should the function return empty lists?
            pass

        return [max_peaks, min_peaks]

    def _movingaverage(self, signal, window_size): #calculate the average of a signal
        window= np.ones(int(window_size))/float(window_size)
        return np.convolve(signal, window, 'same')


    #===========================================================
    # # Inner functions
    #===========================================================
    def __init__(self, experiment_file):
        self.name = experiment_file
        self.data = pd.read_table(experiment_file, skiprows=9)
        if len(self.data.columns)==2:
            self.data.columns = ['time', 'trace']
        else:
            self.data.columns = ['time', 'ekg', 'trace']
        self.sampling_rate = 1/(self.data.time[1]-self.data.time[0])



    def extract_contraction_peaks(self, lookahead=100, delta=0.1,
            sample_average=50, normalization=[600, 1200]):
        #Peak detection
        maxtab, mintab = self._peakdetect(self.data.trace, self.data.time, lookahead=lookahead, delta=delta)
        maxtab = np.array(maxtab)
        mintab = np.array(mintab)
        #periodmax = maxtab[1:, 0]-maxtab[:-1, 0]
        #periodmin = mintab[1:, 0]-mintab[:-1, 0]

        max_min_dict = dict(peakmax_time = maxtab[:, 0],
                            peakmax_val = maxtab[:,1],
                            peakmax_period = maxtab[1:, 0]-maxtab[:-1, 0],
                            peakmin_time = mintab[:, 0],
                            peakmin_val = mintab[:,1],
                            peakmin_period = mintab[1:, 0]-mintab[:-1, 0]
                            )

        self.detected_peaks = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in max_min_dict.items()]))
        #lvp = left ventricular developed pressure (in mV)
        self.lvp = pd.DataFrame({'val': self.detected_peaks.peakmax_val-self.detected_peaks.peakmin_val,
                                 'time': self.detected_peaks.peakmax_time})
        self.lvp['val_avg'] = self._movingaverage(self.lvp.val, sample_average)
        #hr = heart rate
        self.hr = pd.DataFrame({'val': 60*1/self.detected_peaks.peakmax_period,
                                'time': self.detected_peaks.peakmax_time})
        self.hr['val_avg'] = self._movingaverage(self.hr.val, sample_average)
        #rpp = rate pressure product
        self.rpp = pd.DataFrame({'val': self.lvp.val*self.hr.val,
                                 'val_avg': self.lvp.val_avg*self.hr.val_avg,
                                 'time': self.detected_peaks.peakmax_time})
        #normalization over the last 10min of the stabilization period (for a stabilization period of 20min)
        self.rpp_normalized = pd.DataFrame({'val': self.rpp.val/self.rpp[(self.rpp.time>normalization[0]) & (self.rpp.time<normalization[1])].val.mean(),
                                            'val_avg': self.rpp.val_avg/self.rpp[(self.rpp.time>normalization[0]) & (self.rpp.time<normalization[1])].val_avg.mean(),
                                            'time': self.detected_peaks.peakmax_time})

        #return self.detected_peaks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to the file to analyze")
    parser.add_argument("-p", "--plot", help="make graph of raw signal, LVP, HR and RPP and save it", action="store_true")
    parser.add_argument("-n", "--normalize", type=int, help="define time (sec) in between to normalize RPP", nargs='*', default=[600, 1200])
    parser.add_argument("-s", "--csv", help="save the time and average RPP in a csv file", action="store_true")

    args = parser.parse_args()

    time_norm_start = args.normalize[0]
    time_norm_end = args.normalize[1]

    data = Langendorff(args.path)
    data.extract_contraction_peaks(delta=0.01, normalization=[time_norm_start, time_norm_end])

    if args.plot:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(14, 16), sharex=True)
        ax1.set_ylabel('Signal (mV))')
        ax2.set_ylabel('LVP (mV)')
        ax3.set_ylabel('Heart Rate (bpm)')
        ax4.set_ylabel('RPP normalized')
        ax4.set_xlabel('Time (sec)')
        ax4.set_ylim(0, 1.5)

        #plot data
        ax1.plot(data.data.time, data.data.trace, 'k')
        ax2.plot(data.lvp.time, data.lvp.val_avg, 'k')
        ax3.plot(data.hr.time, data.hr.val_avg, 'k')
        ax4.plot(data.rpp_normalized.time, data.rpp_normalized.val_avg, 'k')

        #plt.tight_layout()
        plt.savefig(sys.argv[1].replace('.atf', '_fig.png'))

    if args.csv:
        data.rpp_normalized[["time","val_avg"]].to_csv(
                sys.argv[1].replace('.atf', '.csv'), index=False)
