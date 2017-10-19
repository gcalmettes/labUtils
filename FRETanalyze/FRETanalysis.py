#!/usr/bin/env python
"""
Analyze FRET data contained in the generated excel file from Metafluor.
"""

# Major library imports
import numpy as np
import xlrd

class DataFRET:
    """
    Class to analyze FRET generated excel files.
    To analyze an excel file: dataname = DataFRET("filename.xls")
    """
    def __init__(self, name):
        self.name = name#+".xls"
#===========================================================
# # Define the xls file that contains the data
# # Use xlrd library to access data
#===========================================================
        workbook = xlrd.open_workbook(self.name)
        worksheet = workbook.sheet_by_index(0)
        self.nrows = worksheet.nrows
        self.ncols = worksheet.ncols
        self.cell = worksheet.cell
        self.nbROI = (self.ncols-1)/3
        self.ROIbckg = 1 # default: ROI number 1 is the background

#===========================================================
# # Organization of the raw data
#===========================================================

    def raw_dataM(self):
        """
        Create a numpy matrix of the excel file.
        """
        # Create a matrix of the size of the data
        raw_data = np.zeros((self.nrows, self.ncols))
        # Import value of the cells only if a number (integer or float)
        for rnum in range(self.nrows):
            for cnum in range(self.ncols):
                if isinstance(self.cell(rnum, cnum).value, (int, float)):
                    raw_data[rnum, cnum] = self.cell(rnum,cnum).value
                else:
                    raw_data[rnum, cnum] = np.nan
        return raw_data

    def raw_time(self):
        """
        Extract the time data
        """
        time_data = self.raw_dataM()[:, 0] # first column
        return time_data
    
    def raw_YFPdata(self):
        """
        View of the YFP raw data only.
        """
        YFP_data = self.raw_dataM()[:, 1:self.ncols:3] # every 3 columns starting from 2nd
        return YFP_data

    def raw_CFPdata(self):
        """
        View of the CFP raw data only.
        """
        CFP_data = self.raw_dataM()[:, 2:self.ncols:3] # every 3 columns starting from 2nd
        return CFP_data

    def raw_ratio(self):
        """
        View of the YFP/CFP ratio raw data only.
        """
        ratio_data = self.raw_YFPdata()/self.raw_CFPdata()
        return ratio_data

#===========================================================
# # Calcul of YFP and CFP signals (background soustracted)
# # Calcul of YFP/CFP ratios
#===========================================================
    def YFP_background(self):
        """
        View of the YFP background.
        """
        YFPbckg_data = self.raw_YFPdata()[:, self.ROIbckg-1]
        return YFPbckg_data

    def CFP_background(self):
        """
        View of the YFP background.
        """
        CFPbckg_data = self.raw_CFPdata()[:, self.ROIbckg-1]
        return CFPbckg_data

    def YFP_signal(self):
        """
        Create a numpy matrix of the YFP signal (background substracted).
        """
        YFP_signal = np.delete(self.raw_YFPdata(), self.ROIbckg-1, 1) - self.YFP_background()[:, np.newaxis]
        return YFP_signal
        
    def CFP_signal(self):
        """
        Create a numpy matrix of the CFP signal (background substracted).
        """
        CFP_signal = np.delete(self.raw_CFPdata(), self.ROIbckg-1, 1) - self.CFP_background()[:, np.newaxis]
        return CFP_signal

    def ratio_signal(self):
        """
        Create a numpy matrix of the YFP/CFP signal ratios.
        """
        ratio_signal = self.YFP_signal()/self.CFP_signal()
        return ratio_signal
