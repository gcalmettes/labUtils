# Author: Guillaume Calmettes
# University of Los Angeles California

import os
import sys
import re
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats




#############################################################################
# --------- DATA OBJECT CLASS ----------------------------------------------#
#############################################################################

class MSDataContainer:

  ##################
  ## Init and setup
  ##################

  def __init__(self, fileNames, internalRef="C19:0", nauralAbundanceCorrectionMethod="Control"):
    assert len(fileNames)==2 , "You must choose 2 files!"
    self.internalRef = internalRef
    self.naturalAbundanceCorrection = nauralAbundanceCorrectionMethod
    self.dataFileName, self.templateFileName = self.__getDataAndTemplateFileNames(fileNames)
    self.pathDirName = os.path.dirname(self.dataFileName)
    self.__regexExpression = {"NotLabeled": "([0-9]+)_([0-9]+)_([0-9]+)",
                              "Labeled": "([0-9]+)_([0-9]+).[0-9]+"}
    self.experimentType, self.dataColNames = self.__getExperimentTypeAndDataColumNames()
    self.dataDf = self.__getCleanedUpDataFrames()
    self.__standardDf_template = self.__getStandardsTemplateDf()
    self.volumeMixTotal = 500
    self.volumeMixForPrep = 100
    self.volumeSample = 100
    self.volumeStandards = [1, 5, 10, 20, 40, 80]

    self.standardDf_nMoles = self.computeStandardMoles()

    self.__isotopesToCorrect = {
        'hydrogen': False,
        'carbon':   True,
        'nitrogen': False,
        'oxygen':   True,
        'silicon':  True,
        'sulphur':  False}


  def __getDataAndTemplateFileNames(self, fileNames, templateKeyword="template"):
    '''Classify files (data or template) based on fileName'''
    dataFileName = [fileName for fileName in fileNames if templateKeyword not in fileName][0]
    templateFileName = [fileName for fileName in fileNames if fileName != dataFileName][0]
    return [dataFileName, templateFileName]

  def __getExperimentTypeAndDataColumNames(self):
    """Determine the type of experiment (Labeled/Unlabeled) based on column names"""
    colNames = pd.read_excel(self.dataFileName, nrows=2).filter(regex="[0-9]+[.|_][0-9]+[.|_][0-9]+").columns
    if len(re.findall(self.__regexExpression["NotLabeled"], colNames[0]))==1:
      experimentType = "Not Labeled"
    elif len(re.findall(self.__regexExpression["Labeled"], colNames[0]))==1:
      experimentType = "Labeled"
    else:
      raise Error("The experiment type could not be determined")
    return experimentType,colNames

  def __getCleanedUpDataFrames(self):
    '''Return a simplified pandas dataFrame'''
    
    # load data and template files and isolate data part
    df = pd.read_excel(self.dataFileName, skiprows=1)
    templateMap = pd.read_excel(self.templateFileName, sheet_name="MAP")

    df_Meta,df_Data = self.__getOrderedDfBasedOnTemplate(df, templateMap)
        
    if self.experimentType == "Not Labeled":
      regex = self.__regexExpression["NotLabeled"]
      self.dataColNames = [f"C{carbon[:2]}:{carbon[2:]} ({mass})" for name in self.dataColNames for num,carbon,mass in re.findall(regex, name)]
      df_Data.columns = self.dataColNames
    elif self.experimentType == "Labeled":
      regex = self.__regexExpression["Labeled"]
      ionAndMass = np.array([(int(num), int(mass)) for name in self.dataColNames for num,mass in re.findall(regex, name)])
      assert ionAndMass[0][1] == 242, "Data not starting with 14:0 fatty acid!"
      # get indices that would sort the array by ion
      order = np.argsort([ion for ion,mass in ionAndMass])
      # reorder the columns by ions
      df_Data = df_Data.iloc[:, order]
      FAMasses = ionAndMass[order, 1]
      # get main FA by subtracting ions weights and checking jumps in weights
      # for the same fatty acid, the next weight always increase by 1. or it's other fatty acid
      differences = FAMasses[1:]-FAMasses[:-1]
      idx = np.concatenate([[0], np.where(differences!=1)[0]+1])        
      # list of (idx, carbon, saturation)
      FAparent = [(idx[i], np.ceil((mass-242+14*14)/14).astype(int), [int((desat!=0)*(14-desat)/2) for desat in [(mass-242)%14]][0]) for i,mass in enumerate(FAMasses[idx])]
      self.dataColNames = [ f"C{carbon}:{sat} M.{ion}" for i,(idx,carbon,sat) in enumerate(FAparent[:-1]) for ion in range(idx-idx, FAparent[i+1][0]-idx)]
      # add last carbon ions
      self.dataColNames = self.dataColNames + [f"C{carbon}:{sat} M.{ion}" for ion in range(FAparent[-1][0]-FAparent[-1][0], len(FAMasses)-FAparent[-1][0]) for (carbon,sat) in [[FAparent[-1][1], FAparent[-1][2]]]]
      df_Data.columns = self.dataColNames

    # get sample meta info from template file
    df_TemplateInfo = self.__getExperimentMetaInfoFromMAP(templateMap)

    assert len(df_TemplateInfo)==len(df_Data), \
    f"The number of declared samples in the template (n={len(df_TemplateInfo)}) does not match the number of samples detected in the data file (n={len(df_Data)})"

    return pd.concat([df_Meta, df_TemplateInfo, df_Data], axis=1)

  def __getOrderedDfBasedOnTemplate(self, df, templateMap):
    '''Get new df_Data and df_Meta based on template'''

    # reorder rows based on template and reindex with range
    newOrder = list(map(lambda x: f"F{x.split('_')[1]}", templateMap.SampleID.values))[:len(df)]
    df.index=df["Name"]
    df = df.reindex(newOrder)
    df.index = list(range(len(df)))
    # df_Data["MSsample"] = templateMap["SampleID"]

    df_Meta = df[["Name", "Data File"]]
    df_Data = df.iloc[:, 7:] # 7 first cols are info
    return df_Meta, df_Data

  def __getExperimentMetaInfoFromMAP(self, templateMap):
    '''Return the meta info of the experiment'''
    # keep only rows with declared names
    declaredIdx = templateMap.SampleName.dropna().index
    templateMap = templateMap.loc[declaredIdx]
    # fill in missing weights with 1
    templateMap.loc[templateMap.SampleWeight.isna(), "SampleWeight"]=1
    return templateMap[["SampleID", "SampleName", "SampleWeight", "LabeledCode", "Comments"]]

  def __getStandardsTemplateDf(self, sheetKeyword="STANDARD"):
    sheetName = f"{sheetKeyword}_{'_'.join(self.experimentType.upper().split(' '))}"
    templateStandard = pd.read_excel(self.templateFileName, sheet_name=sheetName)
    return templateStandard

  def __makeResultFolder(self):
    directory = f"{self.pathDirName}/results"
    if not os.path.exists(directory):
      os.mkdir(directory)
    return directory

  def __getNaturalAbundanceDistributions(self):
    '''Return a dictionary of the isotopic proportions at natural abundance 
    desribed in https://www.ncbi.nlm.nih.gov/pubmed/27989585'''
    H1, H2 = 0.999885, 0.000115
    C12, C13 = 0.9893, 0.0107
    N14, N15 = 0.99632, 0.00368
    O16, O17, O18 = 0.99757, 0.00038, 0.00205
    Si28, Si29, Si30 = 0.922297, 0.046832, 0.030872
    S32, S33, S34, S36 = 0.9493, 0.0076, 0.0429, 0.0002

    return {'hydrogen': np.array([H1, H2]),
            'carbon':   np.array([C12, C13]),
            'nitrogen': np.array([N14, N15]),
            'oxygen':   np.array([O16, O17, O18]),
            'silicon':  np.array([Si28, Si29, Si30]),
            'sulphur':  np.array([S32, S33, S34, S36])}

  ##################
  ## Analysis and Updates
  ##################

  def updateInternalRef(self, newInternalRef):
    '''Update FAMES chosen as internal reference and normalize data to it'''
    print(f"Internal Reference changed from {self.internalRef} to {newInternalRef}")
    self.internalRef = newInternalRef
    self.dataDf_norm = self.computeNormalizedData()

  def updateStandards(self, volumeMixForPrep, volumeMixTotal, volumeStandards):
    self.volumeMixForPrep = volumeMixForPrep
    self.volumeMixTotal = volumeMixTotal
    self.volumeStandards = volumeStandards
    self.standardDf_nMoles = self.computeStandardMoles()

  def computeStandardMoles(self):
    '''Calculate nMoles for the standards'''
    template = self.__standardDf_template.copy()
    template["Conc in Master Mix (ug/ul)"] = template["Stock conc (ug/ul)"]*template["Weight (%)"]/100*self.volumeMixForPrep/self.volumeMixTotal
    # concentration of each carbon per standard volume
    for ul in self.volumeStandards:
      template[f"Std-Conc-{ul}"]=ul*(template["Conc in Master Mix (ug/ul)"]+template["Extra"])
    # nMol of each FAMES per standard vol
    for ul in self.volumeStandards:
      template[f"Std-nMol-{ul}"] = 1000*template[f"Std-Conc-{ul}"]/template["MW"]
    # create a clean template with only masses and carbon name
    templateClean = pd.concat([template.Chain, template.filter(regex="Std-nMol")], axis=1).transpose()
    templateClean.columns = list(map(lambda x: "C"+x, templateClean.iloc[0]))
    templateClean = templateClean.iloc[1:]
    return templateClean

  def getStandardAbsorbance(self):
    '''Get normalized absorbance data for standards'''
    return self.dataDf_norm.loc[self.dataDf_norm.SampleName.str.match('S[0-9]+')]


  def updateNaturalAbundanceCorrection(self, newMethod):
    self.naturalAbundanceCorrection = newMethod
    print(f"The method for Natural Abundance Correction has been updated to {newMethod}")
    self.correctForNaturalAbundance()


  def computeNormalizedData(self):
    '''Normalize the data to the internal ref'''
    dataDf_norm = self.dataDf.copy()
    dataDf_norm.iloc[:, 7:] = dataDf_norm.iloc[:, 7:].values/dataDf_norm[self.internalRef].values[:, np.newaxis]
    return dataDf_norm

  def correctForNaturalAbundance(self):
    print(f"{self.naturalAbundanceCorrection} executed")

  def saveStandardCurvesAndResults(self, useMask=False):
    # get current folder and create result folder if needed
    savePath = self.__makeResultFolder()
    
    # will store final results
    resultsDf = pd.DataFrame(index=self.dataDf_norm.index)

    # Plot of Standard
    stdAbsorbance = self.getStandardAbsorbance().filter(regex="C[0-9]")
    assert len(stdAbsorbance) == len(self.standardDf_nMoles),\
    f"The number of standards declared in the STANDARD_{'_'.join(self.experimentType.upper().split(' '))} sheet (n={len(self.standardDf_nMoles)}) is different than the number of standards declared in the data file (n={len(stdAbsorbance)})"
    
    nTotal = len(stdAbsorbance.columns)
    nCols = 4
    if nTotal%4==0:
      nRows = int(nTotal/nCols)
    else:
      nRows = int(np.floor(nTotal/nCols)+1)

    fig1,axes = plt.subplots(ncols=nCols, nrows=nRows, figsize=(20, nRows*4))

    if not useMask:
      self._maskFAMES = {}
      extension = ""
    else:
      extension = "_modified"

    for i,(col,ax) in enumerate(zip(stdAbsorbance.columns,  axes.ravel())):
      carbon = re.findall(r"(C\d+:\d+)", col)[0]
      if carbon in self.standardDf_nMoles.columns:
        # fit of data
        xvals = self.standardDf_nMoles[carbon].values
        yvals = stdAbsorbance[col].values
        # print(carbon)
        
        if not useMask:
          mask = [~np.logical_or(np.isnan(x), np.isnan(y)) for x,y in zip(xvals, yvals)]
          # add carbon to valid standard FAMES and save mask
          self._maskFAMES[col] = {"originalMask": mask}
        else:
          try:
            mask = self._maskFAMES[col]["newMask"]
          except:
            mask = self._maskFAMES[col]["originalMask"]

        slope,intercept = np.polyfit(np.array(xvals[mask], dtype=float), np.array(yvals[mask], dtype=float), 1)
        xfit = [np.min(xvals), np.max(xvals)]
        yfit = np.polyval([slope, intercept], xfit)
        # plot of data
        ax.plot(xvals[mask], yvals[mask], "o")
        ax.plot(xvals[[not i for i in mask]], yvals[[not i for i in mask]], "o", mfc="none", color="black", mew=2)
        ax.plot(xfit, yfit, "red")
        ax.text(ax.get_xlim()[0]+(ax.get_xlim()[1]-ax.get_xlim()[0])*0.05, ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.9, f"R2={stats.pearsonr(xvals[mask], yvals[mask])[0]**2:.4f}", size=14, color="purple")
        ax.text(ax.get_xlim()[0]+(ax.get_xlim()[1]-ax.get_xlim()[0])*0.97, ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.05, f"y={slope:.4f}x+{intercept:.4f}", size=14, color="red", ha="right")
        # calculate final results and save
        resultsDf[col] = ((self.dataDf_norm[col]-intercept)/slope)/self.dataDf_norm["SampleWeight"]
      ax.set_title(col)
      ax.set_xlabel("Quantity (nMoles)")
      ax.set_ylabel("Absorbance")

    fig1.tight_layout()

    # Plot of results on standard curves
    nTotal = len(resultsDf.filter(regex="C[0-9]").columns)
    nCols = 4
    if nTotal%4==0:
      nRows = int(nTotal/nCols)
    else:
      nRows = int(np.floor(nTotal/nCols)+1)

    fig2,axes = plt.subplots(ncols=nCols, nrows=nRows, figsize=(20, nRows*4))

    for i,(col,ax) in enumerate(zip(resultsDf.filter(regex="C[0-9]").columns,  axes.ravel())):
      carbon = re.findall(r"(C\d+:\d+)", col)[0]
      if carbon in self.standardDf_nMoles.columns:
        # fit of data
        xvals = self.standardDf_nMoles[carbon].values
        yvals = stdAbsorbance[col].values
        try:
          mask = self._maskFAMES[col]["newMask"]
        except:
          mask = self._maskFAMES[col]["originalMask"]
        slope,intercept = np.polyfit(np.array(xvals[mask], dtype=float), np.array(yvals[mask], dtype=float), 1)
        xfit = [np.min(xvals), np.max(xvals)]
        yfit = np.polyval([slope, intercept], xfit)
        # plot of data  
        ax.plot(xvals[mask], yvals[mask], "o")
        ax.plot(xvals[[not i for i in mask]], yvals[[not i for i in mask]], "x", color="black", ms=3)
        ax.plot(xfit, yfit, "red")
        # plot values calculated from curve
        ax.plot(resultsDf[col], self.dataDf_norm[col], "o", alpha=0.3)
      ax.set_title(col)
      ax.set_xlabel("Quantity (nMoles)")
      ax.set_ylabel("Absorbance")

    fig2.tight_layout()
    
    # Save data
    resultsDf["SampleID"]=self.dataDf_norm["SampleID"]
    resultsDf["SampleName"]=self.dataDf_norm["SampleName"]
    resultsDf["Comments"]=self.dataDf_norm["Comments"]
    resultsDf = resultsDf[np.concatenate([["SampleID", "SampleName", "Comments"], np.array(resultsDf.filter(regex="C[0-9]").columns)])]
        
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(f"{savePath}/results{extension}.xlsx", engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    resultsDf.to_excel(writer, sheet_name='Results', index=False)
    self.dataDf.to_excel(writer, sheet_name='Initial Data', index=False)
    self.dataDf_norm.to_excel(writer, sheet_name='Normalized Data', index=False)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

    fig1.savefig(f"{savePath}/standard-fit{extension}.pdf")
    fig2.savefig(f"{savePath}/standard-fit-with-data{extension}.pdf")
    
    print(f"The standard curves have been saved at {savePath}/standard-fit{extension}.pdf")
    print(f"The results calculated from the standard regression lines have been saved at {savePath}/standard-fit-with-data{extension}.pdf")
    print(f"The analysis results have been saved at {savePath}/results{extension}.xls")
    
    # close Matplotlib processes
    plt.close('all')




#############################################################################
# --------- GRAPHICAL USER INTERFACE ---------------------------------------#
#############################################################################

def initialFileChoser(directory=False):
  '''Temporary app window to get filenames before building main app'''
  if not directory:
    directory = os.getcwd()
  # Build a list of tuples for each file type the file dialog should display
  appFiletypes = [('excel files', '.xlsx'), ('all files', '.*')]
  # Main window
  appWindow = tk.Tk()
  appWindow.geometry("0x0") # hide the window
  appTitle = appWindow.title("Choose Files")
  # Ask the user to select a one or more file names.
  fileNames = filedialog.askopenfilenames(parent=appWindow,
                                          initialdir=directory,
                                          title="Please select the files:",
                                          filetypes=appFiletypes
                                          )
  appWindow.destroy() # close the app
  return fileNames


# Text widget that can call a callback function when modified
# see https://stackoverflow.com/questions/40617515/python-tkinter-text-modified-callback
class CustomText(tk.Text):
  def __init__(self, *args, **kwargs):
    """A text widget that report on internal widget commands"""
    tk.Text.__init__(self, *args, **kwargs)

    # create a proxy for the underlying widget
    self._orig = self._w + "_orig"
    self.tk.call("rename", self._w, self._orig)
    self.tk.createcommand(self._w, self._proxy)

  def _proxy(self, command, *args):
    cmd = (self._orig, command) + args
    result = self.tk.call(cmd)

    if command in ("insert", "delete", "replace"):
        self.event_generate("<<TextModified>>")

    return result


class MSAnalyzer:
  def __init__(self, dataObject):
    self.window = tk.Tk()
    self.window.title("MS Analyzer")
    self.dataObject = dataObject
    self.FANames = dataObject.dataColNames
    self.internalRef = dataObject.internalRef
    self.create_widgets()

  def create_widgets(self):
    # Create some room around all the internal frames
    self.window['padx'] = 5
    self.window['pady'] = 5

    # - - - - - - - - - - - - - - - - - - - - -
    # The FAMES frame (for internal control)
    FAMESframe = ttk.LabelFrame(self.window, text="Select the internal control", relief=tk.GROOVE)
    FAMESframe.grid(row=1, column=1, columnspan=3, sticky=tk.E + tk.W + tk.N + tk.S, padx=2)

    FAMESListLabel = tk.Label(FAMESframe, text="FAMES")
    FAMESListLabel.grid(row=2, column=1, sticky=tk.W + tk.N)

    # by default, choose internal reference defined in dataObject (C14:0)
    idxInternalRef = [i for i,name in enumerate(self.FANames) if self.dataObject.internalRef in name][0]

    self.FAMESLabelCurrent = tk.Label(FAMESframe, text=f"The current internal control is {self.FANames[idxInternalRef]}", fg="white", bg="#EBB0FF")
    self.FAMESLabelCurrent.grid(row=3, column=1, columnspan=2)

    self.FAMESListValue = tk.StringVar()
    self.FAMESListValue.trace('w', lambda index,value,op : self.__updateInternalRef(FAMESList.get()))
    FAMESList = ttk.Combobox(FAMESframe, height=6, textvariable=self.FAMESListValue)
    FAMESList.grid(row=2, column=2, columnspan=2)
    FAMESList['values'] = self.FANames
    FAMESList.current(idxInternalRef)

    # - - - - - - - - - - - - - - - - - - - - -
    # The standards frame (for fitting)
    Standardframe = ttk.LabelFrame(self.window, text="Standards", relief=tk.RIDGE)
    Standardframe.grid(row=4, column=1, columnspan=3, sticky=tk.E + tk.W + tk.N + tk.S, padx=2, pady=6)
    
    # variables declaration
    self.volTotalVar = tk.IntVar()
    self.volMixVar = tk.IntVar()
    self.volSampleVar = tk.IntVar()
    self.stdVols = self.dataObject.volumeStandards

    self.volTotalVar.set(self.dataObject.volumeMixTotal)
    self.volMixVar.set(self.dataObject.volumeMixForPrep)
    self.volSampleVar.set(self.dataObject.volumeSample)

    # Vol mix total
    self.volTotalVar.trace('w', lambda index,value,op : self.__updateVolumeMixTotal(self.volTotalVar.get()))
    volTotalSpinbox = tk.Spinbox(Standardframe, from_=0, to=1000, width=5, textvariable=self.volTotalVar, command= lambda: self.__updateVolumeMixTotal(self.volTotalVar.get()), justify=tk.RIGHT)
    volTotalSpinbox.grid(row=5, column=2, sticky=tk.W, pady=3)
    volTotalLabel = tk.Label(Standardframe, text="Vol. Mix Total")
    volTotalLabel.grid(row=5, column=1, sticky=tk.W)

    # Vol mix
    self.volMixVar.trace('w', lambda index,value,op : self.__updateVolumeMixForPrep(self.volMixVar.get()))
    volMixSpinbox = tk.Spinbox(Standardframe, from_=0, to=1000, width=5, textvariable=self.volMixVar, command= lambda: self.__updateVolumeMixForPrep(self.volMixVar.get()), justify=tk.RIGHT)
    volMixSpinbox.grid(row=6, column=2, sticky=tk.W, pady=3)
    volMixLabel = tk.Label(Standardframe, text="Vol. Mix")
    volMixLabel.grid(row=6, column=1, sticky=tk.W)

    # Vol sample
    self.volSampleVar.trace('w', lambda index,value,op : self.__updateVolumeSample(self.volSampleVar.get()))
    volSampleSpinbox = tk.Spinbox(Standardframe, from_=0, to=1000, width=5, textvariable=self.volSampleVar, command= lambda: self.__updateVolumeSample(self.volSampleVar.get()), justify=tk.RIGHT)
    volSampleSpinbox.grid(row=7, column=2, sticky=tk.W, pady=3)
    volSampleLabel = tk.Label(Standardframe, text="Vol. Sample")
    volSampleLabel.grid(row=7, column=1, sticky=tk.W)

    # Standards uL
    StandardVols = CustomText(Standardframe, height=7, width=15)
    StandardVols.grid(row=5, rowspan=3, column=3, padx=20)
    StandardVols.insert(tk.END, "Standards (ul)\n"+"".join([f"{vol}\n" for vol in self.stdVols]))
    StandardVols.bind("<<TextModified>>", self.__updateVolumeStandards)

    # Compute Results button
    StandardButton = ttk.Button(Standardframe, text="Compute results", command=lambda: self.computeResults())
    StandardButton.grid(row=8, column=2, columnspan=2, pady=5)

    # - - - - - - - - - - - - - - - - - - - - -
    # Quit button in the upper right corner
    quit_button = ttk.Button(self.window, text="Quit", command=self.window.destroy)
    quit_button.grid(row=1, column=4)

    # - - - - - - - - - - - - - - - - - - - - -
    # The Labeled Correction choice frame 
    Correctionframe = ttk.LabelFrame(self.window, text="Natural Abundance Correction", relief=tk.RIDGE)
    Correctionframe.grid(row=9, column=1, columnspan=3, sticky=tk.E + tk.W + tk.N + tk.S, padx=2, pady=6)

    self.radioCorrectionVariable = tk.StringVar()
    self.radioCorrectionVariable.set("Control")
    self.radioCorrectionVariable.trace('w', lambda index,value,op : self.__updateNaturalAbundanceCorrectionMethod(self.radioCorrectionVariable.get()))
    radiobutton1 = ttk.Radiobutton(Correctionframe, text="Control",
                                   variable=self.radioCorrectionVariable, value="Control")
    radiobutton2 = ttk.Radiobutton(Correctionframe, text="Theoretical",
                                   variable=self.radioCorrectionVariable, value="Theoretical")
    radiobutton1.grid(row=10, column=1, sticky=tk.W)
    radiobutton2.grid(row=10, column=2 , sticky=tk.W)
  

  def popupMsg(self, msg):
    '''Popup message window'''
    popup = tk.Tk()
    popup.wm_title("Look at the standard plots!")
    label = ttk.Label(popup, text=msg, font=("Verdana", 14))
    label.grid(row=1, column=1, columnspan=2, padx=10, pady=10)
      
    B1 = ttk.Button(popup, text="Yes", command = lambda: self.inspectPlots(popup))
    B1.grid(row=2, column=1, pady=10)

    def quitApp():
      popup.destroy()
      app.window.destroy()

    B2 = ttk.Button(popup, text="No", command = quitApp)
    B2.grid(row=2, column=2, pady=10)
    popup.mainloop()

  def __updateInternalRef(self, newInternalRef):
    '''Update FAMES chosen as internal reference'''
    self.dataObject.updateInternalRef(newInternalRef)
    self.FAMESLabelCurrent.config(text=f"The current internal control is {newInternalRef}")

  def __updateVolumeMixTotal(self, newVolumeMixTotal):
    self.dataObject.updateStandards(self.volMixVar.get(), newVolumeMixTotal, self.stdVols)
    print(f"The volumeMixTotal has been updated to {newVolumeMixTotal}")

  def __updateVolumeMixForPrep(self, newVolumeMixForPrep):
    self.dataObject.updateStandards(newVolumeMixForPrep, self.volTotalVar.get(), self.stdVols)
    print(f"The volumeMixForPrep has been updated to {newVolumeMixForPrep}")

  def __updateVolumeSample(self, newVolumeSample):
    self.dataObject.volumeSample = newVolumeSample
    print(f"The volumeSample has been updated to {newVolumeSample}")

  def __updateVolumeStandards(self, event):
    newStdVols = [float(vol) for vol in re.findall(r"(?<!\d)\d+(?!\d)", event.widget.get("1.0", "end-1c"))]
    self.stdVols = newStdVols
    self.dataObject.updateStandards(self.volMixVar.get(), self.volTotalVar.get(), newStdVols)
    print(f"The volumeStandards have been updated to {newStdVols}")

  def __updateNaturalAbundanceCorrectionMethod(self, newMethod):
    self.dataObject.updateNaturalAbundanceCorrection(newMethod)

  def computeResults(self):
    self.dataObject.saveStandardCurvesAndResults()
    self.popupMsg("The results and plots have been saved.\nCheck out the standard plots.\nDo you want to modify the standards?")

  def inspectPlots(self, popup):
    popup.destroy()
        
    selectFrame = tk.Tk()
    selectFrame.wm_title("FAMES standard to modify")
    label = ttk.Label(selectFrame, text="Select the FAMES standard to modify", font=("Verdana", 14))
    label.grid(row=1, column=1, columnspan=2, padx=10, pady=10)

    FAMESlistbox = tk.Listbox(selectFrame, height=10, selectmode='multiple')
    for item in self.dataObject._maskFAMES.keys():
      FAMESlistbox.insert(tk.END, item)
    FAMESlistbox.grid(row=2, column=1, columnspan=2, pady=10)

    FAMESbutton = ttk.Button(selectFrame, text="Select", command = lambda: self.modifySelection(selectFrame, FAMESlistbox))
    FAMESbutton.grid(row=3, column=1, columnspan=2, pady=10)

  def modifySelection(self, frameToKill, selection):
    FAMESselected = [selection.get(i) for i in selection.curselection()]

    # will go over all the selectedFAMES
    currentFAMESidx = 0

    fig,ax = plt.subplots(figsize=(4,3))
    ax.set_xlabel("Quantity (nMoles)")
    ax.set_ylabel("Absorbance")
    fig.tight_layout()

    plotFrame = tk.Tk()
    plotFrame.wm_title("Standard curve inspector")

    pointsListbox = tk.Listbox(plotFrame, height=8, selectmode='multiple')
    pointsListbox.grid(row=1, column=3, columnspan=2, pady=10, padx=5)

    canvas = FigureCanvasTkAgg(fig, plotFrame)

    self.plotIsolatedFAMES(FAMESselected[currentFAMESidx], ax, canvas, pointsListbox)

    figFrame = canvas.get_tk_widget()#.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    figFrame.grid(row=1, column=1, columnspan=2, rowspan=3, pady=10, padx=10)

    plotButton = ttk.Button(plotFrame, text="Remove", command = lambda: self.plotIsolatedFAMES(FAMESselected[currentFAMESidx], ax, canvas, pointsListbox))
    plotButton.grid(row=2, column=3, pady=5)

    def goToNextPlot():
      nonlocal currentFAMESidx
      if currentFAMESidx+1<len(FAMESselected):
        currentFAMESidx = currentFAMESidx+1
        self.plotIsolatedFAMES(FAMESselected[currentFAMESidx], ax, canvas, pointsListbox)
        if currentFAMESidx == len(FAMESselected)-1:
          plotButton2["text"]="Send"
      else:
        plt.close('all')
        plotFrame.destroy()
        self.dataObject.saveStandardCurvesAndResults(useMask=True)
        self.popupMsg("The results and plots have been saved.\nCheck out the standard plots.\nDo you want to modify the standards?")

    if currentFAMESidx == len(FAMESselected)-1:
      textButton = "Send"
    else:
      textButton = "Next"
    plotButton2 = ttk.Button(plotFrame, text=textButton, command = lambda: goToNextPlot())
    plotButton2.grid(row=2, column=4, pady=5)

    def quitCurrent():
      plt.close('all')
      plotFrame.destroy()

    plotButton3 = ttk.Button(plotFrame, text="Quit Inspector", command = quitCurrent)
    plotButton3.grid(row=3, column=3, columnspan=2, pady=5)

    frameToKill.destroy()


  def plotIsolatedFAMES(self, famesName, ax, canvas, pointsListbox):
    ax.clear()

    ax.set_xlabel("Quantity (nMoles)")
    ax.set_ylabel("Absorbance")

    carbon = re.findall(r"(C\d+:\d+)", famesName)[0]
    maskSelected = [not (i in pointsListbox.curselection()) for i in range(len(self.dataObject.standardDf_nMoles[carbon].values))]
    newMask = [(m1 & m2) for m1,m2 in zip(self.dataObject._maskFAMES[famesName]["originalMask"], maskSelected)]
    self.dataObject._maskFAMES[famesName]["newMask"] = newMask
    
    xvals = self.dataObject.standardDf_nMoles[carbon].values
    yvals = self.dataObject.getStandardAbsorbance()[famesName].values

    pointsListbox.delete(0, tk.END)
    for i,(x,y) in enumerate(zip(xvals, yvals)):
      pointsListbox.insert(tk.END, f" point{i}: ({x:.3f}, {y:.3f})")

    ax.plot(xvals[newMask], yvals[newMask], "o")
    ax.plot(xvals[[not i for i in newMask]], yvals[[not i for i in newMask]], "or")
    slope,intercept = np.polyfit(np.array(xvals[newMask], dtype=float), np.array(yvals[newMask], dtype=float), 1)
    xfit = [np.min(xvals), np.max(xvals)]
    yfit = np.polyval([slope, intercept], xfit)
    # plot of data
    ax.plot(xfit, yfit, "purple")
    ax.set_title(famesName)

    canvas.draw()


############################
## MAIN
############################

if __name__ == '__main__':

  # Is the directory to look in for data files defined?
  if len(sys.argv) == 1: # no arguments given to the function
    initialDirectory = False
  else:
    initialDirectory = sys.argv[1]

  # Choose data and template files
  fileNames = initialFileChoser(initialDirectory)


  # The container that will hold all the data
  appData = MSDataContainer(fileNames)

  print(f"""Two files have been loaded:
    \tData file: {appData.dataFileName}
    \tTemplate file: {appData.templateFileName}""")
  print(f"The experiment type detected is '{appData.experimentType}'")


  # Create the entire GUI program and pass in colNames for popup menu
  app = MSAnalyzer(appData)

  # Start the GUI event loop
  app.window.mainloop()


  # print(appData.volumeMixTotal, appData.volumeMixForPrep, appData.volumeSample)
  # print(appData.volumeStandards)

  appData.dataDf.to_excel("test.xlsx")


