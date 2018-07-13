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
from scipy import optimize
from multiprocessing import Pool

#############################################################################
# --------- Natural Abundance Correction CLASS -----------------------------#
#############################################################################

class NAProcess:
  # Adapted from IsoCor code (https://github.com/MetaSys-LISBP/IsoCor)
  
  ##################
  ## Init and setup
  ##################

  def __init__(self, entry, atomTracer, purityTracer=[0, 1], FAMES=True):
    self.NaturalAbundanceDistributions = self.__getNaturalAbundanceDistributions()
    self.formula = self.getFAFormulaString(entry, FAMES)
    self.elementsDict = self.parseFormula(self.formula)
    self.atomTracer = atomTracer
    self.purityTracer = purityTracer
    self.correctionMatrix = self.computeCorrectionMatrix(self.elementsDict, self.atomTracer, self.NaturalAbundanceDistributions, purityTracer)
    
  def getFAFormulaString(self, entry, FAMES):
    ''' Return formula string e.g.: C3H2O3'''
    regex = "C([0-9]+):([0-9]+)"
    carbon,doubleBond = [int(val) for val in re.findall(regex, entry)[0]]
    hydrogen = 3+(carbon-2)*2+1-2*doubleBond
    oxygen = 2
    if (FAMES):
      carbon=carbon+1
      hydrogen=hydrogen-1+3
    return "".join(["".join([letter,str(n)]) for [letter,n] in [
              ["C", carbon], 
              ["H", hydrogen],
              ["O", oxygen]] if n>0])

  def parseFormula(self, formula):
    """
    Parse the elemental formula and return the number
    of each element in a dictionnary d={'El_1':x,'El_2':y,...}.
    """
    regex = f"({'|'.join(self.NaturalAbundanceDistributions.keys())})([0-9]{{0,}})"
    elementDict = dict((element, 0) for element in self.NaturalAbundanceDistributions.keys())
    for element,n in re.findall(regex, formula):
      if n:
        elementDict[element] += int(n)
      else:
        elementDict[element] += 1
    return elementDict

  def __getNaturalAbundanceDistributions(self):
    '''Return a dictionary of the isotopic proportions at natural abundance 
    desribed in https://www.ncbi.nlm.nih.gov/pubmed/27989585'''
    H1, H2 = 0.999885, 0.000115
    C12, C13 = 0.9893, 0.0107
    N14, N15 = 0.99632, 0.00368
    O16, O17, O18 = 0.99757, 0.00038, 0.00205
    Si28, Si29, Si30 = 0.922297, 0.046832, 0.030872
    S32, S33, S34, S36 = 0.9493, 0.0076, 0.0429, 0.0002

    return {'H': np.array([H1, H2]), # hydrogen
            'C': np.array([C12, C13]), # carbon
            'N': np.array([N14, N15]), # nitrogen
            'O': np.array([O16, O17, O18]), # oxygen
            'Si': np.array([Si28, Si29, Si30]), # silicon
            'S': np.array([S32, S33, S34, S36])} # sulphur

  def __calculateMassDistributionVector(self, elementDict, atomTracer, NADistributions):
    """
    Calculate a mass distribution vector (at natural abundancy),
    based on the elemental compositions of metabolite.
    The element corresponding to the isotopic tracer is not taken
    into account in the metabolite moiety.
    """
    result = np.array([1.])
    for atom,n in elementDict.items():
      if atom not in [atomTracer]:
        for i in range(n):
          result = np.convolve(result, NADistributions[atom])
    return result

  def computeCorrectionMatrix(self, elementDict, atomTracer, NADistributions, purityTracer):
    # calculate correction vector used for correction matrix construction
    # it corresponds to the mdv at natural abundance of all elements except the
    # isotopic tracer
    correctionVector = self.__calculateMassDistributionVector(elementDict, atomTracer, NADistributions)

    # check if the isotopic tracer is present in formula
    try:
      nAtomTracer =  elementDict[atomTracer]
    except: 
      print("The isotopic tracer must to be present in the metabolite formula!")

    tracerNADistribution =  NADistributions[atomTracer]

    m = 1+nAtomTracer*(len(tracerNADistribution)-1)
    c = len(correctionVector)

    if m > c + nAtomTracer*(len(tracerNADistribution)-1):
      print("There might be a problem in matrix size.\nFragment does not contains enough atoms to generate this isotopic cluster.")

    if c < m:
      # padd with zeros
      correctionVector.resize(m)

    # create correction matrix
    correctionMatrix = np.zeros((m, nAtomTracer+1))
    for i in range(nAtomTracer+1):
      column = correctionVector[:m]
      for na in range(i):
        column = np.convolve(column, purityTracer)[:m]

      for nb in range(nAtomTracer-i):
        column = np.convolve(column, tracerNADistribution)[:m]                    
      correctionMatrix[:,i] = column
    return correctionMatrix

  ##################
  ## Data processing
  ##################

  def _computeCost(self, currentMID, target, correctionMatrix):
    """
    Cost function used for BFGS minimization.
        return : (sum(target - correctionMatrix * currentMID)^2, gradient)
    """
    difference = target - np.dot(correctionMatrix, currentMID)
    # calculate sum of square differences and gradient
    return (np.dot(difference, difference), np.dot(correctionMatrix.transpose(), difference)*-2)

  def _minimizeCost(self, args):
    '''
    Wrapper to perform least-squares optimization via the limited-memory 
    Broyden-Fletcher-Goldfarb-Shanno algorithm, with an explicit lower boundary
    set to zero to eliminate any potential negative fractions.
    '''
    costFunction, initialMID, target, correctionMatrix = args
    res = optimize.minimize(costFunction, initialMID, jac=True, args=(target, correctionMatrix), 
                            method='L-BFGS-B', bounds=[(0., float('inf'))]*len(initialMID),
                            options={'gtol': 1e-10, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 
                                     'maxcor': 10, 'maxfun': 15000})
    return res

  def correctForNaturalAbundance(self, dataFrame, method="LSC"):
    '''
    Correct the Mass Isotope Distributions (MID) from a given dataFrame.
    Method: SMC (skewed Matrix correction) / LSC (Least Squares Skewed Correction)
    '''
    correctionMatrix = self.computeCorrectionMatrix(self.elementsDict, self.atomTracer, self.NaturalAbundanceDistributions, self.purityTracer)
    nRows, nCols = correctionMatrix.shape

    # ensure compatible sizes (will extend data)
    if nCols<dataFrame.shape[1]:
      print("The measure MID has more clusters than the correction matrix.")
    else:
      dfData = np.zeros((len(dataFrame), nCols))
      dfData[:dataFrame.shape[0], :dataFrame.shape[1]] = dataFrame.values

    if method == "SMC":
      # will mltiply the data by inverse of the correction matrix
      correctionMatrix = np.linalg.pinv(correctionMatrix)
      correctedData = np.matmul(dfData, correctionMatrix.transpose())
      # flatten unrealistic negative values to zero
      correctedData[correctedData<0] = 0

    elif method == "LSC":
      # Prepare multiprocessing optimization
      targetMIDList = dfData.tolist()
      initialMID = np.zeros_like(targetMIDList[0])
      argsList = [(self._computeCost, initialMID, targetMID, correctionMatrix) for targetMID in targetMIDList]

      # Lauch 4 parrallel processes
      processes = Pool(4)
      allRes = processes.map(self._minimizeCost, argsList)
      correctedData = np.vstack([res.x for res in allRes])

    return pd.DataFrame(columns=dataFrame.columns, data=correctedData[:, :dataFrame.shape[1]])



#############################################################################
# --------- DATA OBJECT CLASS ----------------------------------------------#
#############################################################################

class MSDataContainer:

  ##################
  ## Init and setup
  ##################

  def __init__(self, fileNames, internalRef="C19:0", tracer="C", tracerPurity=[0.00, 1.00]):
    assert len(fileNames)==2 , "You must choose 2 files!"
    self.internalRef = internalRef
    self.tracer = tracer
    self.tracerPurity = tracerPurity
    self.NACMethod = "LSC" # least squares skewed matrix correction
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
      self.internalRefList = self.dataColNames
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
      # only parental ions for internalRefList
      self.internalRefList = [ f"C{carbon}:{sat}" for (idx,carbon,sat) in FAparent]

    # get sample meta info from template file
    df_TemplateInfo = self.__getExperimentMetaInfoFromMAP(templateMap)

    assert len(df_TemplateInfo)==len(df_Data), \
    f"The number of declared samples in the template (n={len(df_TemplateInfo)}) does not match the number of samples detected in the data file (n={len(df_Data)})"

    return pd.concat([df_Meta, df_TemplateInfo, df_Data.fillna(0)], axis=1)

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

  def updateTracer(self, newTracer):
    self.tracer = newTracer
    print(f"The tracer has been updated to {newTracer}")
    self.dataDf_corrected = self.correctForNaturalAbundance()

  def updateTracerPurity(self, newPurity):
    self.tracerPurity = newPurity
    self.dataDf_corrected = self.correctForNaturalAbundance()

  def updateNACMethod(self, newMethod):
    self.NACMethod = newMethod
    print(f"The correction method for natural abundance has been updated to {newMethod}")
    self.dataDf_corrected = self.correctForNaturalAbundance()

  def computeNormalizedData(self):
    '''Normalize the data to the internal ref'''
    if self.experimentType == "Not Labeled":
      dataDf_norm = self.dataDf.copy()
      dataDf_norm.iloc[:, 7:] = dataDf_norm.iloc[:, 7:].values/dataDf_norm[self.internalRef].values[:, np.newaxis]
      return dataDf_norm
    else:
      print("need to implement normalization for Labeled experiments")

  def correctForNaturalAbundance(self):
    correctedData = self.dataDf.iloc[:,:7]
    for parentalIon in self.internalRefList:
      ionMID = self.dataDf.filter(regex=parentalIon)
      if ionMID.shape[1]<=1:
        # no clusters, go to next
        print(parentalIon, "doesn't have non parental ions")
        correctedData = pd.concat([correctedData, ionMID], axis=1)
        continue
      ionNA = NAProcess(parentalIon, self.tracer, purityTracer=self.tracerPurity)
      correctedData = pd.concat([correctedData, ionNA.correctForNaturalAbundance(ionMID, method=self.NACMethod)], axis=1)
    print(f"The MIDs have been corrected using the {self.NACMethod} method (tracer: {self.tracer}, purity: {self.tracerPurity})")
    return correctedData


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
    self.FANames = dataObject.internalRefList#.dataColNames
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

    FAMESListLabel = tk.Label(FAMESframe, text="FAMES", fg="black", bg="#ECECEC")
    FAMESListLabel.grid(row=2, column=1, sticky=tk.W + tk.N)

    # by default, choose internal reference defined in dataObject (C19:0)
    idxInternalRef = [i for i,name in enumerate(self.FANames) if self.dataObject.internalRef in name][0]

    self.FAMESLabelCurrent = tk.Label(FAMESframe, text=f"The current internal control is {self.FANames[idxInternalRef]}", fg="white", bg="#EBB0FF")
    self.FAMESLabelCurrent.grid(row=3, column=1, columnspan=3)

    self.FAMESListValue = tk.StringVar()
    self.FAMESListValue.trace('w', lambda index,value,op : self.__updateInternalRef(FAMESList.get()))
    FAMESList = ttk.Combobox(FAMESframe, height=6, textvariable=self.FAMESListValue, state="readonly", takefocus=False)
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
    volTotalLabel = tk.Label(Standardframe, text="Vol. Mix Total", fg="black", bg="#ECECEC")
    volTotalLabel.grid(row=5, column=1, sticky=tk.W)

    # Vol mix
    self.volMixVar.trace('w', lambda index,value,op : self.__updateVolumeMixForPrep(self.volMixVar.get()))
    volMixSpinbox = tk.Spinbox(Standardframe, from_=0, to=1000, width=5, textvariable=self.volMixVar, command= lambda: self.__updateVolumeMixForPrep(self.volMixVar.get()), justify=tk.RIGHT)
    volMixSpinbox.grid(row=6, column=2, sticky=tk.W, pady=3)
    volMixLabel = tk.Label(Standardframe, text="Vol. Mix", fg="black", bg="#ECECEC")
    volMixLabel.grid(row=6, column=1, sticky=tk.W)

    # Vol sample
    self.volSampleVar.trace('w', lambda index,value,op : self.__updateVolumeSample(self.volSampleVar.get()))
    volSampleSpinbox = tk.Spinbox(Standardframe, from_=0, to=1000, width=5, textvariable=self.volSampleVar, command= lambda: self.__updateVolumeSample(self.volSampleVar.get()), justify=tk.RIGHT)
    volSampleSpinbox.grid(row=7, column=2, sticky=tk.W, pady=3)
    volSampleLabel = tk.Label(Standardframe, text="Vol. Sample", fg="black", bg="#ECECEC")
    volSampleLabel.grid(row=7, column=1, sticky=tk.W)

    # Standards uL
    StandardVols = CustomText(Standardframe, height=7, width=15)
    StandardVols.grid(row=5, rowspan=3, column=3, padx=20)
    StandardVols.insert(tk.END, "Standards (ul)\n"+"".join([f"{vol}\n" for vol in self.stdVols]))
    StandardVols.bind("<<TextModified>>", self.__updateVolumeStandards)

    # Compute Results button
    computeResultsButton = ttk.Button(Standardframe, text="Compute results", command=lambda: self.computeResults())
    computeResultsButton.grid(row=8, column=2, columnspan=2, pady=5)

    # - - - - - - - - - - - - - - - - - - - - -
    # Quit button in the upper right corner
    quit_button = ttk.Button(self.window, text="Quit", command=lambda: self.quitApp(self.window))
    quit_button.grid(row=1, column=4)

    if self.dataObject.experimentType == "Labeled":
      # - - - - - - - - - - - - - - - - - - - - -
      # The Natural Abundance Correction frame 
      Correctionframe = ttk.LabelFrame(self.window, text="Natural Abundance Correction", relief=tk.RIDGE)
      Correctionframe.grid(row=9, column=1, columnspan=3, sticky=tk.E + tk.W + tk.N + tk.S, padx=2, pady=6)

      # Natural Abundance Correction Method
      NACLabel = tk.Label(Correctionframe, text="Method:", fg="black", bg="#ECECEC")
      NACLabel.grid(row=10, column=1, columnspan=2, sticky=tk.W)
      self.radioCorrectionMethodVariable = tk.StringVar()
      self.radioCorrectionMethodVariable.set("LSC")
      self.radioCorrectionMethodVariable.trace('w', lambda index,value,op : self.__updateNACorrectionMethod(self.radioCorrectionMethodVariable.get()))
      methodButton1 = ttk.Radiobutton(Correctionframe, text="Least Squares Skewed Matrix",
                                     variable=self.radioCorrectionMethodVariable, value="LSC")
      methodButton2 = ttk.Radiobutton(Correctionframe, text="Skewed Matrix",
                                     variable=self.radioCorrectionMethodVariable, value="SMC")
      methodButton1.grid(row=11, column=1, columnspan=2, sticky=tk.W)
      methodButton2.grid(row=12, column=1 , columnspan=2, sticky=tk.W)

      # isotope tracer
      TracerLabel = tk.Label(Correctionframe, text="Atom tracer:", fg="black", bg="#ECECEC")
      TracerLabel.grid(row=10, column=3, sticky=tk.E)
      self.TracerListValue = tk.StringVar()
      self.TracerListValue.trace('w', lambda index,value,op : self.__updateTracer(TracerList.get()))
      TracerList = ttk.Combobox(Correctionframe, textvariable=self.TracerListValue, width=3, state="readonly", takefocus=False)
      TracerList.grid(row=11, column=3, sticky=tk.E)
      TracerList['values'] = ["C", "H", "O"]
      TracerList.current(0)

      # Standards uL
      self.tracerPurity = self.dataObject.tracerPurity
      TracerPurity = CustomText(Correctionframe, height=3, width=12)
      TracerPurity.grid(row=12, column=3, sticky=tk.E)
      TracerPurity.insert(tk.END, "Purity:\n"+" ".join([f"{pur}" for pur in self.tracerPurity]))
      TracerPurity.bind("<<TextModified>>", self.__updateTracerPurity)

      # Compute Results button
      inspectCorrectionButton = ttk.Button(Correctionframe, text="Inspect NA correction", command=lambda: self.inspectCorrectionPlots())
      inspectCorrectionButton.grid(row=13, column=2, columnspan=2, pady=5)
  
  def quitApp(self, window):
    # close Matplotlib processes if any
    plt.close('all')
    window.destroy()

  def popupMsg(self, msg):
    '''Popup message window'''
    popup = tk.Tk()
    popup.wm_title("Look at the standard plots!")
    label = ttk.Label(popup, text=msg, font=("Verdana", 14))
    label.grid(row=1, column=1, columnspan=2, padx=10, pady=10)
      
    B1 = ttk.Button(popup, text="Yes", command = lambda: self.inspectStandardPlots(popup))
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
    newStdVols = [float(vol) for vol in re.findall(r"(?<!\d)\d+\.?\d*(?!\d)", event.widget.get("1.0", "end-1c"))]
    self.stdVols = newStdVols
    self.dataObject.updateStandards(self.volMixVar.get(), self.volTotalVar.get(), newStdVols)
    print(f"The volumeStandards have been updated to {newStdVols}")

  def __updateTracer(self, newTracer):
    self.dataObject.updateTracer(newTracer)

  def __updateTracerPurity(self, event):
    newPurity = [float(pur) for pur in re.findall(r"(?<!\d)\d+\.?\d*(?!\d)", event.widget.get("1.0", "end-1c"))]
    self.tracerPurity = newPurity
    self.dataObject.updateTracerPurity(newPurity)
    print(f"The tracer purity vector has been updated to {newPurity}")

  def __updateNACorrectionMethod(self, newMethod):
    self.dataObject.updateNACMethod(newMethod)

  def computeResults(self):
    self.dataObject.saveStandardCurvesAndResults()
    self.popupMsg("The results and plots have been saved.\nCheck out the standard plots.\nDo you want to modify the standards?")

  # --------------- Standard plots -------------------
  def inspectStandardPlots(self, popup):
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

    plotFrame = tk.Tk()
    plotFrame.wm_title("Standard curve inspector")

    pointsListbox = tk.Listbox(plotFrame, height=8, selectmode='multiple')
    pointsListbox.grid(row=1, column=3, columnspan=2, pady=10, padx=5)

    canvas = FigureCanvasTkAgg(fig, plotFrame)

    self.plotIsolatedFAMES(FAMESselected[currentFAMESidx], ax, canvas, pointsListbox)

    # make everything fit
    fig.tight_layout()

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
        quitCurrent()
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
    ax.set_xlabel("Quantity (nMoles)")
    ax.set_ylabel("Absorbance")

    canvas.draw()

  # --------------- Correction plots -------------------
  def inspectCorrectionPlots(self):

    def quitCurrent():
      plt.close('all')
      plotFrame.destroy()

    def goToNextPlot(direction):
      nonlocal currentCorrectionIdx
      if (currentCorrectionIdx+direction>=0) & (currentCorrectionIdx+direction<len(self.FANames)):
        currentCorrectionIdx = currentCorrectionIdx+direction
        self.plotIsolatedCorrection(self.FANames[currentCorrectionIdx], SampleList.get(), ax, canvas)
        nextButton["text"]="Next" # in case we come from last plot
        if currentCorrectionIdx == len(self.FANames)-1:
          # last plot
          nextButton["text"]="Finish"
      elif (currentCorrectionIdx+direction<0):
        currentCorrectionIdx = 0
        print("You are already looking at the first correction plot!")
      else:
        quitCurrent()

    def showNewSampleSelection(event):
      self.plotIsolatedCorrection(self.FANames[currentCorrectionIdx], SampleList.get(), ax, canvas)
    
    plotFrame = tk.Tk()
    plotFrame.wm_title("Natural Abundance Correction inspector")

    currentCorrectionIdx = 0 # will be used to go from an FAME to another

    # sample chooser
    SampleList = ttk.Combobox(plotFrame, height=6, state="readonly")
    SampleList.grid(row=1, column=1, columnspan=2)
    SampleList['values'] = [f"{name} - {sample}" for name,sample in self.dataObject.dataDf[["Name", "SampleName"]].values]
    SampleList.current(0)
    # somehow for this one I couldn't just link to a tk.variable and trace to get update working ...
    # so I directly bind to a function on change
    SampleList.bind("<<ComboboxSelected>>", showNewSampleSelection)

    # Main fig
    fig,ax = plt.subplots(figsize=(7,3))
    canvas = FigureCanvasTkAgg(fig, plotFrame)
    self.plotIsolatedCorrection(self.FANames[currentCorrectionIdx], SampleList.get(), ax, canvas)
    fig.tight_layout()
    figFrame = canvas.get_tk_widget()#.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    figFrame.grid(row=2, column=1, columnspan=8, rowspan=3, pady=10, padx=10)

    # buttons and others
    quitButton = ttk.Button(plotFrame, text="Quit", command = lambda: quitCurrent())
    quitButton.grid(row=5, column=1, pady=5)
    
    previousButton = ttk.Button(plotFrame, text="Previous", command = lambda: goToNextPlot(-1))
    previousButton.grid(row=5, column=7, pady=5)
    nextButton = ttk.Button(plotFrame, text="Next", command = lambda: goToNextPlot(1))
    nextButton.grid(row=5, column=8, pady=5)

  def plotIsolatedCorrection(self, famesName, sampleName, ax, canvas):
    ax.clear()

    # get row associated with sampleName provided
    row = np.where(self.dataObject.dataDf["Name"] == sampleName.split(" ")[0])[0][0]
    print(row)

    originalData = self.dataObject.dataDf.filter(regex=famesName).iloc[row]
    correctedData = self.dataObject.dataDf_corrected.filter(regex=famesName).iloc[row]

    # make x label
    xLabels = [f"M.{i}" for i in range(len(originalData))]
    xrange = np.arange(len(originalData))
    barWidth = 0.4

    ax.bar(xrange-barWidth/2, originalData, barWidth, label="Original")
    ax.bar(xrange+barWidth/2, correctedData, barWidth, label="Corrected")
    ax.set_xticks(xrange)
    ax.set_xticklabels(xLabels)
    ax.legend()

    ax.set_ylabel("Absorbance")
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
  # print(appData.internalRefList)
  # appData.dataDf.to_excel("test.xlsx")
  # appData.dataDf_corrected.to_excel("test_correctedGC.xlsx")


  ###################################
  # Natural Abundance Correction test

  # data = pd.read_excel("test.xlsx").filter(regex="C14")
  # naProcess = NAProcess(data.columns[0], "C")

  # df_SMC = naProcess.correctForNaturalAbundance(data, method="SMC")
  # df_LSC = naProcess.correctForNaturalAbundance(data, method="LSC")
  
  # df_SMC.to_excel("test_SMC.xlsx", index=False)
  # df_LSC.to_excel("test_LSC.xlsx", index=False)

  # test = NAProcess("C14:0", "C")
  # print(test.correctionMatrix)


