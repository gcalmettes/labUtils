import os
import sys
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


############################
## GUI
############################

def initialFileChoser(directory=False):
	'''Temporary app to get filenames before building main app'''
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

def popupMsg(msg):
	'''Popup message window'''
	popup = tk.Tk()
	popup.wm_title("Look at the standard plots!")
	label = ttk.Label(popup, text=msg, font=("Verdana", 14))
	label.grid(row=1, column=1, columnspan=2, padx=10, pady=10)
		
	B1 = ttk.Button(popup, text="Yes", command = lambda x=0: inspectPlots(popup))
	B1.grid(row=2, column=1, pady=10)

	def quitApp():
		popup.destroy()
		app.window.destroy()

	B2 = ttk.Button(popup, text="No", command = quitApp)
	B2.grid(row=2, column=2, pady=10)
	popup.mainloop()


def inspectPlots(popup):
	popup.destroy()
			
	selectFrame = tk.Tk()
	selectFrame.wm_title("FAMES standard to modify")
	label = ttk.Label(selectFrame, text="Select the FAMES standard to modify", font=("Verdana", 14))
	label.grid(row=1, column=1, columnspan=2, padx=10, pady=10)

	FAMESlistbox = tk.Listbox(selectFrame, height=10, selectmode='multiple')
	for item in appData["FAMESnames"].keys():
		FAMESlistbox.insert(tk.END, item)
	FAMESlistbox.grid(row=2, column=1, columnspan=2, pady=10)

	FAMESbutton = ttk.Button(selectFrame, text="Select", command = lambda: modifySelection(selectFrame, FAMESlistbox))
	FAMESbutton.grid(row=3, column=1, columnspan=2, pady=10)


def modifySelection(frameToKill, selection):
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

	plotIsolatedFAMES(FAMESselected[currentFAMESidx], ax, canvas, pointsListbox)

	figFrame = canvas.get_tk_widget()#.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
	figFrame.grid(row=1, column=1, columnspan=2, rowspan=3, pady=10, padx=10)

	plotButton = ttk.Button(plotFrame, text="Remove", command = lambda: plotIsolatedFAMES(FAMESselected[currentFAMESidx], ax, canvas, pointsListbox))
	plotButton.grid(row=2, column=3, pady=5)

	def goToNextPlot():
		nonlocal currentFAMESidx
		if currentFAMESidx+1<len(FAMESselected):
			currentFAMESidx = currentFAMESidx+1
			plotIsolatedFAMES(FAMESselected[currentFAMESidx], ax, canvas, pointsListbox)
			if currentFAMESidx == len(FAMESselected)-1:
				plotButton2["text"]="Send"
		else:
			plt.close('all')
			plotFrame.destroy()
			plotStandardAndSaveResults(appData, useMask=True)

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


def plotIsolatedFAMES(famesName, ax, canvas, pointsListbox):
	ax.clear()

	ax.set_xlabel("Quantity (nMoles)")
	ax.set_ylabel("Absorbance")

	carbon,mass = famesName.split(" ")
	maskSelected = [not (i in pointsListbox.curselection()) for i in range(len(appData["Standard-nMol"][carbon].values))]
	newMask = [(m1 & m2) for m1,m2 in zip(appData["FAMESnames"][famesName]["originalMask"], maskSelected)]
	appData["FAMESnames"][famesName]["newMask"] = newMask
	
	xvals = appData["Standard-nMol"][carbon].values
	yvals = appData["Standard-absorbance"][famesName].values


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



class MSAnalyzer:
	def __init__(self, faNames, idxInternalRef):
		self.window = tk.Tk()
		self.window.title("MS Analyzer")
		self.create_widgets(faNames, idxInternalRef)

	def create_widgets(self, faNames, idxInternalRef):
		# Create some room around all the internal frames
		self.window['padx'] = 5
		self.window['pady'] = 5

		# - - - - - - - - - - - - - - - - - - - - -
		# The FAMES frame (for internal control)
		FAMESframe = ttk.LabelFrame(self.window, text="Select the internal control", relief=tk.GROOVE)
		FAMESframe.grid(row=1, column=1, columnspan=3, sticky=tk.E + tk.W + tk.N + tk.S, padx=2)

		FAMESListLabel = tk.Label(FAMESframe, text="FAMES")
		FAMESListLabel.grid(row=2, column=1, sticky=tk.W + tk.N)

		FAMESLabelCurrent = tk.Label(FAMESframe, text=f"The current internal control is {faNames[idxInternalRef]}", fg="white", bg="#EBB0FF")#bg="#E4E4E4")
		FAMESLabelCurrent.grid(row=3, column=1, columnspan=2)

		self.FAMESListValue = tk.StringVar()
		# will update each time that new control is chosen
		self.FAMESListValue.trace('w', lambda index,value,op : updateInternalRef(FAMESList.get(), FAMESLabelCurrent))
		FAMESList = ttk.Combobox(FAMESframe, height=6, textvariable=self.FAMESListValue)
		FAMESList.grid(row=2, column=2, columnspan=2)
		FAMESList['values'] = faNames
		FAMESList.current(idxInternalRef)


		# - - - - - - - - - - - - - - - - - - - - -
		# The standards frame (for fitting)
		Standardframe = ttk.LabelFrame(self.window, text="Standards", relief=tk.RIDGE)
		Standardframe.grid(row=4, column=1, columnspan=3, sticky=tk.E + tk.W + tk.N + tk.S, padx=2, pady=6)
		
		# Vol mix total
		self.volTotalVar = tk.IntVar()
		self.volTotalVar.set(500)
		volTotalSpinbox = tk.Spinbox(Standardframe, from_=0, to=1000, width=5, textvariable=self.volTotalVar, justify=tk.RIGHT)
		volTotalSpinbox.grid(row=5, column=2, sticky=tk.W, pady=3)
		volTotalLabel = tk.Label(Standardframe, text="Vol. Mix Total")
		volTotalLabel.grid(row=5, column=1, sticky=tk.W)

		# Vol mix
		self.volMixVar = tk.IntVar()
		self.volMixVar.set(100)
		volMixSpinbox = tk.Spinbox(Standardframe, from_=0, to=1000, width=5, textvariable=self.volMixVar, justify=tk.RIGHT)
		volMixSpinbox.grid(row=6, column=2, sticky=tk.W, pady=3)
		volMixLabel = tk.Label(Standardframe, text="Vol. Mix")
		volMixLabel.grid(row=6, column=1, sticky=tk.W)

		# Vol sample
		self.volSampleVar = tk.IntVar()
		self.volSampleVar.set(150)
		volSampleSpinbox = tk.Spinbox(Standardframe, from_=0, to=1000, width=5, textvariable=self.volSampleVar, justify=tk.RIGHT)
		volSampleSpinbox.grid(row=7, column=2, sticky=tk.W, pady=3)
		volSampleLabel = tk.Label(Standardframe, text="Vol. Sample")
		volSampleLabel.grid(row=7, column=1, sticky=tk.W)

		# Standards uL
		StandardVols = tk.Text(Standardframe, height=7, width=15)
		StandardVols.grid(row=5, rowspan=3, column=3, padx=20)
		StandardVols.insert(tk.END, "Standards (ul)\n1\n5\n10\n20\n40\n80\n")

		# Do fit button
		StandardButton = ttk.Button(Standardframe, text="Compute results", command=lambda x=1 : fitStandard(StandardVols.get("1.0",tk.END), self.volTotalVar.get(), self.volMixVar.get()))
		StandardButton.grid(row=8, column=2, columnspan=2, pady=5)


		# - - - - - - - - - - - - - - - - - - - - -
		# Quit button in the lower right corner
		quit_button = ttk.Button(self.window, text="Quit", command=self.window.destroy)
		quit_button.grid(row=1, column=4)


############################
## UTILS
############################

def getDataAndTemplateFileNames(fileNames):
	'''Classify data and template files based on fileName'''
	dataFileName = [fileName for fileName in fileNames if "template" not in fileName][0]
	templateFileName = [fileName for fileName in fileNames if fileName != dataFileName][0]
	return [dataFileName, templateFileName]


def getCleanUpDataFrameFromFile(fileName, keyword="Results"):
	'''Return a simplified pandas dataFrame'''
	startIdxCol = 7 # cols before this are just info
	colNames = pd.read_excel(fileName).filter(regex=f"[A-Z0-9]*{keyword}").columns
	colNames = list(map(lambda x: renameFAMES(x), colNames))
	df = pd.read_excel(fileName, skiprows=1)
	df = pd.concat([df.Name, df["Data File"], df.iloc[:, startIdxCol:]], axis=1)
	df.columns = np.concatenate([["Name"], ["File"], colNames])
	return df,colNames


def renameFAMES(name):
	'''Return a more informative name to column'''
	codeID,_ = name.split(" ")
	num,carbon,mass = codeID.split("_")
	return f"C{carbon[:2]}:{carbon[2:]} ({mass})"


def updateInternalRef(newInternalRef, label):
	'''Update FAMES chosen as internal reference and normalize data to it'''
	print(f"Internal Reference changed from {appData['internalRef']} to {newInternalRef}")
	appData["internalRef"] = newInternalRef

	label.config(text=f"The current internal control is {newInternalRef}")

	appData["dataDf_norm"] = appData["dataDf"].copy()
	appData["dataDf_norm"].iloc[:, 2:] = appData["dataDf_norm"].iloc[:, 2:].values/appData["dataDf_norm"][appData["internalRef"]].values[:, np.newaxis]
	# print(appData["dataDf_norm"])
	return


def fitStandard(StandardText, volMixTotal, volMix):
	stdVols = getStandardVolumeFromText(StandardText)#np.array([vol for vol in StandardText.strip().split("\n")][1:], dtype=float)
	appData["Standard-nMol"] = getStandardMoles(appData["templateFileName"], stdVols, volMix, volMixTotal, sheet_name="STANDARD")
	appData["Standard-absorbance"] = getSampleMap(appData["templateFileName"], sheet_name="MAP")
	plotStandardAndSaveResults(appData)
	# print(dataStandard)

def getStandardVolumeFromText(StandardText):
	'''Extract volumes of standards from text'''
	return np.array([vol for vol in StandardText.strip().split("\n")][1:], dtype=float)


def getStandardMoles(fileName, stdVols, volMix, volMixTotal, sheet_name="STANDARD"):
	'''Calculate nMoles for the standards'''
	template = pd.read_excel(fileName, sheet_name=sheet_name)
	template["Conc in Master Mix (ug/ul)"] = template["Stock conc (ug/ul)"]*template["Weight (%)"]/100*float(volMix)/float(volMixTotal)
	# concentration of each carbon per standard volume
	for ul in stdVols:
		template[f"Std-Conc-{ul}"]=ul*(template["Conc in Master Mix (ug/ul)"]+template["Extra"])
	# nMol of each fa per standard vol
	for ul in stdVols:
		template[f"Std-nMol-{ul}"] = 1000*template[f"Std-Conc-{ul}"]/template["MW"]
	# create a clean template with only masses and carbon name
	templateClean = pd.concat([template.Chain, template.filter(regex="Std-nMol")], axis=1).transpose()
	templateClean.columns = list(map(lambda x: "C"+x, templateClean.iloc[0]))
	templateClean = templateClean.iloc[1:]
	return templateClean


def getSampleMap(fileName, sheet_name="MAP"):
	templateMap = pd.read_excel(fileName, sheet_name=sheet_name).dropna()

	#Add column to dataDf_norm with the sample ID defined in the MAP file
	addIDsampleToDataDf(templateMap)

	# index of standards (they will be named S1, S5, etc ...)
	stdIdx = templateMap["SampleName"].loc[templateMap["SampleName"].str.match('S[0-9]+')>0].index

	# get the num of the mass spec cell ID (MS_num) that are holding the standards
	cellStd = templateMap.iloc[stdIdx].SampleID.apply(lambda x: x.split("_")[1])

	# get corresponding data for which the Name (F..) matches the Standards cell ID
	dataStandard = appData["dataDf_norm"].loc[[x for x in range(len(appData["dataDf_norm"])) if appData["dataDf_norm"].Name.apply(lambda x: x.split("F")[1]).loc[x] in cellStd.values]]
	dataStandard["IDsample"]=templateMap.loc[stdIdx,"SampleName"].values
	return dataStandard

def addIDsampleToDataDf(templateMap):
	'''Add column to dataDf_norm with the sample ID defined in the MAP file'''
	appData["dataDf_norm"]["IDsample"] = " "
	for id,sample in templateMap.loc[:, ["SampleID", "SampleName"]].values:
		idNum = id.split("_")[1]
		appData["dataDf_norm"].loc[[x for x in range(len(appData["dataDf_norm"])) if appData["dataDf_norm"].Name.apply(lambda x: x.split("F")[1]).loc[x] == idNum], "IDsample"] = sample

	# reorder rows based on template and reindex with range
	newOrder = list(map(lambda x: f"F{x.split('_')[1]}", templateMap.SampleID.values))
	appData["dataDf_norm"].index=appData["dataDf_norm"]["Name"]
	appData["dataDf_norm"] = appData["dataDf_norm"].reindex(newOrder)
	appData["dataDf_norm"].index = list(range(len(appData["dataDf_norm"])))
	appData["dataDf_norm"]["MSsample"] = templateMap["SampleID"]


def plotStandardAndSaveResults(appData, useMask=False):
	# get current folder and create result folder if needed
	savePath = makeResultFolder()
	
	# will store final results
	resultsDf = pd.DataFrame(index=appData["dataDf_norm"].index)

	# Plot of Standard
	nTotal = len(appData["Standard-absorbance"].filter(regex="C").columns)
	nCols = 4
	if nTotal%4==0:
		nRows = int(nTotal/nCols)
	else:
		nRows = int(np.floor(nTotal/nCols)+1)

	fig1,axes = plt.subplots(ncols=nCols, nrows=nRows, figsize=(20, nRows*4))

	if not useMask:
		appData["FAMESnames"] = {}
		extension = ""
	else:
		extension = "_modified"

	for i,(col,ax) in enumerate(zip(appData["Standard-absorbance"].filter(regex="C").columns,  axes.ravel())):
		carbon,mass = col.split(" ")
		if carbon in appData["Standard-nMol"].columns:
			# fit of data
			xvals = appData["Standard-nMol"][carbon].values
			yvals = appData["Standard-absorbance"][col].values
			
			if not useMask:
				mask = [~np.logical_or(np.isnan(x), np.isnan(y)) for x,y in zip(xvals, yvals)]
				# add carbon to valid standard FAMES and save mask
				appData["FAMESnames"][col] = {"originalMask": mask}
			else:
				try:
					mask = appData["FAMESnames"][col]["newMask"]
				except:
					mask = appData["FAMESnames"][col]["originalMask"]

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
			resultsDf[col] = (appData["dataDf_norm"][col]-intercept)/slope
		ax.set_title(col)
		ax.set_xlabel("Quantity (nMoles)")
		ax.set_ylabel("Absorbance")

	fig1.tight_layout()

	# Plot of results on standard curves
	nTotal = len(resultsDf.filter(regex="C").columns)
	nCols = 4
	if nTotal%4==0:
		nRows = int(nTotal/nCols)
	else:
		nRows = int(np.floor(nTotal/nCols)+1)

	fig2,axes = plt.subplots(ncols=nCols, nrows=nRows, figsize=(20, nRows*4))

	for i,(col,ax) in enumerate(zip(resultsDf.filter(regex="C").columns,  axes.ravel())):
		carbon,mass = col.split(" ")
		if carbon in appData["Standard-nMol"].columns:
			# fit of data
			xvals = appData["Standard-nMol"][carbon].values
			yvals = appData["Standard-absorbance"][col].values
			try:
				mask = appData["FAMESnames"][col]["newMask"]
			except:
				mask = appData["FAMESnames"][col]["originalMask"]
			# mask = [~np.logical_or(np.isnan(x), np.isnan(y)) for x,y in zip(xvals, yvals)]
			slope,intercept = np.polyfit(np.array(xvals[mask], dtype=float), np.array(yvals[mask], dtype=float), 1)
			xfit = [np.min(xvals), np.max(xvals)]
			yfit = np.polyval([slope, intercept], xfit)
			# plot of data	
			ax.plot(xvals[mask], yvals[mask], "o")
			ax.plot(xvals[[not i for i in mask]], yvals[[not i for i in mask]], "x", color="black", ms=3)
			ax.plot(xfit, yfit, "red")
			# plot values calculated from curve
			ax.plot(resultsDf[col], appData["dataDf_norm"][col], "o", alpha=0.3)
		ax.set_title(col)
		ax.set_xlabel("Quantity (nMoles)")
		ax.set_ylabel("Absorbance")

	fig2.tight_layout()
	
	# Save data
	resultsDf["IDsample"]=appData["dataDf_norm"]["IDsample"]
	resultsDf["MSsample"]=appData["dataDf_norm"]["MSsample"]
	resultsDf = resultsDf[np.concatenate([["MSsample", "IDsample"], np.array(resultsDf.filter(regex="C").columns)])]
	resultsDf.to_excel(f"{savePath}/results{extension}.xls", index=False)
	fig1.savefig(f"{savePath}/standard-fit{extension}.pdf")
	fig2.savefig(f"{savePath}/standard-fit-with-data{extension}.pdf")
	
	print(f"The standard plots have been saved at {savePath}/standard-fit{extension}.pdf")
	print(f"The results calculated from the standard regression lines have been saved at {savePath}/standard-fit-with-data{extension}.pdf")
	print(f"The analysis results have been saved at {savePath}/results{extension}.xls")

	
	# close Matplotlib processes
	plt.close('all')

	popupMsg("The results and plots have been saved.\nCheck out the standard plots.\nDo you want to modify the standards?")


def makeResultFolder():
	directory = f"{appData['pathDirName']}/results"
	if not os.path.exists(directory):
		os.mkdir(directory)
	return directory


############################
## MAIN
############################

if __name__ == '__main__':

	appData = {
		"internalRef": "C19:0"
	}
	
	# Directory to look for data files defined?
	if len(sys.argv) == 1: # no arguments given to the function
		initialDirectory = False
	else:
		initialDirectory = sys.argv[1]

	# Choose data and template files
	fileNames = initialFileChoser(initialDirectory)
	appData["dataFileName"],appData["templateFileName"] = getDataAndTemplateFileNames(fileNames)
	appData["pathDirName"] = os.path.dirname(appData["dataFileName"])
	print(f"Path of working directory: {appData['pathDirName']}")
	print(f"Data file: {appData['dataFileName']}")
	print(f"Template file: {appData['templateFileName']}")

	# import data file and clean up
	appData["dataDf"],dataColNames = getCleanUpDataFrameFromFile(appData["dataFileName"], keyword="Results")

	# by default, choose C14:0 as internal reference
	idxInternalRef = [i for i,name in enumerate(dataColNames) if appData["internalRef"] in name][0]

	# Create the entire GUI program and pass in colNames for popup menu
	app = MSAnalyzer(dataColNames, idxInternalRef)

	# Start the GUI event loop
	app.window.mainloop()



