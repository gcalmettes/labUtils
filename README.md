# labUtils

Collection of script/tools to simplify the data analysis workflow of the lab.

### msAnalysis

TKinter app to analyze Mass Spectroscopy data.

Tools: Python

Dependencies: TKinter, Numpy, Pandas, Scipy

### traceViewer

Quickly load data from a csv file and inspect it using a focus+context approach.

Tools: [D3js](https://d3js.org)

### atfFilesAnalyzer

Python class extracting multi-channels data from .atf files generated by the pClamp software, based on peak detection.
Used to analyze files from Langendorff perfused hearts experiments and extract Heart Rate, Developed Pressure and Heart Work (normalized RPP)

Tools: Python

Dependencies: Numpy, Pandas, Matplotlib (if plot option engaged)

### FRETanalysis

Extract and analyze FRET data contained in the generated excel file from Metafluor.

Tools: Python

Dependencies: Numpy, xrld

### fitTools

Convenient functions for data fitting

Tools: Python

### varMap

Simple algorithm to compute the variance map of an image.

An application of the varMap tool to detect and quantify the translocation of the hexokinase protein from mitochondria to cytosol has been published in our 2013 paper [A Quantitative Method to Track Protein Translocation between Intracellular Compartments in Real-Time in Live Cells Using Weighted Local Variance Image Analysis](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0081988)

Tools: Python

### reverseJet

The Jet colormap is [bad](https://jakevdp.github.io/blog/2014/10/16/how-bad-is-your-colormap/).
reverseJet renders the gray-based luminescence version of the original image.

Tools: Python

Dependencies: Skimage 
