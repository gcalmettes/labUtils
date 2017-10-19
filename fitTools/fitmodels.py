import numpy as np
from scipy.optimize import leastsq


##########################
### SEVERAL FUNCTIONS
##########################

def sigmo(x, base, amplitude, xhalf, slope):
    """
    Return the sigmoid curve for the x values.
    """
    return base + amplitude/(1.+np.exp(-slope*(x-xhalf)))


def expo_decay(x, amplitude, tconstant, base):
    """
    Return the exponential decay curve for the x values.
    """
    return amplitude*np.exp(-x/tconstant) + base


def expo_plateau(x, plateau, amplitude, tconstant):
    """
    Return the exponential plateau curve for the x values.
    """
    return plateau - amplitude*np.exp(-x/tconstant)


def hill(x, baseline, amplitude, tconstant, hillcoef):
    """
    Return the hill equation curve for the x values.
    """
    return baseline+amplitude*(x**hillcoef)/(x**hillcoef+tconstant**hillcoef)


def hill2(x, baseline, amplitude, tconstant, hillcoef):
    """
    Return the hill equation curve for the x values.
    """
    return (amplitude/(1.+((x/tconstant)**hillcoef))) + baseline

def double_expo_waveform(x, A, alpha, beta, offset):
    """
    Return the corresponding double exponential waveform curve
    Note: alpha<<beta
    """
    return A*(np.exp(-alpha*x)-np.exp(-beta*x))+offset
