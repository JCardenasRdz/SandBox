import numpy as np

def Lorentz1(xdata,Amp =[1.0],Width = [1.0],Center = [1.0]):
    """
    Estimates sum of Lorentzian functions where:
    Amp   = 1 X N lorentzian of amplitudes
    Width = 1 X N lorentzian of widths
    Center = 1 X N lorentzian of centers
    xdata = 1XN indepedent variable
    """
    # Estimate number of pools
    Num_variables = len(Amp) + len(Width) + len(Center)

    # make sure it is divisible by 3.0
    assert (Num_variables % 3 == 0),"Please provide 3 variables per pool"

    # Preallocate output
    Lsum = np.zeros_like(xdata)

    # define Lorentzian function for one pool
    def Lorentzian(pars,xdata_):
        fwhm = pars[1]**2/4
        L= (pars[0]*fwhm**2)/ ( (xdata_-pars[2])**2 + (fwhm)**2 )
        return L

    # calculate final output
    num_pools = int(Num_variables/3)
    for idx in range( num_pools):
        print(idx)
        # assign each variable
        amp = Amp[idx]
        width = Width[idx]
        center = Center[idx]
        # estimate signal and sum
        Lsum += Lorentzian( [amp,width,center], xdata)

    return Lsum
