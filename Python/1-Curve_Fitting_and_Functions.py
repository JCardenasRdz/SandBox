# Learn several ways of performing curve fitting
import	numpy 				as np
import 	matplotlib.pyplot 	as plt
from 	scipy.optimize	import	curve_fit

# Method 01
def foo(x, a, b, c):               # One input per variable
    return a * np.exp(-b * x) + c

xdata = np.linspace(0, 4, 50);
ydata = foo(xdata, 2.5, 1.3, 0.5) + np.random.normal(0,.1,len(xdata))


#fit
popt1 = curve_fit(foo, xdata, ydata)[0]
yhat_1 = foo(xdata,popt1[0],popt1[1],popt1[2],)

#================================================================
# Method 02
from scipy.optimize import least_squares

def foo2(x_data, p):               # One input per variable
    return p[0] * np.exp(-p[1]  * x_data) + p[2]

def fit_data(x_,y_,function,p0):
    p0 = np.array(p0)
    observed_data   = y_
    def res(x):
        return function(x_,x) - observed_data
    return least_squares(res,  p0 )

fit = fit_data(xdata,ydata,foo2,[1,1,1])
yhat_2 = foo2(xdata,fit.x)

# Plots
plt.plot(xdata,ydata,'o',xdata,yhat_1,'-k',xdata,yhat_2,'xr')
l=("Experimental",'Method 01', 'Method 02')
plt.legend( l )
plt.show()
