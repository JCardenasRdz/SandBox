import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
xdata = np.linspace(0,10,100)
ydata = np.cos(np.pi*xdata)

plt.plot(xdata,ydata)
plt.xlabel('xdata')
plt.ylabel('ydata')
plt.show()
