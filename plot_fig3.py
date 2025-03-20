from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import AutoMinorLocator, FixedLocator
import seaborn as sb
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import seaborn as sb

plt.rcParams['font.family'] = 'Helvetica'


plt.rcParams['text.usetex'] = True
# Ensure the 'amsmath' package is included in the preamble
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

fontsize = 24

plt.close('all')
fig = plt.figure(figsize=(8,6))
ax1 = plt.subplot2grid((1, 1), (0, 0))

left_margin = 0.11   # Adjust this value as needed
right_margin = 0.99  # Adjust this value as needed
bottom_margin = 0.12 # Adjust this value as needed
top_margin = 0.97    # Adjust this value as needed
wspace = 0.0
hspace = 0.0
plt.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin, top=top_margin, wspace=wspace, hspace=hspace)



def Read_Three_Column_File(file_path):
    # Dummy function to represent your file reading
    # You should replace this with your actual file reading logic
    return np.loadtxt(file_path, unpack=True)

def normalize_area_under_curve(x1, x2):
    # Assuming x1 and x2 are numpy arrays
    # Calculate the discrete integral of x2 with respect to x1
    area = np.trapz(x2, x1)
    # Normalize x2 by the total area to ensure the area under the curve is one
    return x2 / area


def set_ticks_and_labelsize(ax, interval=1, labelsize=fontsize-2):
    # Set ticks using MultipleLocator
    #ax.xaxis.set_major_locator(plt.MultipleLocator(interval))
    ax.yaxis.set_major_locator(plt.MultipleLocator(interval))
    
    # Set tick label size
    ax.tick_params(labelsize=labelsize)


# Assuming ax1, ax2, ..., ax5 are your axes objects
set_ticks_and_labelsize(ax1)

ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
ax1.yaxis.set_minor_locator(AutoMinorLocator(5))




def Read_Two_Column_File(file_name):
    with open(file_name, 'r') as data:
        x = []
        y = []
        for line in data:
            p = line.split()
            x.append(float(p[0]))
            y.append(float(p[1]))

    return x, y




def Read_Three_Column_File(file_name):
    with open(file_name, 'r') as data:
        x = []
        y = []
        z = []
        for line in data:
            p = line.split()
            x.append(float(p[0]))
            y.append(float(p[1]))
            z.append(float(p[2]))

    return x, y, z

# Define the function x*exp(-x)
def f(x):
    return x * np.exp(-x)


markersize = 10
linewidth = 0
period = 1

# FIGURE (a)
    
# WT
x1, x2 = Read_Two_Column_File('Scaled_Intensity/WT.dat')
ax1.plot(x1, x2, color='k', marker='o', markersize=markersize, linestyle='-', linewidth=linewidth, alpha=1.0, label = r'$\mathrm{WT}$')
x1, x2 = Read_Two_Column_File('Scaled_Intensity/UBA1.dat')
ax1.plot(x1, x2, color='orange', marker='s', markersize=markersize, linestyle='-', linewidth=linewidth, alpha=1.0, label = r'$\mathit{uba}$-$\mathit{1}$')
x1, x2 = Read_Two_Column_File('Scaled_Intensity/F7.dat')
ax1.plot(x1, x2, color='purple', marker='v', markersize=markersize, linestyle='-', linewidth=linewidth, alpha=1.0, label = r'$\mathit{fbxb}$-$\mathit{65}$')


# Generate x values
x = np.linspace(0, 16, 10000)
y = f(x)
ax1.plot(x, y, color='black', linestyle='-', linewidth=3.5, label=r'$\mathrm{Theory}$')



ax1.set_xlim([-0.2, 16])
ax1.set_ylim([-0.01, 0.42])
ax1.tick_params(which='major', direction='in')
ax1.tick_params(which='minor', direction='in')
ax1.yaxis.set_major_locator(FixedLocator([0, 0.1, 0.2, 0.3, 0.4]))
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.set_ylabel('Intensity',fontsize=fontsize)
#ax1.set_xlabel('Position (\u03bcm)')
ax1.legend(loc='upper right',fontsize=fontsize-2, markerfirst=True, labelspacing=0.02)



ax1.tick_params(which='major', direction='in', bottom=True, top=True, left=True, right=True)
ax1.tick_params(which='minor', direction='in', bottom=True, top=True, left=True, right=True)

ax1.set_ylabel(r'$\tilde{P}_{s}(\mu n)$',fontsize=fontsize)
ax1.set_xlabel(r'$\mu n$',fontsize=fontsize, labelpad=-5)

#plt.tight_layout()
plt.show()


fig.savefig('fig3.pdf', dpi = 600)
