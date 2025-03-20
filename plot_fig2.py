from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator
import seaborn as sb
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

fontsize = 20

plt.close('all')
fig = plt.figure(figsize=(8, 4))
ax1 = plt.subplot2grid((1, 3), (0, 0))
ax2 = plt.subplot2grid((1, 3), (0, 1))
ax3 = plt.subplot2grid((1, 3), (0, 2))

left_margin = 0.09   # Adjust as needed
right_margin = 0.97  # Adjust as needed
bottom_margin = 0.15 # Adjust as needed
top_margin = 0.95    # Adjust as needed
wspace = 0.40
hspace = 0.40
plt.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin,
                    top=top_margin, wspace=wspace, hspace=hspace)

def normalize_area_under_curve(x1, x2):
    """
    Normalize the curve defined by x2 (as a function of x1) so that the total area under the curve is 1.
    Uses np.trapezoid for numerical integration.
    """
    area = np.trapezoid(x2, x1)
    return x2 / area

def set_ticks_and_labelsize(ax, interval=1, labelsize=fontsize-2):
    ax.yaxis.set_major_locator(plt.MultipleLocator(interval))
    ax.tick_params(labelsize=labelsize)

# Apply tick settings to all axes
set_ticks_and_labelsize(ax1)
set_ticks_and_labelsize(ax2)
set_ticks_and_labelsize(ax3)
for ax in [ax1, ax2, ax3]:
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

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
    # This function is defined in case you need to read three-column data.
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

markersize = 4  
linewidth = 0
period = 1

# FIGURE (a): Raw Intensity
x1, x2 = Read_Two_Column_File('Raw_Intensity/WT.dat')
x1 = -(np.array(x1) - np.max(x1))
ax1.plot(x1[::period], x2[::period], color='k', marker='o', markersize=markersize,
         linestyle='-', linewidth=linewidth, alpha=1.0, fillstyle='none')
ax1.set_xlim([-5, 280])
ax1.set_ylim([0, 0.016])
ax1.tick_params(which='major', direction='in')
ax1.tick_params(which='minor', direction='in')
ax1.yaxis.set_major_locator(FixedLocator([0, 0.005, 0.01, 0.015]))
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())

# FIGURE (b): Fluorophore Intensity
x1, x2 = Read_Two_Column_File('Flourophore/WT.dat')
x1 = -(np.array(x1) - np.max(x1))
ax2.plot(x1[::period], x2[::period], color='maroon', marker='o', markersize=markersize,
         linestyle='-', linewidth=linewidth, alpha=1.0, fillstyle='none')
ax2.set_xlim([-5, 280])
ax2.set_ylim([0, 0.016])
ax2.tick_params(which='major', direction='in')
ax2.tick_params(which='minor', direction='in')
ax2.yaxis.set_major_locator(FixedLocator([0, 0.005, 0.01, 0.015]))
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())

# FIGURE (c): Relative Intensity (Normalized)
x1, x2 = Read_Two_Column_File('Relative_Intensity/WT_Normalized.dat')
x1 = -(np.array(x1) - np.max(x1))
x2_normalized = -normalize_area_under_curve(x1/np.max(x1), x2)
ax3.plot(x1[::period], x2_normalized[::period], color='k', marker='o', markersize=markersize,
         linestyle='-', linewidth=linewidth, alpha=1.0, fillstyle='none')
ax3.set_xlim([-5, 280])
ax3.set_ylim([0, 4])
ax3.tick_params(which='major', direction='in')
ax3.tick_params(which='minor', direction='in')
ax3.yaxis.set_major_locator(FixedLocator([0, 2, 4]))
ax3.xaxis.set_minor_locator(AutoMinorLocator())
ax3.yaxis.set_minor_locator(AutoMinorLocator())

# Enhance tick parameters for all axes
for ax in [ax1, ax2, ax3]:
    ax.tick_params(which='major', direction='in', bottom=True, top=True, left=True, right=True)
    ax.tick_params(which='minor', direction='in', bottom=True, top=True, left=True, right=True)

# Annotations for subplots
ax1.text(-0.30, 1.05, '(a)', transform=ax1.transAxes, fontsize=fontsize+2, fontweight='normal', va='top')
ax2.text(-0.30, 1.05, '(b)', transform=ax2.transAxes, fontsize=fontsize+2, fontweight='normal', va='top')
ax3.text(-0.30, 1.05, '(c)', transform=ax3.transAxes, fontsize=fontsize+2, fontweight='normal', va='top')

# Axis labels with LaTeX formatting
ax1.set_ylabel(r'$\langle I_{unc}\rangle$', fontsize=fontsize, labelpad=-65)
ax1.set_xlabel(r'$\mathrm{Axonal~Length}~(\mu \mathrm{m})$', fontsize=fontsize)
ax2.set_ylabel(r'$\langle I_{mscar}\rangle$', fontsize=fontsize, labelpad=-65)
ax2.set_xlabel(r'$\mathrm{Axonal~Length}~(\mu \mathrm{m})$', fontsize=fontsize)
ax3.set_ylabel(r'$\langle I_{unc}/I_{mscar} \rangle$', fontsize=fontsize)
ax3.set_xlabel(r'$\mathrm{Axonal~Length}~(\mu \mathrm{m})$', fontsize=fontsize)

plt.show()
fig.savefig('fig2.pdf', dpi=600)

