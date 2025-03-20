from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter
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

fontsize = 18


def normalize_area_under_curve(x1, x2):
    # Assuming x1 and x2 are numpy arrays
    # Calculate the discrete integral of x2 with respect to x1
    area = np.trapz(x2, x1)
    # Normalize x2 by the total area to ensure the area under the curve is one
    return x2 / area

plt.close('all')
fig = plt.figure(figsize=(12,4))
ax1 = plt.subplot2grid((1, 3), (0, 0))
ax2 = plt.subplot2grid((1, 3), (0, 1))
ax3 = plt.subplot2grid((1, 3), (0, 2))

left_margin = 0.09   # Adjust this value as needed
right_margin = 0.98  # Adjust this value as needed
bottom_margin = 0.15 # Adjust this value as needed
top_margin = 0.94    # Adjust this value as needed
wspace = 0.45
hspace = 0.0
plt.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin, top=top_margin, wspace=wspace, hspace=hspace)


def set_ticks_and_labelsize(ax, interval=1, labelsize=fontsize-2):
    # Set ticks using MultipleLocator
    #ax.xaxis.set_major_locator(plt.MultipleLocator(interval))
    ax.yaxis.set_major_locator(plt.MultipleLocator(interval))
    
    # Set tick label size
    ax.tick_params(labelsize=labelsize)

# Assuming ax1, ax2, ..., ax5 are your axes objects
set_ticks_and_labelsize(ax1)
set_ticks_and_labelsize(ax2)
set_ticks_and_labelsize(ax3)
for ax in [ax1, ax2, ax3]:
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))




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



gamma = 0.0001

def function1(x1):
    L = np.max(x1)  # Calculate L based on x1 if it's not constant
    lgamma = np.sqrt(D / gamma)
    lv = 2 * D / v
    ld = lgamma / np.sqrt(1.0 + (lgamma / lv) ** 2)
    
    exp_term_1 = np.exp(-(L-x1) / ld) / (lv - ld)
    exp_term_2 = np.exp((L - x1) / ld) / (lv + ld)
    exp_term_3 = np.exp(x1 / lv)

    norm = ((lv**2) - (ld**2))/(2.0*lv*ld*np.sinh(L/ld))

    function_ss = norm * (exp_term_1 + exp_term_2) * exp_term_3

    return function_ss


def function2(x1):
    L = np.max(x1)  # Calculate L based on x1 if it's not constant
    lgamma = np.sqrt(D / gamma)
    lv = 2 * D / v
    ld = lgamma / np.sqrt(1.0 + (lgamma / lv) ** 2)
    
    exp_term_1 = np.exp(-(L-x1) / ld) / (lv - ld)
    exp_term_2 = np.exp((L - x1) / ld) / (lv + ld)
    exp_term_3 = np.exp(x1 / lv)

    norm = (Q * lv * ld )/(2.0*D*np.sinh(L/ld))

    function_ss = norm * (exp_term_1 + exp_term_2) * exp_term_3

    return function_ss




markersize = 4   
linewidth = 0
period = 1
    
# Fig. a      WT
x1, x2, x3 = Read_Three_Column_File('Data_Amir/WT_Normalized.dat')
x1 = -(np.array(x1) - np.max(x1))
x2_normalized = -normalize_area_under_curve(x1, x2)
ax1.plot(x1[::period], x2_normalized[::period], color='grey', marker='o', markersize=markersize, linestyle='-', linewidth=linewidth, alpha=1.0, fillstyle='none') #, label=r'$\boldsymbol{\mathrm{exp.}}$')

D = 6.41
v = 0.05

ax1.plot(x1, function1(x1),'r--', linewidth=1.5)
ax1.set_xlim([-10, np.max(x1)+10])
ax1.set_ylim([0, 0.015])
ax1.tick_params(which='major', direction='in')
ax1.tick_params(which='minor', direction='in')
ax1.xaxis.set_major_locator(FixedLocator([0, 100, 200]))
ax1.yaxis.set_major_locator(FixedLocator([0, 0.005, 0.01, 0.015]))

#ax1.set_xlabel('Position (\u03bcm)')

# Fig. b   UBA-1
x1, x2, x3 = Read_Three_Column_File('Data_Amir/UBA1_Normalized.dat')
x1 = -(np.array(x1) - np.max(x1))
x2_normalized = -normalize_area_under_curve(x1, x2)
ax2.plot(x1[::period], x2_normalized[::period], color='grey', marker='o', markersize=markersize, linestyle='-', linewidth=linewidth, alpha=1.0, fillstyle='none') #, label=r'$\boldsymbol{\mathrm{exp.}}$')

D = 5.94
v = 0.05

ax2.plot(x1, function1(x1),'r--', linewidth=1.5)
ax2.set_xlim([-10, np.max(x1)+10])
ax2.set_ylim([0, 0.015])
ax2.tick_params(which='major', direction='in')
ax2.tick_params(which='minor', direction='in')
ax2.xaxis.set_major_locator(FixedLocator([0, 100, 200]))
ax2.yaxis.set_major_locator(FixedLocator([0, 0.005, 0.01, 0.015]))


# Fig. c   FBXB-65
x1, x2, x3 = Read_Three_Column_File('Data_Amir/F7_Normalized.dat')
x1 = -(np.array(x1) - np.max(x1))
x2_normalized = -normalize_area_under_curve(x1, x2)
ax3.plot(x1[::period], x2_normalized[::period], color='grey', marker='o', markersize=markersize, linestyle='-', linewidth=linewidth, alpha=1.0, fillstyle='none') #, label=r'$\boldsymbol{\mathrm{exp.}}$')

D = 6.37
v = 0.04

ax3.plot(x1, function1(x1),'r--', linewidth=1.5)
#x2_normalized = -normalize_area_under_curve(x1, function1(x1))
#ax3.plot(x1, x2_normalized,'r--', linewidth=1.5)
ax3.set_xlim([-10, np.max(x1)+10])
ax3.set_ylim([0, 0.015])
ax3.tick_params(which='major', direction='in')
ax3.tick_params(which='minor', direction='in')
ax3.xaxis.set_major_locator(FixedLocator([0, 100, 200]))
ax3.yaxis.set_major_locator(FixedLocator([0, 0.005, 0.01, 0.015]))



ax1.text(0.75, 0.95, r'$\mathrm{WT}$', transform=ax1.transAxes, fontsize=fontsize-1, fontweight='normal', va='top')
ax2.text(0.65, 0.95, r'$\mathit{uba}$-$\mathit{1}$', transform=ax2.transAxes, fontsize=fontsize-1, fontweight='normal', va='top')
ax3.text(0.60, 0.95, r'$\mathit{fbxb}$-$\mathit{65}$', transform=ax3.transAxes, fontsize=fontsize-1, fontweight='normal', va='top')



ax1.tick_params(which='major', direction='in', bottom=True, top=True, left=True, right=True)
ax1.tick_params(which='minor', direction='in', bottom=True, top=True, left=True, right=True)
ax2.tick_params(which='major', direction='in', bottom=True, top=True, left=True, right=True)
ax2.tick_params(which='minor', direction='in', bottom=True, top=True, left=True, right=True)
ax3.tick_params(which='major', direction='in', bottom=True, top=True, left=True, right=True)
ax3.tick_params(which='minor', direction='in', bottom=True, top=True, left=True, right=True)

#ax1.text(0.75, 0.95, r'$\boldsymbol{\mathrm{WT}}$', transform=ax1.transAxes, fontsize=fontsize-1, fontweight='normal', va='top')
#ax2.text(0.65, 0.95, r'$\boldsymbol{\mathrm{uba}}$-$\boldsymbol{\mathrm{1}}$', transform=ax2.transAxes, fontsize=fontsize-1, fontweight='normal', va='top')

ax1.text(-0.35, 1.05, '(a)', transform=ax1.transAxes,fontsize=fontsize, fontweight='normal', va='top')
ax2.text(-0.35, 1.05, '(b)', transform=ax2.transAxes,fontsize=fontsize, fontweight='normal', va='top')
ax3.text(-0.35, 1.05, '(c)', transform=ax3.transAxes,fontsize=fontsize, fontweight='normal', va='top')


#ax1.legend(loc='upper left',fontsize=fontsize-2, markerfirst=True, labelspacing=-0.25)


ax1.set_ylabel(r'$\rho^{\mathcal{N}}_{s}(x)$',fontsize=fontsize)
ax1.set_xlabel(r'$x~(\mu\mathrm{m})$',fontsize=fontsize)
ax2.set_ylabel(r'$\rho^{\mathcal{N}}_{s}(x)$',fontsize=fontsize)
ax2.set_xlabel(r'$x~(\mu\mathrm{m})$',fontsize=fontsize)
ax3.set_ylabel(r'$\rho^{\mathcal{N}}_{s}(x)$',fontsize=fontsize)
ax3.set_xlabel(r'$x~(\mu\mathrm{m})$',fontsize=fontsize)

plt.show()


fig.savefig('fig5.pdf', dpi = 600)
