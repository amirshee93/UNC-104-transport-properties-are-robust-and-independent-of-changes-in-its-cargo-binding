import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FixedLocator, FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.font_manager import FontProperties



fontsize = 24
cmap5 = plt.cm.turbo
colorbar_width = 0.02  # Set the width of the color bar
colorbar_pad = 0.01    # Set the space between the plot and color bar
# Set Parameters
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

fig = plt.figure(figsize=(8, 3.75))
left_margin = 0.115
right_margin = 0.98
bottom_margin = 0.18
top_margin = 0.90
wspace = 0.40
hspace = 0.0
plt.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin, top=top_margin, wspace=wspace, hspace=hspace)
ax1 = plt.subplot2grid((1, 4), (0, 0))
ax2 = plt.subplot2grid((1, 4), (0, 1))
ax3 = plt.subplot2grid((1, 4), (0, 2))
ax4 = plt.subplot2grid((1, 4), (0, 3))


linewidth = 0.5  # Set the desired line width
for spine in ax1.spines.values():
    spine.set_linewidth(linewidth)
for spine in ax2.spines.values():
    spine.set_linewidth(linewidth)
for spine in ax3.spines.values():
    spine.set_linewidth(linewidth)
for spine in ax4.spines.values():
    spine.set_linewidth(linewidth)

ax1.tick_params(labelsize=fontsize-6)
ax2.tick_params(labelsize=fontsize-6)
ax3.tick_params(labelsize=fontsize-6)
ax4.tick_params(labelsize=fontsize-6)


# Set x-axis major ticks at specific locations
major_locator = ticker.FixedLocator([0, 0.5, 1.0])
ax1.xaxis.set_major_locator(major_locator)
ax2.xaxis.set_major_locator(major_locator)
ax3.xaxis.set_major_locator(major_locator)
ax4.xaxis.set_major_locator(major_locator)

minor_locator = ticker.AutoMinorLocator(2)  # Set minor tick frequency
ax1.xaxis.set_minor_locator(minor_locator)
ax2.xaxis.set_minor_locator(minor_locator)
ax3.xaxis.set_minor_locator(minor_locator)
ax4.xaxis.set_minor_locator(minor_locator)


minor_locator = ticker.AutoMinorLocator(2)  # Set minor tick frequency
ax1.yaxis.set_minor_locator(minor_locator)
minor_locator = ticker.AutoMinorLocator(2)
ax2.yaxis.set_minor_locator(minor_locator)
minor_locator = ticker.AutoMinorLocator(2)
ax3.yaxis.set_minor_locator(minor_locator)
minor_locator = ticker.AutoMinorLocator(2)  # Set minor tick frequency
ax4.yaxis.set_minor_locator(minor_locator)







# Plot kymographs
# Set the color bar range
ngrid = 1000
L = 274.74
gamma = 0.0001
D = 6.41
Q = 0.035
def function_ss(x):
    lgamma = np.sqrt(D / gamma)
    lv = 2 * D / v
    ld = lgamma / np.sqrt(1.0 + (lgamma / lv) ** 2)
    
    exp_term_1 = np.exp(-(L-x) / ld) / (lv - ld)
    exp_term_2 = np.exp((L - x) / ld) / (lv + ld)
    exp_term_3 = np.exp(x / lv)

    norm = ((lv**2) - (ld**2))/(2.0*lv*ld*np.sinh(L/ld))
    #norm = (Q * lv * ld )/(2.0*D*np.sinh(L/ld))


    ss_values = norm * (exp_term_1 + exp_term_2) * exp_term_3
    return ss_values * 100



x_values = np.linspace(0, L, ngrid)
v = 0.000000001
ss = np.zeros((len(x_values)))
for i, x in enumerate(x_values):
    ss[i] = function_ss(x)
ax1.plot(x_values/L, ss, linewidth=2, color='k')

v = 0.01
ss = np.zeros((len(x_values)))
for i, x in enumerate(x_values):
    ss[i] = function_ss(x)
ax2.plot(x_values/L, ss, linewidth=2, color='k')

v = 0.05
ss = np.zeros((len(x_values)))
for i, x in enumerate(x_values):
    ss[i] = function_ss(x)
ax3.plot(x_values/L, ss, linewidth=2, color='r', linestyle = '--')


v = 0.5
ss = np.zeros((len(x_values)))
for i, x in enumerate(x_values):
    ss[i] = function_ss(x)
ax4.plot(x_values/L, ss, linewidth=2, color='k')



ax1.tick_params(which='major', direction='in', bottom=True, top=True, left=True, right=True)
ax1.tick_params(which='minor', direction='in', bottom=True, top=True, left=True, right=True)
ax2.tick_params(which='major', direction='in', bottom=True, top=True, left=True, right=True)
ax2.tick_params(which='minor', direction='in', bottom=True, top=True, left=True, right=True)
ax3.tick_params(which='major', direction='in', bottom=True, top=True, left=True, right=True)
ax3.tick_params(which='minor', direction='in', bottom=True, top=True, left=True, right=True)
ax4.tick_params(which='major', direction='in', bottom=True, top=True, left=True, right=True)
ax4.tick_params(which='minor', direction='in', bottom=True, top=True, left=True, right=True)





ax1.set_ylabel(r'$10^{2}\times\rho^{\mathcal{N}}_{s}$', fontsize=fontsize-2)
ax1.set_xlabel(r'$x/L$', fontsize=fontsize-7, labelpad=2)
#ax2.set_ylabel('$\\rho_{\\mathrm{ss}}$', fontsize=fontsize)
ax2.set_xlabel(r'$x/L$', fontsize=fontsize-7, labelpad=2)
#ax3.set_ylabel('$\\rho_{\\mathrm{ss}}$', fontsize=fontsize)
ax3.set_xlabel(r'$x/L$', fontsize=fontsize-7, labelpad=2)
#ax4.set_ylabel('$\\rho_{\\mathrm{ss}}$', fontsize=fontsize)
ax4.set_xlabel(r'$x/L$', fontsize=fontsize-7, labelpad=2)



ax1.set_xlim([0, 1])
ax2.set_xlim([0, 1])
ax3.set_xlim([0, 1])
ax4.set_xlim([0, 1])



ax1.set_ylim([0.29, 0.51])
ax2.set_ylim([0.335, 0.43])
ax3.set_ylim([0.18, 0.8])
ax4.set_ylim([-0.2, 8])
#ax4.set_ylim([0.06, 0.08])





# Create an inset in panel (d) with size and position relative to ax4
inset_ax = inset_axes(ax4, width="50%", height="40%", loc='center right', borderpad=1)

# Plot data in the inset. This example simply reuses the ss and x_values variables,
# but you might want to plot something different here.
inset_ax.plot(x_values/L, ss, linewidth=1, color='k')

# Optionally, customize the inset axes (e.g., set limits, labels, tick sizes)
inset_ax.set_xlim(0, (L-100)/L)  # Example: Adjust the x-axis limits
inset_ax.set_ylim(0.0192, 0.021)    # Example: Adjust the y-axis limits
inset_ax.tick_params(axis='both', which='major', direction='in', labelsize=14)  # Adjust tick parameters
inset_ax.tick_params(axis='both', which='minor', direction='in', labelsize=14)  # Adjust tick parameters

for spine in inset_ax.spines.values():
    spine.set_linewidth(linewidth)

inset_ax.xaxis.set_major_locator(FixedLocator([0, 0.5]))
#inset_ax.yaxis.set_major_locator(FixedLocator([0.00019, 0.00020, 0.00021]))

minor_locator = ticker.AutoMinorLocator(2)
inset_ax.xaxis.set_minor_locator(minor_locator)
minor_locator = ticker.AutoMinorLocator(2)
inset_ax.yaxis.set_minor_locator(minor_locator)

#ax3.text(0.10, 0.9, r'$\boldsymbol{\mathrm{Experiemental}}$', transform=ax3.transAxes, fontsize=fontsize-10, fontweight='bold', va='top')
#ax3.text(0.10, 0.8, r'$\boldsymbol{\mathrm{Observation}}$', transform=ax3.transAxes, fontsize=fontsize-10, fontweight='bold', va='top')


ax1.text(-0.2, 1.12, r'(a)', transform=ax1.transAxes, fontsize=fontsize-4, fontweight='bold', va='top')
ax2.text(-0.2, 1.12, r'(b)', transform=ax2.transAxes, fontsize=fontsize-4, fontweight='bold', va='top')
ax3.text(-0.2, 1.12, r'(c)', transform=ax3.transAxes, fontsize=fontsize-4, fontweight='bold', va='top')
ax4.text(-0.2, 1.12, r'(d)', transform=ax4.transAxes, fontsize=fontsize-4, fontweight='bold', va='top')


ax1.text(0.18, 0.95, r'$\mathrm{\Omega=0.0}$', transform=ax1.transAxes, fontsize=fontsize-4, fontweight='bold', va='top')
ax2.text(0.14, 0.95, r'$\mathrm{\Omega=0.01}$', transform=ax2.transAxes, fontsize=fontsize-4, fontweight='bold', va='top')
ax3.text(0.14, 0.95, r'$\mathrm{\Omega=0.05}$', transform=ax3.transAxes, fontsize=fontsize-4, fontweight='bold', va='top')
ax4.text(0.18, 0.95, r'$\mathrm{\Omega=0.5}$', transform=ax4.transAxes, fontsize=fontsize-4, fontweight='bold', va='top')

#ax1.set_title(r'$\boldsymbol{\mathrm{\Omega=0.0}}$',fontsize=fontsize-4, pad=10.5)
#ax2.set_title(r'$\boldsymbol{\mathrm{\Omega=0.01}}$',fontsize=fontsize-4, pad=-0.5)
#ax3.set_title(r'$\boldsymbol{\mathrm{\Omega=0.05}}$',fontsize=fontsize-4, pad=-0.5)
#ax4.set_title(r'$\boldsymbol{\mathrm{\Omega=0.5}}$',fontsize=fontsize-4, pad=-0.5)

# Adjust layout and show/save the plot
#plt.tight_layout()
plt.show()
fig.savefig('fig6abcd.pdf', dpi = 600)
