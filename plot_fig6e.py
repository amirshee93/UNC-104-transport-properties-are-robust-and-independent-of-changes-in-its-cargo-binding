import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib import ticker

fontsize = 22
cmap5 = plt.cm.turbo
colorbar_width = 0.02  # Set the width of the color bar
colorbar_pad = 0.01    # Set the space between the plot and color bar
# Set Parameters
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

fig = plt.figure(figsize=(8, 3))
left_margin = 0.10
right_margin = 0.90
bottom_margin = 0.24
top_margin = 0.95
wspace = 0.0
hspace = 0.0
plt.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin, top=top_margin, wspace=wspace, hspace=hspace)
ax5 = plt.subplot2grid((1, 1), (0, 0))


linewidth = 0.4  # Set the desired line width
for spine in ax5.spines.values():
    spine.set_linewidth(linewidth)


ax5.tick_params(labelsize=fontsize-2)

major_locator = ticker.FixedLocator([0, 0.5, 1])
ax5.yaxis.set_major_locator(major_locator)


minor_locator = ticker.AutoMinorLocator(5)
ax5.yaxis.set_minor_locator(minor_locator)




# FIGURE E

ngrid = 1000
L = 274.74
gamma = 0.0001
D = 6.401
Q = 0.035
def function_ss(x,v):
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
#v_values = np.linspace(1, 10, ngrid)
v_values = np.logspace(np.log10(0.0008), np.log10(0.1), ngrid)


ss = np.zeros((len(x_values), len(v_values)))
for i, x in enumerate(x_values):
    for j, v in enumerate(v_values):
        ss[i,j] = function_ss(x,v)

lambdav_values = (2*D) / (v_values*L) 

x_values = x_values / L

#im5 = ax5.pcolormesh(v_values, x_values, ss, shading='auto', cmap=cmap5, norm=colors.LogNorm(vmin=ss.min(), vmax=ss.max()))

#cbar5 = plt.colorbar(im5, ax=ax5, fraction=colorbar_width, pad=colorbar_pad)
#cbar5.ax.set_ylabel(r'$\boldsymbol{\rho_{\mathrm{s}}}$', fontsize=fontsize, labelpad=5)
#cbar5.ax.tick_params(which='both', direction='in', labelsize=fontsize)

im5 = ax5.pcolormesh(v_values, x_values, ss, shading='auto', cmap=cmap5) #, norm=colors.LogNorm())
cbar5 = plt.colorbar(im5, ax=ax5, fraction=colorbar_width, pad=colorbar_pad)
cbar5.ax.set_ylabel(r'$ 10^{2}\times \rho^{\mathcal{N}}_{s}$', fontsize=fontsize, labelpad=1)
cbar5.ax.tick_params(which='both', direction='in', labelsize=fontsize)  # Set color bar ticks inward


for label in (ax5.get_xticklabels() + ax5.get_yticklabels()):
    label.set_fontweight('bold')

for label in cbar5.ax.get_yticklabels():
    label.set_fontweight('bold')


ax5.set_ylabel(r'$x/L$', fontsize=fontsize)
ax5.tick_params(which='major', direction='in', bottom=True, top=True, left=True, right=True)
ax5.tick_params(which='minor', direction='in', bottom=True, top=True, left=True, right=True)

ax5.set_xlabel(r'$\mathrm{\Omega}$', fontsize=fontsize, labelpad=0)

ax5.set_xlim([0.0008, 0.1])
ax5.set_xscale('log')
ax5.set_ylim([0, 1])


ax5.text(-0.12, 1.02, '(e)', transform=ax5.transAxes, fontsize=fontsize, fontweight='bold', va='top')




# Adjust layout and show/save the plot
#plt.tight_layout()
plt.show()
fig.savefig('fig6e.png', dpi = 600)
