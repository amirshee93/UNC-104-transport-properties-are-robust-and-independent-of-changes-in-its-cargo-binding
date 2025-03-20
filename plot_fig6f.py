import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

fontsize = 22
cmap5 = plt.cm.turbo
colorbar_width = 0.02  # Set the width of the color bar
colorbar_pad = 0.01    # Set the space between the plot and color bar
# Set Parameters
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

fig = plt.figure(figsize=(8, 3))
left_margin = 0.095
right_margin = 0.90
bottom_margin = 0.23
top_margin = 0.95
wspace = 0.40
hspace = 0.30
plt.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin, top=top_margin, wspace=wspace, hspace=hspace)
ax6 = plt.subplot2grid((1, 1), (0, 0))
ax7 = ax6.twinx()


linewidth = 0.4  # Set the desired line width
for spine in ax6.spines.values():
    spine.set_linewidth(linewidth)
for spine in ax7.spines.values():
    spine.set_linewidth(linewidth)


ax6.tick_params(labelsize=fontsize-2)
ax7.tick_params(labelsize=fontsize-2)


major_locator = ticker.FixedLocator([0, 0.5, 1])
ax6.yaxis.set_major_locator(major_locator)
#major_locator = ticker.FixedLocator([0, 0.002, 0.004])
#ax7.yaxis.set_major_locator(major_locator)

minor_locator = ticker.AutoMinorLocator(5)
ax6.yaxis.set_minor_locator(minor_locator)
minor_locator = ticker.AutoMinorLocator(2)
ax7.yaxis.set_minor_locator(minor_locator)






# FIGURE F

ngrid = 1000
L = 274.74
gamma = 0.0001
D = 6.41
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
#v_values = np.linspace(0.0000001, 0.05, ngrid)
v_values = np.logspace(np.log10(0.001), np.log10(1), ngrid)

ss = np.zeros((len(x_values), len(v_values)))
for i, x in enumerate(x_values):
    for j, v in enumerate(v_values):
        ss[i,j] = function_ss(x,v)



min_ss_for_each_v = np.min(ss, axis=0)
min_ss_for_each_v_indices = np.argmin(ss, axis=0)
min_x_for_each_v = x_values[min_ss_for_each_v_indices]


# Find local minima or global minima if no local minimum is found
# Initialize arrays to store minima information
minima_x_for_each_v = []
is_local_minimum = []

for v in v_values:
    ss_values = [function_ss(x, v) for x in x_values]
    local_min_found = False

    for i in range(1, len(ss_values) - 1):
        if ss_values[i] < ss_values[i - 1] and ss_values[i] < ss_values[i + 1]:
            minima_x_for_each_v.append(x_values[i])
            is_local_minimum.append(True)
            local_min_found = True
            break

    if not local_min_found:
        global_min_index = np.argmin(ss_values)
        minima_x_for_each_v.append(x_values[global_min_index])
        is_local_minimum.append(False)


# Collect v and minima values for plotting
v_local_minima = []
minima_local = []
v_global_minima = []
minima_global = []

for i, v in enumerate(v_values):
    if is_local_minimum[i]:
        v_local_minima.append(v)
        minima_local.append(minima_x_for_each_v[i] / L)
    else:
        v_global_minima.append(v)
        minima_global.append(minima_x_for_each_v[i] / L)

        #ax6.scatter(v, minima_x_for_each_v[i]/L, color='red', edgecolor='black', linewidth=0.5, s=10)  # Global minima in red

#ax6.scatter(v_global_minima[::10], minima_global[::10], color='red', edgecolor='black', linewidth=0.5, s=10)

ax6.plot(v_global_minima, minima_global, color='maroon', linewidth=2.5)


# Filter v_values_to_plot and minima_values_to_plot for v_values_to_plot < 0.1
filtered_v_values = [v for v in v_local_minima if v <= 0.1]
filtered_minima_values = [minima_local[i] for i, v in enumerate(v_local_minima) if v <= 0.1]

ax6.scatter(filtered_v_values[::10], filtered_minima_values[::10], color='k', linewidth=1.5, s=10)


filtered_v_values = [v for v in v_local_minima if v >= 0.1]
filtered_minima_values = [minima_local[i] for i, v in enumerate(v_local_minima) if v >= 0.1]

ax6.scatter(filtered_v_values[::10], filtered_minima_values[::10], color='k', linewidth=1.5, s=10)


ax6.tick_params(which='major', direction='in', bottom=True, top=True, left=True, right=False)
ax6.tick_params(which='minor', direction='in', bottom=True, top=True, left=True, right=False)
ax7.tick_params(which='major', direction='in', bottom=True, top=True, left=False, right=True)
ax7.tick_params(which='minor', direction='in', bottom=True, top=True, left=False, right=True)


ax6.set_ylabel(r'$x_{\mathrm{min}}/L$', fontsize=fontsize)
ax6.set_xlabel(r'$\mathrm{\Omega}$', fontsize=fontsize, labelpad=-3)


#ax6.set_xlim([0.1, 1000])
ax6.set_xlim([0.001, 1])

ax6.set_ylim([-0.05, 1.05])

ax6.set_xscale('log')
#v_values = (2 * D) / (v_values * L)
 
ax7.plot(v_values, min_ss_for_each_v, color = 'purple', linewidth=2)
ax7.set_ylabel(r'$10^{2} \times \rho_{s}^{\mathcal{N},~\mathrm{min}}$', color='purple', fontsize=fontsize)
ax7.tick_params(axis='y', labelcolor='purple')
ax7.set_ylim([-0.02, 0.4])

ax6.text(-0.1, 1.02, '(f)', transform=ax6.transAxes, fontsize=fontsize, fontweight='bold', va='top')

# Color the region from Omega = 0.1 to Omega = 0.2
ax6.axvspan(0.04, 0.05, color='blue', alpha=0.3)
#ax6.text(0.57, 0.85, 'Experiment', transform=ax6.transAxes, fontsize=fontsize-5, fontweight='bold', va='top', rotation=90)



# Adjust layout and show/save the plot
#plt.tight_layout()
plt.show()
fig.savefig('fig6f.pdf', dpi = 600)
