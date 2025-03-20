from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import gridspec
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
import seaborn as sb
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib as mpl
from PIL import Image
from io import BytesIO
from scipy.special import erf
from scipy.integrate import simpson

fontsize = 20
colorbar_width = 0.02
colorbar_pad = 0.01

plt.rcParams['text.usetex'] = True
# Ensure the 'amsmath' package is included in the preamble
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


plt.rcParams['font.family'] = 'Helvetica'

plt.close('all')
fig = plt.figure(figsize=(8,6))

left_margin = 0.10
right_margin = 0.96
bottom_margin = 0.10
top_margin = 0.98
wspace = 0.30
hspace = 0.30
plt.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin, top=top_margin, wspace=wspace, hspace=hspace)

ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)


linewidth = 0.5  # Set the desired line width
for spine in ax1.spines.values():
    spine.set_linewidth(linewidth)
for spine in ax2.spines.values():
    spine.set_linewidth(linewidth)
for spine in ax3.spines.values():
    spine.set_linewidth(linewidth)


def Read_Two_Column_File(file_name):
    with open(file_name, 'r') as data:
        x = []
        y = []
        for line in data:
            p = line.split()
            x.append(float(p[0]))
            y.append(float(p[1]))

    return x, y



# Raw 

data1 = np.loadtxt('Raw/out2.csv', delimiter=',', skiprows=1)
x, rho1 = data1[:, 0], data1[:, 1]
ax1.plot(x[::3], rho1[::3], marker = 'o', markersize = 4, linestyle='None', markerfacecolor='none', label = r'$1~\mathrm{s}$', color = 'k')

data1 = np.loadtxt('Raw/out4.csv', delimiter=',', skiprows=1)
x, rho1 = data1[:, 0], data1[:, 1]
ax1.plot(x[::3], rho1[::3], marker = 's', markersize = 4, linestyle='None', markerfacecolor='none', label = r'$31~\mathrm{s}$', color = 'maroon')

data1 = np.loadtxt('Raw/out5.csv', delimiter=',', skiprows=1)
x, rho1 = data1[:, 0], data1[:, 1]
ax1.plot(x[::3], rho1[::3], marker = '^', markersize = 4, linestyle='None', markerfacecolor='none', label = r'$92~\mathrm{s}$', color = 'g')


# Raw with bin averaging 10

'''
data1 = np.loadtxt('Raw_10/out2.csv', delimiter=',', skiprows=1)
x, rho1 = data1[:, 0], data1[:, 1]
ax1.plot(x[::3], rho1[::3], linestyle='-', linewidth=2, color='k')

data1 = np.loadtxt('Raw_10/out4.csv', delimiter=',', skiprows=1)
x, rho1 = data1[:, 0], data1[:, 1]
ax1.plot(x[::3], rho1[::3], linestyle='-', linewidth=2, color='maroon')

data1 = np.loadtxt('Raw_10/out5.csv', delimiter=',', skiprows=1)
x, rho1 = data1[:, 0], data1[:, 1]
ax1.plot(x[::3], rho1[::3], linestyle='-', linewidth=2, color='g')




x, y = Read_Two_Column_File('kymograph/experimental/out2.csv')
x = np.array(x) + 1.095
ax1.plot(x, y, marker = 'o', markersize =3, linestyle='None', markerfacecolor='none', label = '$0~\\mathrm{s}$', color = 'k')
x, y = Read_Two_Column_File('WT/out4.csv')
x = np.array(x) + 1.095
ax1.plot(x, y, marker = '^', markersize =3, linestyle='None', markerfacecolor='none', label = '$60~\\mathrm{s}$', color = 'm')
x, y = Read_Two_Column_File('WT/out5.csv')
x = np.array(x) + 1.095
ax1.plot(x, y, marker = 's', markersize =3, linestyle='None', markerfacecolor='none', label = '$120~\\mathrm{s}$', color = 'g')
'''


ngrid = 1000
deltaL = 40 #18.835 * 2
L = np.max(x) - np.min(x)
a = deltaL / 2.0

# Solid line with t 1/4
# t1/4 = 31.8370
D = 4.0305
x_values = np.linspace(-L/2, L/2, ngrid)

t = 1
def function_ss(x):
    ss_values = 0.5 * (2 - erf((a - x) / np.sqrt(4.0 * D * t)) - erf((a + x) / np.sqrt(4.0 * D * t)))
    return ss_values

ss = np.zeros((len(x_values)))
for i, x in enumerate(x_values):
      ss[i] = function_ss(x)

ax1.plot(x_values, ss, color = 'k')

t = 31
def function_ss(x):
    ss_values = 0.5 * (2 - erf((a - x) / np.sqrt(4.0 * D * t)) - erf((a + x) / np.sqrt(4.0 * D * t)))
    return ss_values

ss = np.zeros((len(x_values)))
for i, x in enumerate(x_values):
      ss[i] = function_ss(x)

ax1.plot(x_values, ss, color = 'maroon')
 

t = 92
def function_ss(x):
    ss_values = 0.5 * (2 - erf((a - x) / np.sqrt(4.0 * D * t)) - erf((a + x) / np.sqrt(4.0 * D * t)))
    return ss_values

ss = np.zeros((len(x_values)))
for i, x in enumerate(x_values):
      ss[i] = function_ss(x)

ax1.plot(x_values, ss, color = 'green')
 

# Dashed line plot with t 1/2
# t1/2 = 58.0255
D = 6.4695 
x_values = np.linspace(-L/2, L/2, ngrid)

t = 1
def function_ss(x):
    ss_values = 0.5 * (2 - erf((a - x) / np.sqrt(4.0 * D * t)) - erf((a + x) / np.sqrt(4.0 * D * t)))
    return ss_values

ss = np.zeros((len(x_values)))
for i, x in enumerate(x_values):
      ss[i] = function_ss(x)

ax1.plot(x_values, ss, linestyle='--', linewidth=2, color = 'k')

t = 31
def function_ss(x):
    ss_values = 0.5 * (2 - erf((a - x) / np.sqrt(4.0 * D * t)) - erf((a + x) / np.sqrt(4.0 * D * t)))
    return ss_values

ss = np.zeros((len(x_values)))
for i, x in enumerate(x_values):
      ss[i] = function_ss(x)

ax1.plot(x_values, ss, linestyle='--', linewidth=2, color = 'maroon')
 

t = 92
def function_ss(x):
    ss_values = 0.5 * (2 - erf((a - x) / np.sqrt(4.0 * D * t)) - erf((a + x) / np.sqrt(4.0 * D * t)))
    return ss_values

ss = np.zeros((len(x_values)))
for i, x in enumerate(x_values):
      ss[i] = function_ss(x)

ax1.plot(x_values, ss, linestyle='--', linewidth=2, color = 'green')
 


ax1.axvspan(-deltaL/2, deltaL/2, facecolor='#a484ac', alpha=0.5)
ax1.axvspan(-2.015, 2.015, facecolor='0.1', alpha=0.5)
ax1.set_xlim([-45, 47])
ax1.set_ylim([-.15, 1.75])
ax1.tick_params(which='major', direction='in',labelsize=16)
ax1.tick_params(which='minor', direction='in')
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.set_ylabel(r'$\phi(x,t)$',fontsize=fontsize)
ax1.set_xlabel(r'$x~(\mu\mathrm{m})$',fontsize=fontsize, labelpad=-0.5)
ax1.legend(loc='upper right',prop={"size":12}, labelspacing=-0.1)



def function_ss(x,t):
    ss_values = 0.5 * (2 - erf((a - x) / np.sqrt(4.0 * D * t)) - erf((a + x) / np.sqrt(4.0 * D * t)))
    return ss_values

x_values = np.linspace(-2.015, 2.015, ngrid)
t_values = np.linspace(0.00001, 300, ngrid)

ss = np.zeros((len(x_values), len(t_values)))
for i, x in enumerate(x_values):
    for j, t in enumerate(t_values):
        ss[i,j] = function_ss(x,t) 

# Calculate spatial average over x for each t
spatial_averages = np.mean(ss, axis=0)  # Averaging across the x dimension

# Define a function to get spatial average as a function of time
def spatial_average_as_function_of_time(t):
    return spatial_averages[t_values-t]

#ax2.plot(t_values, spatial_averages, label='Spatial Average of ss')

data1 = np.loadtxt('Raw/out6.csv', delimiter=',', skiprows=1)
time, rho1 = data1[:, 0], data1[:, 1]
ax2.plot(time, rho1, marker = 'o', markersize = 4, linestyle='None', markerfacecolor='none', label = r'$120~\mathrm{s}$', color = 'g')

'''
# t1/2

D = 6.4695 

def function_ss(x,t):
    ss_values = 0.5 * (2 - erf((a - x) / np.sqrt(4.0 * D * t)) - erf((a + x) / np.sqrt(4.0 * D * t)))
    return ss_values

# Define the window range
x_start = -2.0  # example value, change as needed
x_end = 2.0  # example value, change as needed
num_points = 100  # example value, change as needed

# Generate the x values over the window
x_values = np.linspace(x_start, x_end, num_points)

# Define the time range
t_start = 1  # start time
t_end = 150  # end time
t_steps = 150  # number of time steps

# Generate the time values
t_values = np.linspace(t_start, t_end, t_steps)

# Calculate the mean integrated intensity as a function of time
mean_intensity_values = []

for t in t_values:
    ss_values = function_ss(x_values, t)
    integrated_intensity = simpson(y=ss_values, x=x_values)
    mean_intensity = integrated_intensity / (x_end - x_start)
    mean_intensity_values.append(mean_intensity)
# Convert the result to a NumPy array for easier handling
mean_intensity_values = np.array(mean_intensity_values)

ax2.plot(t_values, mean_intensity_values, linestyle = '--',  linewidth = 2.5 , color = 'k')


# t1/4
D = 4.0305

def function_ss(x,t):
    ss_values = 0.5 * (2 - erf((a - x) / np.sqrt(4.0 * D * t)) - erf((a + x) / np.sqrt(4.0 * D * t)))
    return ss_values

# Define the window range
x_start = -2.0  # example value, change as needed
x_end = 2.0  # example value, change as needed
num_points = 100  # example value, change as needed

# Generate the x values over the window
x_values = np.linspace(x_start, x_end, num_points)

# Define the time range
t_start = 1  # start time
t_end = 150  # end time
t_steps = 150  # number of time steps

# Generate the time values
t_values = np.linspace(t_start, t_end, t_steps)

# Calculate the mean integrated intensity as a function of time
mean_intensity_values = []

for t in t_values:
    ss_values = function_ss(x_values, t)
    integrated_intensity = simpson(y=ss_values, x=x_values)
    mean_intensity = integrated_intensity / (x_end - x_start)
    mean_intensity_values.append(mean_intensity)

# Convert the result to a NumPy array for easier handling
mean_intensity_values = np.array(mean_intensity_values)

ax2.plot(t_values, mean_intensity_values, linestyle = '-' , linewidth = 2.5, color = 'k')

'''

# t 1/2
D = 6.4695 
def function_ss(t):
    ss_values = 1 - erf(a / np.sqrt(4.0 * D * t))
    return ss_values

# Define the time range
t_start = 1  # start time
t_end = 150  # end time
t_steps = 150  # number of time steps

# Generate the time values
t_values = np.linspace(t_start, t_end, t_steps)

intensity_values = []
for t in t_values:
    ss_values = function_ss(t)
    intensity_values.append(ss_values)

# Convert the result to a NumPy array for easier handling
intensity_values = np.array(intensity_values)

ax2.plot(t_values, intensity_values, linestyle = '--' , linewidth = 2.5, color = 'k')

# t 1/4
D = 4.0305
def function_ss(t):
    ss_values = 1 - erf(a / np.sqrt(4.0 * D * t))
    return ss_values

# Define the time range
t_start = 1  # start time
t_end = 150  # end time
t_steps = 150  # number of time steps

# Generate the time values
t_values = np.linspace(t_start, t_end, t_steps)

intensity_values = []
for t in t_values:
    ss_values = function_ss(t)
    intensity_values.append(ss_values)

# Convert the result to a NumPy array for easier handling
intensity_values = np.array(intensity_values)

ax2.plot(t_values, intensity_values, linestyle = '-' , linewidth = 2.5, color = 'k')



#x, y = Read_Two_Column_File('WT/out6.csv')
#ax2.plot(x, y)
#ax2.plot(31.8370,0.10, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
ax2.axvline(x = 31.8370, ymin = 0, ymax = 0.24, color = 'r', linestyle='dashed', alpha=0.5)
ax2.text(0.22, 0.16, '$t_{1/4}$', transform=ax2.transAxes, fontsize=16, fontweight='normal', va='top')
#ax2.plot(58.0255,0.275, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
ax2.axvline(x = 58.0255, ymin = 0, ymax = 0.50, color = 'r', linestyle='dashed', alpha=0.5)
ax2.text(0.40, 0.16, '$t_{1/2}$', transform=ax2.transAxes, fontsize=16, fontweight='normal', va='top')
ax2.set_xlim([0, 150])
ax2.set_ylim([0.0, 0.9])
ax2.tick_params(which='major', direction='in',labelsize=16)
ax2.tick_params(which='minor', direction='in')
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.set_ylabel(r'$\phi(0,t)$',fontsize=fontsize)
ax2.set_xlabel(r'$t~(\mathrm{s})$',fontsize=fontsize, labelpad=-0.5)


data1 = np.loadtxt('D_WT.txt')
data2 = np.loadtxt('D_UBA-1.txt')
data3 = np.loadtxt('D_1-F7.txt')
x = [r"$\mathrm{WT}$",r"$\mathit{uba}$-$\mathit{1}$",r"$\mathit{fbxb}$-$\mathit{65}$"]
data = [data1, data2, data3]
sb.violinplot(data=data, ax=ax3)
ax3.set_xticks(range(len(data)), x)
ax3.tick_params(which='major', direction='in',labelsize=16)
ax3.tick_params(which='minor', direction='in')
ax3.set_ylabel(r'$D(\mu\mathrm{m}^2\mathrm{s}^{-1})$',fontsize=fontsize)


ax1.text(-0.22, 1.025, '(a)', transform=ax1.transAxes,
      fontsize=20, fontweight='normal', va='top')
ax2.text(-0.25, 1.025, '(b)', transform=ax2.transAxes,
      fontsize=20, fontweight='normal', va='top')
ax3.text(-0.10, 1.08, '(c)', transform=ax3.transAxes,
      fontsize=20, fontweight='normal', va='top')





#plt.tight_layout()
plt.show()


fig.savefig('fig4.pdf', dpi = 600)
