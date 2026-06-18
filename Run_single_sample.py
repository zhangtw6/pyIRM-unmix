# %% Step 1: Import necessary packages

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import re
import scipy
import matplotlib.pyplot as plt
from packaging import version
import importlib

# %% Step 2: Set .py path and add to working path

root = tk.Tk()
root.withdraw()
root.lift()
root.attributes("-topmost", 1)
messagebox.showinfo("","Select the path of Function file")
current_path = filedialog.askdirectory()

print(current_path)

os.chdir(current_path)
if current_path not in sys.path:
    sys.path.append(current_path)

import pyIRM_unmix_functions as fc

importlib.reload(fc)

# %% Step 3: Select files (single file only)

files = fc.open_files()

file = files[2]
print (file)

# %% Step 4: Read IRM data and plot
# Interp IRM data into equal intervals in log scale.
# Smooth data using savgol_filter

D = fc.read_file_AIO(file)
x = D["raw_data"]["field_irm"].copy()
y = D["raw_data"]["remanence_irm"].copy()
print (f"Maximum field: {np.max(x)}")
data = fc.interp_IRM(x, y, xmax=D["raw_data"]["max_field"])
smooth_window = 7
data["smooth"] = fc.smooth_IRM(data["remanence"], smooth_window=smooth_window, polyorder=0)
data["gradient_raw"] = np.gradient(data["remanence"], data["field_log"])
# data["gradient_smooth"] = np.gradient(data["smooth"], data["field_log"])
data["gradient_smooth"] = fc.smooth_IRM(data["gradient_raw"], smooth_window=smooth_window, polyorder=0)

x = np.array(data["field_log"])
yr = np.array(np.maximum(data["gradient_raw"], 0))
ys = np.array(np.maximum(data["gradient_smooth"], 0))

if version.parse(np.__version__) < version.parse("2.0.0"):
    yr = yr/np.trapz(yr, x)
    ys = ys/np.trapz(ys, x)
else:
    yr = yr/np.trapezoid(yr, x)
    ys = ys/np.trapezoid(ys, x)

fig, ax = plt.subplots()
ax.scatter(x, yr, c="gray", s=20)
ax.plot(x, ys, "b", linewidth=1)

# %% Step 5: Exploring optimum number of components using BIC model
# Using BayesianGaussianMixture to get the initial params

gmm = fc.calculate_comps_BGMM_BIC(x, yr, ys, nmin=1, nmax=5, n_iters=10)

# %% Step 6: Set number of components and perform params optimize
# Using scipy.optimize.minize

n = 5 # set the number of components to optimization  

params = gmm[f"comp_nums_{n}"]["params"]

cons = {"type": "eq", "fun": fc.weight_constrain}
results = fc.nls_calculate(x, ys, params, n, cons)

colors = plt.cm.tab20.colors
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs = axs.ravel()

axs[0].scatter(x, yr, c="gray", s=10)
axs[0].plot(x, ys, "black", linewidth=1.5)
for i in range(n):
    color = colors[i*2]
    tmp = fc.calculate_sum_component_SGG_model(x=x, params=results["params_initial"])
    axs[0].plot(x, tmp[f"comp_{i}"], color=color)
    axs[0].plot(x, tmp[f"comp_sum"], color="r")
axs[0].text(
    0.05, 0.95, "RSS: " + f"{results['rss_initial']:.4e}", 
    color="b", fontsize=12,
    transform=axs[0].transAxes,
    verticalalignment='top',
)
# popt_comps = fc.calculate_individual_component_SGG_model(x, params)

axs[1].scatter(x, yr, c="gray", s=10, label="Data")
axs[1].plot(x, ys, "black", linewidth=1.5, label="Smooth")
for i in range(n):
    color = colors[i*2]
    tmp = fc.calculate_sum_component_SGG_model(x=x, params=results["params_popt"])
    axs[1].plot(x, tmp[f"comp_{i}"], color=color, label=f"Comp_{i+1}")
axs[1].plot(x, tmp[f"comp_sum"], color="r", label=f"Sum")
axs[1].text(
    0.05, 0.95, "RSS: " + f"{results['rss_popt']:.4e}",
    color="b", fontsize=12,
    transform=axs[1].transAxes,
    verticalalignment='top',
)
axs[1].legend()

for ax in axs:
    ax.set_title(D["fname"], fontsize=14)
    ax.set_xlabel("Field (Log$_{10}$(mT))", fontsize=12)
    ax.set_ylabel("IRM Gradient", fontsize=12)
    ax.tick_params(axis='both', direction="in", labelsize=10)
plt.show()

print (fc.reshape_parameters(results["params_popt"]))

# %% Step 7: Perform bootstrap sampling procedure and calculate confidence intervals

params = results["params_popt"]

all_results = fc.run_mc_simulation(n_iterations=100, x=x, y=yr, params=params, 
                                   cons=cons, smooth_window=smooth_window,
                                   proportion=0.98, std=0.01)

_, icomp, scomp, confidence  = fc.extract_confidence_intervals(x, y, all_results)
ncomp, nrow, ncol = len(icomp.keys()), len(icomp[list(icomp.keys())[0]]), len(x)

colors = plt.cm.tab20.colors
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(x, yr, c="gray", s=10, label="Data")
ax.plot(x, ys, "black", linewidth=1.5, label="Smooth")
for i in range(ncomp):
    s = list(confidence.keys())[i]
    color_light = colors[i*2+1]
    color_dark = colors[i*2]

    ax.fill_between(
        x,
        confidence[s]["icomp_lower"],
        confidence[s]["icomp_upper"],
        color=color_light,
    )
    ax.plot(x, confidence[s]["icomp_middle"], color=color_dark, linewidth=1.5, label=f"Comp_{i+1}")
   
ax.fill_between(
    x,
    confidence[s]["scomp_lower"],
    confidence[s]["scomp_upper"],
    color=[1.0, 0.5, 0.5],
    )
ax.plot(x, confidence[s]["scomp_middle"], color="r", linewidth=1.5, label=f"Sum")

ax.legend()
ax.set_title(D["fname"], fontsize=14)
ax.set_xlabel("Field (Log$_{10}$(mT))", fontsize=12)
ax.set_ylabel("IRM Gradient", fontsize=12)
ax.tick_params(axis='both', direction="in", labelsize=10)
plt.show()

d = fc.reshape_parameters(params)

print (d)

# %% Step 8: Export data and figure

# Export figure
fpath = D["fpath"]
fname = D["fname"]
expath = os.path.join(fpath, "pyIRM_unmix_processed")

if not os.path.exists(expath):
        os.makedirs(expath)

exfname = os.path.join(expath, fname+".eps")
fig.savefig(exfname, dpi=300)

exfname = os.path.join(expath, fname+".png")
fig.savefig(exfname, dpi=300)

plt.close(fig)
print (exfname)

# Export fit data
d.index.name = fname
exfname = os.path.join(expath, fname+".csv")
d.to_csv(exfname, sep=",", index=True)


# %%
