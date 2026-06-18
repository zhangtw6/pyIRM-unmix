# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:23:14 2026

@author: ZTW
"""

# %% Step 1: Import necessary packages

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
from tqdm import tqdm
import importlib
import copy
from joblib import Parallel, delayed


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
# importlib.reload(fc)

# %% Step 3: Select files (multiple) and pre process

files = fc.open_files()

# file = files[0]

total = (len(files))
pbar = tqdm(total=total)

D = {}
for i in range(len(files)):
    file = files[i]
    fname = os.path.split(file)[1]
    fname = os.path.splitext(fname)[0]
    tmp = fc.read_file_AIO(file)
    D[f"{fname}"] = tmp.copy()
    tx, ty = tmp["raw_data"]["field_irm"], tmp["raw_data"]["remanence_irm"]
    tmp = fc.interp_IRM(x=tx, y=ty, xmax=1000.0)
    D[f"{fname}"]["process_data"] = tmp.copy()
    tx, ty = tmp["field_log"], tmp["remanence"]
    D[f"{fname}"]["process_data"]["smooth"] = fc.smooth_IRM(ty, smooth_window=9, polyorder=0)
    D[f"{fname}"]["process_data"]["gradient_raw"] = np.gradient(ty, tx)
    ty = D[f"{fname}"]["process_data"]["smooth"]
    D[f"{fname}"]["process_data"]["gradient_smooth"] = np.gradient(ty, tx)
    pbar.update(1)
pbar.close()
print (f"Successful load {len(files)} files")

files_name = list(D.keys())

# %% Set initial parameters

n = 4

means = np.array([1.0, 1.25, 1.61, 2.5]) # Need adjust this parameters mannually
stds = np.array([0.25, 0.25, 0.25, 0.25]) # Need adjust this parameters mannually
weights = np.array([0.25, 0.25, 0.25, 0.25])   # Need adjust this parameters mannually

params_initial = {"means": means,
                  "stds": stds,
                  "weights": weights,
                  }

params_initial = fc.params_transfer_GMM_NLS(params_initial)

# i = 0
# print (files[i])
# Results = fc.multiple_process_single_nls_calculate(i=i, file=files[i], params=params_initial, 
#                                                    d=D[f"{files_name[i]}"]["process_data"],
#                                                    comp_nums=n,cons=None, 
#                                                    showfig=True, savefig=False)
# d = fc.reshape_parameters(Results["params_popt"])
# print (d)

# %%

cons = {"type": "eq", "fun": fc.weight_constrain}

Results_list = fc.multiple_process(D, files[:50], files_name[:50], params_initial, comp_nums=n, savefig=True)

Results = {
    result["fname"]: result
    for result in Results_list
}


# %%

data = []    
for i in range(len(Results)):
    fname = list(Results.keys())[i]
    results = Results[f"{fname}"]
    data.append(results["params_popt"])
data = np.array(data)

d = fc.reshape_bounds_multiple(data)
d = fc.reshape_parameters_multiple(data)
d.insert(0, "Sample ID", files_name[:50])

file = files[0]
fpath = os.path.split(file)[0]
fname = os.path.split(file)[1]
fname = os.path.splitext(fname)[0]

expath = os.path.join(fpath, "pyIRM_UnMix_processed")

if not os.path.exists(expath):
    os.makedirs(expath)

exfname = os.path.join(expath, "export.dat")
d.to_csv(exfname, sep="\t", index=False)
    

# %%
