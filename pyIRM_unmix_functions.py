# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:26:40 2026

@author: ZHANGTW
"""

import os
import sys
import time
import numpy as np
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import re
import scipy
import xml.etree.ElementTree as ET
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from joblib import Parallel, delayed
from tqdm import tqdm
import copy
from packaging import version
import psutil

def set_work_path():
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", 1)
    return filedialog.askdirectory()


def open_file():
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)
    files = filedialog.askopenfilename(title="Select data file")
    root.destroy()
    return files

def open_files():
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)
    files = filedialog.askopenfilenames(title="Select data files")
    root.destroy()
    return files

# %%
def read_file_AIO(file):
    
    # files = open_file()
    # file = files[0]

    fpath = os.path.split(file)[0]
    fname = os.path.split(file)[1]
    fname = os.path.splitext(fname)[0]

    nbytes = 4096
    with open(file, 'rb') as f:
        raw = str(f.read(nbytes))

    if "MicroMag 2900/3900" in raw:
        D = read_VSM3900_irm(file)
        # print ("1")
    elif "CoilSet" in raw:
        D = read_VSM8600_vers_irm(file)
        # print ("2")
    elif "Coil set" in raw:
        D = read_VSM8600_csv_irm(file)
        # print ("3")
    else:
        D = read_generic_irm(file)
        # print ("4")

    D["fpath"] = fpath
    D["fname"] = fname

    return D

def read_VSM8600_vers_irm(file):

    data = ET.parse(file)
    root = data.getroot()
           
    DiagnosticInfo = root.findall('.//ResultsProfileSequence/Nodes/ResultNode/DiagnosticInfo')
    num = len(DiagnosticInfo)
    for j in range(0, num):
        DiagnosticInfo.append(str.split(DiagnosticInfo[j].text, sep=" ")[0])
    del DiagnosticInfo[:int(len(DiagnosticInfo)/2)]
    del DiagnosticInfo[-1]
        
    Title = root.findall('.//ResultCharts/MeasurementChart/Title')
    num = len(Title)
    for j in range(0, num):
        Title.append(Title[j].text)
    del Title[:int(len(Title)/2)]
    
    idx =  DiagnosticInfo.index("Remanence")+1
    idx =  Title.index(str(idx))
    
    PlotSeries = root.findall('.//ResultCharts/MeasurementChart/PlotSeries')
    PlotSeries = PlotSeries[idx]

    ChartDataPoint = PlotSeries.findall(".//MeasurementChartPlotSeries/Measurements/ChartDataPoint")
    
    d = np.full((len(ChartDataPoint),3),fill_value=np.nan)
    names = ["Segment","X","Y"]
    d = pd.DataFrame(d, columns=names)
    diagnostcinfo = "Remanence"
        
    D = {}
    D["DiagnosticInfo"] = diagnostcinfo
    D["XAxisType"] = ChartDataPoint[0].findtext('MeasurementXAxisType')
    D["YAxisType"] = ChartDataPoint[0].findtext('MeasurementYAxisType')
    D["StartTimestamp"] = ChartDataPoint[0].findtext('Timestamp')
    D["EndTimestamp"] = ChartDataPoint[len(ChartDataPoint)-1].findtext('Timestamp')
    D['StepIteration'] = float(ChartDataPoint[0].findtext('StepIteration'))
    D['Step'] = float(ChartDataPoint[0].findtext('Step'))
        
    for k in range(0, len(ChartDataPoint)):
        d.loc[k, "Segment"] = float(ChartDataPoint[k].findtext('Segment'))
        d.loc[k, "X"] = float(ChartDataPoint[k].findtext('X'))
        d.loc[k, "Y"] = float(ChartDataPoint[k].findtext('Y'))
    
    segment = pd.unique(d["Segment"])

    if len(segment) != 1:
        for i in segment:
            tmp = d[d["Segment"]==i]
            if tmp["X"].iloc[-1] < 0:
                d = d[d["Segment"] != i]

    if np.max(np.abs(d["X"])) < 10:
        max_field = np.power(10, np.round(np.log10(np.nanmax(d["X"])*1e+3)))
        d["X"] = d["X"] * 1e+3
    else:
        max_field = np.power(10, np.round(np.log10(np.nanmax(d["X"]))))

    D["raw_data"] = {
        "field_irm": np.array(d["X"]),
        "remanence_irm": np.array(d["Y"]),
        "max_field": max_field,
    }

    return D


def read_VSM8600_csv_irm(file):
    
    D = {}
    D["dtype"] = "VSM_8600_csv"
    
    coding = "ISO-8859-15"
    with open(file, "r", encoding=coding) as fr:
        for i, line in enumerate(fr):
            if "Number of points" in str(line):
                l = str(line)
                p = re.split(r"\s*:\s*", l.strip(), maxsplit=1)
                if len(p) == 2:
                    k, v = p
                else:
                    k, v = p[0], None
                try:
                    v = float(v)
                except ValueError:
                    v = v
                number_of_points = v

            if "##DATA TABLE" in line:
                skiprows = i+2
                # print (skiprows)

    names = ["Step","Iteration","Segment","Field","Moment",
             "Time Stamp", "Field Status", "Moment Status"]
    rD = pd.read_csv(
        file,
        sep=None,
        names=names,
        # na_values=0,
        skiprows=skiprows,
        nrows=number_of_points,
        engine="python",
        encoding=coding,
    )

    if np.max(np.abs(rD["field_irm"])) < 10:
        max_field = np.power(10, np.round(np.log10(np.nanmax(rD["field_irm"])*1e+3)))
        rD["field_irm"] = rD["field_irm"] * 1e+3
    else:
        max_field = np.power(10, np.round(np.log10(np.nanmax(rD["field_irm"]))))
    D["raw_data"]={"field_irm":rD["field_irm"],
                   "remanence_irm":rD["remanence_irm"],
                   "max_field":max_field,
                   }

    return D

def read_generic_irm(file):
    
    D = {}
    D["dtype"] = "Generic"
    
    tmp = pd.read_csv(file, sep=None, nrows=1, 
                      names=["field_irm","remanence_irm"], 
                      engine="python",
                      )
    try:
        float(tmp.loc[0,"field_irm"])
        rD = pd.read_csv(file, sep=None, names=["field_irm","remanence_irm"],
                         engine="python")
    except:
        rD = pd.read_csv(file, sep=None, names=["field_irm","remanence_irm"],
                         skiprows=1, engine="python")
        

    if np.max(np.abs(rD["field_irm"])) < 10:
        max_field = np.power(10, np.round(np.log10(np.nanmax(rD["field_irm"])*1e+3)))
        rD["field_irm"] = rD["field_irm"] * 1e+3
    else:
        max_field = np.power(10, np.round(np.log10(np.nanmax(rD["field_irm"]))))
    
    D["raw_data"]={"field_irm":rD["field_irm"],
                   "remanence_irm":rD["remanence_irm"],
                   "max_field":max_field,
                   }
    
    return D

def read_VSM3900_irm(file):
    """
    Read IRM data measured in VSM3900/2900 

    Parameters:
        file: str

    Returns:
        D: shaped data
    """
    D = {}
    D["dtype"] = "VSM3900/2900"
    script = {}
    coding = "ISO-8859-15"

    s, e, idx = [], [], []
    with open(file, "r", encoding=coding) as fr:
        for i, line in enumerate(fr):
            l = line.strip()
            s.append(l)
            if len(l) == 0:
                e.append(i)
            if "Number      Time         Field" in str(line):
                rows_idx = i + 2
                names_idx = [
                    "Segment number",
                    "Averaging time",
                    "Initial Field",
                    "Field Increment",
                    "Final Field",
                    "Pause",
                    "Final Index",
                ]
            if "Field       Remanence" in str(line):
                raws_data = i + 2
                break
    e = np.array(e)
    e = np.delete(e, np.argwhere(e > rows_idx))
    e = np.concatenate(([-1], e))
    s = np.array(s, dtype=object)

    for i in range(0, len(e) - 1):
        idx = np.arange(e[i] + 1, e[i + 1])
        t = s[idx]
        if i == 0:
            script["Header info"] = list(t)
        else:
            script[t[0]] = {}
            for j in range(1, len(t)):
                l = t[j]
                p = re.split(r"\s{2,}", l.strip(), maxsplit=1)
                if len(p) == 2:
                    k, v = p
                else:
                    k, v = p[0], None
                try:
                    v = float(v)
                except ValueError:
                    v = v
                script[t[0]][k] = v

    D["script"] = script

    segments = int(script["SCRIPT"]["Number of segments"])
    number = int(script["SCRIPT"]["Number of data"])
    data_line = int(number + segments - 1)

    irm, bfd, direct = [
        script["SCRIPT"]["Include IRM?"] == "Yes",
        script["SCRIPT"]["Include DCD?"] == "Yes",
        script["SCRIPT"]["Include direct moment?"] == "Yes",
    ]

    index = pd.read_csv(
        file,
        delimiter=",",
        names=names_idx,
        skiprows=rows_idx,
        nrows=segments,
        engine="python",
        encoding=coding,
    )
    data_line = int(number + segments - 1)
    raws_data = raws_data if isinstance(raws_data, int) else 1

    if script["SCRIPT"]["Include direct moment?"] == "Yes":
        names = [
            "field",
            "remanence",
            "direct_moment",
            "adjust_field",
            "adjust_remanence",
            "adjust_direct_moment",
        ]
    elif script["PROCESSING"]["Normalization"] != "No":
        names = ["field", "remanence", "norm_remanence"]
    else:
        names = ["field", "remanence"]

    rD = pd.read_csv(
        file,
        delimiter=",",
        names=names,
        na_values=0,
        skiprows=raws_data,
        nrows=data_line,
        engine="python",
        encoding=coding,
    )
    Seg_idx = {}
    for i in range(0, segments):
        if i == 0:
            Seg_idx[f"seg_{i}"] = np.arange(0, index.loc[i, "Final Index"])
        else:
            Seg_idx[f"seg_{i}"] = np.arange(
                index.loc[i - 1, "Final Index"], index.loc[i, "Final Index"]
            )

    raw_data, adjust_data = {}, {}
    if segments == 1:
        if irm == True:
            raw_data["field_irm"] = rD.loc[Seg_idx["seg_0"], "field"].values
            raw_data["remanence_irm"] = rD.loc[Seg_idx["seg_0"], "remanence"].values
            if direct == True:
                raw_data["direct_moment_irm"] = rD.loc[
                    Seg_idx["seg_0"], "direct_moment"
                ].values
                adjust_data["adjust_field_irm"] = rD.loc[
                    Seg_idx["seg_0"], "adjust_field"
                ].values
                adjust_data["adjust_remanence_irm"] = rD.loc[
                    Seg_idx["seg_0"], "adjust_remanence"
                ].values
                adjust_data["adjust_direct_moment_irm"] = rD.loc[
                    Seg_idx["seg_0"], "adjust_direct_moment"
                ].values
        elif bfd == True:
            raw_data["field_bfd"] = rD.loc[Seg_idx["seg_0"], "field"].values
            raw_data["remanence_bfd"] = rD.loc[Seg_idx["seg_0"], "remanence"].values
            if direct == True:
                raw_data["direct_moment_bfd"] = rD.loc[
                    Seg_idx["seg_0"], "direct_moment"
                ].values
                adjust_data["adjust_field_bfd"] = rD.loc[
                    Seg_idx["seg_0"], "adjust_field"
                ].values
                adjust_data["adjust_remanence_bfd"] = rD.loc[
                    Seg_idx["seg_0"], "adjust_remanence"
                ].values
                adjust_data["adjust_direct_moment_bfd"] = rD.loc[
                    Seg_idx["seg_0"], "adjust_direct_moment"
                ].values
    elif segments == 2:
        if direct == True:
            raw_data["direct_moment_irm"] = rD.loc[
                Seg_idx["seg_0"], "direct_moment"
            ].values
            adjust_data["adjust_field_irm"] = rD.loc[
                Seg_idx["seg_0"], "adjust_field"
            ].values
            adjust_data["adjust_remanence_irm"] = rD.loc[
                Seg_idx["seg_0"], "adjust_remanence"
            ].values
            adjust_data["adjust_direct_moment_irm"] = rD.loc[
                Seg_idx["seg_0"], "adjust_direct_moment"
            ].values

            raw_data["direct_moment_bfd"] = rD.loc[
                Seg_idx["seg_1"], "direct_moment"
            ].values
            adjust_data["adjust_field_bfd"] = rD.loc[
                Seg_idx["seg_1"], "adjust_field"
            ].values
            adjust_data["adjust_remanence_bfd"] = rD.loc[
                Seg_idx["seg_1"], "adjust_remanence"
            ].values
            adjust_data["adjust_direct_moment_bfd"] = rD.loc[
                Seg_idx["seg_1"], "adjust_direct_moment"
            ].values

        raw_data["field_irm"] = rD.loc[Seg_idx["seg_0"], "field"].values
        raw_data["remanence_irm"] = rD.loc[Seg_idx["seg_0"], "remanence"].values
        raw_data["field_bfd"] = rD.loc[Seg_idx["seg_1"], "field"].values
        raw_data["remanence_bfd"] = rD.loc[Seg_idx["seg_1"], "remanence"].values
    else:
        print("data segments Error!")

    for i in raw_data.keys():
        if "irm" in i:
            sort_idx_irm = np.argsort(raw_data["field_irm"])
        if "bfd" in i:
            sort_idx_bfd = np.argsort(raw_data["field_bfd"])

    for i in raw_data.keys():
        if "irm" in i:
            raw_data[i] = raw_data[i][sort_idx_irm]
        if "bfd" in i:
            raw_data[i] = raw_data[i][sort_idx_bfd]
        if "field" in i:
            raw_data[i] = raw_data[i] * 1e3
    for i in adjust_data.keys():
        if "irm" in i:
            adjust_data[i] = adjust_data[i][sort_idx_irm]
        if "bfd" in i:
            adjust_data[i] = adjust_data[i][sort_idx_bfd]
        if "field" in i:
            adjust_data[i] = adjust_data[i] * 1e3

    D["raw_data"] = raw_data
    D["raw_data"]["max_field"] = D["script"]["SCRIPT"]["Final field"] * 1e+3
    if direct == True:
        D["asjust_data"] = adjust_data

    # plt.plot(D["adjust_field_irm"], D["adjust_moment_irm"])
    # plt.xscale("log")
    return D


def proces_IRM_multiple(d, n=9):
    
    x = d["raw_data"]["field_irm"]
    y = d["raw_data"]["remanence_irm"]

    data = interp_IRM(x, y, xmax=d["raw_data"]["max_field"])
    data["smooth"] = smooth_IRM(data["remanence"], n=n, polyorder=0)
    data["gradient_raw"] = np.gradient(data["remanence"], data["field_log"])
    data["gradient_smooth"] = np.gradient(data["smooth"], data["field_log"])

    x = np.array(data["field_log"])
    yr = np.array(np.maximum(data["gradient_raw"], 0))
    ys = np.array(np.maximum(data["gradient_smooth"], 0))

    # yr = yr / np.trapezoid(yr, x)
    # ys = ys / np.trapezoid(ys, x)

    return x, yr, ys


def interp_IRM(x=None, y=None, xmax=None):
    """
    Interpolate IRM data in logarithmic scale with interval of 0.025

    Parameters:
        x: field in normal scale
        y: moments
        kwargs: max field (optional)

    Returns:
        data: interpolated data in DataFrame type.

    Note:
        Only data above 1 mT (log10(x)>=0) are retained.
    """
    x, y = x.copy(), y.copy()
    x, y = x[x >= 0], y[x >= 0]
    lgx = np.log10(x)
    inx = np.arange(0, np.log10(xmax) + 0.025, 0.025)
    f = interp1d(lgx, y, kind="linear", fill_value="extrapolate", bounds_error=False)
    iny = f(inx)
    iny = norm(iny)
    data = pd.DataFrame(
        data={"field": np.power(10, inx), "field_log": inx, "remanence": iny}
    )
    idx = np.where(data["field"] <= xmax)
    data = data.iloc[idx]

    return data


def smooth_IRM(data=None, smooth_window=9, polyorder=3, mode="nearest"):
    """
    smooth IRM data using savgol_filter (scipy.signal.savgol_filter)

    Parameters:
        data: array, remanence data
        smooth_window: int, window_length of savgol_filter
        mode: str, optional

    Returns:
        data: smoothed data
    """

    data = scipy.signal.savgol_filter(
        data, window_length=smooth_window, polyorder=polyorder, mode=mode
    )
    return data


def norm(s):
    ns = s / np.nanmax(s)
    return ns


# from numba import njit
# @njit


def SGG(x=None, mu=None, sigma=None, weight=1, q=1, p=2, mode="PDF"):
    pdf = np.array([])
    # if np.abs(q) < 0.1:
    #     q = 0.1 if q>=0 else -0.1
    for i in range(0, len(x)):
        xi = (x[i] - mu) / sigma
        sgg = (
            (1 / (math.pow(2, 1 + 1 / p) * sigma * math.gamma(1 + 1 / p)))
            * (
                math.fabs(q * math.exp(q * xi) + math.pow(q, -1) * math.exp(xi / q))
                / (math.exp(q * xi) + math.exp(xi / q))
            )
            * (
                math.exp(
                    -1
                    / 2
                    * math.pow(
                        math.fabs(math.log((math.exp(q * xi) + math.exp(xi / q)) / 2)),
                        p,
                    )
                )
            )
        )
        pdf = np.append(pdf, sgg)
    cdf = norm(np.cumsum(pdf))
    if (mode == "PDF") | (mode == "pdf"):
        return pdf * weight
    elif (mode == "CDF") | (mode == "cdf"):
        return cdf * weight
    else:
        return print("mode error!")


def GMM(x, y, n_components):

    y = np.maximum(y, 0)
    n_samples = 10000
    probs = y / np.sum(y)
    X_resampled = np.random.choice(x, size=n_samples, p=probs).reshape(-1, 1)

    gmm = GaussianMixture(n_components=n_components, random_state=None).fit(X_resampled)

    GMM = {}
    GMM["gmm"] = gmm
    GMM["weights"] = gmm.weights_
    GMM["means"] = gmm.means_.flatten()
    GMM["stds"] = np.sqrt(gmm.covariances_.flatten())

    return GMM


def BGMM(x, y, n_components, **kwargs):

    rc = kwargs.pop("reg_covar", 1e-8)
    dfp = kwargs.pop("degrees_of_freedom_prior", None)
    n_init = kwargs.pop("n_init", 1)
    # print ("n_init:\t"+ str(n_init))

    y = np.maximum(y, 0)
    n_samples = 10000

    probs = y / np.sum(y)

    X_resampled = np.random.choice(x, size=n_samples, p=probs).reshape(-1, 1)

    bgmm = BayesianGaussianMixture(
        n_components=n_components,
        weight_concentration_prior=1e-2,
        max_iter=10000,
        random_state=None,
        reg_covar=rc,
        degrees_of_freedom_prior=dfp,
        ).fit(X_resampled)

    BGMM = {}
    BGMM["gmm"] = bgmm
    BGMM["weights"] = bgmm.weights_
    BGMM["means"] = bgmm.means_.flatten()
    BGMM["stds"] = np.sqrt(bgmm.covariances_.flatten())

    return BGMM


def calculate_comps_BGMM_BIC(x, yr, ys, **kwargs):
    """ """

    nmin = kwargs.pop("nmin", 1)
    nmax = kwargs.pop("nmax", 10)
    rc = kwargs.pop("reg_covar", 1e-8)
    dfp = kwargs.pop("degrees_of_freedom_prior", None)
    n_iters = kwargs.pop("n_iters", 1)
    show_fig = kwargs.pop("show_fig", True)

    ncols = 3
    nplots = int(nmax - nmin) + 2

    if nplots % ncols == 0:
        nrows = int(nplots / ncols)
    else:
        nrows = int(nplots / ncols) + 1

    # print (nmin, nmax, rc, dfp)

    rmse = np.full((int(nmax - nmin + 1), 1), fill_value=np.nan)
    rss = rmse.copy()
    bic = np.full((int(nmax - nmin + 1), 1), fill_value=np.nan)

    gmm={
        "rss": rss,
        "rmse": rmse,
        "bic": bic
    }

    comp_nums = np.arange(nmin, nmax + 1)

    total = (nplots - 1) * (n_iters)
    pbar = tqdm(total=total)

    for i in range(nplots - 1):
        n = comp_nums[i]
        # print (f"Calculating components number: {n}")
        m, s, w, p, = [], [], [], []
        rsst = np.full((n_iters), fill_value=np.nan)
        bict = rsst.copy()
        for j in range(0, n_iters):
            
            f = BGMM(
                x,
                ys,
                n_components=n,
                reg_covar=rc,
                degrees_of_freedom_prior=dfp,
                )
            mt, st, wt = f["means"], f["stds"], f["weights"]
            m.append(mt)
            s.append(st) 
            w.append(wt)
            tmp = {
                "means":mt,
                "stds":st,
                "weights":wt,
                }
            params = params_transfer_GMM_NLS(tmp)
            p.append(params)
            comps = calculate_sum_component_SGG_model(x=x, params=params)
            
            rsst[j] = np.sum((comps["comp_sum"] - ys) ** 2)
            
            bict[j] = (
                len(x) * np.log(rsst[j] / len(x))
                + (3 * j) * np.log(len(x))
                + len(x) * np.log(10) * penalty_mu(mt) 
                + penalty_sigma(st)
                )
            pbar.update(1)

        if np.all(np.isinf(bict)):
            idx = np.nanargmin(rsst)
        else:
            idx = np.nanargmin(bict)

        comps = calculate_sum_component_SGG_model(x=x, params=p[idx])
        rmse = np.sqrt(np.mean((comps["comp_sum"] - ys) ** 2))
        gmm[f"comp_nums_{n}"] = {
            "means": m[idx],
            "stds": s[idx],
            "weights": w[idx],
            "params": p[idx],
            "comps": comps,
            "rss": rsst[idx],
            "bic": bict[idx],
            "rmse": rmse,
            }
        gmm["rss"][i] = rsst[idx]
        gmm["rmse"][i] = rmse
        gmm["bic"][i] = bict[idx]
        # print (f"Components number {n} finished\n")
    pbar.close()
    
    if show_fig == True:
        colors = plt.cm.Set2.colors
        fig, axs = plt.subplots(nrows, ncols, figsize=(12,8))
        axs = axs.ravel()

    # for ax in np.ravel(axs):
    #     ax.set_box_aspect(3/4)

        for i in range(nplots - 1):
            n = comp_nums[i]
            axs[i].scatter(x, yr, c="gray", s=10)
            axs[i].plot(x, ys, "k", linewidth=1.5)
            comps = gmm[f"comp_nums_{n}"]["comps"]
            for j in range(n):
                color = colors[j]
                axs[i].plot(x, comps[f"comp_{j}"], color=color, alpha=1.0)
            axs[i].plot(x, comps[f"comp_sum"], color="r", alpha=1.0)
            axs[i].set_title(f"Component = {n}")
    
        axs[i + 1].plot(np.arange(nmin, nmax+1, step=1), bic)
        axs[i + 1].set_title(f"BIC scores")
        for ax in axs[nplots:]:
            ax.remove()
        plt.xticks(np.arange(nmin, nmax+1, step=1))
        plt.tight_layout()
        plt.show()
        
    idx = np.nanargmin(bic)
    print(f"Optimitize component number: {idx+nmin}")

    return gmm        

def penalty_mu(means):
    penalty = 0

    if len(means) > 1:
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                diff = np.abs(means[i] - means[j])
                penalty = penalty + np.exp(-(diff**2) / (2 * 0.1**2))
        
    return penalty

def penalty_sigma(stds):
    penalty = 0

    for s in stds:
        if s < 0.10 or s > 0.6:
            penalty = np.inf

    return penalty

def params_transfer_GMM_NLS(params):
    # params = gmm[f"comp_nums_{n}"]
    means, stds, weights = params["means"], params["stds"], params["weights"]
    idx = np.argsort(means)
    qs, ps = np.full((len(means)), fill_value=int(1)), np.full(
        (len(means)), fill_value=int(2)
    )
    params = []
    for i in idx:
        params.extend([means[i], stds[i], weights[i], qs[i], ps[i]])

    return params


def calculate_sum_component_SGG_model(x=None, y=None, params=None, **kwargs):
    """
    Params:
        x: array-like, field
        params: array-like, shape parameters of components

    Returns:
        y_pred: prediction sum pdf curve
    """
    mode = kwargs.get("mode", "PDF")

    y_pred = np.zeros_like(x)
    n = len(params) // 5
    # params = params_transfer_GMM_NLS(n, gmm)

    # plt.figure()
    # plt.plot(x, y, 'k')
    sum_component = {}
    for i in range(n):
        mu, sigma, weight, q, p = params[i * 5 : i * 5 + 5]
        y_comp = SGG(x=x, mu=mu, sigma=sigma, weight=weight, q=q, p=p, mode=mode)
        y_pred += SGG(x=x, mu=mu, sigma=sigma, weight=weight, q=q, p=p, mode=mode)
        # plt.plot(x, y_comp)
        # plt.plot(x, y_pred)
        sum_component[f"comp_{i}"] = y_comp
    sum_component[f"comp_sum"] = y_pred

    return sum_component


def calculate_individual_component_SGG_model(x=None, params=None):
    ncomp = len(params) // 5
    iparams = np.reshape(params, (ncomp, 5))
    results = {}
    for i in range(ncomp):
        m, s, w, q, p = iparams[i, :]
        results[f"comp_{i}"] = SGG(x, m, s, w, q, p, mode="PDF")
    return results


def nls_func(params, x, y, penalty_scale=None):
    y_pred = calculate_sum_component_SGG_model(x, y, params)["comp_sum"]
    rss = np.sum((y - y_pred) ** 2)
    y_scale = np.sum(y**2) + 1e-30  # 防止y=0的情况，加一个极小的常数

    if penalty_scale==None:
        return rss / y_scale
    else:
        penalty = penalty_mu(params[0::5])
        return rss / y_scale + penalty_scale * penalty
    

def weight_constrain(params):
    weights = params[2::5]
    return np.sum(weights) - 1.0

def skewness_constrain(params):
    q = params[3::5]
    return q**2 - 0.0001

def params_perturb(params=None, std=0.02, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    p = params.copy()
    p[0::5] = p[0::5] * rng.normal(1.0, std, size=len(p[0::5]))
    p[1::5] = p[1::5] * rng.normal(1.0, std, size=len(p[1::5]))
    p[2::5] = p[2::5] / np.sum(p[2::5])
    p[3::5] = p[3::5] * rng.normal(1.0, 2*std, size=len(p[3::5]))

    return p

def reshape_bounds(params):
    n = len(params) // 5
    bounds_lower, bounds_upper = [], []
    for i in range(n):
        mu = params[i * 5]
        bounds_lower.extend([mu - 0.25, 0.15, 0.0, 0.1, 1.5])
        bounds_upper.extend([mu + 0.25, 0.35, 1.0, 1.0, 3.5])
    bounds = list(zip(bounds_lower, bounds_upper))

    return bounds

def reshape_bounds_multiple(params):
    n = len(params) // 5
    bounds_lower, bounds_upper = [], []
    for i in range(n):
        mu = params[i * 5]
        bounds_lower.extend([mu - 0.15, 0.15, 0.0, 0.1, 1.5])
        bounds_upper.extend([mu + 0.15, 0.35, 1.0, 1.0, 3.5])
    bounds = list(zip(bounds_lower, bounds_upper))

    return bounds

def nls_calculate(x, y, params, n, cons={"type": "eq", "fun": weight_constrain}, mode="single", **kwargs):

    param_initial = params.copy()
    param = params.copy()
    y_pred_before = calculate_sum_component_SGG_model(x, y, params)
    rss_before = np.sum((y - y_pred_before["comp_sum"]) ** 2)
    y_scale = np.sum(y**2) + 1e-30
    penalty_scale = rss_before/y_scale * 0.10
    # print(f"Initial RSS: {rss_before:.4f}\n")
    # print(f"Initial RSS scale: {rss_before/y_scale:.4f}\n")
    # print(f"Penalty_scale: {penalty_scale:.4f}\n")
          
    param = params_perturb(param)
    if mode=="single":
        bounds = reshape_bounds(param)
    elif mode=="multiple":
        bounds = reshape_bounds_multiple(param)
    else:
        raise ValueError(
            f"Invalid mode: {mode!r}. Expected 'single' or 'multiple'."
            )

    res = minimize(
        nls_func,
        x0=param,
        args=(x, y, penalty_scale),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 10000, "ftol": 1e-6, "eps": 1e-4},
    )

    if res.success:
        popt = res.x
    else:
        print("Failed:", res.message)
        print("Using the last step params")
        popt = res.x

    y_pred_after = calculate_sum_component_SGG_model(x, y, popt)
    rss_after = np.sum((y - y_pred_after["comp_sum"]) ** 2)

    # print(f"RSS before NLS: {rss_before}")
    # print(f"RSS after NLS: {rss_after}")

    results = {
        "params_initial": param_initial,
        "params_popt": popt,
        "rss_initial": rss_before,
        "rss_popt": rss_after,
        "y_pred": y_pred_after,
    }

    return results


def extract_confidence_intervals(x=None, y=None, results=None):

    params_collection = np.array([res["popt"] for res in results if res is not None])
    rss_collection = np.array([res["rss"] for res in results if res is not None])
    idx = np.argsort(np.nanmin(rss_collection))
    param_popt = params_collection[idx].T
    ncomp = np.shape(params_collection)[1] // 5

    nrows, ncols = len(params_collection), len(x)
    icomp = {}
    scomp = {}

    for i in range(ncomp):
        icomp[f"comp_{i}"] = []
        scomp[f"comp_{i}"] = []

    for i in range(nrows):
        params = params_collection[i, :]
        tmp = calculate_sum_component_SGG_model(x=x, y=y, params=params)
        for j in range(ncomp):
            icomp[f"comp_{j}"].append(tmp[f"comp_{j}"])
            scomp[f"comp_{j}"].append(tmp[f"comp_sum"])

    for i in range(ncomp):
        icomp[f"comp_{i}"] = np.array(icomp[f"comp_{i}"])
        scomp[f"comp_{i}"] = np.array(scomp[f"comp_{i}"])

    confidence = {}

    for i in icomp.keys():
        t, s = icomp[i], scomp[i]
        ilower, slower = np.percentile(t, 2.5, axis=0), np.percentile(s, 2.5, axis=0)
        iupper, supper = np.percentile(t, 97.5, axis=0), np.percentile(s, 97.5, axis=0)
        imiddle, smiddle = np.mean(t, axis=0), np.mean(s, axis=0)

        confidence[i] = {
            "icomp_lower": ilower,
            "icomp_middle": imiddle,
            "icomp_upper": iupper,
            "scomp_lower": slower,
            "scomp_upper": supper,
            "scomp_middle": smiddle,
        }
    
    return param_popt, icomp, scomp, confidence

def process_mc_seed(x, y, nx, ny, smooth_window=9, polyorder=0, **kwargs):

    xmax = np.nanmax(x)
    inx = np.arange(0, xmax + 0.025, 0.025)
    f = interp1d(nx, ny, kind="linear", fill_value="extrapolate", bounds_error=False)
    iny = f(inx)
    iny = norm(iny)
    sy = scipy.signal.savgol_filter(iny, window_length=smooth_window, polyorder=polyorder)
    if version.parse(np.__version__) < version.parse("2.0.0"):
        sy = sy/np.trapz(sy, inx)
    else:
        sy = sy/np.trapezoid(sy, inx)

    return inx, sy

def run_mc_iteration(
    seed=None, x=None, y=None, params=None, cons=None, 
    proportion=0.95, std=0.02, smooth_window=9, **kwargs
):
    """ """
    rng = np.random.default_rng(seed)
    m = int(len(x) * proportion)
    idx = np.sort(rng.choice(np.arange(0, len(x)), size=m, replace=False))
    nx, ny = x[idx], y[idx]
    sx, sy = process_mc_seed(x, y, nx, ny, smooth_window=smooth_window)

    param = copy.deepcopy(params)
    param = params_perturb(param, std=std, rng=rng)
    bounds = reshape_bounds(param)

    res = minimize(
        nls_func,
        x0=param,
        args=(sx, sy),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-6, "eps": 1e-4},
    )

    if res.success:
        popt = res.x
    else:
        print("Failed:", res.message)
        print("Using the last step params")
        popt = res.x

    y_pred = calculate_sum_component_SGG_model(x, y, popt)
    rss = np.sum((y - y_pred["comp_sum"]) ** 2)

    results = {"params": param, "popt": popt, "rss":rss, "data":[nx, ny]}
    return results


def run_mc_simulation(
    n_iterations=100, x=None, y=None, params=None, cons=None, 
    n_jobs=None, smooth_window=9, **kwargs
):
    """
    n_iterations: number of mc iterations
    n_jobs: Number of parallel cores, -1 represents using all cores.
    """

    if n_jobs is None:
        n_jobs = psutil.cpu_count(logical=True)-2
    else:
        n_jobs = int(n_jobs)

    print(f"Monte Carlo simulation startin: total of {n_iterations} simulations...")
    
    start_time = time.time()

    tasks = (
        delayed(run_mc_iteration)(seed=i, x=x, y=y, params=params, cons=cons,
                                  smooth_window=smooth_window, **kwargs)
        for i in range(n_iterations)
    )

    parallel_generator = Parallel(n_jobs=n_jobs, verbose=0, return_as="generator")(
        tasks
    )

    results = []

    with tqdm(
        total=n_iterations,
        desc="Monte Carlo Simulation",
        position=0,
        leave=True,
        file=sys.stdout,
    ) as pbar:
        for result in parallel_generator:
            results.append(result)
            pbar.update(1)  
        # pbar.close()

    sys.stdout.flush()
    duration = time.time() - start_time
    print(f"并行计算结束，总共用时: {duration:.2f}s")

    return results

def reshape_parameters(params):
    p = copy.deepcopy(params)
    n = len(p) // 5
    d = np.reshape(p, (n, 5)).T
    names = []
    for i in range(n):
        names.append(f"Comp_{i+1}")
    index = ["Bh", "DP", "Proportion", "skewness", "kurtosis"]
    d = pd.DataFrame(d, columns=names, index=index)
    d.loc["Bh"] = 10**d.loc["Bh"] 

    return d

def reshape_parameters_multiple(params):
    p = copy.deepcopy(params)
    n = len(p[0]) // 5
    d = np.concatenate([p[:,0::5], 
                        p[:,1::5], 
                        p[:,2::5],
                        p[:,3::5],
                        p[:,4::5]], 
                        axis=1
                        )
    names = [
        f"{prefix}{i+1}"
        for prefix in ["mean", "dp", "weight", "q", "p"]
        for i in range(n)
        ]
    d = pd.DataFrame(d, columns=names)
    for i in range(n):
        d[f"mean{i+1}"] = 10**d[f"mean{i+1}"]
    return d

def multiple_process_single_nls_calculate(i=None, file=None, params=None, d=None, 
                                          comp_nums=None, cons=None, showfig=False, 
                                          savefig=False, **kwargs):

    params_initial = copy.deepcopy(params)

    fpath = os.path.split(file)[0]
    fname = os.path.split(file)[1]
    fname = os.path.splitext(fname)[0]

    x, yr, ys = d["field_log"], d["gradient_raw"], d["gradient_smooth"]

    results = nls_calculate(x, ys, params_initial, comp_nums, cons, mode="multiple")

    results[f"fpath"] = fpath
    results[f"fname"] = fname
    results[f"i"] = i
    if showfig or savefig:
        multiple_process_single_plot(x=x, yr=yr, ys=ys, n=comp_nums,
                                     results=results, showfig=showfig, savefig=savefig)

    print(type(results))
    
    return results

def multiple_process_single_plot(x=None, yr=None, ys=None, n=None, results=None, showfig=False, savefig=False, **kwargs):
    # print (file)
    fpath = results["fpath"]
    fname = results["fname"]

    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, yr, c="gray", s=10, label="Data")
    ax.plot(x, ys, "black", linewidth=1.5, label="Smooth")

    for i in range(n):
        color = colors[i*2]
        tmp = calculate_sum_component_SGG_model(x=x, params=results["params_popt"])
        ax.plot(x, tmp[f"comp_{i}"], color=color, label=f"Comp_{i+1}")

    ax.plot(x, tmp[f"comp_sum"], color="r", label=f"Sum")
    ax.text(
        0.05, 0.98, "RSS: " + f"{results['rss_popt']:.4e}",
        color="b", fontsize=12,
        transform=ax.transAxes,
        verticalalignment='top',
    )
    ax.legend()
    ax.set_title(f"{fname}", fontsize=14)
    ax.set_xlabel("Field (Log$_{10}$(mT))", fontsize=12)
    ax.set_ylabel("IRM Gradient", fontsize=12)
    ax.tick_params(axis='both', direction="in", labelsize=10)

    if savefig:
        expath = os.path.join(fpath, "pyIRM_UnMix_processed")
        if not os.path.exists(expath):
            os.makedirs(expath)
        exfname = os.path.join(expath, fname+".pdf")
        fig.savefig(exfname, dpi=300)
        exfname = os.path.join(expath, fname+".png")
        fig.savefig(exfname, dpi=300)
        
    if showfig:
        plt.show()
    else:
        plt.close(fig)

    return 


def multiple_process(D=None, files=None, files_name=None, params_initial=None,comp_nums=None,
                     cons=None, n_jobs=-1, showfig=False, savefig=False, **kwargs):
    
    params = copy.deepcopy(params_initial)

    tasks = (
    delayed(multiple_process_single_nls_calculate)(
        i=i,
        file=file,
        params=params,
        d=D[file_name]["process_data"],
        comp_nums=comp_nums,
        cons=None,
        showfig=showfig,
        savefig=savefig,
    )
    for i, (file, file_name) in enumerate(zip(files, files_name))
    )

    results_generator = Parallel(
        n_jobs=-1,
        return_as="generator"
    )(tasks)

    Results = list(tqdm(
        results_generator,
        total=len(files),
        desc="Processing files"
    ))

    return Results