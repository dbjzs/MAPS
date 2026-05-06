# Built-in
import time
import random
import os
from typing import List, NamedTuple, Optional, TYPE_CHECKING, Union, Tuple
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix

# Third-party
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import pyimzml
from pyimzml.ImzMLParser import ImzMLParser
import intervaltree


# Spatial metabolome preprocessing
# Reference: https://github.com/WanluLiuLab/SpatialMETA/blob/master/spatialmeta/data/_dataloader.py
def _merge_and_get_unique_arrays(mass_data):
    all_values = [value for array in mass_data.values() for value in array]
    unique_values = list(set(all_values))
    return unique_values


def _calculate_ppm_range(observed_mz, ppm_tolerance=5):
    ppm_range = observed_mz * (ppm_tolerance / 1e6)
    lower_limit = observed_mz - ppm_range
    upper_limit = observed_mz + ppm_range
    return pd.Series(
        [ppm_range, lower_limit, upper_limit],
        index=["ppm_range", "lower_limit", "upper_limit"],
    )


def get_mz_reference(
    p: ImzMLParser, 
    ppm_tolerance: int = 5
) -> pd.DataFrame:
    """
    Get m/z reference from ImzMLParser object.

    :param p: ImzMLParser. The ImzMLParser object.
    :param ppm_tolerance: int. The ppm tolerance, default is 5.

    :return: pd.DataFrame. The m/z reference for the SM data, and the column names are "m/z", "Interval Width (+/- Da)".
    """
    my_spectra = []
    mass_data = {}
    for idx, (x, y, z) in enumerate(p.coordinates):
        mzs, intensities = p.getspectrum(idx)
        mass_data[idx] = mzs
        my_spectra.append([mzs, intensities, (x, y, z)])
    unique_values = _merge_and_get_unique_arrays(mass_data)
    unique_values_ppm = [
        _calculate_ppm_range(x, ppm_tolerance) for x in unique_values
    ]
    unique_values_ppm_df = pd.DataFrame(unique_values_ppm)
    unique_mz_df = pd.DataFrame(
        {
            "mz": unique_values,
            "ppm_range": unique_values_ppm_df["ppm_range"].values,
            "lower_limit": unique_values_ppm_df["lower_limit"].values,
            "upper_limit": unique_values_ppm_df["upper_limit"].values,
        }
    )
    unique_mz_df = unique_mz_df.sort_values(by="mz").reset_index(drop=True)

    intervals_list = []
    current_lower = unique_mz_df["lower_limit"].iloc[0]
    current_upper = unique_mz_df["upper_limit"].iloc[0]
    for i in range(1, len(unique_mz_df)):
        if unique_mz_df["lower_limit"].iloc[i] <= current_upper:
            current_upper = max(current_upper, unique_mz_df["upper_limit"].iloc[i])
        else:
            intervals_list.append({"lower_limit": current_lower, "upper_limit": current_upper})
            current_lower = unique_mz_df["lower_limit"].iloc[i]
            current_upper = unique_mz_df["upper_limit"].iloc[i]
    intervals_list.append({"lower_limit": current_lower, "upper_limit": current_upper})
    merged_intervals = pd.DataFrame(intervals_list)

    merged_intervals["new_mz"] = (
        merged_intervals["lower_limit"] + merged_intervals["upper_limit"]
    ) / 2
    merged_intervals["new_ppm_range"] = (
        merged_intervals["upper_limit"] - merged_intervals["lower_limit"]
    ) / 2
    mz_reference = merged_intervals[["new_mz", "new_ppm_range"]]
    mz_reference.columns = ["m/z", "Interval Width (+/- Da)"]
    return mz_reference



def _get_interval(t, v):
    r = t.at(v)
    if len(r) > 0:
        return list(r)[0][-1]
    else:
        return None


def read_sm_imzml_as_anndata(
    p: ImzMLParser, 
    mz_reference: pd.DataFrame
) -> AnnData:
    """
    Read SM imzML file as AnnData object.

    :param p: ImzMLParser. The ImzMLParser object.
    :param mz_reference: pd.DataFrame. The m/z reference.

    :return: AnnDataSM. The AnnData object with SM data.
    """
    my_spectra = []
    for idx, (x, y, z) in enumerate(p.coordinates):
        mzs, intensities = p.getspectrum(idx)
        my_spectra.append([mzs, intensities, (x, y, z)])
    t = intervaltree.IntervalTree()
    for i, j in mz_reference.iloc[:, :2].to_numpy():
        t[i - j : i + j] = str(i)
    mz_mapping_result = [[_get_interval(t, x) for x in y[0]] for y in my_spectra]
    mz_reorder_indices = dict(
        zip(list(map(str, mz_reference.iloc[:, 0])), range(len(mz_reference)))
    )
    new_signal_merged = []
    # Reorder the signal to match the mz_reference
    for e, m in enumerate(mz_mapping_result):
        signal = my_spectra[e][1]
        new_signal = np.zeros(len(mz_reference))
        for r, v in enumerate(m):
            if v in mz_reorder_indices.keys():
                new_signal[mz_reorder_indices[v]] += signal[r]
        new_signal_merged.append(new_signal)
    new_signal_merged = np.vstack(new_signal_merged)
    x_coord_original = [spectra[2][0] for spectra in my_spectra]
    y_coord_original = [spectra[2][1] for spectra in my_spectra]
    var_list = mz_reference["m/z"].values
    adata_SM = AnnData(
        X=sparse.csr_matrix(new_signal_merged),
        obs=pd.DataFrame(
            np.array([x_coord_original, y_coord_original]).T,
            columns=["x_coord_original", "y_coord_original"],
        ),
        var=pd.DataFrame(var_list, index=var_list, columns=["m/z"]),
    )
    adata_SM.obsm["spatial"] = adata_SM.obs.loc[
        :, ["x_coord_original", "y_coord_original"]
    ].to_numpy()
    adata_SM.obs["total_intensity"] = np.array(adata_SM.X.sum(1)).flatten()
    adata_SM.obs["mean_intensity"] = np.array(adata_SM.X.mean(1)).flatten()
    adata_SM = adata_SM[adata_SM.obs["total_intensity"] > 0]
    return adata_SM