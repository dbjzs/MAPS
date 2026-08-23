from scipy.spatial import cKDTree
import numpy as np
def evaluation(src_cor, tgt_cor, src_cell_type, tgt_cell_type):
    kd_tree = cKDTree(src_cor)
    distances, indices = kd_tree.query(tgt_cor, k=1) 
    src_arr = src_cell_type.to_numpy()
    tgt_arr = tgt_cell_type.to_numpy()
    CI = np.mean(tgt_arr == src_arr[indices])
    return CI