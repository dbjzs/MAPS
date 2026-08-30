from __future__ import print_function, absolute_import
import os

from tqdm import tqdm
import h5py
import numpy as np
import scanpy as sc
import pickle
import pandas as pd
import warnings
from scipy import sparse

warnings.filterwarnings("ignore")


class Single_omics(object):
    def __init__(self, path_omics, train_slices, test_slices):
        
        self.path_omics = path_omics
        
        train_adata_list = [sc.read_h5ad(os.path.join(path_omics, slice_name)) for slice_name in train_slices]
        test_adata_list = [sc.read_h5ad(os.path.join(path_omics, slice_name)) for slice_name in test_slices]

        ##########
        train_adata = sc.concat(train_adata_list, axis=0)
        test_adata = sc.concat(test_adata_list, axis=0)

        self.test_adata_spatial = test_adata.obsm['spatial']

        sc.pp.log1p(train_adata)
        sc.pp.log1p(test_adata)

        self.max, self.min = np.max(train_adata.X.toarray(), axis=0), np.min(train_adata.X.toarray(), axis=0)
        ##########

        self.panel = train_adata.var_names.tolist()
        self.panel_size = len(self.panel)
        self.fea_size = train_adata.obsm['UNI_embedding'].shape[1]

        self.datasets_train = self._process_data(train_adata)
        self.datasets_test = self._process_data(test_adata)

        print("=> Liver cancer loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  Panel    |  Omics1:{}| ".format(len(self.panel)))
        print("  ------------------------------")
        print("  train    |  {:5d} cells from {:5d} slides ".format(len(self.datasets_train), len(train_slices)))
        print("  test     |  {:5d} cells from {:5d} slides".format(len(self.datasets_test), len(test_slices)))
        print("  ------------------------------")
 
    def _process_data(self, adata, slice_ID='slice'):
        dataset = []

        #############
        # data preprocessing
        array = adata.X.toarray()
        array = (array - self.min[None, :]) / (self.max - self.min)[None, :]
        img_fea = adata.obsm['UNI_embedding']
        
        indexes = adata.obs.index.tolist()

        dataset = list(zip(array,
                           img_fea,
                           adata.obs['batch_id'].to_numpy(),
                           indexes))

        return dataset


class Multi_omics(object):
    def __init__(self, path_omics, train_slices, test_slices):
        
        self.path_omics = path_omics
        
        train_adata_omics1_list = [sc.read_h5ad(os.path.join(path_omics, slice_name)) for slice_name in train_slices['omics1']]
        train_adata_omics2_list = [sc.read_h5ad(os.path.join(path_omics, slice_name)) for slice_name in train_slices['omics2']]

        test_adata_list = [sc.read_h5ad(os.path.join(path_omics, slice_name)) for slice_name in test_slices]

        ##########
        train_adata_omics1 = sc.concat(train_adata_omics1_list, axis=0)
        train_adata_omics2 = sc.concat(train_adata_omics2_list, axis=0)

        test_adata = sc.concat(test_adata_list, axis=0)
        self.test_adata_spatial = test_adata.obsm['spatial']

        sc.pp.log1p(train_adata_omics1)
        sc.pp.log1p(train_adata_omics2)
        sc.pp.log1p(test_adata)

        self.max_omics1, self.min_omics1 = np.max(train_adata_omics1.X.toarray() if sparse.issparse(train_adata_omics1.X) else train_adata_omics1.X, axis=0), np.min(train_adata_omics1.X.toarray() if sparse.issparse(train_adata_omics1.X) else train_adata_omics1.X, axis=0)
        self.max_omics2, self.min_omics2 = np.max(train_adata_omics2.X.toarray() if sparse.issparse(train_adata_omics2.X) else train_adata_omics2.X, axis=0), np.min(train_adata_omics2.X.toarray() if sparse.issparse(train_adata_omics2.X) else train_adata_omics2.X, axis=0)
        ##########

        self.panel_omics1 = train_adata_omics1.var_names.tolist()
        self.panel_omics2 = train_adata_omics2.var_names.tolist()
        self.panel_size_omics1 = len(self.panel_omics1)
        self.panel_size_omics2 = len(self.panel_omics2)

        self.fea_size = train_adata_omics1.obsm['UNI_embedding'].shape[1]

        self.datasets_train = self._process_data(train_adata_omics1, train_adata_omics2)
        self.datasets_test = self._process_data(test_adata)

        print("=> Liver cancer loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  Panel    |  Omics1:{} Omics2:{}| ".format(len(self.panel_omics1), len(self.panel_omics2)))
        print("  ------------------------------")
        print("  train    |  {:5d} cells from {:5d} slides ".format(len(self.datasets_train), len(train_slices)))
        print("  test     |  {:5d} cells from {:5d} slides".format(len(self.datasets_test), len(test_slices)))
        print("  ------------------------------")
 
    def _process_data(self, adata, adata_omics2=None):
        dataset = []

        #############
        # data preprocessing
        array = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
        array = (array - self.min_omics1[None, :]) / (self.max_omics1 - self.min_omics1)[None, :]
        img_fea = adata.obsm['UNI_embedding']
        
        indexes = adata.obs.index.tolist()

        if adata_omics2 is not None:
            array_omics2 = adata_omics2.X.toarray() if sparse.issparse(adata_omics2.X) else adata_omics2.X
            array_omics2 = (array_omics2 - self.min_omics2[None, :]) / (self.max_omics2 - self.min_omics2)[None, :]
            
            dataset = list(zip(array,
                               array_omics2,
                               img_fea,
                               adata.obs['batch_id'].to_numpy(),
                               indexes))    
        else:
            dataset = list(zip(array,
                               np.zeros((array.shape[0], self.panel_size_omics2)),  # Placeholder for omics2 data
                               img_fea,
                               adata.obs['batch_id'].to_numpy(),
                               indexes))

        return dataset
    

if __name__ == '__main__':
    # dataset = Single_omics(path_omics='/home/wzk/Code_SRT/Round35_bingjie/datasets_10x_hbc/hbc313', train_slices=['hbc313_uni2_cls_rep1.h5ad'], test_slices=['hbc313_uni2_cls_rep2.h5ad'])
    dataset = Multi_omics(path_omics='/home/wzk/Code_SRT/Round35_bingjie/datasets_10x_hbc/hbc313', train_slices={'omics1': ['hbc313_uni2_cls_rep1.h5ad'], 'omics2': ['hbc313_uni2_cls_rep1_protein.h5ad']}, test_slices=['hbc313_uni2_cls_rep2.h5ad'])
    # breakpoint()
