from __future__ import print_function, absolute_import
import os

from tqdm import tqdm
import h5py
import numpy as np
import scanpy as sc
import pickle
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def load_h5ad(path_rna, path_protein, shape, hvg):
    adata_rna = sc.read_h5ad(path_rna)
    adata_protein = sc.read_h5ad(path_protein)

    ##############
    # filter cells with too few genes
    mask = (adata_rna.X > 0).sum(axis=1) >= 50
    adata_rna = adata_rna[mask, :]

    sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=hvg)
    sc.pp.log1p(adata_rna)

    adata_protein = adata_protein[mask, :]
    sc.pp.log1p(adata_protein)

    ##############
    # filter cells at the boundary of the image
    mask = [adata_rna.obsm['spatial'][:, 0] - 128>0, 
            adata_rna.obsm['spatial'][:, 1] - 128>0, 
            adata_rna.obsm['spatial'][:, 0] + 128<shape[1], 
            adata_rna.obsm['spatial'][:, 1] + 128<shape[0]]

    aggregated_mask = np.all(mask, axis=0)
    adata_rna = adata_rna[aggregated_mask, :]
    adata_protein = adata_protein[aggregated_mask, :]

    return adata_rna, adata_protein

class LiverCancer(object):
    def __init__(self, path_omics, path_img, hvg, return_data):
        
        self.path_img = path_img
        
        shapes = {
            'ID1': (34266, 45588),
            'ID10': (30874, 48338),
            'ID100': (30906, 54072)
        }

        rna_adata_list, protein_adata_list = [], []

        for ID in ['ID1', 'ID10', 'ID100']:
            rna_adata, protein_adata = load_h5ad(os.path.join(path_omics, ID + '_rna.h5ad'), os.path.join(path_omics, ID + '_protein.h5ad'), shapes[ID], hvg=hvg)
            rna_adata_list.append(rna_adata)
            protein_adata_list.append(protein_adata)

        gene_mask = np.any([rna_adata_list[i].var.highly_variable.values for i in range(3)], axis=0)

        gene_path = os.path.join(os.path.dirname(__file__), 'gene.csv')
        df = pd.read_csv(gene_path, header=None)
        gene_panel = df.iloc[:, 0].values
        gene_mask_bj = np.isin(rna_adata_list[0].var_names, gene_panel)
        gene_mask = np.logical_or(gene_mask, gene_mask_bj)

        self.gene_mask = gene_mask
        rna_adata_list = [rna_adata_list[i][:, gene_mask] for i in range(3)]
        self.spatial_coordinates = [rna_adata_list[i].obsm['spatial'] for i in range(3)]

        ##########
        rna_list, protein_list = [], []
        for i in range(3):
            rna_list.append(rna_adata_list[i].X.toarray())
            protein_list.append(protein_adata_list[i].X.toarray())

        rna_list = np.concatenate(rna_list, axis=0)
        protein_list = np.concatenate(protein_list, axis=0)

        self.rna_max, self.rna_min = np.max(rna_list, axis=0), np.min(rna_list, axis=0)
        self.protein_max, self.protein_min = np.max(protein_list, axis=0), np.min(protein_list, axis=0)
        ##########

        rna_panel = rna_adata_list[0].var_names
        protein_panel = protein_adata_list[0].var_names

        self.omics1_size = len(rna_panel)
        self.omics2_size = len(protein_panel)

        ###############
        self.omics1_panel = rna_panel
        self.omics2_panel = protein_panel
        ################

        if return_data == True:
            data_ID1, data_ID10, data_ID100 = \
                self._process_data(rna_adata_list[0], protein_adata_list[0], 'ID1'), \
                self._process_data(rna_adata_list[1], protein_adata_list[1], 'ID10'), \
                self._process_data(rna_adata_list[2], protein_adata_list[2], 'ID100')

            self.datasets = data_ID1 + data_ID100
            self.datasets_test = data_ID10

            print("=> Liver cancer loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  Panel    |  Omics1:{}; Omics2:{}; | ".format(len(rna_panel), len(protein_panel)))
            print("  ------------------------------")
            print("  train    |  {:5d} cells from {:5d} slides ".format(len(self.datasets), 2))
            print("  test     |  {:5d} cells from {:5d} slides".format(len(self.datasets_test), 1))
            print("  ------------------------------")
 
    def _process_data(self, rna_adata, protein_adata, slice_ID):
        dataset = []

        #############
        # data preprocessing
        array_rna, array_protein = rna_adata.X.toarray(), protein_adata.X.toarray()

        array_rna = (array_rna - self.rna_min[None, :]) / (self.rna_max - self.rna_min)[None, :]
        array_protein = (array_protein - self.protein_min[None, :]) / (self.protein_max - self.protein_min)[None, :]
        
        indexes = rna_adata.obs.index.tolist()
        
        ##################
        
        file = h5py.File(os.path.join(self.path_img, slice_ID, 'patches_256_features_local_global.h5'), 'r')
        fea_all = [file[i][:] for i in tqdm(indexes)]

        # with open(f"/mnt/sdf/zhikangwang/fudan_Xenium_liver_cancer_organize/{slice_ID}_new.pkl", "wb") as f1:
        #     pickle.dump(fea_all, f1)

        # with open(f"/mnt/sdf/zhikangwang/fudan_Xenium_liver_cancer_organize/{slice_ID}_new.pkl", "rb") as f:
        #     fea_all = pickle.load(f)

        ##################

        dataset = list(zip(fea_all,
                           array_rna,
                           array_protein,
                           [slice_ID] * len(indexes),
                           indexes))

        return dataset

if __name__ == '__main__':
    dataset = LiverCancer()
