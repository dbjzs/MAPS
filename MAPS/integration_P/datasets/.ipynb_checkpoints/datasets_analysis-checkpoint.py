from __future__ import print_function, absolute_import
import os

import h5py
import numpy as np
import scanpy as sc


def load_h5ad(path, shape, hvg):
    adata = sc.read_h5ad(path + '.h5ad')

    ##############
    # filter cells with NaN values in obs
    mask = []
    for i in adata.obs.columns:
        values = adata.obs[i].values
        mask.append(~np.isnan(values))

    aggregated_mask = np.all(mask, axis=0)
    adata = adata[aggregated_mask, :]

    ##############
    # filter cells with too few genes
    sc.pp.filter_cells(adata, min_genes=50)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hvg)

    ##############
    # filter cells at the boundary of the image
    mask = [adata.obsm['spatial'][:, 0] - 128>0, 
            adata.obsm['spatial'][:, 1] - 128>0, 
            adata.obsm['spatial'][:, 0] + 128<shape[1], 
            adata.obsm['spatial'][:, 1] + 128<shape[0]]
    aggregated_mask = np.all(mask, axis=0)
    adata = adata[aggregated_mask, :]

    return adata

class LiverCancer(object):
    def __init__(self, 
                 path_img='/mnt/sdf/zhikangwang/fudan_Xenium_liver_cancer_organize/Image_features_registration_v2', 
                 path_all='/mnt/sdf/zhikangwang/fudan_Xenium_liver_cancer_organize',
                 hvg=500):
        
        self.path_img, self.path_all = path_img, path_all
        
        shapes = {
            'ID1': (34266, 45588),
            'ID10': (30874, 48338),
            'ID100': (30906, 54072)
        }

        rna_adata_list = [load_h5ad(os.path.join(path_all, ID), shapes[ID], hvg=hvg) for ID in ['ID1', 'ID10', 'ID100']]

        ##################
        # self.test_ratio = 0.2
        # for i in range(3):
        #     width = rna_adata_list[i].obsm['spatial'].max(axis=0)[0]
        #     threshold = self.test_ratio * width
        #     temp_mask = rna_adata_list[i].obsm['spatial'][:, 0] < threshold
        #     rna_adata_list[i] = rna_adata_list[i][temp_mask, :]
        ##################
            
        gene_mask = np.any([rna_adata_list[i].var.highly_variable.values for i in range(3)], axis=0)
        self.gene_mask = gene_mask
        rna_adata_list = [rna_adata_list[i][:, gene_mask] for i in range(3)]

        ##########
        rna_list, protein_list = [], []
        for i in range(3):
            rna_list.append(rna_adata_list[i].X.toarray())
            protein_list.append(rna_adata_list[i].obs.values)

        rna_list = np.log1p(np.concatenate(rna_list, axis=0))
        protein_list = np.log1p(np.concatenate(protein_list, axis=0))
        self.rna_max = np.max(rna_list, axis=0)
        self.protein_max = np.max(protein_list, axis=0)
        self.protein_min = np.min(protein_list, axis=0)
        ##########

        self.rna_panel = rna_adata_list[0].var_names
        self.protein_panel = rna_adata_list[0].obs.columns

        self.omics1_size = len(self.rna_panel)//2
        self.omics2_size = len(self.rna_panel) - self.omics1_size
        self.omics3_size = len(self.protein_panel)

        ###############
        self.omics1_panel = self.rna_panel[:self.omics1_size]
        self.omics2_panel = self.rna_panel[self.omics1_size:]
        self.omics3_panel = self.protein_panel
        ################

      

if __name__ == '__main__':
    dataset = LiverCancer()
