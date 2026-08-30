from __future__ import print_function, absolute_import
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class single_omics_loader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        # self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        omics, fea, slide_id, cell_id = self.dataset[index]

        omics = torch.Tensor(omics)
        fea = torch.Tensor(fea)

        # return img, fea, rna, protein, key
        return fea, omics, slide_id, cell_id


class multi_omics_loader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        # self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        omics1, omics2, fea, slide_id, cell_id = self.dataset[index]

        omics1 = torch.Tensor(omics1)
        omics2 = torch.Tensor(omics2)
        fea = torch.Tensor(fea)

        # return img, fea, rna, protein, key
        return fea, omics1, omics2, slide_id, cell_id