import torch
import scipy
import math
import numpy as np
import warnings

import matplotlib.pyplot as plt


def draw_dot_plots(predict_list, target_list, pcc_list, spcc_list, panel, omics_type):
    for i in range(predict_list.shape[1]):
        predict, target = predict_list[:, i], target_list[:, i]

        max_val = max(max(predict), max(target))
        lim = (0, max_val)

        plt.figure(figsize=(8, 6))
        plt.scatter(predict, target, color='blue', alpha=0.5, s=5)
        plt.plot(lim, lim, color='red', linestyle='--', linewidth=2)

        plt.xlim(lim)
        plt.ylim(lim)

        plt.xlabel('Predicted')
        plt.ylabel('Measured')
        plt.title('Panel: {} {};pcc {:.2f} and spcc {:.2f}'.format(omics_type, panel[i], pcc_list[i], spcc_list[i]))
        plt.grid(False)
        plt.savefig('./plots/{}_{}.jpg'.format(omics_type, panel[i]))
        plt.close()


def evaluator(predict_list, target_list, panel=None, omics_type=None, draw_plots=True):
    if isinstance(predict_list, list):
        predict_list = torch.cat(predict_list, dim=0).cpu().detach().numpy()
    if isinstance(target_list, list):
        target_list = torch.cat(target_list, dim=0).cpu().detach().numpy()

    pcc_list, spcc_list, rmse_list = [], [], []

    for i in range(target_list.shape[1]):
        pcc, _ = scipy.stats.pearsonr(predict_list[:, i], target_list[:, i])
        spcc, _ = scipy.stats.spearmanr(predict_list[:, i], target_list[:, i])
        rmse =  np.sqrt(np.mean( (predict_list[:, i] - target_list[:, i])**2 ))
   
        pcc_list.append(pcc)
        spcc_list.append(spcc)
        rmse_list.append(rmse)

    pcc_list, spcc_list, rmse_list = np.array(pcc_list), np.array(spcc_list), np.array(rmse_list)
    
    if draw_plots == True:
        draw_dot_plots(predict_list, target_list, pcc_list, spcc_list, panel, omics_type)
    return pcc_list, spcc_list, rmse_list