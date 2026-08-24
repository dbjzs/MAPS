import torch
import numpy as np
import pandas as pd
import random
import torch.nn as nn
from utils.utils import AverageMeter
from utils.evaluation import evaluator


def weighted_mse_loss(input, target, weight, type='mean'):
    if type == 'mean':
        return torch.mean((weight * (input - target) ** 2).sum(dim=-1))
    else:
        return torch.sum(weight * (input - target) ** 2)


def train(model, criterion_mse, criterion_ce, optimizer, trainloader, fea_list=None):
    model.train()
    losses = AverageMeter()

    for batch_idx, (img, rna_omics, protein, slide_id, _) in enumerate(trainloader):
        img, rna_omics, protein, slide_id = img.cuda(), rna_omics.cuda(), protein.cuda(), np.array(slide_id)
        ID1_mask, ID100_mask = slide_id=='ID1', slide_id=='ID100'

        target1, target2 = rna_omics, protein
        outputs1 = model(img)

        bs = img.shape[0]
        #######################
        # here, I change
        # sample_indices = np.random.choice(fea_list.shape[0], size=bs//3, replace=False)
        # img_others = torch.Tensor(fea_list[sample_indices]).cuda()

        sample_indices = [  fea_list[i][np.random.choice(fea_list[i].shape[0], size=bs//2, replace=False)] for i in range(2)  ]
        img_others = torch.Tensor(  np.concatenate(  sample_indices, axis=0    )  ).cuda()
        

        outputs2 = model(img_others)

        loss, loss1, loss2, loss3 = 0.0, 0.0, 0.0, 0.0

        loss1 = weighted_mse_loss(outputs1[0], target1, weight=torch.exp(target1))
        loss2 = criterion_mse(outputs1[1], target2)
        
        domain_label = torch.cat([torch.tensor( ID1_mask) * 0 + torch.tensor(ID100_mask) * 1, torch.ones(bs//2 * 2)*2]).long().cuda()
        domain_loss = criterion_ce(torch.cat([outputs1[2], outputs2[2]], dim=0), domain_label)

        total_loss = loss1 + loss2 + domain_loss
        loss = loss1 / (loss1 / total_loss).detach() + loss2 / (loss2 / total_loss).detach() + 0.3 * domain_loss / (domain_loss / total_loss).detach()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.data, img.size(0))

        if (batch_idx+1) == len(trainloader):
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))
    

def test(model, testloader, omics1_panel, omics2_panel):
    model.eval()

    predict_list_omics1, target_list_omics1 = [], []
    predict_list_omics2, target_list_omics2 = [], []

    with torch.no_grad():
        for _, (img, rna_omics, protein, slide_id, _) in enumerate(testloader):
            img, rna_omics, protein, slide_id = img.cuda(), rna_omics.cuda(), protein.cuda(), np.array(slide_id)

            target1, target2 = rna_omics, protein

            outputs = model(img)
            outputs_omics1, outputs_omics2 = outputs[0], outputs[1]

            predict_list_omics1.append(outputs_omics1)
            target_list_omics1.append(target1)
            predict_list_omics2.append(outputs_omics2)
            target_list_omics2.append(target2)


    predict_list_omics1 = torch.cat(predict_list_omics1, dim=0).cpu().numpy().astype(np.float32)
    target_list_omics1  = torch.cat(target_list_omics1, dim=0).cpu().numpy().astype(np.float32)
    predict_list_omics2 = torch.cat(predict_list_omics2, dim=0).cpu().numpy().astype(np.float32)
    target_list_omics2  = torch.cat(target_list_omics2, dim=0).cpu().numpy().astype(np.float32)

    pcc_o1, spcc_o1, rmse_o1 = evaluator(predict_list_omics1, target_list_omics1, panel=omics1_panel, omics_type='omics1')
    pcc_o2, spcc_o2, rmse_o2 = evaluator(predict_list_omics2, target_list_omics2, panel=omics2_panel, omics_type='omics2')


    results_omics1 = pd.DataFrame({'pcc_o1': pcc_o1, 'spcc_o1': spcc_o1, 'rmse_o1': rmse_o1})
    results_omics2 = pd.DataFrame({'pcc_o2': pcc_o2, 'spcc_o2': spcc_o2, 'rmse_o2': rmse_o2})

    results_omics1.to_csv('results_omics1.csv', index=False)
    results_omics2.to_csv('results_omics2.csv', index=False)
        
    print('Slice 10 Omics 1: PCC {:.4f}; SPCC {:.4f}; rmse {:.4f}'.format(pcc_o1.mean(), spcc_o1.mean(), rmse_o1.mean()), flush=True)
    print('Slice 10 Omics 2: PCC {:.4f}; SPCC {:.4f}; rmse {:.4f}'.format(pcc_o2.mean(), spcc_o2.mean(), rmse_o2.mean()), flush=True)
    
    return None