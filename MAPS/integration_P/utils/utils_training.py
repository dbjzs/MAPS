import torch
import numpy as np
import pandas as pd
import random
import torch.nn as nn
from MAPS.integration_P.utils.utils import AverageMeter
from MAPS.integration_P.utils.evaluation import evaluator


def weighted_mse_loss(input, target, weight, type='mean'):
    if type == 'mean':
        return torch.mean((weight * (input - target) ** 2).sum(dim=-1))
    else:
        return torch.sum(weight * (input - target) ** 2)


def train_single_omics(model, criterion_mse, criterion_ce, optimizer, trainloader, fea_list=None):
    model.train()
    losses = AverageMeter()

    for batch_idx, (img, omics, slide_id, _) in enumerate(trainloader):
        img, omics, slide_id = img.cuda(), omics.cuda(), np.array(slide_id)

        target1 = omics
        outputs1 = model(img)

        #######################
        loss = weighted_mse_loss(outputs1, target1, weight=torch.exp(target1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.data, img.size(0))

        if (batch_idx+1) == len(trainloader):
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))
    

def test_single_omics(model, testloader, omics1_panel):
    model.eval()

    predict_list_omics1, target_list_omics1 = [], []
    with torch.no_grad():
        for _, (img, omics, slide_id, _) in enumerate(testloader):
            img, omics, slide_id = img.cuda(), omics.cuda(), np.array(slide_id)

            target1 = omics
            outputs = model(img)

            predict_list_omics1.append(outputs)
            target_list_omics1.append(target1)
           

    predict_list_omics1 = np.clip(torch.cat(predict_list_omics1, dim=0).cpu().numpy().astype(np.float32), 0, 1)
    target_list_omics1  = torch.cat(target_list_omics1, dim=0).cpu().numpy().astype(np.float32)

    return predict_list_omics1


def train_multi_omics(model, criterion_mse, criterion_ce, optimizer, trainloader, fea_list=None):
    model.train()
    losses = AverageMeter()

    for batch_idx, (img, omics1, omics2, slide_id, _) in enumerate(trainloader):
        img, omics1, omics2, slide_id = img.cuda(), omics1.cuda(), omics2.cuda(), np.array(slide_id)

        target1, target2 = omics1, omics2
        outputs1 = model(img)

        loss1 = weighted_mse_loss(outputs1[0], target1, weight=torch.exp(target1))
        loss2 = criterion_mse(outputs1[1], target2)
        
        total_loss = loss1 + loss2
        loss = loss1 / (loss1 / total_loss).detach() + loss2 / (loss2 / total_loss).detach()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.data, img.size(0))

        if (batch_idx+1) == len(trainloader):
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))
    

def test_multi_omics(model, testloader, omics1_panel, omics2_panel):
    model.eval()

    predict_list_omics1, target_list_omics1 = [], []
    predict_list_omics2, target_list_omics2 = [], []

    with torch.no_grad():
        for _, (img, omics1, _, slide_id, _) in enumerate(testloader):
            img, omics1, slide_id = img.cuda(), omics1.cuda(), np.array(slide_id)

            outputs1, outputs2 = model(img)

            predict_list_omics1.append(outputs1)
            target_list_omics1.append(omics1)
            predict_list_omics2.append(outputs2)

    predict_list_omics1 = np.clip(torch.cat(predict_list_omics1, dim=0).cpu().numpy().astype(np.float32), 0, 1)
    target_list_omics1  = torch.cat(target_list_omics1, dim=0).cpu().numpy().astype(np.float32)
    predict_list_omics2 = np.clip(torch.cat(predict_list_omics2, dim=0).cpu().numpy().astype(np.float32), 0, 1)
    
    return predict_list_omics1, predict_list_omics2