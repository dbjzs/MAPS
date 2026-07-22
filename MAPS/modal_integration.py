import anndata
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from MAPS.utils import set_seed


# The reference code is from https://github.com/dbjzs/SpaLP
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, bn=False, activation_fn=None):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.activation = activation_fn

    def forward(self, x):  # x: (N, C_in)
        x = self.linear(x)  # (N, C_out)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x  # (N, C_out)


def batch_gather(data, index):
    return data[index]


def gather_neighbour(point_features, neighbor_idx):
    point_features_t = point_features
    gathered_features = batch_gather(point_features_t, neighbor_idx)
    return gathered_features


class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=1)  # softmax over k (neighbor dim)
        )
        self.mlp = MLP(in_channels, out_channels, bn=False, activation_fn=None)

    def forward(self, x):  # x: (N, k, C)
        scores = self.score_fn(x)  # (N, k, C)
        
        feat = torch.sum(scores * x, dim=1)  # (N, C)
        return self.mlp(feat)  # (N, C_out)


class LocalFeatureAggregation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp1 = MLP(in_channels, 2*out_channels, bn=False, activation_fn=nn.ReLU())
        self.bn_after_gather = nn.BatchNorm1d(2*out_channels)
        self.pool1 = AttentivePooling(2*out_channels, out_channels)

    def forward(self, features, neighbor_idx):
        """
        coords: (N, coord_dim)
        features: (N, C_in)
        neighbor_idx: (N, k)
        """
        x = self.mlp1(features)  # (N, 128)
        x = gather_neighbour(x, neighbor_idx)#(N, k, 128)

        x = x.permute(0, 2, 1)  
        x = self.bn_after_gather(x) 
        x = x.permute(0, 2, 1)
        
        x = self.pool1(x)  # (N, out_channels)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, bn=False, activation_fn=nn.ReLU()):
        super().__init__()
        self.mlp = MLP(in_channels, out_channels, bn=bn, activation_fn=activation_fn)

    def forward(self, x):  # x: (N, C)
        return self.mlp(x)


class COI(nn.Module):#COI(Cross-omics integration)
    def __init__(self, in_channels1,in_channels2, out_channels):
        super().__init__()
        self.encoder1 = LocalFeatureAggregation(in_channels1, out_channels)
        self.encoder2 = LocalFeatureAggregation(in_channels2, out_channels)
        
        self.decoder1 = Decoder(out_channels, in_channels1, bn=False, activation_fn=nn.ReLU())
        self.decoder2 = Decoder(out_channels, in_channels2, bn=False, activation_fn=nn.ReLU())
        
    def forward(self, features1, neighbor_idx1,features2,neighbor_idx2,neighbor_idx1_2):
        
        embedding1 = self.encoder1(features1, neighbor_idx1)
        embedding2 = self.encoder2(features2, neighbor_idx2)
        
        embedding1_2 = self.encoder2(features2, neighbor_idx1_2)
        reconstructed1 = self.decoder1(embedding1)
        reconstructed2 = self.decoder2(embedding2)
        reconstructed1_2 = self.decoder1(embedding1_2)
        return reconstructed1, reconstructed2, reconstructed1_2, embedding1, embedding2, embedding1_2

    def Train(self,graph1,graph2,idx_1_to_2,epochs=200,lr=1e-3,device=None,seed=7):
        set_seed(seed)
        self.to(device)
            
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        self.train()
        pbar = tqdm(range(epochs), desc="Cross-modal integration", ncols=150)
        for epoch in pbar:
            optimizer.zero_grad()
            reconstructed1, reconstructed2, reconstructed1_2, embedding1, embedding2, embedding1_2 = self(graph1.features, graph1.neighbor_idx, graph2.features, graph2.neighbor_idx, idx_1_to_2)
            loss1 = criterion(reconstructed1, graph1.features)
            loss2 = criterion(reconstructed2, graph2.features)
            loss3 = criterion(reconstructed1_2, graph1.features)
            loss_align = criterion(embedding1, embedding1_2)
            loss=loss1+loss2+loss3+5*loss_align
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"Epoch": epoch,"Loss": f"{loss.item():.3f}","Loss1": f"{loss1.item():.3f}","Loss2": f"{loss2.item():.3f}","Loss3": f"{loss3.item():.3f}",  "Loss4": f"{loss_align.item():.3f}"}) #"Loss3": f"{loss3.item():.3f}"


    @torch.no_grad()
    def get_embedding(self,graph1,graph2,idx_1_to_2,device=None):
        self.eval()
        reconstructed1, reconstructed2, reconstructed1_2, embedding1, embedding2, embedding1_2 = self(graph1.features, graph1.neighbor_idx, graph2.features, graph2.neighbor_idx, idx_1_to_2)
        embedding1=embedding1.cpu().numpy()
        embedding2=embedding2.cpu().numpy()
        reconstructed1=reconstructed1.cpu().numpy()
        reconstructed2=reconstructed2.cpu().numpy()
        
        return embedding1, embedding2,reconstructed1,reconstructed2


class TripleCOI(nn.Module):#TripleCOI(TripleCross-omics integration)
    def __init__(self, in_channels1,in_channels2,in_channels3, out_channels):
        super().__init__()
        self.encoder1 = LocalFeatureAggregation(in_channels1, out_channels)
        self.encoder2 = LocalFeatureAggregation(in_channels2, out_channels)
        self.encoder3 = LocalFeatureAggregation(in_channels3, out_channels)
        
        self.decoder1 = Decoder(out_channels, in_channels1, bn=False, activation_fn=nn.ReLU())
        self.decoder2 = Decoder(out_channels, in_channels2, bn=False, activation_fn=nn.ReLU())
        self.decoder3 = Decoder(out_channels, in_channels3, bn=False, activation_fn=nn.ReLU())
        
    def forward(self, features1, neighbor_idx1,features2,neighbor_idx2,features3,neighbor_idx3,neighbor_idx1_3,neighbor_idx2_3):
        
        embedding1 = self.encoder1(features1, neighbor_idx1)
        embedding2 = self.encoder2(features2, neighbor_idx2)
        embedding3 = self.encoder3(features3, neighbor_idx3)


        
        embedding1_3 = self.encoder3(features3, neighbor_idx1_3)
        embedding2_3 = self.encoder3(features3, neighbor_idx2_3)
        
        reconstructed1 = self.decoder1(embedding1)
        reconstructed2 = self.decoder2(embedding2)
        reconstructed3 = self.decoder3(embedding3)
        
        reconstructed1_3 = self.decoder1(embedding1_3)
        reconstructed2_3 = self.decoder2(embedding2_3)
        
        return reconstructed1, reconstructed2,reconstructed3, reconstructed1_3,reconstructed2_3, embedding1, embedding2,embedding3, embedding1_3,embedding2_3

    def Train(self,graph1,graph2,graph3,idx_1_to_3,idx_2_to_3,epochs=200,lr=1e-3,device=None,seed=7):
        set_seed(seed)
        self.to(device)
            
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        self.train()
        pbar = tqdm(range(epochs), desc="Cross-modal integration", ncols=150)
        for epoch in pbar:
            optimizer.zero_grad()
            reconstructed1, reconstructed2,reconstructed3, reconstructed1_3,reconstructed2_3, embedding1, embedding2,embedding3, embedding1_3,embedding2_3 = self(graph1.features, graph1.neighbor_idx, graph2.features, graph2.neighbor_idx,graph3.features,
                                                                                                                                                                                                graph3.neighbor_idx,idx_1_to_3,idx_2_to_3)
            loss1 = criterion(reconstructed1, graph1.features)
            loss2 = criterion(reconstructed2, graph2.features)
            loss3 = criterion(reconstructed3, graph3.features)
            
            loss4 = criterion(reconstructed1_3, graph1.features)
            loss5 = criterion(reconstructed2_3, graph2.features)
            
            loss_align1 = criterion(embedding1, embedding1_3)
            loss_align2 = criterion(embedding2, embedding2_3)
            
            loss=loss1+loss2+loss3+loss4+loss5+5*loss_align1+5*loss_align2
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"Epoch": epoch,"Total_Loss": f"{loss.item():.3f}","Loss_rec": f"{loss1+loss2+loss3+loss4+loss5.item():.3f}","Loss_align": f"{loss_align1+loss_align2.item():.3f}"})


    @torch.no_grad()
    def get_embedding(self,graph1,graph2,graph3,idx_1_to_3,idx_2_to_3,device=None):
        self.eval()
        reconstructed1, reconstructed2,reconstructed3, reconstructed1_3,reconstructed2_3, embedding1, embedding2,embedding3, embedding1_3,embedding2_3 = self(graph1.features, graph1.neighbor_idx, graph2.features, graph2.neighbor_idx,graph3.features,
                                                                                                                                                                                            graph3.neighbor_idx,idx_1_to_3,idx_2_to_3)
        embedding1=embedding1.cpu().numpy()
        embedding2=embedding2.cpu().numpy()
        embedding3=embedding3.cpu().numpy()
        reconstructed1=reconstructed1.cpu().numpy()
        reconstructed2=reconstructed2.cpu().numpy()
        reconstructed3=reconstructed3.cpu().numpy()
        return embedding1, embedding2,embedding3,reconstructed1,reconstructed2,reconstructed3
