from __future__ import absolute_import

import math
import torch
from torch import nn

from MAPS.integration_P.model.attention import *
import torch.nn.init as init
from torch.autograd import Function


def init_weights(module):
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.constant_(module.bias, 0)
    
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)
    
    elif isinstance(module, nn.LayerNorm):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)
    
    elif isinstance(module, nn.MultiheadAttention):
        init.xavier_uniform_(module.in_proj_weight)
        if module.in_proj_bias is not None:
            init.constant_(module.in_proj_bias, 0)
    
        init.xavier_uniform_(module.out_proj.weight)
        if module.out_proj.bias is not None:
            init.constant_(module.out_proj.bias, 0)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class MLP(nn.Module):
    def __init__(self, n_inp, n_out, activation=None):
        super().__init__()
        self.linear = nn.Linear(n_inp, n_out)        
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        y = self.linear(x)
        y = self.activation(y)
        return y

class ELU(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta

    def forward(self, x):
        return self.activation(x) + self.beta

class Transformer(nn.Module):
    def __init__(self, fea_size, num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.mhsa = Self_Attention(query_dim=fea_size, context_dim=fea_size, heads=num_heads, dim_head=fea_size//num_heads, dropout=dropout_rate)
        self.ffn = FeedForward(fea_size, mult=2)
        self.ln1, self.ln2 = nn.LayerNorm(fea_size), nn.LayerNorm(fea_size)
    
    def forward(self, x):
        x = self.mhsa(self.ln1(x)) + x
        x = self.ffn(self.ln2(x)) + x
        return x
    
class Transformer_Cross(nn.Module):
    def __init__(self, fea_size, num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.mhca = Cross_Attention(query_dim=fea_size, context_dim=fea_size, heads=num_heads, dim_head=fea_size//num_heads, dropout=dropout_rate)
        self.ffn = FeedForward(fea_size, mult=2)
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(fea_size), nn.LayerNorm(fea_size), nn.LayerNorm(fea_size)
    
    def forward(self, x1, x2):
        x1 = self.mhca(self.ln1(x1), self.ln2(x2)) + x1
        x1 = self.ffn(self.ln3(x1)) + x1
        return x1

# 1. GRL Implementation (remains the same)
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output.neg() * ctx.lambda_), None

class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, x, lambda_=1.0):
        return GradientReversalFunction.apply(x, lambda_)

class Network_multi_omics(nn.Module):
    def __init__(self, omics1_size, omics2_size, input_dimension, noise_rate=0.2, dropout_rate=0.1):
        super(Network_multi_omics, self).__init__()

        self.input_dimension = input_dimension
        self.fea_size = 256
        self.num_heads = 4
        self.omics1_size, self.omics2_size = omics1_size, omics2_size

        self.noise_dropout = nn.Dropout(noise_rate)

        # Encoder initialization
        self.net_lat1 = nn.Sequential(MLP(self.input_dimension//2, 512), MLP(512, 512))
        self.net_lat2 = nn.Sequential(MLP(self.input_dimension//2, 512), MLP(512, 256))

        self.net_lat_omics1 = MLP(256+512, self.fea_size)
        self.net_lat_omics2 = MLP(256+512, self.fea_size)
        self.net_lat_omics3 = MLP(256+512, self.fea_size)

        ########################
        # Token initialization
        self.omics12 = nn.Parameter(torch.randn((1, self.omics1_size + self.omics2_size, self.fea_size), requires_grad=True))
        trunc_normal_(self.omics12, mean=0.0, std=.02)

        ########################
        # Transformers
        self.trans_omics1 = Transformer(fea_size=self.fea_size, num_heads=self.num_heads, dropout_rate=dropout_rate)
        self.trans_omics2 = Transformer(fea_size=self.fea_size, num_heads=self.num_heads, dropout_rate=dropout_rate)

        self.trans_cross1 = Transformer_Cross(fea_size=self.fea_size, num_heads=self.num_heads, dropout_rate=dropout_rate)
        self.trans_cross2 = Transformer_Cross(fea_size=self.fea_size, num_heads=self.num_heads, dropout_rate=dropout_rate)
        
        ########################
        # Others initialization
        self.dropout = nn.Dropout(dropout_rate)

        self.recalibrate_layers = nn.Sequential(nn.Linear(self.fea_size, self.fea_size), 
                                                # nn.BatchNorm1d(3),
                                                nn.LeakyReLU(0.1, inplace=True))

        #########################
        # Domain classifier initialization
        self.domain_classifier1 = nn.Sequential(nn.Linear(self.fea_size*3, self.fea_size), 
                                                nn.LeakyReLU(0.1, inplace=True), 
                                                nn.Linear(self.fea_size, 3, bias=False))
        self.grl1 = GradientReversalLayer()

        self.domain_classifier2 = nn.Sequential(nn.Linear(self.fea_size*2, self.fea_size),
                                                nn.LeakyReLU(0.1, inplace=True),
                                                nn.Linear(self.fea_size, 3, bias=False))
        self.grl2 = GradientReversalLayer()
        
    def extraction(self, x):
        f1, f2 = self.net_lat_omics1(x), self.net_lat_omics2(x)
        return f1, f2
    
    def prediction(self, x1, x2):
        st1_omics1, st1_omics2 = self.final_predict1(x1), self.final_predict2(x2)
        return st1_omics1, st1_omics2

    def forward(self, img):

        ########################### stage 1: Encode
        b = img.shape[0]
        img = self.noise_dropout(img)

        f = torch.cat([self.net_lat2(img[:, 0:self.input_dimension//2]), self.net_lat1(img[:, self.input_dimension//2:self.input_dimension])], dim=1)

        f1, f2 = self.extraction(f)

        # breakpoint()
        ############################
        # Domain Classifier 1
        # f_reverse1 = self.grl1(f.view(b, -1), lambda_=1.0)  # Apply gradient reversal
        # y_domain1 = self.domain_classifier1(f_reverse1)
        ############################

        f = torch.stack([f1, f2], dim=1)

        omics12 = self.omics12.expand(b, -1, -1)
        f_ = self.trans_cross1(f, omics12) + f
        f_ = self.trans_omics1(f_) + f

        f_ = self.trans_cross2(f_, omics12) + f
        f_ = self.trans_omics2(f_) + f

        ############################
        # Domain Classifier 2
        # f_reverse2 = self.grl2(f_.view(b, -1), lambda_=1.0)  # Apply gradient reversal
        # y_domain2 = self.domain_classifier2(f_reverse2)
        ############################

        f_overal = self.recalibrate_layers(f_)
        f1, f2 = torch.chunk(f_overal, 2, dim=1)
        f1, f2 = self.dropout(f1).squeeze(), self.dropout(f2).squeeze()

        omics1, omics2 = self.omics12[:, :self.omics1_size, :], self.omics12[:, self.omics1_size: , :]
        st2_omics1, st2_omics2 = f1@omics1.squeeze().T, f2@omics2.squeeze().T

        return st2_omics1, st2_omics2, 


class Network_single_omics(nn.Module):
    def __init__(self, omics_size, input_dimension, noise_rate=0.2, dropout_rate=0.1):
        super(Network_single_omics, self).__init__()

        self.input_dimension = input_dimension
        self.fea_size = 256
        self.num_heads = 4
        self.omics_size = omics_size

        self.noise_dropout = nn.Dropout(noise_rate)

        # Encoder initialization
        self.net_lat1 = nn.Sequential(MLP(input_dimension//2, 512), MLP(512, 512))
        self.net_lat2 = nn.Sequential(MLP(input_dimension//2, 512), MLP(512, 256))

        self.net_lat_omics1 = MLP(256+512, self.fea_size)

        ########################
        # Token initialization
        self.omics = nn.Parameter(torch.randn((1, self.omics_size, self.fea_size), requires_grad=True))
        trunc_normal_(self.omics, mean=0.0, std=.02)

        ########################
        # Transformers
        self.trans_omics1 = Transformer(fea_size=self.fea_size, num_heads=self.num_heads, dropout_rate=dropout_rate)
        self.trans_omics2 = Transformer(fea_size=self.fea_size, num_heads=self.num_heads, dropout_rate=dropout_rate)

        self.trans_cross1 = Transformer_Cross(fea_size=self.fea_size, num_heads=self.num_heads, dropout_rate=dropout_rate)
        self.trans_cross2 = Transformer_Cross(fea_size=self.fea_size, num_heads=self.num_heads, dropout_rate=dropout_rate)
        
        ########################
        # Others initialization
        self.dropout = nn.Dropout(dropout_rate)

        self.recalibrate_layers = nn.Sequential(nn.Linear(self.fea_size, self.fea_size), 
                                                # nn.BatchNorm1d(3),
                                                nn.LeakyReLU(0.1, inplace=True))

        #########################
        # Domain classifier initialization
        self.domain_classifier1 = nn.Sequential(nn.Linear(self.fea_size*3, self.fea_size), 
                                                nn.LeakyReLU(0.1, inplace=True), 
                                                nn.Linear(self.fea_size, 3, bias=False))
        self.grl1 = GradientReversalLayer()

        self.domain_classifier2 = nn.Sequential(nn.Linear(self.fea_size*2, self.fea_size),
                                                nn.LeakyReLU(0.1, inplace=True),
                                                nn.Linear(self.fea_size, 3, bias=False))
        self.grl2 = GradientReversalLayer()
        
    def extraction(self, x):
        f1 = self.net_lat_omics1(x)
        return f1
    

    def forward(self, img):

        ########################### stage 1: Encode
        b = img.shape[0]
        img = self.noise_dropout(img)
        f = torch.cat([self.net_lat2(img[:, 0:self.input_dimension//2]), self.net_lat1(img[:, self.input_dimension//2:self.input_dimension])], dim=1)

        f = self.extraction(f)[:, None, :]

        omics = self.omics.expand(b, -1, -1)
        f_ = self.trans_cross1(f, omics) + f
        f_ = self.trans_omics1(f_) + f

        f_ = self.trans_cross2(f_, omics) + f
        f_ = self.trans_omics2(f_) + f


        f_overal = self.recalibrate_layers(f_)
        f1 = self.dropout(f_overal).squeeze()

        st2_omics = f1@self.omics.squeeze().T

        return st2_omics