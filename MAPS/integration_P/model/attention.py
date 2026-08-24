import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum


def exists(val):
    return val is not None


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim=512, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Cross_Attention(nn.Module):
    def __init__(self, query_dim=512, context_dim=512, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Sequential(nn.Linear(query_dim, inner_dim, bias=False))
        self.to_k = nn.Sequential(nn.Linear(context_dim, inner_dim, bias=False))
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Linear(inner_dim, inner_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, context):
        h = self.heads

        q = self.to_q(x1)
        k, v = self.to_k(context), self.to_v(context)
       
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        attn_ori = sim.softmax(dim=-1)
        attn = self.dropout(attn_ori)
 

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)
    


class Self_Attention(nn.Module):
    def __init__(self, query_dim=512, context_dim=512, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Sequential(nn.Linear(query_dim, inner_dim, bias=False))
        self.to_k = nn.Sequential(nn.Linear(context_dim, inner_dim, bias=False))
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, inner_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, matrix=None):
        h = self.heads

        q = self.to_q(x1)
        k, v = self.to_k(x1), self.to_v(x1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if matrix is not None:
            # attn = (sim.softmax(dim=-1) +  matrix[None, :, :].softmax(dim=-1))/2
            attn = sim + matrix[None, :, :]
            attn = attn.softmax(dim=-1)
        else:
            attn = sim.softmax(dim=-1)

        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)