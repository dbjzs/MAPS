import torch

def chamfer_distance_torch(x, y):
    if x.dim() == 2:
        x = x.unsqueeze(0)  # [N, D] -> [1, N, D]
    if y.dim() == 2:
        y = y.unsqueeze(0)
    
    dist_matrix = torch.cdist(x, y, p=2)           # [B, N_x, N_y]
    dist_src_to_tgt = torch.min(dist_matrix, dim=2)[0]  # [B, N_x]
    return torch.mean(dist_src_to_tgt)



def chamfer_distance_torch_bidirectional(x, y):
    if x.dim() == 2:
        x = x.unsqueeze(0)  # (1, N, D)
    if y.dim() == 2:
        y = y.unsqueeze(0)  # (1, M, D)

    dist_matrix = torch.cdist(x, y, p=2)

    dist_src_to_tgt = torch.min(dist_matrix, dim=2)[0]   # (B, N)
    loss_src_to_tgt = torch.mean(dist_src_to_tgt)
    
    dist_tgt_to_src = torch.min(dist_matrix, dim=1)[0]   # (B, M)
    loss_tgt_to_src = torch.mean(dist_tgt_to_src)

    chamfer_loss = (loss_src_to_tgt + loss_tgt_to_src) / 2.0
    return chamfer_loss