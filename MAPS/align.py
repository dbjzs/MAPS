import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
from scipy.spatial import cKDTree 
from MAPS.loss import chamfer_distance_torch
from MAPS.loss import chamfer_distance_torch_bidirectional
from MAPS.utils import set_seed



def Rigid_alignment(source, target, epochs=2000, sample_size=20000, lr_rot=0.01, lr_trans=0.5, lr_scale=0.0001, enable_scale=False, device=None):
    """
    Align a source slice to a target slice using rigid (or similarity) transformation.
    The alignment minimizes the Chamfer distance between the transformed source points and the target points via gradient‑based optimization (Adam).  
    A coarse angle scan is performed before optimization to initialize the rotation.

    Parameters
    ----------
    source : np.ndarray
        Source slice spatial coordinates, shape ``(Nsource, 2)``.  
        Each row contains ``(x, y)`` coordinates.
    target : np.ndarray
        Target slice spatial coordinates, shape ``(Ntarget, 2)``.  
        Each row contains ``(x, y)`` coordinates.
    epochs : int, optional (default: 2000)
        Number of optimization iterations.
    sample_size : int, optional (default: 20000)
        Number of cells randomly sampled from each slice per iteration.
        If the slice has fewer cells, all points are used.
    lr_rot : float, optional (default: 0.01)
        Learning rate for the rotation angle (in radians).
    lr_trans : float, optional (default: 0.5)
        Learning rate for the translation vector ``(dx, dy)``.
    lr_scale : float, optional (default: 0.0001)
        Learning rate for the scaling factor (used only when ``enable_scale=True``).
    enable_scale : bool, optional (default: False)
        If ``True``, allows isotropic scaling.  
        Otherwise, only rotation and translation are applied.
    device : str or torch.device, optional (default: None)
        Computation device, e.g. ``"cuda"``, ``"cpu"``.  
        If ``None``, the default PyTorch device is used.

    Returns
    -------
    aligned : np.ndarray
        Aligned source slice spatial coordinates, same shape as ``source``.
    rotation_deg : float
        Estimated rotation angle in degrees.  
        The value is normalized to the range ``(-180, 180]``.
    translation : np.ndarray
        Estimated translation vector, shape ``(2,)``.
    scale : float
        Estimated scaling factor (``1.0`` if ``enable_scale=False``).

    Notes
    -----
    The transformation is applied in the following order:

    1. Center the source slice by subtracting its centroid.
    2. Apply isotropic scaling (if enabled).
    3. Apply rotation.
    4. Translate back to the original centroid plus the estimated translation.
    """
    set_seed(7)
    source_t = torch.tensor(source, dtype=torch.float32, device=device).unsqueeze(0)
    target_t = torch.tensor(target, dtype=torch.float32, device=device).unsqueeze(0)
    if torch.cuda.is_available() and (device is None or torch.device(device).type == 'cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    
    s_center = source_t.mean(dim=1, keepdim=True)
    t_center = target_t.mean(dim=1, keepdim=True)
    s_centered = source_t - s_center
    
    mean_dist_s = torch.norm(s_centered, dim=2).mean()
    mean_dist_t = torch.norm(target_t - t_center, dim=2).mean()
    init_scale_val = (mean_dist_t / mean_dist_s).item() if mean_dist_s > 0 else 1.0

    candidate_angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    best_init_theta = 0.0
    min_coarse_loss = float('inf')
    
    print("Scanning angles to find best starting angle...")
    with torch.no_grad():
        idx_s_c = torch.randperm(s_centered.shape[1], device=device)[:sample_size]
        idx_t_c = torch.randperm(target_t.shape[1], device=device)[:sample_size]
        
        for ang in candidate_angles:
            ang_t = torch.tensor(ang, dtype=torch.float32, device=device)
            cos_a, sin_a = torch.cos(ang_t), torch.sin(ang_t)
            R_c = torch.stack([torch.stack([cos_a, -sin_a]), torch.stack([sin_a, cos_a])])
    
            test_pos = init_scale_val * torch.matmul(s_centered[:, idx_s_c, :], R_c.T) + t_center
            loss_c = chamfer_distance_torch(test_pos, target_t[:, idx_t_c, :])
            if loss_c < min_coarse_loss:
                min_coarse_loss = loss_c
                best_init_theta = ang

    theta = torch.tensor([best_init_theta], dtype=torch.float32, device=device, requires_grad=True)
    translation = (t_center - s_center).clone().detach().requires_grad_(True)
    
    if enable_scale:
        scale = torch.tensor([init_scale_val], dtype=torch.float32, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([{'params': [theta], 'lr': lr_rot},{'params': [translation], 'lr': lr_trans},{'params': [scale], 'lr': lr_scale}], betas=(0.9, 0.999))
    else:
        scale = torch.tensor([1.0], dtype=torch.float32, device=device)  
        optimizer = torch.optim.Adam([{'params': [theta], 'lr': lr_rot},{'params': [translation], 'lr': lr_trans}], betas=(0.9, 0.999))

    pbar = tqdm(range(epochs), desc="Global Align", ncols=150)

    for epoch in pbar:
        optimizer.zero_grad()
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        R = torch.stack([torch.stack([cos_t[0], -sin_t[0]]), torch.stack([sin_t[0], cos_t[0]])])
        
        idx_s = torch.randperm(s_centered.shape[1], device=device)[:sample_size]
        idx_t = torch.randperm(target_t.shape[1], device=device)[:sample_size]
        
        if enable_scale:
            transformed = scale * torch.matmul(s_centered[:, idx_s, :], R.T) + s_center + translation
        else:
            transformed = torch.matmul(s_centered[:, idx_s, :], R.T) + s_center + translation
        loss = chamfer_distance_torch(transformed, target_t[:, idx_t, :])
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            curr_theta_deg = theta.item() * 180 / np.pi
            curr_theta_norm = curr_theta_deg % 360
            curr_trans = translation.squeeze().detach().cpu().numpy()
            curr_scale = scale.item() if enable_scale else 1.0
            pbar.set_postfix({"Loss": f"{loss.item():.2f}", 
                "Scale": f"{curr_scale:.3f}", 
                "Rot": f"{curr_theta_norm-360 if curr_theta_norm > 180 else curr_theta_norm:.2f}°", 
                "Trans": f"({curr_trans[0]:.1f}, {curr_trans[1]:.1f})"
            })

    with torch.no_grad():
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        final_R = torch.stack([torch.stack([cos_t[0], -sin_t[0]]), torch.stack([sin_t[0], cos_t[0]])])
        
        if enable_scale:
            final_coords = scale * torch.matmul(s_centered, final_R.T) + s_center + translation
        else:
            final_coords = torch.matmul(s_centered, final_R.T) + s_center + translation

    aligned_np = final_coords.squeeze(0).cpu().numpy()
    theta_deg = theta.item() * 180 / np.pi
    translation_np = translation.squeeze().detach().cpu().numpy()
    final_scale_np = scale.item() if enable_scale else 1.0
    
    GPU_peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    
    print(f"Peak GPU memory usage: {GPU_peak_memory:.3f} GB")
    print(f"Scale factor: {final_scale_np:.3f}, Rotation angle: {theta_deg - 360 if theta_deg > 180 else theta_deg:.2f}°, Translation (x, y): ({translation_np[0]:.2f}, {translation_np[1]:.2f})")
    
    return aligned_np, theta_deg, translation_np, final_scale_np










def partial_alignment(source, target, epochs=2000, sample_size=20000, lr_rot=0.01, lr_trans=0.5, device=None):
    """
    Partially align a source slice to a target slice using region‑of‑interest (ROI) guided rigid transformation.

    The function first extracts dense regions (ROIs) from the target cloud by sliding a window of size proportional to the source bounding box. 
    For each ROI, a coarse rotation search is performed on a subsampled source, and the best‑matching ROI (minimizing Chamfer distance) is selected to initialize the translation and rotation.  
    Fine‑tuning is then carried out via Adam optimization of the rigid transformation (rotation + translation).

    Parameters
    ----------
    source : np.ndarray
        Source slice spatial coordinates, shape ``(Nsource, 2)``.  
        Each row contains ``(x, y)`` coordinates.
    target : np.ndarray
        Target slice spatial coordinates, shape ``(Ntarget, 2)``.  
        Each row contains ``(x, y)`` coordinates.
    epochs : int, optional (default: 2000)
        Number of optimization iterations for fine‑tuning.
    sample_size : int, optional (default: 20000)
        Number of cells randomly sampled from each slice per iteration during fine‑tuning.
    lr_rot : float, optional (default: 0.01)
        Learning rate for the rotation angle (in radians).
    lr_trans : float, optional (default: 0.5)
        Learning rate for the translation vector ``(dx, dy)``.
    device : str or torch.device, optional (default: None)
        Computation device, e.g. ``"cuda"``, ``"cpu"``.  
        If ``None``, the default PyTorch device is used.

    Returns
    -------
    aligned : np.ndarray
        Aligned source slice, same shape as ``source``.
    rotation_deg : float
        Estimated rotation angle in degrees.  
        The value is normalized to the range ``(-180, 180]``.
    translation : np.ndarray
        Estimated translation vector, shape ``(2,)``.

    Notes
    -----
    The transformation is applied in the following order:

    1. Center the source cloud by subtracting its centroid.
    2. Apply rotation.
    3. Translate back to the original centroid plus the estimated translation.
    """
    set_seed(7)
    source_t = torch.tensor(source, dtype=torch.float32, device=device).unsqueeze(0)
    target_t = torch.tensor(target, dtype=torch.float32, device=device).unsqueeze(0)
    
    s_center = source_t.mean(dim=1, keepdim=True)
    s_centered = source_t - s_center
    
    # ==========================================
    # ROI feature extraction
    # ==========================================
    target_pts = target_t.squeeze(0) # [N, 2]
    
    s_min = source_t.min(dim=1)[0].squeeze()
    s_max = source_t.max(dim=1)[0].squeeze()
    roi_size = torch.max(s_max - s_min).item() * 2 
    
    t_min = target_t.min(dim=1)[0].squeeze()
    t_max = target_t.max(dim=1)[0].squeeze()
    step_size = roi_size * 0.03 # 3% overleap
    
    x_coords = torch.arange(t_min[0].item(), t_max[0].item() + step_size, step_size, device=device)
    y_coords = torch.arange(t_min[1].item(), t_max[1].item() + step_size, step_size, device=device)
    
    valid_centroids = []
    valid_rois_padded = []     
    all_roi_pts_raw = []       
    roi_sample_size = 500     
    
    for x in x_coords:
        for y in y_coords:
            mask_x = (target_pts[:, 0] >= x - roi_size/2) & (target_pts[:, 0] <= x + roi_size/2)
            mask_y = (target_pts[:, 1] >= y - roi_size/2) & (target_pts[:, 1] <= y + roi_size/2)
            roi_pts = target_pts[mask_x & mask_y]
            
            if len(roi_pts) > 50:
                centroid = roi_pts.mean(dim=0)
                valid_centroids.append(centroid)
                all_roi_pts_raw.append(roi_pts) # 
                
                if len(roi_pts) >= roi_sample_size:
                    idx = torch.randperm(len(roi_pts), device=device)[:roi_sample_size]
                    sampled_roi = roi_pts[idx]
                else:
                    idx = torch.randint(0, len(roi_pts), (roi_sample_size,), device=device)
                    sampled_roi = roi_pts[idx]
                valid_rois_padded.append(sampled_roi)
                
    if len(valid_centroids) == 0:
        print("Warning: No valid ROIs found, falling back to target center.")
        valid_centroids.append(target_pts.mean(dim=0))
        valid_rois_padded.append(target_pts[:roi_sample_size] if len(target_pts) > roi_sample_size else target_pts)
        all_roi_pts_raw.append(target_pts)
        
    candidate_centers = torch.stack(valid_centroids).unsqueeze(0) 
    candidate_rois = torch.stack(valid_rois_padded)               
    num_trans_candidates = candidate_centers.shape[1]
    
    # ==========================================
    #  Coarse search
    # ==========================================
    num_angles = 180
    candidate_angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    batch_size = 200 
    
    best_init_theta = 0.0
    best_init_trans = torch.zeros((1, 1, 2), device=device)
    min_coarse_loss = float('inf')
    best_roi_idx = 0 
    
    print(f"ROI Extracted: {num_trans_candidates} valid physical centroids.")
    
    with torch.no_grad():
        idx_s_c = torch.randperm(s_centered.shape[1], device=device)[:2000]
        s_c_sampled = s_centered[:, idx_s_c, :]
        
        for ang in candidate_angles:
            ang_t = torch.tensor(ang, dtype=torch.float32, device=device)
            cos_a, sin_a = torch.cos(ang_t), torch.sin(ang_t)
            R_c = torch.stack([torch.stack([cos_a, -sin_a]), torch.stack([sin_a, cos_a])])
            
            rotated_s = torch.matmul(s_c_sampled, R_c.T)
            
            for i in range(0, num_trans_candidates, batch_size):
                t_batch = candidate_centers[0, i:i+batch_size, :] 
                roi_batch = candidate_rois[i:i+batch_size, :, :]  
                
                B = t_batch.shape[0]
                test_pos = rotated_s.expand(B, -1, -1) + t_batch.unsqueeze(1) 

                dist_matrix = torch.cdist(test_pos, roi_batch, p=2)**2
                dist_src_to_tgt = torch.min(dist_matrix, dim=2)[0]
                loss_batch = torch.mean(dist_src_to_tgt, dim=1)
                
                min_loss_idx = torch.argmin(loss_batch)
                if loss_batch[min_loss_idx] < min_coarse_loss:
                    min_coarse_loss = loss_batch[min_loss_idx]
                    best_init_theta = ang
                    best_init_trans = t_batch[min_loss_idx].view(1, 1, 2) - s_center
                    best_roi_idx = i + min_loss_idx.item() 
                    
    print(f"Coarse init done! Locked on ROI #{best_roi_idx}")

    theta = torch.tensor([best_init_theta], dtype=torch.float32, device=device, requires_grad=True)
    translation = best_init_trans.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([{'params': [theta], 'lr': lr_rot},{'params': [translation], 'lr': lr_trans} ], betas=(0.9, 0.999))

    pbar = tqdm(range(epochs), desc="Align",ncols=150)
    for epoch in pbar:
        optimizer.zero_grad()
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        R = torch.stack([torch.stack([cos_t[0], -sin_t[0]]), torch.stack([sin_t[0], cos_t[0]])])
        idx_s = torch.randperm(s_centered.shape[1], device=device)[:sample_size]
        idx_t = torch.randperm(target_t.shape[1], device=device)[:sample_size]
        transformed = torch.matmul(s_centered[:, idx_s, :], R.T) + s_center + translation
        loss = chamfer_distance_torch(transformed, target_t[:, idx_t, :])
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            curr_theta_deg = theta.item() * 180 / np.pi
            curr_theta_norm = curr_theta_deg % 360
            curr_trans = translation.squeeze().detach().cpu().numpy()
            pbar.set_postfix({"Loss": f"{loss.item():.2f}", "Rot": f"{curr_theta_norm-360:.2f}°", "Trans": f"({curr_trans[0]:.1f}, {curr_trans[1]:.1f})"})


    with torch.no_grad():
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        final_R = torch.stack([torch.stack([cos_t[0], -sin_t[0]]), torch.stack([sin_t[0], cos_t[0]])])
        final_coords = torch.matmul(s_centered, final_R.T) + s_center + translation

    aligned_np=final_coords.squeeze(0).cpu().numpy()
    theta_deg=theta.item() * 180 / np.pi
    translation_np=translation.squeeze().detach().cpu().numpy()
    
    print(f"Rotation angle: {theta_deg - 360:.2f}")
    print(f"Translation (x, y): {translation_np[0]:.2f}, {translation_np[1]:.2f}")
    return aligned_np,theta_deg,translation_np




def transform_full_source(source_np, roi, theta_deg, translation_np, scale=1.0):
    """
    Apply a similarity transformation (scaling + rotation + translation) to the full source slice.

    The transformation is defined relative to an anchor point (the centroid of a reference ROI) 
    and consists of scaling and rotation followed by a translation.  
    This function is typically used after :func:`partial_alignment` to align the entire source slice 
    using the transformation estimated from a partially overlapping region (ROI).

    Parameters
    ----------
    source_np : np.ndarray
        Full source slice, shape ``(Nsource, 2)``.  
        Each row contains ``(x, y)`` coordinates.
    roi : np.ndarray
        Reference region of interest (ROI) from the source or target, shape ``(Nroi, 2)``.
        Its centroid is used as the rotation/scaling anchor.  
        Usually this is the ROI in the source that matched a target ROI during partial alignment.
    theta_deg : float
        Rotation angle in degrees. Positive values indicate counter‑clockwise rotation.
    translation_np : np.ndarray
        Translation vector, shape ``(2,)``, to be applied after rotation and scaling.
    scale : float, optional
        Scale factor to be applied to the centered coordinates. Default is 1.0.

    Returns
    -------
    aligned_full_source : np.ndarray
        Transformed source slice, same shape as ``source_np``.

    Notes
    -----
    The transformation is applied in the following order:

    1. Center the source cloud by subtracting the anchor point (centroid of ``roi``).
    2. Scale and rotate the centered points by ``scale`` and angle ``theta_deg``.
    3. Add back the anchor point.
    4. Apply the translation vector.
    """
    anchor_center = np.mean(roi, axis=0)
    theta_rad = theta_deg * np.pi / 180.0
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    R = np.array([[cos_t, -sin_t], [sin_t,  cos_t]])
    full_centered = source_np - anchor_center
    aligned_full_source = scale * np.dot(full_centered, R.T) + anchor_center + translation_np
    
    return aligned_full_source








def Multi_slices_rigid_alignment(adata_list, mode='reference', ref_idx=0, epochs=2000,sample_size=20000,enable_scale=False,device=None):
    """
    Align spatial coordinates of slices either to a fixed reference or sequentially.

    Parameters
    ----------
    adata_list : list of AnnData
        List of AnnData objects, each must contain .obsm['spatial'] coordinates.
    mode : str, optional
        'reference' : all slices are aligned to a fixed reference slice (specified by ref_idx).
        'sequential' : sequential alignment, slice i is aligned to slice i-1 (first slice fixed).
    ref_idx : int, optional
        Index of the reference slice, only used when mode='reference'.
    epochs : int, optional (default: 2000)
        Number of optimization iterations.
    sample_size : int, optional (default: 20000)
        Number of cells randomly sampled from each slice per iteration.
        If the slice has fewer cells, all points are used.
    enable_scale : bool, optional (default: False)
        If ``True``, allows isotropic scaling.  
        Otherwise, only rotation and translation are applied.
    device : str or torch.device, optional (default: None)
        Computation device, e.g. ``"cuda"``, ``"cpu"``.  
        If ``None``, the default PyTorch device is used.

    Returns
    -------
    aligned_list : list of AnnData
        A new list of AnnData objects after alignment (original unchanged).
    """
    # Create a deep copy of each AnnData object
    # Using .copy() ensures independent objects (spatial coordinates are also copied)
    aligned_list = [adata.copy() for adata in adata_list]

    if mode == 'reference':
        target = aligned_list[ref_idx].obsm['spatial'].astype(np.float32)
        for i in range(len(aligned_list)):
            if i == ref_idx:
                print(f"\n--- Slice {i+1} is the reference, skipping ---")
                continue
            print(f"\n--- Aligning Slice {i+1} to Reference (Slice {ref_idx+1}) ---")
            source = aligned_list[i].obsm['spatial'].astype(np.float32)
            aligned_spatial, theta_deg, translation, scale = Rigid_alignment(
                source, target, epochs=epochs, sample_size=sample_size,
                enable_scale=enable_scale, device=device
            )
            aligned_list[i].obsm['spatial'] = aligned_spatial
            

    elif mode == 'sequential':
        print(f"\n--- Slice 1 is the first slice ---")
        for i in range(1, len(aligned_list)):
            print(f"\n--- Aligning Slice {i+1} to Slice {i} (sequential alignment) ---")
            source = aligned_list[i].obsm['spatial'].astype(np.float32)
            target = aligned_list[i-1].obsm['spatial'].astype(np.float32)
            aligned_spatial, theta_deg, translation, scale = Rigid_alignment(
                source, target, epochs=epochs, sample_size=sample_size,
                enable_scale=enable_scale, device=device
            )
            aligned_list[i].obsm['spatial'] = aligned_spatial

    else:
        raise ValueError("mode must be 'reference' or 'sequential'")

    print("\n--- All slices have been aligned! ---")
    return aligned_list







def Rigid_alignment_atlas(source, target, epochs=2000, sample_size=20000, lr_rot=0.01, lr_trans=0.5, lr_scale=0.0001, enable_scale=False, device=None):
    set_seed(7)
    source_t = torch.tensor(source, dtype=torch.float32, device=device).unsqueeze(0)
    target_t = torch.tensor(target, dtype=torch.float32, device=device).unsqueeze(0)
    if torch.cuda.is_available() and (device is None or torch.device(device).type == 'cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    
    s_center = source_t.mean(dim=1, keepdim=True)
    t_center = target_t.mean(dim=1, keepdim=True)
    s_centered = source_t - s_center
    
    mean_dist_s = torch.norm(s_centered, dim=2).mean()
    mean_dist_t = torch.norm(target_t - t_center, dim=2).mean()
    init_scale_val = (mean_dist_t / mean_dist_s).item() if mean_dist_s > 0 else 1.0

    candidate_angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    best_init_theta = 0.0
    min_coarse_loss = float('inf')
    
    with torch.no_grad():
        idx_s_c = torch.randperm(s_centered.shape[1], device=device)[:sample_size]
        idx_t_c = torch.randperm(target_t.shape[1], device=device)[:sample_size]
        
        for ang in candidate_angles:
            ang_t = torch.tensor(ang, dtype=torch.float32, device=device)
            cos_a, sin_a = torch.cos(ang_t), torch.sin(ang_t)
            R_c = torch.stack([torch.stack([cos_a, -sin_a]), torch.stack([sin_a, cos_a])])
    
            test_pos = init_scale_val * torch.matmul(s_centered[:, idx_s_c, :], R_c.T) + t_center
            loss_c = chamfer_distance_torch_bidirectional(test_pos, target_t[:, idx_t_c, :])
            if loss_c < min_coarse_loss:
                min_coarse_loss = loss_c
                best_init_theta = ang

    theta = torch.tensor([best_init_theta], dtype=torch.float32, device=device, requires_grad=True)
    translation = (t_center - s_center).clone().detach().requires_grad_(True)
    
    if enable_scale:
        scale = torch.tensor([init_scale_val], dtype=torch.float32, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([{'params': [theta], 'lr': lr_rot},{'params': [translation], 'lr': lr_trans},{'params': [scale], 'lr': lr_scale}], betas=(0.9, 0.999))
    else:
        scale = torch.tensor([1.0], dtype=torch.float32, device=device)  
        optimizer = torch.optim.Adam([{'params': [theta], 'lr': lr_rot},{'params': [translation], 'lr': lr_trans}], betas=(0.9, 0.999))

    pbar = tqdm(range(epochs), desc="Global Align", ncols=150)

    for epoch in pbar:
        optimizer.zero_grad()
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        R = torch.stack([torch.stack([cos_t[0], -sin_t[0]]), torch.stack([sin_t[0], cos_t[0]])])
        
        idx_s = torch.randperm(s_centered.shape[1], device=device)[:sample_size]
        idx_t = torch.randperm(target_t.shape[1], device=device)[:sample_size]
        
        if enable_scale:
            transformed = scale * torch.matmul(s_centered[:, idx_s, :], R.T) + s_center + translation
        else:
            transformed = torch.matmul(s_centered[:, idx_s, :], R.T) + s_center + translation
        loss = chamfer_distance_torch_bidirectional(transformed, target_t[:, idx_t, :])

        final_loss = loss.item()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            curr_theta_deg = theta.item() * 180 / np.pi
            curr_theta_norm = curr_theta_deg % 360
            curr_trans = translation.squeeze().detach().cpu().numpy()
            curr_scale = scale.item() if enable_scale else 1.0
            pbar.set_postfix({"Loss": f"{loss.item():.2f}", 
                "Scale": f"{curr_scale:.3f}", 
                "Rot": f"{curr_theta_norm-360 if curr_theta_norm > 180 else curr_theta_norm:.2f}°", 
                "Trans": f"({curr_trans[0]:.1f}, {curr_trans[1]:.1f})"
            })

    with torch.no_grad():
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        final_R = torch.stack([torch.stack([cos_t[0], -sin_t[0]]), torch.stack([sin_t[0], cos_t[0]])])
        
        if enable_scale:
            final_coords = scale * torch.matmul(s_centered, final_R.T) + s_center + translation
        else:
            final_coords = torch.matmul(s_centered, final_R.T) + s_center + translation

    aligned_np = final_coords.squeeze(0).cpu().numpy()
    theta_deg = theta.item() * 180 / np.pi
    translation_np = translation.squeeze().detach().cpu().numpy()
    final_scale_np = scale.item() if enable_scale else 1.0
    
    GPU_peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return aligned_np, theta_deg, translation_np, final_scale_np, final_loss



def Align_slice_to_atlas(source_input, atlas_adata_list, top_k=3, epochs=2000, sample_size=20000, enable_scale=False, device=None):
    set_seed(7)
    
    if hasattr(source_input, 'obsm'):
        is_adata = True
        source_raw = source_input.obsm['spatial'].astype(np.float32)
    else:
        is_adata = False
        source_raw = source_input.astype(np.float32)
        
    s_min = source_raw.min(axis=0)
    s_zeroed = source_raw - s_min
    s_max_span = s_zeroed.max()  
    source = (s_zeroed / s_max_span) * 500
    
    source_t = torch.tensor(source, dtype=torch.float32, device=device).unsqueeze(0)
    s_center = source_t.mean(dim=1, keepdim=True)
    s_centered = source_t - s_center
    mean_dist_s = torch.norm(s_centered, dim=2).mean()
    
    coarse_losses = []
    candidate_angles = np.linspace(0, 2 * np.pi, 180, endpoint=False) 
    
    print(">>> Atlas Scanning...")
    for idx, atlas_adata in enumerate(tqdm(atlas_adata_list, desc="Atlas Scanning", ncols=150)):
        target_raw = atlas_adata.obsm['spatial'].astype(np.float32)
        
        t_min = target_raw.min(axis=0)
        t_zeroed = target_raw - t_min
        t_max_span = t_zeroed.max()
        target_scaled = (t_zeroed / t_max_span) * 1000
        
        target_t = torch.tensor(target_scaled, dtype=torch.float32, device=device).unsqueeze(0)
        t_center = target_t.mean(dim=1, keepdim=True)
        mean_dist_t = torch.norm(target_t - t_center, dim=2).mean()
        init_scale_val = (mean_dist_t / mean_dist_s).item() if (mean_dist_s > 0 and enable_scale) else 1.0
        
        with torch.no_grad():
            actual_s_size = min(s_centered.shape[1], sample_size)
            actual_t_size = min(target_t.shape[1], sample_size)
            idx_s_c = torch.randperm(s_centered.shape[1], device=device)[:actual_s_size]
            idx_t_c = torch.randperm(target_t.shape[1], device=device)[:actual_t_size]
            
            min_c_loss = float('inf')
            for ang in candidate_angles:
                ang_t = torch.tensor(ang, dtype=torch.float32, device=device)
                cos_a, sin_a = torch.cos(ang_t), torch.sin(ang_t)
                R_c = torch.stack([torch.stack([cos_a, -sin_a]), torch.stack([sin_a, cos_a])])
                
                test_pos = init_scale_val * torch.matmul(s_centered[:, idx_s_c, :], R_c.T) + t_center
                loss_c = chamfer_distance_torch_bidirectional(test_pos, target_t[:, idx_t_c, :])
                
                norm_loss_c = loss_c.item() / actual_t_size 
                if norm_loss_c < min_c_loss:
                    min_c_loss = norm_loss_c
                    
            coarse_losses.append((idx, min_c_loss))
            
    coarse_losses.sort(key=lambda x: x[1])
    top_candidates = coarse_losses[:top_k]
    print(f"Coarse screening complete！The most likely slices is: {[x[0] for x in top_candidates]} in adata_list")
    
    print(f"\n>>> Top {top_k} candidate slices...")
    best_atlas_idx = None
    best_atlas_label = None
    best_final_loss = float('inf')
    best_alignment_results = None
    
    best_t_min = None
    best_t_max_span = None
    
    for rank, (atlas_idx, _) in enumerate(top_candidates):
        target_adata = atlas_adata_list[atlas_idx]
        if 'batch' in target_adata.obs.columns:
            batch_label = target_adata.obs['batch'].iloc[0]
        else:
            batch_label = f"Slice_{atlas_idx}"
            
        print(f"\nEvaluating candidates {rank+1}/{top_k} (Atlas batch: {batch_label})")
        
        target_raw = target_adata.obsm['spatial'].astype(np.float32)
        t_min = target_raw.min(axis=0)
        t_zeroed = target_raw - t_min
        t_max_span = t_zeroed.max()
        target_scaled = (t_zeroed / t_max_span) * 1000

        aligned_spatial, theta_deg, translation, scale, current_loss = Rigid_alignment_atlas(
            source, target_scaled, epochs=epochs, sample_size=sample_size,
            enable_scale=enable_scale, device=device
        )
        
        actual_t_size = min(target_raw.shape[0], sample_size)
        norm_current_loss = current_loss / actual_t_size
        
        if norm_current_loss < best_final_loss:
            best_final_loss = norm_current_loss
            best_atlas_idx = atlas_idx
            best_atlas_label = batch_label
            best_alignment_results = (aligned_spatial, theta_deg, translation, scale)
            best_t_min = t_min
            best_t_max_span = t_max_span
            
    print(f"\n Match done！The best match is adata_list[{best_atlas_idx}] slice in the atlas (Label: {best_atlas_label})")

    aligned_spatial_canvas = best_alignment_results[0] 
    aligned_roi_np = ((aligned_spatial_canvas / 1000.0) * best_t_max_span + best_t_min).astype(np.float32)
    
    best_target_raw = atlas_adata_list[best_atlas_idx].obsm['spatial'].astype(np.float32)
    t_center_raw = best_target_raw.mean(axis=0)
    s_center_raw = source_raw.mean(axis=0)
    
    fine_translation_raw = (best_alignment_results[2] / 1000.0) * best_t_max_span
    absolute_translation_np = t_center_raw - s_center_raw + fine_translation_raw
    
    raw_scale = best_alignment_results[3] * 0.5 * (best_t_max_span / s_max_span)
    theta_deg = best_alignment_results[1]

    if is_adata:
        aligned_adata = source_input.copy()
        aligned_adata.obsm['spatial'] = aligned_roi_np
        aligned_adata.uns['atlas_match'] = {'best_atlas_slice_idx': best_atlas_label,'rotation_deg': theta_deg,'translation': absolute_translation_np, 'scale': raw_scale}
        return aligned_adata
    else:
        return best_atlas_label, aligned_roi_np, theta_deg, absolute_translation_np, raw_scale










