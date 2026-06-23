import pandas as pd
import anndata
import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tifffile
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def rotate_and_translate_spatial(adata, angle_deg, dx=0, dy=0, scale=1.0):
    coords = adata.obsm['spatial']
    theta = np.deg2rad(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),  np.cos(theta)]])
    center = coords.mean(axis=0)
    coords_centered = coords - center
    scaled_and_rotated = (scale * coords_centered) @ R.T 
    translated = scaled_and_rotated + center + np.array([dx, dy])
    adata.obsm['spatial'] = translated 


class Graph:
    def __init__(self, features, neighbor_idx):
        """
        Graph data structure to store coordinates, features, and neighbor indices.

        Args:
            features (torch.Tensor): Features, shape (N, C).
            neighbor_idx (torch.Tensor): Neighbor indices, shape (N, k).
            batches (list, optional): Preprocessed batch data for training. Defaults to None.
        """
        self.features = features
        self.neighbor_idx = neighbor_idx
        
    def get_node(self, node_idx):
        return {
            "features": self.features[node_idx], 
            "neighbor_idx": self.neighbor_idx[node_idx]
        }

def build_neighbor_idx(coords, k):
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    N = coords.shape[0]
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k, workers=-1)  # indices: (N, k)
    return torch.tensor(indices, dtype=torch.long)

def build_cross_slice_idx(coords1, coords2, k):
    if not isinstance(coords1, np.ndarray): coords1 = np.array(coords1)
    if not isinstance(coords2, np.ndarray): coords2 = np.array(coords2)
    tree2 = cKDTree(coords2)
    _, indices = tree2.query(coords1, k=k, workers=-1)
    return torch.tensor(indices, dtype=torch.long)


def prepare_inputs(adata,k, device):
    if "spatial" not in adata.obsm:
        raise KeyError("spatial coordinates not found in adata.obsm['spatial'].")
    if 'feat' not in adata.obsm:
        raise KeyError("Processed feature matrix not found in adata.obsm['feat'].")
    features = adata.obsm['feat']
    if hasattr(features, 'toarray'):
        features = features.toarray()
    coords = adata.obsm['spatial']
    features = torch.tensor(features, dtype=torch.float32)  # (N, C) 
    neighbor_idx = build_neighbor_idx(coords, k)
    
    features = features.to(device)
    neighbor_idx = neighbor_idx.to(device)
    return Graph(features, neighbor_idx)

def prepare_paired_inputs(source_adata, target_adata, k_intra, k_inter, device):
    graph1 = prepare_inputs(source_adata, k_intra, device)
    graph2 = prepare_inputs(target_adata, k_intra, device)
    
    coords1 = source_adata.obsm['spatial']
    coords2 = target_adata.obsm['spatial']
    idx_1_to_2 = build_cross_slice_idx(coords1, coords2, k_inter).to(device)
    
    return graph1, graph2, idx_1_to_2


def Triple_prepare_paired_inputs(source_adata1,source_adata2, target_adata, k_intra, k_inter, device):
    graph1 = prepare_inputs(source_adata1, k_intra, device)
    graph2 = prepare_inputs(source_adata2, k_intra, device)
    graph3 = prepare_inputs(target_adata, k_intra, device)
    
    coords1 = source_adata1.obsm['spatial']
    coords2 = source_adata2.obsm['spatial']
    coords3 = target_adata.obsm['spatial']
    
    idx_1_to_3 = build_cross_slice_idx(coords1, coords3, k_inter).to(device)
    idx_2_to_3 = build_cross_slice_idx(coords2, coords3, k_inter).to(device)
    
    return graph1, graph2,graph3, idx_1_to_3,idx_2_to_3






default_color_dict = {
    "0": "#66C5CC",
    "1": "#F6CF71",
    "2": "#F89C74",
    "3": "#DCB0F2",
    "4": "#87C55F",
    "5": "#9EB9F3",
    "6": "#FE88B1",
    "7": "#C9DB74",
    "8": "#8BE0A4",
    "9": "#B497E7",
    "10": "#D3B484",
    "11": "#B3B3B3",
    "12": "#276A8C", # Royal Blue
    "13": "#DAB6C4", # Pink
    "14": "#C38D9E", # Mauve-Pink
    "15": "#9D88A2", # Mauve
    "16": "#FF4D4D", # Light Red
    "17": "#9B4DCA", # Lavender-Purple
    "18": "#FF9CDA", # Bright Pink
    "19": "#FF69B4", # Hot Pink
    "20": "#FF00FF", # Magenta
    "21": "#DA70D6", # Orchid
    "22": "#BA55D3", # Medium Orchid
    "23": "#8A2BE2", # Blue Violet
    "24": "#9370DB", # Medium Purple
    "25": "#7B68EE", # Medium Slate Blue
    "26": "#4169E1", # Royal Blue
    "27": "#FF8C8C", # Salmon Pink
    "28": "#FFAA80", # Light Coral
    "29": "#48D1CC", # Medium Turquoise
    "30": "#40E0D0", # Turquoise
    "31": "#00FF00", # Lime
    "32": "#7FFF00", # Chartreuse
    "33": "#ADFF2F", # Green Yellow
    "34": "#32CD32", # Lime Green
    "35": "#228B22", # Forest Green
    "36": "#FFD8B8", # Peach
    "37": "#008080", # Teal
    "38": "#20B2AA", # Light Sea Green
    "39": "#00FFFF", # Cyan
    "40": "#00BFFF", # Deep Sky Blue
    "41": "#4169E1", # Royal Blue
    "42": "#0000CD", # Medium Blue
    "43": "#00008B", # Dark Blue
    "44": "#8B008B", # Dark Magenta
    "45": "#FF1493", # Deep Pink
    "46": "#FF4500", # Orange Red
    "47": "#006400", # Dark Green
    "48": "#FF6347", # Tomato
    "49": "#FF7F50", # Coral
    "50": "#CD5C5C", # Indian Red
    "51": "#B22222", # Fire Brick
    "52": "#FFB83F",  # Light Orange
    "53": "#8B0000", # Dark Red
    "54": "#D2691E", # Chocolate
    "55": "#A0522D", # Sienna
    "56": "#800000", # Maroon
    "57": "#808080", # Gray
    "58": "#A9A9A9", # Dark Gray
    "59": "#C0C0C0", # Silver
    "60": "#9DD84A",
    "61": "#F5F5F5", # White Smoke
    "62": "#F17171", # Light Red
    "63": "#000000", # Black
    "64": "#FF8C42", # Tangerine
    "65": "#F9A11F", # Bright Orange-Yellow
    "66": "#FACC15", # Golden Yellow
    "67": "#E2E062", # Pale Lime
    "68": "#BADE92", # Soft Lime
    "69": "#70C1B3", # Greenish-Blue
    "70": "#41B3A3", # Turquoise
    "71": "#5EAAA8", # Gray-Green
    "72": "#72B01D", # Chartreuse
    "73": "#9CD08F", # Light Green
    "74": "#8EBA43", # Olive Green
    "75": "#FAC8C3", # Light Pink
    "76": "#E27D60", # Dark Salmon
    "77": "#C38D9E", # Mauve-Pink
    "78": "#937D64", # Light Brown
    "79": "#B1C1CC", # Light Blue-Gray
    "80": "#88A0A8", # Gray-Blue-Green
    "81": "#4E598C", # Dark Blue-Purple
    "82": "#4B4E6D", # Dark Gray-Blue
    "83": "#8E9AAF", # Light Blue-Grey
    "84": "#C0D6DF", # Pale Blue-Grey
    "85": "#97C1A9", # Blue-Green
    "86": "#4C6E5D", # Dark Green
    "87": "#95B9C7", # Pale Blue-Green
    "88": "#C1D5E0", # Pale Gray-Blue
    "89": "#ECDB54", # Bright Yellow
    "90": "#E89B3B", # Bright Orange
    "91": "#CE5A57", # Deep Red
    "92": "#C3525A", # Dark Red
}

def create_new_color_dict(
        adata,
        cat_key,
        color_palette="default",
        overwrite_color_dict={"-1" : "#E1D9D1"},
        skip_default_colors=0):
    """
    Create a dictionary of color hexcodes for a specified category.

    Parameters
    ----------
    adata:
        AnnData object.
    cat_key:
        Key in ´adata.obs´ where the categories are stored for which color
        hexcodes will be created.
    color_palette:
        Type of color palette.
    overwrite_color_dict:
        Dictionary with overwrite values that will take precedence over the
        automatically created dictionary.
    skip_default_colors:
        Number of colors to skip from the default color dict.

    Returns
    ----------
    new_color_dict:
        The color dictionary with a hexcode for each category.
    """
    new_categories = adata.obs[cat_key].unique().tolist()
    if color_palette == "cell_type_30":
        # https://github.com/scverse/scanpy/blob/master/scanpy/plotting/palettes.py#L40
        new_color_dict = {key: value for key, value in zip(
            new_categories,
            ["#023fa5",
             "#7d87b9",
             "#bec1d4",
             "#d6bcc0",
             "#bb7784",
             "#8e063b",
             "#4a6fe3",
             "#8595e1",
             "#b5bbe3",
             "#e6afb9",
             "#e07b91",
             "#d33f6a",
             "#11c638",
             "#8dd593",
             "#c6dec7",
             "#ead3c6",
             "#f0b98d",
             "#ef9708",
             "#0fcfc0",
             "#9cded6",
             "#d5eae7",
             "#f3e1eb",
             "#f6c4e1",
             "#f79cd4",
             '#7f7f7f',
             "#c7c7c7",
             "#1CE6FF",
             "#336600"])}
    elif color_palette == "cell_type_20":
        # https://github.com/vega/vega/wiki/Scales#scale-range-literals (some adjusted)
        new_color_dict = {key: value for key, value in zip(
            new_categories,
            ['#1f77b4',
             '#ff7f0e',
             '#279e68',
             '#d62728',
             '#aa40fc',
             '#8c564b',
             '#e377c2',
             '#b5bd61',
             '#17becf',
             '#aec7e8',
             '#ffbb78',
             '#98df8a',
             '#ff9896',
             '#c5b0d5',
             '#c49c94',
             '#f7b6d2',
             '#dbdb8d',
             '#9edae5',
             '#ad494a',
             '#8c6d31'])}
    elif color_palette == "cell_type_10":
        # scanpy vega10
        new_color_dict = {key: value for key, value in zip(
            new_categories,
            ['#7f7f7f',
             '#ff7f0e',
             '#279e68',
             '#e377c2',
             '#17becf',
             '#8c564b',
             '#d62728',
             '#1f77b4',
             '#b5bd61',
             '#aa40fc'])}
    elif color_palette == "batch":
        # sns.color_palette("colorblind").as_hex()
        new_color_dict = {key: value for key, value in zip(
            new_categories,
            ['#0173b2', '#d55e00', '#ece133', '#ca9161', '#fbafe4',
             '#949494', '#de8f05', '#029e73', '#cc78bc', '#56b4e9',
             '#F0F8FF', '#FAEBD7', '#00FFFF', '#7FFFD4', '#F0FFFF',
             '#F5F5DC', '#FFE4C4', '#000000', '#FFEBCD', '#0000FF',
             '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0', '#7FFF00',
             '#D2691E', '#FF7F50', '#6495ED', '#FFF8DC', '#DC143C'])}
    elif color_palette == "default":
        new_color_dict = {key: value for key, value in zip(new_categories, list(default_color_dict.values())[skip_default_colors:])}
    for key, val in overwrite_color_dict.items():
        new_color_dict[key] = val
    return new_color_dict






def extract_rois(source_np, target_np, grid_size=(4, 4), min_pts=100,s=0.3):
    p_min, p_max = source_np.min(axis=0), source_np.max(axis=0)
    num_x, num_y = grid_size
    
    step_x = (p_max[0] - p_min[0]) / num_x
    step_y = (p_max[1] - p_min[1]) / num_y
    
    source_rois = []
    roi_id = 0
    
    for i in range(num_x):
        for j in range(num_y):
            x_start = p_min[0] + i * step_x
            x_end = p_min[0] + (i + 1) * step_x
            y_start = p_min[1] + j * step_y
            y_end = p_min[1] + (j + 1) * step_y
            
            mask = (source_np[:, 0] >= x_start) & (source_np[:, 0] <= x_end) & \
                   (source_np[:, 1] >= y_start) & (source_np[:, 1] <= y_end)
            p = source_np[mask]
            
            if len(p) > min_pts:
                source_rois.append({
                    'id': roi_id,
                    'points': p,
                    'center': p.mean(axis=0),
                    'bbox': (x_start, y_start, step_x, step_y)
                })
                roi_id += 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].scatter(source_np[:, 0], source_np[:, 1], s=s, c='whitesmoke', zorder=0)
    
    for roi in source_rois:
        r_id_str = str(roi['id'])
        color = default_color_dict.get(r_id_str, "#000000") 
        
        x, y, w, h = roi['bbox']
        
        axes[0].scatter(roi['points'][:, 0], roi['points'][:, 1], s=s, c=color, zorder=1)
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none', alpha=0.8, zorder=2)
        axes[0].add_patch(rect)
        
        axes[0].text(x + w/2, y + h/2, r_id_str, color='black', fontsize=12, fontweight='bold', 
                     ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2), zorder=3)
        
    axes[0].set_title(f"Source Grid ROIs Colored (Total: {len(source_rois)})", fontsize=14)
    axes[0].axis('equal')

    axes[1].scatter(target_np[:, 0], target_np[:, 1], s=s, c='#2ca02c', alpha=0.5)
    axes[1].set_title("Target Reference", fontsize=14)
    axes[1].axis('equal')

    plt.tight_layout()
    plt.show()

    return source_rois


def extract_rois_atlas(source_np, grid_size=(4, 4), min_pts=100, s=0.3):
    p_min, p_max = source_np.min(axis=0), source_np.max(axis=0)
    num_x, num_y = grid_size
    
    step_x = (p_max[0] - p_min[0]) / num_x
    step_y = (p_max[1] - p_min[1]) / num_y
    
    source_rois = []
    roi_id = 0
    
    for i in range(num_x):
        for j in range(num_y):
            x_start = p_min[0] + i * step_x
            x_end = p_min[0] + (i + 1) * step_x
            y_start = p_min[1] + j * step_y
            y_end = p_min[1] + (j + 1) * step_y
            
            mask = (source_np[:, 0] >= x_start) & (source_np[:, 0] <= x_end) & \
                   (source_np[:, 1] >= y_start) & (source_np[:, 1] <= y_end)
            p = source_np[mask]
            
            if len(p) > min_pts:
                source_rois.append({
                    'id': roi_id,
                    'points': p,
                    'center': p.mean(axis=0),
                    'bbox': (x_start, y_start, step_x, step_y)
                })
                roi_id += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.scatter(source_np[:, 0], source_np[:, 1], s=s, c='whitesmoke', zorder=0)
    
    for roi in source_rois:
        r_id_str = str(roi['id'])
        color = default_color_dict.get(r_id_str, "#000000") 
        
        x, y, w, h = roi['bbox']
        
        ax.scatter(roi['points'][:, 0], roi['points'][:, 1], s=s, c=color, zorder=1)
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none', alpha=0.8, zorder=2)
        ax.add_patch(rect)
        
        ax.text(x + w/2, y + h/2, r_id_str, color='black', fontsize=12, fontweight='bold', 
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2), zorder=3)
        
    ax.set_title(f"Source Grid ROIs Colored (Total: {len(source_rois)})", fontsize=14)
    ax.axis('equal')

    plt.tight_layout()
    plt.show()

    return source_rois
    


def plot_2d_alignment_to_3d_flow(
    source_adata, target_adata, label_key='label', palette=None,
    n_lines=500, height_scale=1.0, size=2, alpha_points=0.6, alpha_lines=0.6,alpha=0.5,
    xlim=None, ylim=None, save_path=None
):
    source_coords = source_adata.obsm['spatial']
    target_coords = target_adata.obsm['spatial']
    
    source_labels = source_adata.obs[label_key].values
    target_labels = target_adata.obs[label_key].values
    
    #3D
    source_3d = np.hstack((source_coords, np.zeros((len(source_coords), 1))))
    target_3d = np.hstack((target_coords, np.full((len(target_coords), 1), height_scale)))
    
    tree = cKDTree(target_coords)
    _, indices = tree.query(source_coords, k=1)
    
    matched_target_3d = target_3d[indices]
    matched_target_labels = target_labels[indices]
    
    #palette
    if palette is None:
        unique_labels = sorted(list(set(source_labels) | set(target_labels)))
        cmap = plt.get_cmap('tab20')
        palette = {lbl: cmap(i / len(unique_labels)) for i, lbl in enumerate(unique_labels)}
        
    source_colors = np.array([palette.get(lbl, '#cccccc') for lbl in source_labels])
    target_colors = np.array([palette.get(lbl, '#cccccc') for lbl in target_labels])
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    ax.scatter(source_3d[:, 0], source_3d[:, 1], source_3d[:, 2],c=source_colors, s=size, alpha=alpha_points)
    ax.scatter(target_3d[:, 0], target_3d[:, 1], target_3d[:, 2],c=target_colors, s=size, alpha=alpha_points)
    
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(source_coords), size=min(n_lines, len(source_coords)), replace=False)
    
    correct_count = 0
    incorrect_count = 0
    
    for idx in sample_indices:
        start = source_3d[idx]
        end = matched_target_3d[idx]
        
        if source_labels[idx] == matched_target_labels[idx]:
            line_color = 'black'
            correct_count += 1
        else:
            line_color = 'red'
            incorrect_count += 1
            
        ax.plot(
            [start[0], end[0]], 
            [start[1], end[1]], 
            [start[2], end[2]],
            color=line_color, 
            linewidth=0.8,         
            alpha=alpha_lines,
            linestyle='--'       
        )
        
    print(f"Connected sampling (total {len(sample_indices)}):")
    print(f"   - Correct pairing (Black): {correct_count}")
    print(f"   - Incorrect pairing (red): {incorrect_count}")
    
    ax.view_init(elev=25, azim=-45)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.set_zlim(-0.1, height_scale + 0.1)
    
    ax.set_axis_off() 
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved in: {save_path}")
        
    plt.show()





def plot_aligned_adata_on_if(image_path, adata, level=0, color_by='cell_type', is_gene=False, origin='lower', s=0.5, alpha=0.6,figsize=(9,8),dpi=300,palette=None,cmap='Blues',save_path=None):
    with tifffile.TiffFile(image_path) as tif:
        full_shape = tif.series[0].shape
        preview_level = tif.series[0].levels[level]
        low_shape = preview_level.shape
        img = preview_level.asarray()
        if img.ndim == 3:
            if img.shape[0] < img.shape[2]: # (C, H, W) -> (H, W, C)
                full_w = full_shape[2]
                low_w = low_shape[2]
                img = img.transpose(1, 2, 0)
            else:
                full_w = full_shape[1]
                low_w = low_shape[1]
        else:
            full_w = full_shape[1]
            low_w = low_shape[1]
            
    scale_factor = low_w / full_w
    
    small_img = img[::8, ::8] if img.ndim==2 else img[::8, ::8, :]  
    p_low, p_high = np.percentile(small_img, [1, 99.7])
    img_bright = np.clip((img - p_low) / (p_high - p_low), 0, 1)
    spatial_coords = adata.obsm['spatial'] * scale_factor
    x_dots = spatial_coords[:, 0]
    y_dots = spatial_coords[:, 1]
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    ax.imshow(img_bright, origin=origin)
    if is_gene:
        gene_expr = adata[:, color_by].X
        if hasattr(gene_expr, "toarray"):
            gene_expr = gene_expr.toarray()
        gene_expr = gene_expr.flatten()
        keep = gene_expr > 0
        
        scatter = ax.scatter(x_dots[keep], y_dots[keep], c=gene_expr[keep], cmap=cmap, s=s*0.5, alpha=alpha)
        
        plt.colorbar(scatter, ax=ax, label=f'{color_by} Expression', shrink=0.645, pad=0.02)
        
    else:
        cell_categories = adata.obs[color_by].astype('category')
        codes = cell_categories.cat.codes
        categories = cell_categories.cat.categories
        n_cats = len(categories)
        
        if palette is None:
            cmap = plt.cm.tab20 
        elif isinstance(palette, str):
            cmap = plt.get_cmap(palette)
        elif isinstance(palette, list):
            cmap = ListedColormap(palette)
        elif isinstance(palette, dict):
            ordered_colors = [palette[cat] for cat in categories]
            cmap = ListedColormap(ordered_colors)
        else:
            raise ValueError("palette must be None, str, list, or dict")
        
        scatter = ax.scatter(x_dots, y_dots, c=codes, cmap=cmap, s=s*0.5, alpha=alpha)
        handles = []
        for i, cat in enumerate(categories):
            if isinstance(cmap, ListedColormap):
                color = cmap(i % cmap.N)
            else:
                color = cmap(i / max(1, n_cats-1))
            handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=cat))
            
        ax.legend(handles=handles, title=color_by, bbox_to_anchor=(0.99, 1.04), loc='upper left',frameon=False)
  
    ax.axis('off')
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig, ax


