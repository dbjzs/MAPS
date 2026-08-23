<div align=center>
<img src="https://github.com/dbjzs/MAPS/blob/main/Logo.jpg" width="200"  alt="MAPS-logo" >
</div>
MAPS, a modality-agnostic platform for spatial biology comprising MAPS-alignment for ultrafast alignment of any modality, MAPS-integration for unpaired cross-modal integration across orthogonal modalities and paired 3D multimodal reconstruction, and MAPS-Explorer for large-scale interactive 3D analysis.

[![License](https://img.shields.io/badge/License-Apache-blue.svg)](https://github.com/dbjzs/MAPS/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/dbjzs/MAPS?logo=GitHub&color=yellow)](https://github.com/dbjzs/MAPS/stargazers)
[![Docs](https://readthedocs.org/projects/MAPS/badge/?version=latest)](https://maps-tools.readthedocs.io/en/latest/)
[![Forks](https://img.shields.io/github/forks/dbjzs/MAPS?logo=GitHub&color=yellow)](https://github.com/dbjzs/MAPS/forks)
![Python 3.10.13](https://img.shields.io/badge/python->=3.10-blue.svg)

## Installation via Github
#### 📥 Download
```
git clone https://github.com/dbjzs/MAPS.git
cd MAPS
```
#### 🔧 environment
MAPS is available for Python 3.10. We recommend to train MAPS models on a device with GPU support.  
* Using the conda install environment
```
conda create -n MAPS -c conda-forge python==3.10.13 libopenblas=0.3.25 -y
conda activate MAPS
```
#### 🛠️package
* Then using pip install MAPS.
```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install .
```
### 📁 Tutorial h5ad file
- All h5ad files have been uploaded to the [zenodo repository]()
  
### 🚀Getting started Tutorial
- Jupyter tutorials and API documentation are available at [Tutorial](https://maps-tools.readthedocs.io/en/latest/).
- The expected data input format of MAPS is [AnnData](https://anndata.readthedocs.io/en/stable/) object.
- Please use [issues](https://github.com/dbjzs/MAPS/issues) to submit bug reports.  

### Reference
- If you find MAPS useful for your research, please consider citing the MAPS manuscript [bioRxiv]().

```
@article{,
  title = {Unifying physical and molecular coordinates systems across modalities in spatial biology},
  author = {Dai, Yan, Wang*, Liang,....Zuo*, Qian*, Yuan* et al.},
  year = {2026},
  journal = {bioRxiv : Unifying physical and molecular coordinates systems across modalities in spatial biology},
  eprint = {},
  doi = {},
}
```
