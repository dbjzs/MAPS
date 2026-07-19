<div align=center>
<img src="https://github.com/dbjzs/MAPS/blob/main/Logo.jpg" width="200"  alt="MAPS-logo" >
</div>

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
### 🚀Getting started Tutorial
- Jupyter tutorials and API documentation are available at [Tutorial](https://maps-tools.readthedocs.io/en/latest/).
- The expected data input format of SpaLP is [AnnData](https://anndata.readthedocs.io/en/stable/) object.
- Please use [issues](https://github.com/dbjzs/MAPS/issues) to submit bug reports.  

