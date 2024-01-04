# Machine learning toolkits for spectrum mapping data

This project is an integral component of the [2DMat.ChemDX.org](https://2dMat.ChemDX.org/) platform, which stands as a dedicated hub for the repository and comprehensive analysis of 2D materials data. The core objective of this initiative is to enhance the resolution of spectroscopy mapping data, including photoluminescence (PL) and Raman spectra, related to 2D materials. Our toolkit's primary aim is to amplify data clarity, thus enabling researchers to perform deeper and more precise analyses in a reduced timeframe. Leveraging the capabilities of the state-of-the-art super-resolution (SR) model, [SwinIR](https://github.com/jingyunliang/swinir), users can expect high-quality mapping data in just a few minutes.

Installation
------------
```bash
git clone https://github.com/KRICT-DATA/2D_Spectrum_ML.git
conda env create -f environment.yml
conda activate SR_toolkit
```
How to Use
----------
For an in-depth guide on executing the SR process, please check out our detailed Jupyter notebook titled [super_resolution.ipynb](./super_resolution.ipynb).

Within the file, input parameters can be modified within the first block as shown below.

```python
import torch

# main input file
input_file = './example_data/128.txt'

# default parameters
batch_size = 2
kernel_intensity = 4
target_resolution = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```
- `input_file`: Input mapping data in the format (`M`, `N`*`N`+1), where `M` represents the spectra length, and `N` stands for the resolution.
- `batch_size`: Batch size for prediction
- `kernel_intensity`: Sharpening kernel intensity
- `target_resolution`: Desired resolution after enhancemet. The iteration continues until the enhanced resolution surpasses or equals this target value. 
- `device`: Device to run SR model. It will automatically detect GPU device if available.

Project Resource
-----------------
- Dive into the broader context on our primary project webpage [![2DMat.ChemDX.org](https://img.shields.io/badge/2DMat.ChemDX.org-gray)](https://2dMat.ChemDX.org/toolkits)

- To understand the foundation of our work, explore the SwinIR model's project page [![Link](https://img.shields.io/badge/SwinIR-gray)](https://github.com/jingyunliang/swinir)

