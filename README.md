# TUS-REC2025-Challenge_baseline
This is the official baseline repository for TUS-REC2025 Challenge - MICCAI2025


#### 3. Install conda environment

``` bash
conda create -n freehand-US python=3.9.13
conda activate freehand-US
pip install -r requirements.txt
conda install pytorch3d --no-deps -c pytorch3d
```
If you encounter a "Segmentation fault" error during the installation of pytorch3d, please refer to this <a href="https://github.com/facebookresearch/pytorch3d/issues/1891" target="_blank">link</a>.