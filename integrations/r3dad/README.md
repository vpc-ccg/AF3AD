# R3D-AD: Reconstruction via Diffusion for 3D Anomaly Detection
### [Website](https://zhouzheyuan.github.io/r3d-ad) | [Paper](https://arxiv.org/abs/2407.10862)
> [**R3D-AD: Reconstruction via Diffusion for 3D Anomaly Detection**](https://doi.org/10.1007/978-3-031-72764-1_6),  
> Zheyuan Zhou∗, Le Wang∗, Naiyu Fang, Zili Wang, Lemiao Qiu, Shuyou Zhang
> **ECCV 2024**

## Installation
```sh
pip install easydict faiss-gpu ninja numpy open3d==0.16.0 opencv-python-headless pyyaml scikit-learn scipy tensorboard timm torch tqdm 
pip install "git+https://github.com/unlimblue/KNN_CUDA.git#egg=knn_cuda&subdirectory=."
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

## Datasets

### Anomaly-ShapeNet
Download dataset from [google drive](https://drive.google.com/file/d/16R8b39Os97XJOenB4bytxlV4vd_5dn0-/view?usp=sharing) and extract `pcd` folder into `./data/shapenet-ad/`
```
shapenet-ad
├── ashtray0
    ├── train
        ├── ashtray0_template0.pcd
        ...
    ├── test
        ├── ashtray0_bulge0.pcd
        ...
    ├── GT
        ├── ashtray0_bulge0.txt
        ... 
├── bag0
...
...
├── vase9
```

## Training and Testing
```bash
python train_test.py PATH_TO_CONFIG
```

## Visualization
```bash
python vis_result.py PATH_TO_LOGS
```

## Acknowledgement
Thanks to previous open-sourced repo:

[PVD](https://github.com/alexzhou907/PVD)

[diffusion-point-cloud](https://github.com/luost26/diffusion-point-cloud)

## Citation 
If you find this project useful in your research, please consider cite:

```bibtex
@inproceedings{zhou2024r3dad,
  title={R3D-AD: Reconstruction via Diffusion for 3D Anomaly Detection},
  author={Zhou, Zheyuan and Wang, Le and Fang, Naiyu and Wang, Zili and Qiu, Lemiao and Zhang, Shuyou},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```