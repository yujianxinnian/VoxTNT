## A 3D Object Detection Approach from Point Cloud based on Voxel-level Transformer iN Transformer

**Authors**: [Qiangwen Wen](https://github.com/yujianxinnian), [Sheng Wu*](http://adcfj.cn/sirc/door/team/TeacherList/Detail?personId=%20422), Jinghui Wei.

**Institution**: Introduction of The Academy of Digital China (Fujian), Fuzhou University, China

This project is built on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). 

## Introduction
<img src="diagram.png" alt="drawing" width="900" height="400"/>
This study proposes a Transformer-in-Transformer based 3D object detection method that effectively addresses multi-scale object detection challenges through a local-global feature coordination mechanism. The core concept models point cloud processing as a set-to-set transformation to better preserve original point cloud information. The designed 3D feature learning module focuses on discovering context-adaptive hidden code combinations by balancing local and global receptive fields. Demonstrating robust performance across both single-stage detection frameworks and extended two-stage architectures, the proposed method achieves accurate identification of multi-scale targets in complex environments while maintaining an effective balance between improved detection accuracy and real-time processing capabilities.

### 1. Recommended Environment
- OpenPCDet Version: 0.5.2
- Linux (tested on Ubuntu 22.04)
- Python 3.7
- PyTorch 1.9 or higher (tested on PyTorch 1.13.0)
- CUDA 9.0 or higher (tested on CUDA 11.7)


### 2. Set the Environment

```shell
pip install -r requirements.txt
python setup.py build_ext --inplace 
```
The [torch_geometric, torch_scatter, torch_sparse, torch_cluster, torch_spline_conv](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) package is required



### 3. Data Preparation

- Prepare [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and [road planes](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing)

```shell
# Download KITTI and organize it into the following form:
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2

# Generatedata infos:
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

### 4. Pretrain model

PLEASE NOTE: For the voxel-based methods, the point clouds are randomly sampled, which results in some deviation in the prediction outcomes for each instance. However, the deviation is not expected to be too large. This is a normal phenomenon.


The performance (Best combination，using 11 recall poisitions) on KITTI validation set is as follows(single-stage):
```
		
Car  AP@0.70, 0.70, 0.70:

3d   AP: 88.52 78.20 77.03

Pedestrian AP@0.50, 0.50, 0.50:

3d   AP: 65.76 59.56 53.20

Cyclist AP@0.50, 0.50, 0.50:

3d   AP: 85.66 70.07 66.04
```
The runtime is about **36 ms** per sample. (RTX 4090 GPU)


The performance (using 40 recall poisitions) on the KITTI test set (two-stage).
In two-stage models are not suitable to directly report results on KITTI test set, please use slightly lower score threshold and train the models on all or 90% training data to achieve a desirable performance on KITTI test set.
```
Car  AP@0.70, 0.70, 0.70:

3D   AP: 90.51 81.74 77.22	
	
Pedestrian AP@0.50, 0.50, 0.50:

3D   AP: 50.92 43.87 40.53

Cyclist AP@0.50, 0.50, 0.50:

3D   AP: 83.37 68.53 62.13
```
Due to the voxel based method, each sampling point is random, so the results may vary during each training or testing.

### 5. Train

- Train with a single GPU

```shell

cd VoxTNT/tools
python train.py --cfg_file cfgs/kitti_models/voxt_gnn.yaml


```

### 6. Test with a pretrained model

```shell
cd VoxTNT/tools
python test.py --cfg_file --cfg_file ./cfgs/kitti_models/voxt_gnn.yaml --ckpt ${CKPT_FILE}
```
### 7. Acknowledgement

Some codes are from VoxSeT(https://github.com/skyhehe123/VoxSeT).


