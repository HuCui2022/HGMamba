# HGMamba
PyTorch implementation of "HGMamba: Enhancing 3D Human Pose Estimation with a HyperGCN-Mamba Network", IJCNN2025..

## 🛠️ Environment

The project is developed under the following environment:
- Linux
- Python 3.8.10
- PyTorch 1.12.0
- CUDA 11.6+

For installation of the project dependencies, please run:

- [Option] `pip install causal-conv1d>=1.4.0`: an efficient implementation of a simple causal Conv1d layer used inside the Mamba block.
- `pip install mamba-ssm`: the core Mamba package.
-  `pip install mamba-ssm[causal-conv1d]`: To install core Mamba package and causal-conv1d.


## 📂 Dataset
### Human3.6M
#### Preprocessing
1. Download the fine-tuned Stacked Hourglass detections of [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md)'s preprocessed H3.6M data [here](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ) and unzip it to 'data/motion3d'.
2. Slice the motion clips by running the following python code in `data/preprocess` directory:
**For HGMamba-Base**:
```text
python h36m.py  --n-frames 243
```

**For HGMamba-Small**:
```text
python h36m.py --n-frames 81
```

**For HGMamba-XSmall**:
```text
python h36m.py --n-frames 27
```



### MPI-INF-3DHP
#### Preprocessing
Please refer to [P-STMO](https://github.com/paTRICK-swk/P-STMO#mpi-inf-3dhp) for dataset setup. After preprocessing, the generated .npz files (`data_train_3dhp.npz` and `data_test_3dhp.npz`) should be located at `data/motion3d` directory.





# 🔧 Code Release in Progress  

We are currently **organizing and cleaning** the code before making it fully available to the public.  

### 📅 Expected Release Timeline  
- ✅ Initial data preparation  
- 🔄 Code refactoring and cleanup (In Progress)  
- 🔜 Documentation and final review  

Stay tuned for updates, and feel free to ⭐ the repository for notifications!  


## 📄 Citation
If you find our work useful, please cite our paper:

```bibtex
@article{cui2025hgmamba,
  title={HGMamba: Enhancing 3D Human Pose Estimation with a HyperGCN-Mamba Network},
  author={Hu Cui and Tessai Hayama},
  journal={arXiv preprint arXiv:2504.06638},
  year={2025},
  url={https://arxiv.org/abs/2504.06638}
}
```
You can find the paper on arXiv.
