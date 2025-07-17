This repo is

(1) a PyTorch library that provides classical knowledge distillation algorithms on asymmetric image retrieval benchmarks.

(2) the official implementation of the CVPR-2024 paper: [D3still: Decoupled Differential Distillation for Asymmetric Image Retrieval](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_D3still_Decoupled_Differential_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2024_paper.html).




NOTE: Unlike our CVPR 2024 paper, which employs only distillation loss, this repository follows common practices in prior knowledge distillation research by incorporating standard losses (e.g., cross-entropy loss and triplet loss) during the distillation of the student network (query network). Correspondingly, the hyperparameters in all distillation methods are also adjusted accordingly. As a result, the experimental outcomes reported in this repository demonstrate significant performance improvements across multiple datasets compared to those presented in the CVPR 2024 paper. For example, on the In-Shop dataset, [FitNet](https://arxiv.org/abs/1412.6550) performance is improved from 62.84% mAP to 65.99% mAP. To offer a more comprehensive evaluation of the effectiveness of our approach, this repository presents ablation experiment results on new benchmarks.

(3) the official implementation of the Neural Networks-2025 paper: [Unambiguous granularity distillation for asymmetric image retrieval](https://www.sciencedirect.com/science/article/pii/S0893608025001820).

## D3still: Decoupled Differential Distillation for Asymmetric Image Retrieval

### Framework
<div style="text-align:center"><img src="/AIR_Distiller/.github/D3still_framework.png" width="100%" ></div>

### Ablation Experiments

Gallery Network: ResNet101 &nbsp; Gallery Network Input Resolution: $256\times256$
 
Query Network: ResNet18  &nbsp; Query Network Input Resolution: CUB-200-2011 ($128\times128$) &nbsp; In-Shop ($64\times 64$) &nbsp; SOP ($64\times 64$)

<div style="text-align:center"><img src="/AIR_Distiller/.github/D3still_ablation_study.png" width="100%" ></div> 

## Unambiguous granularity distillation for asymmetric image retrieval

### Framework
<div style="text-align:center"><img src="/AIR_Distiller/.github/UGD_framework.png" width="100%" ></div> 

## SOTA Experiments

### On the Caltech-UCSD Birds 200 (CUB-200-2011) dataset

Performance on different resolutions
| Teacher <br> Student | ResNet101 ($256\times256$) <br> ResNet18 ($128\times128$) | ResNet101 ($384\times384$) <br> ResNet18 ($128\times128$) |
|:---------------:|:-----------------:|:-----------------:|
| VanillaKD | 0.99% mAP &nbsp; 0.62% R1 | 0.80% mAP &nbsp; 0.28% R1 |
| RKD | 0.85% mAP &nbsp; 0.36% R1 | 0.90% mAP &nbsp; 0.52% R1 |
| PKT | 0.87% mAP &nbsp; 0.71% R1 | 0.95% mAP &nbsp; 0.55% R1 |
| FitNet | 54.73% mAP &nbsp; 60.96% R1 | 56.61% mAP &nbsp; 61.82% R1 |
| CC | 57.93% mAP &nbsp; 62.89% R1 | 59.91% mAP &nbsp; 64.26% R1 |
| CSD | 58.86% mAP &nbsp; 64.08% R1 | 60.44% mAP &nbsp; 64.31% R1 |
| RAML | 57.87% mAP &nbsp; 62.89% R1 | 60.22% mAP &nbsp; 64.79% R1 |
| ROP | 55.61% mAP &nbsp; 63.00% R1 | 57.23% mAP &nbsp; 63.12% R1 |
| D3still (Ours) | **59.40% mAP &nbsp; 64.46% R1** | **61.03% mAP &nbsp; 64.84% R1** |

Performance on different network architectures
| Teacher <br> Student | ResNet101 ($256\times256$) <br> MobileNet-V3-Small ($128\times128$) | Swin-Transformer-V2-Small ($256\times256$) <br> ResNet18 ($128\times128$) |
|:---------------:|:-----------------:|:-----------------:|
| VanillaKD |0.90% mAP &nbsp; 0.45% R1 | 0.97% mAP &nbsp; 0.33% R1|
| RKD | 0.93% mAP &nbsp; 0.59% R1| 1.01% mAP &nbsp; 0.38% R1 |
| PKT | 0.88% mAP &nbsp; 0.40% R1| 1.09% mAP &nbsp; 0.57% R1 |
| FitNet | 39.50% mAP &nbsp; 44.56% R1 | 60.02% mAP &nbsp; 61.67% R1|
| CC | 44.79%   mAP &nbsp; 49.21% R1| 60.74% mAP &nbsp; 61.55% R1|
| CSD | **46.45% mAP &nbsp; 50.45% R1**| 60.44% mAP &nbsp; 62.08% R1|
| RAML | 45.57% mAP &nbsp; 49.53% R1| 60.51% mAP &nbsp; 61.60% R1|
| ROP | 41.90% mAP &nbsp; 48.10% R1|  57.97% mAP &nbsp; 60.70% R1|
| D3still (Ours) |45.90% mAP &nbsp; 50.83% R1| **61.35% mAP &nbsp; 62.03% R1**|

### On the In-Shop Clothes Retrieval (In-Shop) dataset

Performance on different resolutions
| Teacher <br> Student | ResNet101 ($256\times256$) <br> ResNet18 ($64\times64$)|  ResNet101 ($384\times384$) <br> ResNet18 ($64\times64$)|
|:---------------:|:-----------------:|:-----------------:|
| VanillaKD | 0.15% mAP &nbsp; 0.02% R1|  0.12% mAP &nbsp; 0.03% R1|
| RKD |  0.15% mAP &nbsp; 0.06% R1| 0.15% mAP &nbsp; 0.04% R1|
| PKT | 0.13% mAP &nbsp; 0.02% R1 |  0.13% mAP &nbsp; 0.04% R1|
| FitNet|  65.99% mAP &nbsp; 80.50% R1| 66.07% mAP &nbsp; 79.58% R1|
| CC| 66.60% mAP &nbsp; 81.21% R1 | 66.09% mAP &nbsp; 79.54% R1|
| CSD| 66.64% mAP &nbsp; 81.00% R1| 65.73% mAP &nbsp; 78.55% R1|
| RAML| 67.18% mAP &nbsp; 81.85% R1| 65.95% mAP &nbsp; 79.45% R1|
| ROP| 65.58% mAP &nbsp; 80.24% R1| 64.20% mAP &nbsp; 77.62% R1|
| D3still| 68.56% mAP &nbsp; 83.96% R1 | 67.90% mAP &nbsp; 82.18% R1|
| UGD | 69.20% mAP &nbsp; 84.05% R1 | 68.74% mAP &nbsp; 82.39% R1|

Performance on different network architectures
| Teacher <br> Student | ResNet101 ($256\times256$) <br> MobileNet-V3-Small ($64\times64$) | Swin-Transformer-V2-Small ($256\times256$) <br> ResNet18 ($64\times64$)|
|:---------------:|:-----------------:|:-----------------:|
| VanillaKD | 0.16%  mAP &nbsp; 0.06% R1  | 0.13% mAP &nbsp; 0.03% R1|
| RKD |  0.15% mAP &nbsp;  0.03% R1| 0.14% mAP &nbsp; 0.04% R1|
| PKT | 0.15% mAP &nbsp; 0.08% R1 | 0.16% mAP &nbsp; 0.04% R1|
| FitNet| 60.41% mAP &nbsp; 74.61% R1| 56.35% mAP &nbsp; 65.77% R1|
| CC| 61.53% mAP &nbsp; 76.25% R1| 56.55% mAP &nbsp; 65.51% R1|
| CSD| 62.27% mAP &nbsp; 76.48% R1| 57.58% mAP &nbsp; 67.87% R1|
| RAML| 62.29% mAP &nbsp; 76.13% R1| 57.34 mAP &nbsp; 67.42% R1|
| ROP| 61.43% mAP &nbsp; 75.91% R1| 53.87% mAP &nbsp; 63.52% R1|
| D3still | 64.14% mAP &nbsp; 80.08% R1 |60.19% mAP &nbsp; 72.07 R1|
| UGD| 64.50% mAP &nbsp; 79.55% R1 ||

### On the Stanford Online Products (SOP) dataset

Performance on different resolutions
| Teacher <br> Student | ResNet101 ($256\times256$) <br> ResNet18 ($64\times64$) | ResNet101 ($384\times384$) <br> ResNet18 ($64\times64$) |
|:---------------:|:-----------------:|:-----------------:|
| VanillaKD | 0.04% mAP &nbsp; 0.00% R1 | 0.04% mAP &nbsp; 0.01% R1 |
| RKD |  0.03% mAP &nbsp; 0.00% R1 |  0.03% mAP &nbsp; 0.00% R1 |
| PKT | 0.04% mAP &nbsp; 0.02% R1 | 0.03% mAP &nbsp; 0.00% R1 |
| FitNet |  48.87% mAP &nbsp; 65.35% R1 | 45.70% mAP &nbsp; 60.53% R1 |
| CC | 49.11% mAP &nbsp; 66.05% R1 | 46.57% mAP &nbsp; 62.16% R1 |
| CSD | 49.43% mAP &nbsp; 65.96% R1 | 46.22% mAP &nbsp; 61.03% R1 |
| RAML | 49.46% mAP &nbsp; 66.24% R1 | 46.79% mAP &nbsp; 62.12% R1 |
| ROP | 48.03% mAP &nbsp; 64.66% R1 | 44.53% mAP &nbsp; 59.66% R1 |
| D3still| 51.12% mAP &nbsp; 68.42% R1 | 48.70% mAP &nbsp; 64.94% R1 |
| UGD | 50.31% mAP &nbsp; 67.43% R1 | 47.30% mAP &nbsp; 63.13% R1 |


Performance on different network architectures
| Teacher <br> Student | ResNet101 ($256\times256$) <br> MobileNet-V3-Small ($64\times64$) | Swin-Transformer-V2-Small ($256\times256$) <br> ResNet18 ($64\times64$) |
|:---------------:|:-----------------:|:-----------------:|
| VanillaKD | 0.03% mAP &nbsp; 0.00% R1 | 0.03% mAP &nbsp; 0.01% R1|
| RKD | 0.03% mAP &nbsp; 0.00% R1 |  0.04% mAP &nbsp; 0.02% R1|
| PKT |  0.04% mAP &nbsp; 0.01% R1 | 0.03% mAP &nbsp; 0.00% R1 |
| FitNet | 44.80% mAP &nbsp; 60.30% R1 | 37.06% mAP &nbsp; 46.57% R1|
| CC | 44.37% mAP &nbsp; 60.09% R1 | 40.27% mAP &nbsp; 51.94% R1|
| CSD | 44.98% mAP &nbsp; 60.72% R1 |  40.50% mAP &nbsp; 51.63% R1|
| RAML | 45.52% mAP &nbsp; 61.48% R1 | 40.64% mAP &nbsp; 51.97% R1|
| ROP | 43.67% mAP &nbsp; 59.35% R1 | 37.90 mAP &nbsp; 49.45% R1|
| D3still (Ours) | 46.59% mAP &nbsp; 63.60% R1 | 43.32% mAP &nbsp; 56.98% R1|
| UGD| 44.13% mAP &nbsp; 60.29% R1| % mAP &nbsp; % R1|


### On the MSMT17 dataset

Performance on different resolutions
| Teacher <br> Student | ResNet101 ($320\times160$) <br> ResNet18 ($160\times80$) | ResNet101 ($480\times240$) <br> ResNet18 ($160\times80$) |
|:---------------:|:-----------------:|:-----------------:|
| VanillaKD | 0.06% mAP &nbsp; 0.01% R1 | 0.07% mAP &nbsp; 0.02% R1 |
| RKD | 0.07% mAP &nbsp; 0.03% R1 | 0.07% mAP &nbsp; 0.03% R1 |
| PKT | 0.04% mAP &nbsp; 0.02% R1 | 0.07% mAP &nbsp; 0.03% R1 |
| FitNet | 36.46% mAP &nbsp; 56.45% R1 | 36.88% mAP &nbsp; 56.57% R1 |
| CC | 36.57% mAP &nbsp; 56.82% R1 | 37.26% mAP &nbsp; 56.65% R1 |
| CSD | 38.30% mAP &nbsp; 59.04% R1 | 38.39% mAP &nbsp; 57.85% R1 |
| RAML | 38.15% mAP &nbsp; 58.89% R1 | 38.32% mAP &nbsp; 58.21% R1 |
| ROP | 36.02% mAP &nbsp; 57.17% R1 | 36.08% mAP &nbsp; 56.45% R1 |
| D3still (Ours) | **39.54% mAP &nbsp; 61.37% R1** | **39.42% mAP &nbsp; 60.13% R1** |

Performance on different network architectures
| Teacher <br> Student | ResNet101 ($320\times160$) <br> MobileNet-V3-Small ($160\times80$) | ResNet101-IBN ($320\times160$) <br> ResNet18 ($160\times80$) |
|:---------------:|:-----------------:|:-----------------:|
| VanillaKD |0.06% mAP &nbsp; 0.03% R1  | 0.07% mAP &nbsp; 0.04% R1 |
| RKD |0.06% mAP &nbsp; 0.02% R1 | 0.07% mAP &nbsp; 0.04% R1 | 
| PKT | 0.06% mAP &nbsp; 0.03% R1  | 0.07% mAP &nbsp; 0.01% R1 | 
| FitNet | 31.97% mAP &nbsp; 49.53% R1 |38.97% mAP &nbsp; 57.36% R1 | 
| CC | 32.08% mAP &nbsp; 49.88% R1|38.65% mAP &nbsp; 56.65% R1 | 
| CSD |32.63%  mAP &nbsp; 50.91% R1 |39.54% mAP &nbsp; 57.90% R1 |
| RAML | 32.93% mAP &nbsp; 50.94% R1 |39.72% mAP &nbsp; 58.85% R1 |
| ROP | 30.12% mAP &nbsp; 48.12% R1 |37.56% mAP &nbsp; 57.35% R1 |
| D3still (Ours) | **33.77%  mAP &nbsp; 53.80% R1** | **41.93% mAP &nbsp; 61.51% R1** |


# AIR-Distiller

### Introduction

AIR-Distiller supports the following distillation methods on  Caltech-UCSD Birds 200 (CUB-200-2011), In-Shop Clothes Retrieval (In-Shop), Stanford Online Products (SOP) and MSMT17:
|Method|Publication|YEAR|
|:---:|:---:|:---:|
|[VanillaKD](https://arxiv.org/abs/1503.02531) |NIPS Workshop|2014|
|[FitNet](https://arxiv.org/abs/1412.6550) |ICLR|2015 |
|[PKT](https://openaccess.thecvf.com/content_ECCV_2018/papers/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.pdf) | ECCV | 2018 |
|[RKD](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html) |CVPR| 2019|
|[CC](https://openaccess.thecvf.com/content_ICCV_2019/html/Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.html) |ICCV| 2019|
|[CSD](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Contextual_Similarity_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2022_paper.html) |CVPR|2023 |
|[RAML](https://openaccess.thecvf.com/content/WACV2023/html/Suma_Large-to-Small_Image_Resolution_Asymmetry_in_Deep_Metric_Learning_WACV_2023_paper.html)|WACV|2023|
|[ROP](https://openreview.net/forum?id=dYHYXZ3uGdQ)|ICLR|2023|
|[D3still](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_D3still_Decoupled_Differential_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2024_paper.html) |CVPR|2024|
|[UGD](https://www.sciencedirect.com/science/article/pii/S0893608025001820) |Neural Networks|2025|

### Installation

Environments:

- Python 3.10
- PyTorch 2.4.1
- torchvision 0.19.1
- ptflops 0.7.4

Install the package:

```
sudo pip3 install -r requirements.txt
```

### Getting started

0. download data
- The dataset has been prepared in the format we read at the link: https://pan.baidu.com/s/1ySKEmn8WVm2efJVvJ_vMBQ?pwd=ebyx. Please download the data and untar it to `XXXX/data` via `unzip XXXX`. For example,  `unzip CUB_200_2011.zip`. Finally, the data file directory should be as follows:


  XXXX/data/  
    &nbsp; &nbsp; &nbsp; &nbsp; └── CUB_200_2011  
    &nbsp; &nbsp; &nbsp; &nbsp; └── InShop  
    &nbsp; &nbsp; &nbsp; &nbsp; └── Stanford_Online_Products  
    &nbsp; &nbsp; &nbsp; &nbsp; └── MSMT17

1. download teacher models
- Our teacher models are at https://pan.baidu.com/s/1X8urI8_bDfmdapSaNGYbtA?pwd=if2i, please download the checkpoints to `./download_ckpts`

2. Path setting
- Please modify the following line in `AIR_Distiller/tools/train.py` and `AIR_Distiller/tools/test.py`:  
`sys.path.append(os.path.abspath("XXXXX/AIR_Distiller"))`  
Replace `"XXXXX/AIR_Distiller"` with the absolute path of your project to ensure correct module imports.

 **Example** (assuming the project path is `/home/user/AIR_Distiller`):  
```python
import sys  
import os  
sys.path.append(os.path.abspath("/home/user/AIR_Distiller"))
```
- Please set the `ROOT_DIR` path in the configuration file, i.e., XXX.yaml to the absolute path of the `data` folder.  
- 
**Example** (assuming the data path is `/home/user/data`):  
```yaml
DATASETS:
  NAMES: "SOP"
  ROOT_DIR: "/home/user/data"
```


3. Training 

 ```bash
  # for instance, when the gallery network is ResNet101 and the query network is ResNet18, our D3 method.
  python AIR_Distiller/tools/train.py --cfg Training_Configs/SOP/ResNet101_256x256_ResNet18_64x64/D3.yaml 
  ```

  - By default, the ImageNet pre-trained model will be used for training. The model will be automatically downloaded from the internet on the first run.  
  If you want to use a different pre-trained model, modify the `STUDENT_PRETRAIN_PATH` in the YAML configuration file.  


4. Evaluation

 ```bash
  # for instance, when the gallery network is ResNet101 and the query network is ResNet18, our D3 method.
  python AIR_Distiller/tools/test.py --cfg Training_Configs/SOP/ResNet101_256x256_ResNet18_64x64/D3.yaml 

 ```
 - During inference, you can first navigate to `AIR_Distiller/utils/rank_cylib` and run the following commands to enable sorting with C language, which helps reduce inference time:  

```bash
python3 setup.py build_ext --inplace
rm -rf build
```

### Custom Distillation Method

1. create a python file at `AIR_Distiller/distillers/` and define the distiller
  
  ```python
  from ._base import Distiller

  class MyDistiller(Distiller):
      def __init__(self, student, teacher, cfg):
          super(MyDistiller, self).__init__(student, teacher)
          self.hyper1 = cfg.MyDistiller.hyper1
          ...

      def forward_train(self, image, kd_student_image, kd_teacher_image, target, kd_target, **kwargs):
          # return the output logits and a Dict of losses
          ...
      # rewrite the get_learnable_parameters function if there are more nn modules for distillation.
      # rewrite the get_extra_parameters if you want to obtain the extra cost.
    ...
  ```

2. regist the distiller in `distiller_dict` at `AIR_Distiller/distillers/__init__.py`

3. regist the corresponding hyper-parameters at `AIR_Distiller/config/defaults.py`

4. create a new config file and test it.

### Experimental Note
During training, the batch size of the distillation dataloader (256) is larger than that of the student dataloader (96), resulting in fewer iterations per epoch. This discrepancy may lead to suboptimal performance  for some methods due to insufficient training steps. Future researchers can reduce the distillation batch size or increase the number of training epochs to address this concern.

Across all methods we experimented with, we found that using a distillation batch size of either 256 or 96 yields comparable best performance. However, some method-specific hyperparameters may need to be tuned accordingly.

# Citation

If this repo is helpful for your research, please consider citing the paper:

```BibTeX
@InProceedings{Xie_2024_CVPR,
    author    = {Xie, Yi and Lin, Yihong and Cai, Wenjie and Xu, Xuemiao and Zhang, Huaidong and Du, Yong and He, Shengfeng},
    title     = {D3still: Decoupled Differential Distillation for Asymmetric Image Retrieval},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {17181-17190}
}
```

```BibTeX
@article{zhang2025unambiguous,
  title={Unambiguous granularity distillation for asymmetric image retrieval},
  author={Zhang, Hongrui and Xie, Yi and Zhang, Haoquan and Xu, Cheng and Luo, Xuandi and Chen, Donglei and Xu, Xuemiao and Zhang, Huaidong and Heng, Pheng Ann and He, Shengfeng},
  journal={Neural Networks},
  volume={187},
  pages={107303},
  year={2025},
  publisher={Elsevier}
}
```

# License

AIR_Distiller is released under the MIT license. See [LICENSE](LICENSE) for details.

# Acknowledgement
- Thanks for DKD. We build this library based on the [DKD's codebase](https://github.com/megvii-research/mdistiller).
- Thanks Yihong Lin for the discussion about D3still.
