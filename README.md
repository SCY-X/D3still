# D3still


This repo is

(1) a PyTorch library that provides classical knowledge distillation algorithms on asymmetric image retrieval benchmarks.

(2) the official implementation of the CVPR-2024 paper: [D3still: Decoupled Differential Distillation for Asymmetric Image Retrieval](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_D3still_Decoupled_Differential_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2024_paper.html).




NOTE: Unlike our CVPR 2024 paper, which employs only distillation loss, this repository follows common practices in prior knowledge distillation research by incorporating standard losses (e.g., cross-entropy loss and triplet loss) during the distillation of the student network (query network). Correspondingly, the hyperparameters in all distillation methods are also adjusted accordingly. As a result, the experimental outcomes reported in this repository demonstrate significant performance improvements across multiple datasets compared to those presented in the CVPR 2024 paper. For example, on the In-Shop dataset, [FitNet](https://arxiv.org/abs/1412.6550) performance is improved from 62.84% mAP to 65.99% mAP. To offer a more comprehensive evaluation of the effectiveness of our approach, this repository presents ablation experiment results on new benchmarks.




### Ablation Experiments

Gallery Network: ResNet101 &nbsp; Gallery Network Input Resolution: $256\times256$
 
Query Network: ResNet18  &nbsp; Query Network Input Resolution: CUB-200-2011 ($128\times128$) &nbsp; In-Shop ($64\times 64$) &nbsp; SOP ($64\times 64$)

<div style="text-align:center"><img src=".github/Ablation_Study.png" width="100%" ></div> 


### SOTA Experiments

#### On the Caltech-UCSD Birds 200 (CUB-200-2011) dataset

| Teacher <br> Student | ResNet101 ($256\times256$) <br> ResNet18 ($64\times64$)|  <br> |  <br> |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|
| VanillaKD | 0.99% mAP &nbsp; 0.62% R1 1 |  |  |
| RKD | 0.85% mAP &nbsp; 0.36% R1 1|  |  |
| PKT | 0.87% mAP &nbsp; 0.71% R1 30000 |  |  |
| FitNet|  54.73% mAP &nbsp; 60.96% R1 2|  |  |
| CC| 57.93% mAP &nbsp; 62.89% R1  5|  |  |
| CSD| 58.86% mAP &nbsp; 64.08% R1  10|  |  |
| RAML| 57.87% mAP &nbsp; 62.89% R1 10|  |  |
| ROP| 55.61% mAP &nbsp; 63.00% R1 1|  |  |
| D3still (Ours) | 59.40% mAP &nbsp; 64.46% R1 100 5 1 | | |

#### On the In-Shop Clothes Retrieval (In-Shop) dataset

| Teacher <br> Student | ResNet101 ($256\times256$) <br> ResNet18 ($64\times64$)|  <br> |  <br> |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|
| VanillaKD | 0.15% mAP &nbsp; 0.02% R1 1 |  |  |
| RKD |  0.15% mAP &nbsp; 0.06% R1 1|  |  |
| PKT | 0.13% mAP &nbsp; 0.02% R1 30000 |  |  |
| FitNet|  65.99% mAP &nbsp; 80.50% R1 2|  |  |
| CC| 66.60% mAP &nbsp; 81.21% R1  5|  |  |
| CSD| 66.64% mAP &nbsp; 81.00% R1  10|  |  |
| RAML| 67.18% mAP &nbsp; 81.85% R1 10|  |  |
| ROP| 65.58% mAP &nbsp; 80.24% R1 1|  |  |
| D3still (Ours) | 68.56% mAP &nbsp; 83.96% R1 100 5 1 | | |


#### On the Stanford Online Products (SOP) dataset

| Teacher <br> Student | ResNet101 ($256\times256$) <br> ResNet18 ($64\times64$)|  <br> |  <br> |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|
| VanillaKD | 0.04% mAP &nbsp; 0.00% R1 1 |  |  |
| RKD |  0.03% mAP &nbsp; 0.00% R1 1 |  |  |
| PKT | 0.04% mAP &nbsp; 0.02% R1 30000 |  |  |
| FitNet|  48.87% mAP &nbsp; 65.35% R1 2|  |  |
| CC| 49.11% mAP &nbsp; 66.05% R1  5|  |  |
| CSD| 49.43% mAP &nbsp; 65.96% R1  10|  |  |
| RAML| 49.46% mAP &nbsp; 66.24% R1 10|  |  |
| ROP| 48.03% mAP &nbsp;% 64.66% R1 1|  |  |
| D3still (Ours) | 51.12% mAP &nbsp; 68.42% R1 100 5 1 | | |
