<div align="center">
<h3>VCMamba: Bridging Convolutions with Multi-Directional Mamba for Efficient Visual Representation</h3>

[Mustafa Munir](https://github.com/mmunir127)\*, [Alex Zhang](https://github.com/Wertyuui345)\*, [Radu Marculescu](https://scholar.google.com/citations?user=ZCmYP5cAAAAJ&hl=en)

The University of Texas at Austin

(\*) equal contribution

[ArXiv Preprint](https://arxiv.org/abs/2509.04669)

ICCV 2nd Workshop on Efficient Computing under Limited Resources: <https://eclr-workshop.github.io/> 


</div>


## Abstract
Recent advances in Vision Transformers (ViTs) and State Space Models (SSMs) have challenged the dominance of Convolutional Neural Networks (CNNs) in computer vision. ViTs excel at capturing global context, and SSMs like Mamba offer linear complexity for long sequences, yet they do not capture fine-grained local features as effectively as CNNs. Conversely, CNNs possess strong inductive biases for local features but lack the global reasoning capabilities of transformers and Mamba. To bridge this gap, we introduce \textit{VCMamba}, a novel vision backbone that integrates the strengths of CNNs and multi-directional Mamba SSMs. VCMamba employs a convolutional stem and a hierarchical structure with convolutional blocks in its early stages to extract rich local features. These convolutional blocks are then processed by later stages incorporating multi-directional Mamba blocks designed to efficiently model long-range dependencies and global context. This hybrid design allows for superior feature representation while maintaining linear complexity with respect to image resolution. We demonstrate VCMamba's effectiveness through extensive experiments on ImageNet-1K classification and ADE20K semantic segmentation. Our VCMamba-B achieves 82.6\% top-1 accuracy on ImageNet-1K, surpassing PlainMamba-L3 by 0.3\% with 37\% fewer parameters, and outperforming Vision GNN-B by 0.3\% with 64\% fewer parameters. Furthermore, VCMamba-B obtains 47.1 mIoU on ADE20K, exceeding EfficientFormer-L7 by 2.0 mIoU while utilizing 62\% fewer parameters.

## Overview
<div align="center">
<img src="assets/Hybrid VMambaBG.png" />
</div>

## Prerequisites

- Python 3.9.7

```shell
conda create -n your_env_name python=3.9.7
```

- torch 2.1.1

```shell
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118`
```

Requirements 
```shell
pip install -U openmim
mim install mmcv-full
pip install mamba-ssm
pip install mlflow fvcore timm
pip install -r requirements.txt
```
  
## Train and Test
- Train classification

```
bash test.slurm
```

- For eval and load model weights add the following to python launch classification
```
--resume "weight_file.pth" --eval
```


## Classification
<div align="center">

| Name | Dataset | Top-1 Acc | Params | GMACs | Weights |
| :---: | :---: | :---: | :---: | :---: | :---: |
| VCMamba-S | ImageNet-1K | 78.7 | 10.5M | 1.1 | [VCMamba-S](https://huggingface.co/SLDGroup/VCMamba/blob/main/VCMamba-S.pth) |
| VCMamba-M | ImageNet-1K | 81.5 | 21.0M | 2.3 | [VCMamba-M](https://huggingface.co/SLDGroup/VCMamba/blob/main/VCMamba-M.pth) |
| VCMamba-B | ImageNet-1K | 82.6 | 31.5M | 4.0 | [VCMamba-B](https://huggingface.co/SLDGroup/VCMamba/blob/main/VCMamba-B.pth) |

</div>

## Acknowledgement

This project is based on Mamba ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)).

## Citation

```bibtex
@misc{Munir2025vcmamba_arxiv,
      title={VCMamba: Bridging Convolutions with Multi-Directional Mamba for Efficient Visual Representation}, 
      author={Mustafa Munir and Alex Zhang and Radu Marculescu},
      year={2025},
      eprint={2509.04669},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.04669}, 
}
```
