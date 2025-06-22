<div align="center">
<h1>Vision CNN Mamba </h1>
<h3>Bridging Convolutions with Multi-Directional Mamba for Efficient
Visual Representation</h3>

[Mustafa Munir](https://github.com/mmunir127)<sup>1</sup> \*,[Alex Zhang](https://github.com/Wertyuui345)<sup>1</sup> \*,[Radu Marculescu](https://scholar.google.com/citations?user=ZCmYP5cAAAAJ&hl=en)<sup>1</sup>

<sup>1</sup>  University of Texas at Austin

(\*) equal contribution, (<sup>:email:</sup>) corresponding author.

ArXiv Preprint (TBD)


</div>


## Abstract
Recent advances in Vision Transformers (ViTs) and State Space Models (SSMs) have challenged the dominance of Convolutional Neural Networks (CNNs) in computer vision. ViTs excel at capturing global context, and SSMs like Mamba offer linear complexity for long sequences, yet they do not capture fine-grained local features as effectively as CNNs. Conversely, CNNs possess strong inductive biases for local features but lack the global reasoning capabilities of transformers and Mamba. To bridge this gap, we introduce \textit{VCMamba}, a novel vision backbone that integrates the strengths of CNNs and multi-directional Mamba SSMs. VCMamba employs a convolutional stem and a hierarchical structure with convolutional blocks in its early stages to extract rich local features. These convolutional blocks are then processed by later stages incorporating multi-directional Mamba blocks designed to efficiently model long-range dependencies and global context. This hybrid design allows for superior feature representation while maintaining linear complexity with respect to image resolution. We demonstrate VCMamba's effectiveness through extensive experiments on ImageNet-1K classification and ADE20K semantic segmentation. Our VCMamba-B achieves 82.6\% top-1 accuracy on ImageNet-1K, surpassing PlainMamba-L3 by 0.3\% with 37\% fewer parameters, and outperforming Vision GNN-B by 0.3\% with 64\% fewer parameters. Furthermore, VCMamba-B obtains 47.1 mIoU on ADE20K, exceeding EfficientFormer-L7 by 2.0 mIoU while utilizing 62\% fewer parameters.

## Overview
<div align="center">
<img src="assets/Hybrid VMamba.png" />
</div>

## Prerequisites

- Python 3.9.7

  - `conda create -n your_env_name python=3.9.7`

Requirements TBD
  
  
## Train 

`bash test.slurm`

## Evaluation

TBD


## Acknowledgement

TBD


## Citation

TBD
