[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

<div align="center">
<h1>
<b>
Rethinking Image Color Aesthetics Assessment: Models, Datasets and Benchmarks
</b>
</h1>
<h4>
<b>
Shuai He, Yi Xiao, Anlong Ming, Huadong Ma
    
Beijing University of Posts and Telecommunications
</b>
</h4>
</div>

This repo contains the official implementation and the new dataset ICAA20K of the **Information Fusion 2024** paper.


## Introduction

* **Brief Version**: Impressive zero-shot and fine-tuning capabilities on sub-attribute ICAA tasks, and even supports user-customized scenarios.

* **DR Version**: Image color aesthetics assessment (ICAA) aims to assess color aesthetics based on human perception, which is crucial for various applications such as imaging measurement and image analysis. The ceiling of previous methods is constrained to a holistic evaluation approach, which hinders their ability to offer explainability from multiple perspectives. Moreover, existing ICAA datasets often lack multi-attribute annotations beyond holistic scores, which are necessary to provide effective supervision for training or validating models' multi-perspective assessment capabilities, thereby hindering their capacity for effective generalization.To advance ICAA research:
    * We propose an ``all-in-one'' model called the Prompt-Guided Delegate Transformer (Prompt-DeT). Prompt-DeT utilizes dedicated prompt strategies and an Aesthetic Adapter (Aes-Adapter), to exploit the rich visual language prior embedded in large pre-trained vision-language models. It enhances the model's perception of multiple attributes, enabling impressive zero-shot and fine-tuning capabilities on sub-attribute tasks, and even supports user-customized scenarios.
    * We elaborately construct a color-oriented dataset, ICAA20K, containing 20K images and 6 annotated dimensions to support both holistic and sub-attribute ICAA tasks.
    * We develop a comprehensive benchmark comprising of 17 methods, which is the most extensive to date, based on four datasets (ICAA20K, ICAA17K, SPAQ, and PARA) for evaluating the holistic and sub-attribute performance of ICAA methods. Our work, not only achieves state-of-the-art (SOTA) performance, but also offers the community a roadmap to explore solutions for ICAA.

<div align="center">
<img src="https://github.com/user-attachments/assets/79a7eccb-abdf-465b-b1a3-e59d3bf27887" alt="Image text" width="700px" />
</div>


## ICAA20K Dataset
To enhance the ICAA17K+ dataset, we have incorporated 2,000 detailed labels concerning color attributes including colorfulness, harmony, and temperature annotations. `The dataset can be downloaded from: [ICAA20K Dataset](https://drive.google.com/file/d/1tUo9o--ls18phooYHZhAPrkYf8IKqrGr/view?usp=sharing).'

<div align="center">
<img src="https://github.com/woshidandan/DeT-Plus/assets/15050507/561076c4-d14d-4f06-afab-82e986e64901" alt="Image text" width="700px" />
</div>

## Prompt-DeT
we propose the Prompt-Guided Delegate Transformer (Prompt-DeT). Firstly, it employs the proposed Aes-Adapter and contrastive learning to align attribute-related features with CLIP’s vast knowledge. Guided by our explainable attribute-aware prompts, it extracts attribute-related features to adapt to downstream tasks, and enhances the understanding of diverse aesthetic attributes.
Secondly, Prompt-DeT simulates human behavior in color space segmentation and adaptively assigns different attention weights based on color importance. This approach further extracts explainable features to express the color information. `The weight can be downloaded from: [Prompt-DeT_Weights](https://drive.google.com/drive/folders/1E7aOnGsvu1ogk-pDmEu4V9DFFq_XJcZI).'

<div align="center">
<img src="https://github.com/user-attachments/assets/61265d61-48f0-417d-8e1b-f3a04fbe9cf5" alt="Image text" width="700px" />
</div>



## Performance and Benchmark

We develop a comprehensive benchmark comprising of 17 methods, which is the most extensive to date, based on three datasets (ICAA20K, ICAA17K, SPAQ, and PARA) for evaluating the holistic and sub-attribute performance of ICAA methods. Our work achieves state-of-the-art (SOTA) performance on all benchmarks.

<div align="center">
<img src="https://github.com/user-attachments/assets/9eebb014-4f02-4396-818b-72cb7a96f5a9" alt="Image text" width="700px" />
    <img src="https://github.com/user-attachments/assets/ba27ba75-f244-45dd-b922-88c3fd3cbb34" alt="Image text" width="700px" />
    <img src="https://github.com/user-attachments/assets/1f959023-5811-4f4d-9f9e-18b346da4933" alt="Image text" width="700px" />
    <img src="https://github.com/user-attachments/assets/f6f7dd4c-c586-4eee-b148-f3c5bd4f8600" alt="Image text" width="700px" />
</div>

## Requirement
```
einops==0.8.0
ftfy==6.3.1
matplotlib==3.8.0
numpy==2.1.3
pandas==2.2.3
Pillow==9.0.1
Pillow==11.0.0
pytorch_model_summary==0.1.2
PyYAML==6.0.1
PyYAML==6.0.2
regex==2023.8.8
Requests==2.32.3
scikit_learn==1.3.1
scipy==1.14.1
seaborn==0.13.2
setuptools==68.0.0
setuptools==74.1.2
tensorboardX==2.6.2.2
tensorboardX==2.6.2.2
thop==0.1.1.post2209072238
timm==0.9.7
torch==2.0.1
torchvision==0.15.2
tqdm==4.65.2
yacs==0.1.8
```


## If you find our work is useful, pleaes cite our paper:
```
@article{he2025prompt,
  title={Prompt-guided image color aesthetics assessment: Models, datasets and benchmarks},
  author={He, Shuai and Xiao, Yi and Ming, Anlong and Ma, Huadong},
  journal={Information Fusion},
  volume={114},
  pages={102706},
  year={2025},
  publisher={Elsevier}
}
```

## Related Work from Our Group
<table>
  <thead align="center">
    <tr>
      <td><b>🎁 Projects</b></td>
      <td><b>📚 Publication</b></td>
      <td><b>🌈 Content</b></td>
      <td><b>⭐ Stars</b></td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://github.com/woshidandan/Pixel-level-No-reference-Image-Exposure-Assessment"><b>Pixel-level image exposure assessment【首个像素级曝光评估】</b></a></td>
      <td><b>NIPS 2024</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Pixel-level-No-reference-Image-Exposure-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Long-Tail-image-aesthetics-and-quality-assessment"><b>Long-tail solution for image aesthetics assessment【美学评估数据不平衡解决方案】</b></a></td>
      <td><b>ICML 2024</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Long-Tail-image-aesthetics-and-quality-assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Prompt-DeT"><b>CLIP-based image aesthetics assessment【基于CLIP多因素色彩美学评估】</b></a></td>
      <td><b>Information Fusion 2024</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Prompt-DeT?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/SR-IAA-image-aesthetics-and-quality-assessment"><b>Compare-based image aesthetics assessment【基于对比学习的多因素美学评估】</b></a></td>
      <td><b>ACMMM 2024</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/SR-IAA-image-aesthetics-and-quality-assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Image-Color-Aesthetics-and-Quality-Assessment"><b>Image color aesthetics assessment【首个色彩美学评估】</b></a></td>
      <td><b>ICCV 2023</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Image-Color-Aesthetics-and-Quality-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Image-Aesthetics-and-Quality-Assessment"><b>Image aesthetics assessment【通用美学评估】</b></a></td>
      <td><b>ACMMM 2023</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Image-Aesthetics-and-Quality-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/TANet-image-aesthetics-and-quality-assessment"><b>Theme-oriented image aesthetics assessment【首个多主题美学评估】</b></a></td>
      <td><b>IJCAI 2022</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/TANet-image-aesthetics-and-quality-assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/AK4Prompts"><b>Select prompt based on image aesthetics assessment【基于美学评估的提示词筛选】</b></a></td>
      <td><b>IJCAI 2024</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/AK4Prompts?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/mRobotit/M2Beats"><b>Motion rhythm synchronization with beats【动作与韵律对齐】</b></a></td>
      <td><b>IJCAI 2024</b></td>
      <td><b>Code, Dataset</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/mRobotit/M2Beats?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Champion-Solution-for-CVPR-NTIRE-2024-Quality-Assessment-on-AIGC"><b>Champion Solution for AIGC Image Quality Assessment【NTIRE AIGC图像质量评估赛道冠军】</b></a></td>
      <td><b>CVPRW NTIRE 2024</b></td>
      <td><b>Code</b></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Champion-Solution-for-CVPR-NTIRE-2024-Quality-Assessment-on-AIGC?style=flat-square&labelColor=343b41"/></td>
    </tr>
  </tbody>
</table>
