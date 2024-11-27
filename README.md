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



# Introduction
Our refined work of ICCV 2023 work [DeT](https://github.com/woshidandan/Image-Color-Aesthetics-Assessment), weights and dataset can be downloaded from: 
- [Prompt-DeT_Weights](https://drive.google.com/drive/folders/1E7aOnGsvu1ogk-pDmEu4V9DFFq_XJcZI),
- [ICAA20K Dataset](https://drive.google.com/file/d/1tUo9o--ls18phooYHZhAPrkYf8IKqrGr/view?usp=sharing),

# ICAA20K
To enhance the ICAA17K+ dataset, we have incorporated 2,000 detailed labels concerning color attributes including colorfulness, harmony, and temperature annotations.
<div align="center">
    
![image](https://github.com/woshidandan/DeT-Plus/assets/15050507/561076c4-d14d-4f06-afab-82e986e64901)

</div>

# Performance of Prompt-DeT
We develop a comprehensive benchmark comprising of 17 methods, which is the most extensive to date, based on three datasets (ICAA20K, ICAA17K, SPAQ, and PARA) for evaluating the holistic and sub-attribute performance of ICAA methods. Our work achieves state-of-the-art (SOTA) performance on all benchmarks.
<div align="center">
    
![ICAA20K](https://github.com/user-attachments/assets/9eebb014-4f02-4396-818b-72cb7a96f5a9)

![ICAA17K](https://github.com/user-attachments/assets/ba27ba75-f244-45dd-b922-88c3fd3cbb34)

![SPAQ](https://github.com/user-attachments/assets/1f959023-5811-4f4d-9f9e-18b346da4933)

![PARA](https://github.com/user-attachments/assets/f6f7dd4c-c586-4eee-b148-f3c5bd4f8600)

</div>

# Requirement
```
einops==0.6.1
ftfy==6.1.1
nni==2.10.1
numpy==1.25.2
pandas==2.1.0
Pillow==10.0.0
PyYAML==6.0.1
regex==2023.8.8
Requests==2.31.0
scikit_learn==1.3.0
scipy==1.11.2
setuptools==65.5.1
tensorboardX==2.6.2.2
timm==0.9.7
tqdm==4.66.1
yacs==0.1.8
```
