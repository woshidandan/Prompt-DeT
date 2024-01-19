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
Shuai He, Yi Xiao，Anlong Ming, Huadong Ma
    
Beijing University of Posts and Telecommunications
</b>
</h4>
</div>



# Introduction
Our refined work of ICCV 2023 work [DeT](https://github.com/woshidandan/Image-Color-Aesthetics-Assessment), weights and dataset can be downloaded from: 
- [DeT_plus_ICAA](https://mega.nz/file/I1UjlYoD#fKkLdtubl5OXpQQXB37_OOy0A9kw8HpuQwW_x80MU-s),
- [DeT_plus_PARA](https://mega.nz/file/Mp82UKoI#7a4ik8jFBLnQewE0-bPNg5rz1F6-XMOp1WSXw_FG3XQ),
- [DeT_plus_SPAQ](https://mega.nz/file/Z9tUVJKI#HqgW77gxdtHF6QG5GfnLA31druLONw9IFlsvY3w4W_Q),
- [ICAA17K+ dataset](https://mega.nz/file/V8kwgLaD#ty6IG7gyQduanfd1ViloZVlREq18e0MuWwqCnvTMtiM),
- [RN50](https://mega.nz/file/9sUAzZIZ#Cu4C5QamEn41abU6yz_39IN0by-qIuMkRmh4YFBTy8I).

Details will be published after the acceptance of the paper, we aspire for our work to make a valuable contribution to the ongoing research on ICAA within the community!

# ICAA17K+
To enhance the ICAA17K+ dataset, we have incorporated 2,000 detailed labels concerning color attributes including colorfulness, harmony, and temperature annotations.
<div align="center">
    
![image](https://github.com/woshidandan/DeT-Plus/assets/15050507/561076c4-d14d-4f06-afab-82e986e64901)

</div>

# Performance of DeT-Plus
We develop a comprehensive benchmark comprising of 17 methods, which is the most extensive to date, based on three datasets (ICAA17K+, SPAQ, and PARA) for evaluating the holistic and sub-attribute performance of ICAA methods. Our work achieves state-of-the-art (SOTA) performance on all benchmarks.
<div align="center">
  
![image](https://github.com/woshidandan/DeT-Plus/assets/15050507/7269c1a4-8381-4b79-94e8-b9df5340f994)

![image](https://github.com/woshidandan/DeT-Plus/assets/15050507/589cfcaf-05c9-4af5-9083-87aae294e9ca)

![image](https://github.com/woshidandan/DeT-Plus/assets/15050507/81db22d9-6319-4da8-8708-0b96188e854a)

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
