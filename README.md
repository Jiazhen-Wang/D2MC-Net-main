# D2MC-Net

***********************************************************************************************************

Pytorch codes for model in "Dual Domain Motion Artifacts Correction for MR Imaging Under Guidance of K-space Uncertainty" (MICCAI 2023)
 
If you use these codes, please cite our paper:

[1] Jiazhen Wang, Yizhe Yang, Yan Yang, Jian Sun. Dual Domain Motion Artifacts Correction for MR Imaging Under Guidance of K-space Uncertainty (MICCAI 2023).

http://gr.xjtu.edu.cn/web/jiansun/publications

All rights are reserved by the authors.

Jiazhen Wang and Yizhe Yang -2023/07/04. For more detail or traning data, feel free to contact: jzwang@stu.xjtu.edu.cn


***********************************************************************************************************
## Installation
This installation guide shows you how to set up the environment for running our code using conda.

First clone the D2MC-Net repository
```
git clone https://github.com/Jiazhen-Wang/D2MC-Net.git
cd D2MC-Net
```
Then start a virtual environment with new environment variables
```
conda create --name D2MC-Net python=3.8
conda activate D2MC-Net 
```
Install PyTorch 
```
pip install torch torchvision
```
Install all requirements
```
pip install -r requirements.txt
```

## Usage:
The net of training was implemented by end-to-end in D2MC-Net of pytorch version. For retraining you should simulate motion data and change the data path in train.py and saved in the logs_D2MCNet directory. 

Training:
```
python train.py
```
***********************************************************************************************************