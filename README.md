## Dynamic Feature Integration for Simultaneous Detection of Salient Object, Edge and Skeleton

### This is a demo PyTorch implementation of our IEEE TIP 2020 [paper](http://mftp.mmcheng.net/Papers/20TIP-DFI.pdf).
### We also provide an [Online Demo](http://mc.nankai.edu.cn/dfi).

<p align="center">
  <img src="https://github.com/backseason/DFI/blob/master/demo/demo.gif" alt="animated" />
</p>

## Prerequisites

- [Pytorch 0.4.1+](http://pytorch.org/)
- [torchvision](http://pytorch.org/)
- [opencv](https://opencv.org/)


## Demo usage
### 1. Clone the repository
```shell
git clone https://github.com/backseason/DFI.git
cd DFI/
```

### 2. Download the pretrained model 
`dfi.pth` [GoogleDrive](https://drive.google.com/file/d/1N29cJghKKJOHbKgpwR2_Ui64umCE-XG3/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1WPQiUPo7t8REK3LtmG9_KA) (pwd: **wkeb**)
and move it to the `pretrained` folder.


### 3. Test (demo)
The source images are in the `demo/images` folder.
By running 
```shell
python main.py
```
you'll get the predictions under
the `demo/predictions` folder. The predictions of all the three tasks are performed simultaneously.


## Pre-computed results and evaluation results

You can find the pre-computed predictions maps of all the three tasks and 
their corresponding evaluation scores with
the following link:
`Results reported in the paper` [GoogleDrive](https://drive.google.com/file/d/17SBs3v3h_FnImbHOZk0zy4JzDUHSK1zv/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1WP3WP5oaNWRuaUcKH4oZ7g) (pwd: **7eg3**)

## Contact
If you have any questions, feel free to contact me via: `j04.liu(at)gmail.com`.

### If you think this work is helpful, please cite
```latex
@article{liu2020dynamic,
  title={Dynamic Feature Integration for Simultaneous Detection of Salient Object, Edge and Skeleton},
  author={Jiang-Jiang Liu and Qibin Hou and Ming-Ming Cheng},
  journal={IEEE Transactions on Image Processing},
  year={2020},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TIP.2020.3017352},
}
```
