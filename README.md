## Focal Loss
This is the keras implementation of [focal loss](https://arxiv.org/abs/1708.02002) with the backend of tensorflow. The Focal Loss is proposed for dealing with `foreground-backgrou class` imbalance.
![论文中的结果图Fig1](https://github.com/Atomwh/FocalLoss_Keras/blob/master/images/fig1-focal%20loss%20results.png)

## Usage
Compile your model with focal loss as sample:
>model.compile(optimizer = Adam(lr = 1e-4), loss = [focal_loss(gamma=2,alpha=0.6)], metrics = ['accuracy'])

## Experiments
We implement [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) with Focal Loss and get the results of different parameters as follows:
![不同的图像fig2](https://github.com/Atomwh/FocalLoss_Keras/blob/master/images/fig2-differern%20result.png)
fig1

![曲线图fig3](https://github.com/Atomwh/FocalLoss_Keras/blob/master/images/fig3.png)

fig2

From fig2, we found the improvement of focal is not ideal. And different parameters have a greater impact.
## Breaks
We found that the Focal Loss is not stable and I think the main reason is parameters initialization. I wil try to fix it.

## Others
The implemented code is based [@zhixuhao' code](https://github.com/zhixuhao/unet) 
