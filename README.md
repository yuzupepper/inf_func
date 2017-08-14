# inf_func
implementation of influence function
https://arxiv.org/abs/1703.04730

Requirements
* chianer 1.24.0
Performance using GPU is not tested.

# How to use

1. train a logistic regression model

```
python train_model.py
```
It outputs result/model.npz.

2. calculate influence function and output influential example

```
python inf_func.py -m result/model.npz [--dont_log]
```

It outputs result/model.npz_image.png.

result/model.npz_image.png
