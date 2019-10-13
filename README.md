FC<sup>2</sup>N: Fully Channel-Concatenated Network for Single Image Super-Resolution (under modification)
================
![pic](./pictures/FC2N.png)
e-print: <a href="https://arxiv.org/abs/1907.03221?context=eess.IV">https://arxiv.org/abs/1907.03221?context=eess.IV</a>

Xiaole Zhao, Ying Liao, Ye Li, Tao Zhang, and Xueming Zou

Introduction
--------
Most current image super-resolution (SR) methods based on deep convolutional neural networks (CNNs) use residual learning in network structural design, which contributes to effective back propagation, thus improving SR performance by increasing model scale. However, deep residual network suffers some redundancy in model representational capacity by introducing short paths and therefore cannot make full mining of model capacity. In addition, blindly enlarging the model scale will cause more problems in model training, even with residual learning. We present a novel CNN structure to make full use of model representational capacity, i.e., fully channel-concatenated network (FC<sup>2</sup>N)， whose key structural design is the **weighted channel-concatenation**. To our best knowledge, FC<sup>2</sup>N is the first CNN-based SR model that **does not use residual learning and reaches network depth over 400 layers**, and the first CNN model that **achieves state-of-the-art SR performance with less than 10M model parameters**. Moreover, it shows good SR performance in both large-scale and small-scale implementations.

Dependencies
--------
* TensorFlow >= 1.9.0
* Python >= 3.5
* Numpy >= 1.15.0
* Imageio 2.3.0
* Termcolor


Pretrained models
--------
Some pretrained models are available at the following link. Note these results may be slightly different from those in the paper. 

[https://zxlation.github.io/xiaole.github.com/](https://zxlation.github.io/xiaole.github.com/ "This is my homepage.")


Model training
--------




Testing
--------




Examples of super-resolution
--------
![example1](./pictures/exp1.png)
![example2](./pictures/exp2.png)


Citations
--------

```citation
    @article{Zhao2019FC2N,
        title   = {FC2N: Fully channel-concatenated network for single image super-resolution},
        author  = {Zhao, Xiaole and Liao, Ying and Li, Ye and Zhang, Tao and Zou, Xueming},
        journal = {arXiv preprint arXiv:1907.03221?context=eess.IV},
        year    = {2019}
    }
```



