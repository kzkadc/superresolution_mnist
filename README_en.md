# Simple Superresolution

## Requirements (Chainer)
Uses Chainer, NumPy, OpenCV.

Installation:  
```bash
$ sudo pip install chainer opencv-python numpy
```

## Requirements (PyTorch)
Uses PyTorch, Ignite, Numpy, OpenCV.

Installation:  
PyTorch: see the [official document](https://pytorch.org/get-started/locally/).

```bash
$ sudo pip install pytorch-ignite opencv-python numpy
```

## Run

```bash
$ python train-chainer.py
$ python train-pytorch.py
```

You can specify number of epochs and GPU (if available).

```bash
$ python train-xx.py -e epochs -g GPUID
```

Outputs of the trained model w.r.t test data are saved after training.