# シンプルな超解像

## 準備 (Chainer version)
Chainer, OpenCV, NumPyを使います。

インストール：
```bash
$ sudo pip install chainer opencv-python numpy
```

## 準備 (PyTorch version)
PyTorch, Ignite, OpenCV, NumPyを使います。

インストール：  
PyTorch: [公式](https://pytorch.org/get-started/locally/)を参照してください。

```bash
$ sudo pip install pytorch-ignite opencv-python numpy
```

## 実行
```bash
$ python train-chainer.py
$ python train-pytorch.py
```

エポック数とGPU（使える場合）を指定できます。
```bash
$ python train-xx.py -e エポック数 -g GPUID
```

学習後、テストデータをモデルに入力した結果が画像として保存されます。
