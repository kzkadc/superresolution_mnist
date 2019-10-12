# シンプルな超解像

## 準備
Chainer, OpenCV, NumPyを使います。

インストール：
```bash
$ sudo pip install chainer opencv-python numpy
```

## 実行
```bash
$ python train.py
```

エポック数とGPU（使える場合）を指定できます。
```bash
$ python train.py -e エポック数 -g GPUID
```

学習後、テストデータをモデルに入力した結果が画像として保存されます。
