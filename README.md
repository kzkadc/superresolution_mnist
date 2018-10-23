# ディープラーニングで簡単な超解像をやってみた
[Qiitaに書いた記事](https://qiita.com/kzkadc/items/321f2196bfd4fe0f4132)のコードです。 

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
