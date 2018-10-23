# coding: utf-8

import argparse
import numpy as np
import cv2
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable, cuda
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class CNNAE(Chain):
	def __init__(self):
		super().__init__()
		with self.init_scope():
			N = 16
			kwds = {"ksize": 3, "stride": 1, "pad": 1, "nobias": True}
			self.conv1 = L.Convolution2D(1, N, **kwds)
			self.bn1 = L.BatchNormalization(N)
			self.conv2 = L.Convolution2D(N, N*2, **kwds)
			self.bn2 = L.BatchNormalization(N*2)
			self.conv3 = L.Convolution2D(N*2, N*4, **kwds)
			self.bn3 = L.BatchNormalization(N*4)
			self.conv4 = L.Convolution2D(N*4, 1, ksize=3, stride=1, pad=1)
			
	def forward(self, x):
		# 低画質画像をもとに戻す
		h = F.relu(self.bn1(self.conv1(x)))
		h = F.relu(self.bn2(self.conv2(h)))
		h = F.relu(self.bn3(self.conv3(h)))
		h = F.sigmoid(self.conv4(h))
		
		return h
		
	def __call__(self, x, t):
		# 高解像度化
		h = self.forward(x)
		# オリジナルとの誤差を算出
		loss = F.mean_squared_error(h, t)
		report({"loss": loss}, self)
		return loss

		
parser = argparse.ArgumentParser()
parser.add_argument("-e", type=int, default=5, help="epoch")
parser.add_argument("-g", type=int, default=-1, help="GPU ID (negative value indicates CPU mode)")
args = parser.parse_args()

# MNISTデータセットの取得
train, test = datasets.get_mnist(withlabel=False, ndim=3)

# 入力画像を作成する
# 7x7に縮小してからもとのサイズに拡大して低画質化
train_boke = F.resize_images(train, (10,10))
train_boke = F.resize_images(train_boke, (28,28)).array
test_boke = F.resize_images(test, (10,10))
test_boke = F.resize_images(test_boke, (28,28)).array

# 低画質画像とオリジナル画像のペアにする
train = datasets.TupleDataset(train_boke, train)
train_iter = iterators.SerialIterator(train, 64, shuffle=True, repeat=True)
test = datasets.TupleDataset(test_boke, test)
test_iter = iterators.SerialIterator(test, 64, shuffle=False, repeat=False)

chainer.config.user_gpu = (args.g >= 0)
if chainer.config.user_gpu:
	cuda.get_device_from_id(args.g).use()
	print("GPU mode")
else:
	print("CPU mode")

model = CNNAE()	# 超解像モデル作成
if chainer.config.user_gpu:
	model.to_gpu()
opt = optimizers.Adam()	# 最適化アルゴリズムとしてAdamを選択
opt.setup(model)

# 学習の準備
updater = training.updaters.StandardUpdater(train_iter, opt, device=args.g)
trainer = training.Trainer(updater, (args.e,"epoch"))
# テストの設定
evaluator = extensions.Evaluator(test_iter, model, device=args.g)
trainer.extend(evaluator)
# 学習経過の表示設定
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(["epoch", "main/loss", "validation/main/loss"]))
trainer.extend(extensions.ProgressBar())

# 学習開始
trainer.run()

chainer.config.train = False
if chainer.config.user_gpu:
	model.to_cpu()

# 学習終了後に実際にテスト画像を高解像度化してみる
indices = np.random.choice(len(test), 10).tolist()
test_samples = [test[i][0] for i in indices]
for i,img in enumerate(test_samples):
	img = np.expand_dims(img, axis=0)
	output = model.forward(img)
	
	# 画像として保存
	img = np.squeeze(img*255).astype(np.uint8)
	output = np.squeeze(output.array*255).astype(np.uint8)
	cv2.imwrite("input_{:02d}.png".format(i), img)
	cv2.imwrite("output_{:02d}.png".format(i), output)
