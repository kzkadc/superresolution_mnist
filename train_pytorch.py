# coding: utf-8

import argparse

import numpy as np
import cv2

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torchvision.datasets import MNIST
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite import metrics


class CNNAE(nn.Module):
    def __init__(self):
        super().__init__()
        N = 16
        kwds = {"kernel_size": 3, "stride": 1, "padding": 1, "bias": False}
        self.conv1 = nn.Conv2d(1, N, **kwds)
        self.bn1 = nn.BatchNorm2d(N)
        self.conv2 = nn.Conv2d(N, N * 2, **kwds)
        self.bn2 = nn.BatchNorm2d(N * 2)
        self.conv3 = nn.Conv2d(N * 2, N * 4, **kwds)
        self.bn3 = nn.BatchNorm2d(N * 4)
        self.conv4 = nn.Conv2d(N * 4, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = torch.sigmoid(self.conv4(h))
        return h


class SuperresolutionDataset(data.Dataset):
    # 低画質画像とオリジナル画像のペアにしたデータセット
    def __init__(self, train: bool = True):
        # MNISTデータセットの取得
        self.mnist = MNIST(root=".", download=True, train=train,
                           transform=lambda x: np.asarray(x, dtype=np.float32) / 255)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, i):
        img_orig, _ = self.mnist[i]

        # 入力画像を作成する
        # 10x10に縮小してからもとのサイズに拡大して低画質化
        img_lowres = cv2.resize(img_orig, (10, 10))
        img_lowres = cv2.resize(img_lowres, (28, 28))

        img_orig = np.expand_dims(img_orig, 0)
        img_lowres = np.expand_dims(img_lowres, 0)

        return img_lowres, img_orig


def evaluate(evaluator, val_loader):
    def _evaluate(engine):
        evaluator.run(val_loader)
        print(f"epoch {engine.state.epoch:d}, mse_loss: {evaluator.state.metrics['mse_loss']:f}")

    return _evaluate


parser = argparse.ArgumentParser()
parser.add_argument("-e", type=int, default=5, help="epoch")
parser.add_argument("-g", type=int, default=-1, help="GPU ID (negative value indicates CPU mode)")
args = parser.parse_args()

if args.g >= 0 and torch.cuda.is_available():
    device = torch.device(f"cuda:{args.g:d}")
    print(f"GPU mode: {args.g:d}")
else:
    device = torch.device("cpu")
    print("CPU mode")

train_dataset = SuperresolutionDataset(train=True)
train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = SuperresolutionDataset(train=False)
val_loader = data.DataLoader(val_dataset, batch_size=64, shuffle=False)

model = CNNAE().to(device)
opt = torch.optim.Adam(model.parameters())

trainer = create_supervised_trainer(model, opt, F.mse_loss, device=device)

metrics = {"mse_loss": metrics.Loss(F.mse_loss)}
evaluator = create_supervised_evaluator(model, metrics, device=device)

trainer.add_event_handler(Events.EPOCH_COMPLETED, evaluate(evaluator, val_loader))

trainer.run(train_loader, max_epochs=args.e)

# 学習終了後に実際にテスト画像を高解像度化してみる
model.eval()

n_samples = 10
indices = np.random.choice(len(val_dataset), size=n_samples, replace=False)
test_inputs = np.stack([val_dataset[i][0] for i in indices])
test_inputs = torch.from_numpy(test_inputs).to(device)
with torch.no_grad():
    outputs = model(test_inputs)
outputs = outputs.cpu().detach().numpy() * 255
outputs = outputs.squeeze().astype(np.uint8)
test_inputs = test_inputs.cpu().numpy() * 255
test_inputs = test_inputs.squeeze().astype(np.uint8)

# 入出力を左右に並べて出力
output_images = np.concatenate([test_inputs, outputs], axis=2)
for i, img in enumerate(output_images):
    cv2.imwrite(f"test_img_{i:02d}.png", img)
