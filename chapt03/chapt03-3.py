import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import numpy as np
import os
import math
import sys
from PIL import Image
from PIL import ImageDraw

uses_device = 0  # GPU#0を使用

# GPU使用時とCPU使用時でデータ形式が変わる
if uses_device >= 0:
    import cupy as cp
    import chainer.cuda
else:
    cp = np


class SuperResolution_NN(chainer.Chain):

    def __init__(self):
        # 重みデータの初期値を指定する
        w1 = chainer.initializers.Normal(scale=0.0378, dtype=None)
        w2 = chainer.initializers.Normal(scale=0.3536, dtype=None)
        w3 = chainer.initializers.Normal(scale=0.1179, dtype=None)
        w4 = chainer.initializers.Normal(scale=0.189, dtype=None)
        w5 = chainer.initializers.Normal(scale=0.0001, dtype=None)
        super(SuperResolution_NN, self).__init__()
        # 全ての層を定義する
        with self.init_scope():
            self.c1 = L.Convolution2D(1, 56, ksize=5, stride=1, pad=0, initialW=w1)
            self.l1 = L.PReLU()
            self.c2 = L.Convolution2D(56, 12, ksize=1, stride=1, pad=0, initialW=w2)
            self.l2 = L.PReLU()
            self.c3 = L.Convolution2D(12, 12, ksize=3, stride=1, pad=1, initialW=w3)
            self.l3 = L.PReLU()
            self.c4 = L.Convolution2D(12, 12, ksize=3, stride=1, pad=1, initialW=w3)
            self.l4 = L.PReLU()
            self.c5 = L.Convolution2D(12, 12, ksize=3, stride=1, pad=1, initialW=w3)
            self.l5 = L.PReLU()
            self.c6 = L.Convolution2D(12, 12, ksize=3, stride=1, pad=1, initialW=w3)
            self.l6 = L.PReLU()
            self.c7 = L.Convolution2D(12, 56, ksize=1, stride=1, pad=1, initialW=w4)
            self.l7 = L.PReLU()
            self.c8 = L.Deconvolution2D(56, 1, ksize=9, stride=3, pad=4, initialW=w5)

    def __call__(self, x, t=None, train=True):
        h1 = self.l1(self.c1(x))
        h2 = self.l2(self.c2(h1))
        h3 = self.l3(self.c3(h2))
        h4 = self.l4(self.c4(h3))
        h5 = self.l5(self.c5(h4))
        h6 = self.l6(self.c6(h5))
        h7 = self.l7(self.c7(h6))
        h8 = self.c8(h7)
        # 損失か結果を返す
        return F.mean_squared_error(h8, t) if train else h8


# ニューラルネットワークを作成
model = SuperResolution_NN()

# 学習結果を読み込む
chainer.serializers.load_hdf5('chapt03.hdf5', model)

if uses_device >= 0:
    # GPUを使う
    chainer.cuda.get_device_from_id(0).use()
    chainer.cuda.check_cuda_available()
    # GPU用データ形式に変換
    model.to_gpu()

# 入力ファイル
in_file = 'test.png'
if len(sys.argv) >= 2:
    in_file = str(sys.argv[1])

# 出力ファイル
dest_file = 'dest.png'
if len(sys.argv) >= 3:
    dest_file = str(sys.argv[2])

# 入力画像を開く
img = Image.open(in_file).convert('YCbCr')

# 画像サイズが16ピクセルの倍数でない場合、16ピクセルの倍数にする
org_w = w = img.size[0]
org_h = h = img.size[1]
if w % 16 != 0:
    w = (math.floor(w / 16) + 1) * 16
if h % 16 != 0:
    h = (math.floor(h / 16) + 1) * 16
if w != img.size[0] or h != img.size[1]:
    img = img.resize((w, h))

# 出力画像
dst = Image.new('YCbCr', (10 * w // 4, 10 * h // 4), 'white')

# 入力画像を分割
cur_x = 0
while cur_x <= img.size[0] - 16:
    cur_y = 0
    while cur_y <= img.size[1] - 16:
        # 画像から切りだし
        rect = (cur_x, cur_y, cur_x + 16, cur_y + 16)
        cropimg = img.crop(rect)
        # YCbCrのY画素のみを使う
        hpix = cp.array(cropimg, dtype=cp.float32)
        hpix = hpix[:, :, 0] / 255
        x = cp.array([[hpix]], dtype=cp.float32)
        # 超解像を実行
        t = model(x, train=False)
        # YCbCrのCbCrはBICUBICで拡大
        dstimg = cropimg.resize((40, 40), Image.BICUBIC)
        hpix = np.array(dstimg, dtype=np.float32)
        # YCbCrのY画素をコピー
        hpix.flags.writeable = True
        if uses_device >= 0:
            hpix[:, :, 0] = chainer.cuda.to_cpu(t.data[0]) * 255
        else:
            hpix[:, :, 0] = t.data[0] * 255
        # 画像を結果に配置
        bytes = np.array(hpix.clip(0, 255), dtype=np.uint8)
        himg = Image.fromarray(bytes, 'YCbCr')
        dst.paste(himg, (10 * cur_x // 4, 10 * cur_y // 4, 10 * cur_x // 4 + 40, 10 * cur_y // 4 + 40))
        # 次の切りだし場所へ
        cur_y += 16
    cur_x += 16

# 結果を保存する
dst = dst.convert('RGB')
dst.save(dest_file)
