import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import numpy as np
from PIL import Image

uses_device = -1  # GPU#0を使用,CPUの場合-1

# GPU使用時とCPU使用時でデータ形式が変わる
if uses_device >= 0:
    import cupy as cp
else:
    cp = np


class NMIST_Conv_NN(chainer.Chain):

    def __init__(self):
        super(NMIST_Conv_NN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 8, ksize=3)  # フィルタサイズ＝3で出力数8
            self.linear1 = L.Linear(1352, 10)  # 出力数10

    def __call__(self, x, t=None, train=True):
        # 畳み込みニューラルネットワークによる画像認識
        h1 = self.conv1(x)  # 畳み込み層
        h2 = F.relu(h1)  # 活性化関数
        h3 = F.max_pooling_2d(h2, 2)  # プーリング層
        h4 = self.linear1(h3)  # 全結合層
        # 損失か結果を返す
        return F.softmax_cross_entropy(h4, t) if train else F.softmax(h4)


# ニューラルネットワークを定義
model = NMIST_Conv_NN()

if uses_device >= 0:
    # GPUを使う
    chainer.cuda.get_device_from_id(0).use()
    chainer.cuda.check_cuda_available()
    # GPU用データ形式に変換
    model.to_gpu()

# 学習結果を読み込む
chainer.serializers.load_hdf5('chapt02.hdf5', model)

# 画像を読み込む
image = Image.open('test/mnist-0.png').convert('L')
# ニューラルネットワークの入力に合わせて成形する
pixels = cp.asarray(image).astype(cp.float32).reshape(1, 1, 28, 28)
pixels = pixels / 255

# ニューラルネットワークを実行する
result = model(pixels, train=False)
# 実行結果を表示する
for i in range(len(result.data[0])):
    print(str(i) + '\t' + str(result.data[0][i]))
