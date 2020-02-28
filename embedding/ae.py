import chainer
import chainer.functions as F
import chainer.links as L


class AE(chainer.Chain):
    def __init__(self, size, in_channels, n_channels=8, ksize=4, stride=2):
        super(AE, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels, n_channels, ksize=ksize, stride=stride)
            self.deconv1 = L.Deconvolution2D(n_channels, in_channels, ksize=ksize, stride=stride, outsize=size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.deconv1(x)
        return x

    def embed(self, x):
        x = F.relu(self.conv1(x))
        return x.array
