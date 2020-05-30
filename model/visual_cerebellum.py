import chainer
from chainer import serializers

from model.vgg import VGG11
from model.cerebellum import Cerebellum


class VisualCerebellum(chainer.Chain):
    def __init__(self, in_size, out_size, args):
        super(VisualCerebellum, self).__init__()

        with self.init_scope():
            # visual model
            self.visual = VGG11()
            # cerebellum model
            self.cerebellum = Cerebellum(in_size, out_size, args)

    def forward(self, x, attack=False):
        batch_size = x.shape[0]
        if not attack:
            with chainer.no_backprop_mode():
                h = self.visual.project(x)
        else:
            h = self.visual.project(x)
        h = self.cerebellum.forward(h.reshape(batch_size, -1), attack=attack)
        return h

    def init_visual(self, visual_dir='./pretrain/vgg11_cifar10.pkl'):
        serializers.load_npz(visual_dir, self.visual)
        print("initialized: " + visual_dir)
