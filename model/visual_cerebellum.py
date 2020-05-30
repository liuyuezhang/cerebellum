import chainer
from chainer import serializers


from model.resnet import ResNet50
from model.cerebellum import Cerebellum


class VisualCerebellum(chainer.Chain):
    def __init__(self, in_size, out_size, args):
        super(VisualCerebellum, self).__init__()

        with self.init_scope():
            # visual model
            self.visual = ResNet50()
            # cerebellum model
            self.cerebellum = Cerebellum(in_size, out_size, args)

    def forward(self, x, attack=False):
        if not attack:
            with chainer.no_backprop_mode():
                h = self.visual.project(x)
        else:
            h = self.visual.project(x)
        h = self.cerebellum.forward(h, attack=attack)
        return h

    def init_visual(self, visual_dir='./pretrain/resnet50_cifar10.pkl'):
        serializers.load_npz(visual_dir, self.visual)
