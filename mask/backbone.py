import math

import torch
from detectron2.layers.batch_norm import get_norm
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, cfg, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = get_norm(cfg.MODEL.CENTER_MASK.NORM, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = get_norm(cfg.MODEL.CENTER_MASK.NORM, planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, cfg, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, bias=False,
                              padding=(kernel_size - 1) // 2)
        self.bn = get_norm(cfg.MODEL.CENTER_MASK.NORM, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, cfg, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(cfg, in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(cfg, out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(cfg, levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(cfg, levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(cfg, root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                get_norm(cfg.MODEL.CENTER_MASK.NORM, out_channels)
            )

    def forward(self, x, residual=None, children=None):
        if self.training and residual is not None:
            x = x + residual.sum() * 0.0
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(Backbone):
    def __init__(self, cfg, levels, channels, block=BasicBlock, residual_root=False):
        super(DLA, self).__init__()
        self.cfg = cfg
        self.channels = channels

        self._out_features = ["level{}".format(i) for i in range(6)]
        self._out_feature_channels = {k: channels[i] for i, k in enumerate(self._out_features)}
        self._out_feature_strides = {k: 2 ** i for i, k in enumerate(self._out_features)}

        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            get_norm(cfg.MODEL.CENTER_MASK.NORM, channels[0]),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(cfg, levels[2], block, channels[1], channels[2], 2,
                           level_root=False, root_residual=residual_root)
        self.level3 = Tree(cfg, levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(cfg, levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(cfg, levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                get_norm(self.cfg.MODEL.CENTER_MASK.NORM, planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.base_layer(x)
        y = {}
        for i in range(6):
            name = 'level{}'.format(i)
            x = getattr(self, name)(x)
            y[name] = x
        return y


@BACKBONE_REGISTRY.register()
def dla34(cfg, input_shape):
    model = DLA(cfg, [1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], block=BasicBlock)
    return model
