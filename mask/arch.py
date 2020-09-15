import math

import numpy as np
import torch
from detectron2.layers import DeformConv
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from torch import nn


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConvV2(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConvV2, self).__init__()
        self.actf = nn.Sequential(nn.BatchNorm2d(cho), nn.ReLU(inplace=True))
        self.conv = DeformConv(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)
        nn.init.uniform_(self.actf[0].weight.data)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConvV2(c, o)
            node = DeformConvV2(o, o)
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, padding=f // 2, output_padding=0, groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i), IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


@META_ARCH_REGISTRY.register()
class CenterMask(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_classes = cfg.MODEL.CENTER_MASK.NUM_CLASSES
        self.down_ratio = cfg.MODEL.CENTER_MASK.DOWN_RATIO
        self.last_level = cfg.MODEL.CENTER_MASK.LAST_LEVEL
        self.head_conv = cfg.MODEL.CENTER_MASK.HEAD_CONV
        self.s = cfg.MODEL.CENTER_MASK.S
        self.norm = cfg.MODEL.CENTER_MASK.NORM
        self.focal_alpha = cfg.MODEL.CENTER_MASK.FOCAL_LOSS_ALPHA
        self.focal_gamma = cfg.MODEL.CENTER_MASK.FOCAL_LOSS_GAMMA
        self.lambda_p = cfg.MODEL.CENTER_MASK.LAMBDA_P
        self.lambda_off = cfg.MODEL.CENTER_MASK.LAMBDA_OFF
        self.lambda_size = cfg.MODEL.CENTER_MASK.LAMBDA_SIZE
        self.lambda_mask = cfg.MODEL.CENTER_MASK.LAMBDA_MASK
        self.score_threshold = cfg.MODEL.CENTER_MASK.SCORE_THRESH_TEST
        self.mask_threshold = cfg.MODEL.CENTER_MASK.MASK_THRESH_TEST
        self.top_k_candidates = cfg.MODEL.CENTER_MASK.TOP_K_CANDIDATES_TEST

        # backbone
        self.backbone = build_backbone(cfg)
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        # up sample
        channels = self.backbone.channels
        self.first_level = int(np.log2(self.down_ratio))
        out_channel = channels[self.first_level]
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)
        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])

        # heads
        for head_name, num_channel in [('saliency', self.num_classes), ('shape', self.s ** 2), ('size', 2),
                                       ('heatmap', self.num_classes), ('offset', 2)]:
            fc = nn.Sequential(
                nn.Conv2d(out_channel, self.head_conv, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.head_conv, num_channel, kernel_size=1, stride=1, padding=0, bias=True))
            if head_name == 'heatmap':
                fc[-1].bias.data.fill_(-2.19)
            setattr(self, '{}_head'.format(head_name), fc)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        size = images.tensor.size()[-2:]
        features = self.backbone(images.tensor)

        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
        else:
            targets = None
        results, losses = self.sem_seg_head(features, size, targets)

        if self.training:
            return losses

        processed_results = []
        for result, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = sem_seg_postprocess(result, image_size, height, width)
            processed_results.append({"sem_seg": r})
        return processed_results
