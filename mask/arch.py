import numpy as np
import torch
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from torch import nn

from .head import IDAUp, DLAUp
from .utils import gen_heatmap


@META_ARCH_REGISTRY.register()
class CenterMask(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # configs
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
        self.head_names = ['saliency', 'shape', 'size', 'heatmap', 'offset']
        for head_name, num_channel in zip(self.head_names, [self.num_classes, self.s ** 2, 2, self.num_classes, 2]):
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
        images, instances = self.preprocess_image(batched_inputs)
        x = self.backbone(images.tensor)
        x = self.dla_up(x)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        z = {}
        for head_name in self.head_names:
            if head_name == 'heatmap':
                head_output = torch.clamp(torch.sigmoid(getattr(self, '{}_head'.format(head_name))(y[-1])), min=1e-4,
                                          max=1 - 1e-4)
                z[head_name] = head_output
            else:
                z[head_name] = getattr(self, '{}_head'.format(head_name))(y[-1])

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

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) / 255.0 for x in batched_inputs]
        if self.training:
            instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            instances = []

        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 32)
        input_shape = np.array(images.tensor.shape[2:])
        output_shape = input_shape // self.down_ratio
        if not self.training:
            return images, []
        instances = [gen_heatmap(x, output_shape, self.num_classes) for x in instances]
        return images, instances
