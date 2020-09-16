import numpy as np
import torch
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from torch import nn

from .head import IDAUp, DLAUp, FocalLoss, L1Loss
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
        self.focal_beta = cfg.MODEL.CENTER_MASK.FOCAL_LOSS_BETA
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

        # losses
        self.focal_loss = FocalLoss(self.focal_alpha, self.focal_beta)
        self.l1_loss = L1Loss()

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
                head_output = torch.sigmoid(getattr(self, '{}_head'.format(head_name))(y[-1]))
                z[head_name] = head_output
            else:
                z[head_name] = getattr(self, '{}_head'.format(head_name))(y[-1])

        if self.training:
            losses = self.losses(z, instances)
            return losses
        else:
            results = self.inference(z, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
            ):
                original_height = input_per_image.get("height", image_size[0])
                original_width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, original_height, original_width)
                processed_results.append({"instances": r})
            return processed_results

    def preprocess_image(self, batched_inputs):
        images = [x['image'].to(self.device) / 255.0 for x in batched_inputs]
        if self.training:
            instances = [x['instances'].to(self.device) for x in batched_inputs]
        else:
            instances = []

        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 32)
        input_shape = np.array(images.tensor.shape[2:])
        output_shape = input_shape // self.down_ratio
        if not self.training:
            return images, []
        instances = [gen_heatmap(x, output_shape, self.num_classes, self.down_ratio) for x in instances]
        return images, instances

    def losses(self, outputs, instances):
        # get ground truth from batched_inputs
        gt_heatmap = torch.cat([x['heatmap'].unsqueeze(0).to(self.device) for x in instances], dim=0)
        gt_offset = torch.cat([x['wh_offset'].unsqueeze(0).to(self.device) for x in instances], dim=0)
        gt_size = torch.cat([x['wh_map'].unsqueeze(0).to(self.device) for x in instances], dim=0)
        gt_reg = torch.cat([x['reg_mask'].unsqueeze(0).to(self.device) for x in instances], dim=0)
        gt_ind = torch.cat([x['ind'].unsqueeze(0).to(self.device) for x in instances], dim=0)

        heatmap_loss = self.focal_loss(outputs['heatmap'], gt_heatmap)
        size_loss = self.l1_loss(outputs['size'], gt_size, gt_ind, gt_offset)
        offset_loss = self.l1_loss(outputs['offset'], gt_size, gt_ind, gt_reg)

        return {
            "heatmap_loss": heatmap_loss * self.lambda_p,
            "size_loss": size_loss * self.lambda_size,
            "offset_loss": offset_loss * self.lambda_off
        }

    def inference(self, outputs, image_sizes):
        """
        Inference on outputs.
        :param outputs: outputs of centermask
        :param image_sizes: 
        :return: 
        """"""results (List[Instances]): a list of #images elements.
        """
        results = []
        for img_idx, image_size in enumerate(image_sizes):
            output = {
                'hm': outputs['hm'][img_idx].unsqueeze(0),
                'wh': outputs['wh'][img_idx].unsqueeze(0),
                'reg': outputs['reg'][img_idx].unsqueeze(0)
            }
            results_per_image = self.inference_single_image(
                output, tuple(image_size)
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, output, image_size):
        """
        Inference on one image.
        :param output:
        :param image_size:
        :return:
        """

        # decode centernet output and keep top k top scoring indices.
        boxes_all, scores_all, class_idxs_all = ctdet_decode(output['hm'],
                                                             output['wh'],
                                                             reg=output['reg'],
                                                             down_ratio=self.down_ratio,
                                                             K=self.topk_candidates)

        # take max number of detections per image
        max_num_detections_per_image = min(self.max_detections_per_image, self.topk_candidates)
        scores_all = scores_all[:max_num_detections_per_image]
        boxes_all = boxes_all[:max_num_detections_per_image]
        class_idxs_all = class_idxs_all[:max_num_detections_per_image]

        # filter out by threshold
        keep_idxs = scores_all > self.score_threshold
        scores_all = scores_all[keep_idxs]
        boxes_all = boxes_all[keep_idxs]
        class_idxs_all = class_idxs_all[keep_idxs]
        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all)
        result.scores = scores_all
        result.pred_classes = class_idxs_all
        return result
