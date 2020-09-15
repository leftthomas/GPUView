import numpy as np
import torch


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Calculate gaussian radius of a bounding box.
    :param det_size: bounding box size in (h, w)
    :param min_overlap:
    :return:
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gen_heatmap(instances, output_shape, num_classes):
    """
    Generate heatmap for centernet or cornernet, which use heatmap as backbone.
    :param instances: instances that used in detectron2
    :param output_shape: output of the model, say centernet is input / 4
    :param num_classes: dataset num_classes
    :return:
    """
    hm = np.zeros((num_classes, output_shape[0], output_shape[1]),
                  dtype=np.float32)
    wh = np.zeros((128, 2), dtype=np.float32)
    reg_mask = np.zeros(128, dtype=np.uint8)
    reg = np.zeros((128, 2), dtype=np.float32)
    ind = np.zeros(128, dtype=np.int64)
    num_objs = instances.gt_classes.shape[0]
    num_objs = min(num_objs, 128)
    for k in range(num_objs):
        bbox = instances.gt_boxes.tensor[k] / 4
        cls_id = instances.gt_classes[k]
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if h > 0 and w > 0:
            import math
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            ct = np.array(
                [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm[cls_id], ct_int, radius)
            wh[k] = 1. * w, 1. * h
            ind[k] = ct_int[1] * output_shape[1] + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1
    instance_dict = {'hm': torch.tensor(hm), 'reg_mask': torch.tensor(reg_mask), 'ind': torch.tensor(ind),
                     'wh': torch.tensor(wh), 'reg': torch.tensor(reg)}

    return instance_dict
