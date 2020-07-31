import torch
from detectron2.data import MetadataCatalog
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
from torch import nn
from torchtext.vocab import GloVe


class SemanticRelation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.eta = cfg.MODEL.SEMANTIC_RELATION.ETA
        self.gamma = cfg.MODEL.SEMANTIC_RELATION.GAMMA
        self.tau = cfg.MODEL.SEMANTIC_RELATION.TAU
        # init vocabs
        vocabs = GloVe(name='6B')
        categories = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
        self.word_embedding = []
        # process missing words
        for category in categories:
            if ' ' in category:
                tokens = category.split(' ')
            else:
                if category == 'diningtable':
                    tokens = ['dining', 'table']
                elif category == 'pottedplant':
                    tokens = ['potted', 'plant']
                elif category == 'tvmonitor':
                    tokens = ['tv', 'monitor']
                else:
                    tokens = category
            embeddings = vocabs.get_vecs_by_tokens(tokens)
            if embeddings.size(0) == 2:
                embeddings = embeddings.mean(dim=0)
            self.word_embedding.append(embeddings)
        self.word_embedding = torch.stack(self.word_embedding, dim=0).to(cfg.MODEL.DEVICE)

    def forward(self, features, classes):
        nodes = [self.word_embedding[index] for index in classes]
        return nodes


class SpatialRelation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.k = cfg.MODEL.SPATIAL_RELATION.K

    def forward(self, x):
        return x


@ROI_HEADS_REGISTRY.register()
class RelationROIHeads(Res5ROIHeads):
    """
    The RelationROIHeads in a typical "C4" R-CNN model, where
    the box head uses the cropping and the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.semantic = SemanticRelation(cfg)
        self.spatial = SpatialRelation(cfg)

    def forward(self, images, features, proposals, targets=None):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        gt_classes = [x.gt_classes for x in targets]
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        box_features = box_features.mean(dim=[2, 3])
        # obtain relations
        semantic_relation = self.semantic(box_features, gt_classes)
        spatial_relation = self.spatial(box_features)
        box_features = torch.cat((semantic_relation, spatial_relation), dim=-1)
        predictions = self.box_predictor(box_features)

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
