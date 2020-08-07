import math

import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
from torch import nn
from torchtext.vocab import GloVe


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SemanticRelation(nn.Module):
    def __init__(self, cfg, dimension):
        super().__init__()
        self.eta = cfg.MODEL.SEMANTIC_RELATION.ETA
        self.gamma = cfg.MODEL.SEMANTIC_RELATION.GAMMA
        self.tau = cfg.MODEL.SEMANTIC_RELATION.TAU

        categories = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
        # generate A
        Y = torch.zeros((len(categories), len(categories)))
        Z = torch.zeros(len(categories))
        for dataset in cfg.DATASETS.TRAIN:
            for data in DatasetCatalog.get(dataset):
                # for each image to calculate the number of occurrences of label and pairs
                labels = set()
                for annotation in data['annotations']:
                    labels.add(annotation['category_id'])
                labels = list(labels)
                Z[labels] += 1
                for i in range(len(labels)):
                    for j in range(i + 1, len(labels)):
                        Y[labels[i], labels[j]] += 1
                        Y[labels[j], labels[i]] += 1
        A = Y / Z.unsqueeze(dim=-1)
        A.fill_diagonal_(1.0)
        A[A >= self.eta] = 1.0
        A[A < self.eta] = 0.0
        A.fill_diagonal_(0.0)
        A /= torch.sum(A, dim=-1, keepdim=True)
        A[torch.isnan(A)] = 0.0
        A.fill_diagonal_(1.0 - self.gamma)
        self.A = A.to(cfg.MODEL.DEVICE)
        # init vocabs
        vocabs = GloVe(name='6B')
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

        self.conv1 = GraphConvolution(self.word_embedding.size(-1), self.word_embedding.size(-1))
        self.conv2 = GraphConvolution(self.word_embedding.size(-1), dimension)
        self.fc1 = nn.Linear(self.word_embedding.size(0), int(dimension * self.tau))
        self.fc2 = nn.Linear(int(dimension * self.tau) + dimension, dimension)
        self.relu = nn.LeakyReLU()

    def forward(self, features):
        x = self.relu(self.conv1(self.word_embedding, self.A))
        x = self.relu(self.conv2(x, self.A))
        x = self.fc1(torch.relu(torch.mm(features, x.t())))
        x = torch.relu(self.fc2(torch.cat((x, features), dim=-1)))
        return x


class SpatialRelation(nn.Module):
    def __init__(self, cfg, input_shape):
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
        self.semantic_relation = SemanticRelation(cfg, cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * 8)
        self.spatial_relation = SpatialRelation(cfg, input_shape)

    def forward(self, images, features, proposals, targets=None):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

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
        semantic_features = self.semantic_relation(box_features)
        spatial_features = self.spatial_relation(box_features)
        box_features = torch.cat((semantic_features, spatial_features), dim=-1)
        predictions = self.box_predictor(box_features)

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
