import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from bidict import bidict
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.backends import cudnn
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm


def parse_common_args():
    # for reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    cudnn.deterministic = True
    cudnn.benchmark = False
    # common args
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--data_root', default='data', type=str, help='Datasets root path')
    parser.add_argument('--data_name', default='pacs', type=str, choices=['pacs', 'office_31', 'office_home'],
                        help='Dataset name')
    parser.add_argument('--method_name', default='zsco', type=str,
                        choices=['zsco', 'simsiam', 'simclr', 'npid', 'proxyanchor', 'softtriple'],
                        help='Compared method name')
    parser.add_argument('--hidden_dim', default=512, type=int, help='Hidden feature dim for projection head')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=32, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--total_iter', default=10000, type=int, help='Number of bp to train')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')
    return parser


def obtain_style_code(style_num, size):
    code = F.one_hot(torch.arange(style_num), num_classes=style_num)
    style = code.view(*code.size(), 1, 1).expand(*code.size(), *size)
    return style


class AddStyleCode(torch.nn.Module):
    def __init__(self, style_num=0):
        super().__init__()
        self.style_num = style_num

    def forward(self, tensor):
        if self.style_num != 0:
            tensor = tensor.unsqueeze(dim=0).expand(self.style_num, *tensor.size())
            styles = obtain_style_code(self.style_num, tensor.size()[-2:])
            tensor = torch.cat((tensor, styles), dim=1)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(style_num={0})'.format(self.style_num)


def get_transform(data_type='train', style_num=0):
    if data_type == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, (1.0, 1.14)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            AddStyleCode(style_num)])
    else:
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            AddStyleCode(style_num)])


class DomainDataset(Dataset):
    def __init__(self, data_root, data_name, data_type='train', style_num=0):
        super(DomainDataset, self).__init__()

        self.data_name = data_name
        # which image
        self.images = sorted(glob.glob(os.path.join(data_root, data_name, '*', data_type, '*', '*.jpg')))
        self.categories, self.labels, self.domains, self.classes, i, j = [], [], {}, {}, 0, 0
        for image in self.images:
            domain = os.path.dirname(image).split('/')[-3]
            if domain not in self.domains:
                self.domains[domain] = i
                i += 1
            # which domain
            self.categories.append(self.domains[domain])

            label = os.path.dirname(image).split('/')[-1]
            if label not in self.classes:
                self.classes[label] = j
                j += 1
            # which label
            self.labels.append(self.classes[label])

        self.domains = bidict(self.domains)
        self.classes = bidict(self.classes)
        self.transform = get_transform(data_type, style_num)

    def __getitem__(self, index):
        img_name = self.images[index]
        img = Image.open(img_name)
        img_1 = self.transform(img)
        img_2 = self.transform(img)
        category = self.categories[index]
        label = self.labels[index]
        return img_1, img_2, img_name, category, label, index

    def __len__(self):
        return len(self.images)

    def refresh(self, style_num):
        images, names, categories, labels = [], [], [], []
        # need reverse del index to avoid the del index not exist error
        indexes = sorted(random.sample(range(0, len(self.images)), k=style_num), reverse=True)
        for i in indexes:
            name = self.images.pop(i)
            names.append(name)
            images.append(Image.open(name))
            categories.append(self.categories.pop(i))
            labels.append(self.labels.pop(i))
        return images, names, categories, labels


def compute_map(vectors, domains, categories, labels):
    computer = AccuracyCalculator(include=['mean_average_precision'])
    domain_vectors, domain_labels, acc, value, num = [], [], {}, 0.0, 0
    for i, domain in enumerate(domains):
        domain_vectors.append(vectors[torch.as_tensor(categories) == i])
        domain_labels.append(torch.as_tensor(labels, device=vectors.device)[torch.as_tensor(categories) == i])
    for i in range(len(domain_vectors)):
        for j in range(i + 1, len(domain_vectors)):
            domain_a_vectors = domain_vectors[i]
            domain_b_vectors = domain_vectors[j]
            domain_a_labels = domain_labels[i]
            domain_b_labels = domain_labels[j]
            # A -> B
            map_ab = computer.get_accuracy(domain_a_vectors, domain_b_vectors, domain_a_labels, domain_b_labels, False)
            # B -> A
            map_ba = computer.get_accuracy(domain_b_vectors, domain_a_vectors, domain_b_labels, domain_a_labels, False)

            acc['{}->{}'.format(domains[i], domains[j])] = map_ab['mean_average_precision']
            acc['{}->{}'.format(domains[j], domains[i])] = map_ba['mean_average_precision']
            value += map_ab['mean_average_precision']
            value += map_ba['mean_average_precision']
            num += 2
    # the mean map is chosen as the representative of precise
    acc['val_precise'] = value / num
    return acc


# val for all val data
def val_contrast(net, data_loader, results, current_iter, total_iter):
    net.eval()
    vectors = []
    with torch.no_grad():
        for data, _, _, _, _, _ in tqdm(data_loader, desc='Feature extracting', dynamic_ncols=True):
            vectors.append(net(data.cuda())[0])
        vectors = torch.cat(vectors, dim=0)
        acc = compute_map(vectors, sorted(data_loader.dataset.domains.keys()), data_loader.dataset.categories,
                          data_loader.dataset.labels)
        precise = acc['val_precise'] * 100
        print('Val Iter: [{}/{}] Precise: {:.2f}%'.format(current_iter, total_iter, precise))
        for key, value in acc.items():
            if key in results:
                results[key].append(value * 100)
            else:
                results[key] = [value * 100]
    net.train()
    return precise, vectors


class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.detach().cpu():
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.0)
