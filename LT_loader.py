import numpy as np
import random

import torch
import torchvision
from torchvision import transforms
from PIL import Image
import torchvision.datasets
from torchvision.datasets import CIFAR10, CIFAR100

def corrupted_labels(targets, r=0.4, noise_type='sym'):
    transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}  # class transition for asymmetric noise
    size = int(len(targets) * r)
    idx = list(range(len(targets)))
    random.shuffle(idx)
    noise_idx = idx[:size]
    noisy_label = []
    for i in range(len(targets)):
        if i in noise_idx:
            if noise_type == 'sym':
                noisy_label.append(random.randint(0, 9))
            elif noise_type == 'asym':
                noisy_label.append(transition[targets[i]])
        else:
            noisy_label.append(targets[i])
    return np.array(noisy_label)


class IMB_CIFAR10_LT(CIFAR10):
    cls_num = 10

    def __init__(self, root, transform, imb_type='exp', imb_factor=0.01, noise_type='sym', r=0.4):
        super(IMB_CIFAR10_LT, self).__init__(root, download=True)
        np.random.seed(3407)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)
        self.noise_targets = corrupted_labels(self.targets, r, noise_type)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                print("(cls_idx / (cls_num - 1.0))", type((cls_idx / (cls_num - 1.0))))
                print("imb_factor", type(imb_factor))
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __getitem__(self, index):
        img, target, true_target = self.data[index], self.noise_targets[index], self.targets[index]
        img = self.data[index]
        
        img = Image.fromarray(img)
        
        img1 = self.transform(img)
        return img1, target, true_target, index


