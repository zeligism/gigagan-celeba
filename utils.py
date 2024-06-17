

import torch.nn as nn
import numpy as np
import os
import PIL
from typing import Any, Callable, List, Optional, Tuple, Union
from collections import defaultdict
import csv

import torch.nn.functional as F
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset
from torchvision.datasets import CelebA


def remove_leading_zeros(filename):
    i = 0
    while filename[i] == "0":
        i += 1
    return filename[i:]


def filter_none(mylist):
    return list(filter(lambda x: x is not None, mylist))


class CelebAHQ(CelebA):
    base_folder = "CelebAMask-HQ"
    img_folder_orig = "img_align_celeba"
    img_folder_hq = "CelebA-HQ-img"
    file_list = CelebA.file_list[1:]

    def __init__(self, min_occurences=1, *args, **kwargs) -> None:
        split_orig = "train" if "split" not in kwargs else kwargs["split"]
        kwargs["split"] = "all"  # do split manually to avoid losing index mapping ability
        super().__init__(*args, **kwargs)
        self.min_occurences = min_occurences

        # Mapping orig to hq
        filename = "CelebA-HQ-to-CelebA-mapping.txt"
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))
        # self.hq_to_orig = {int(row[0]): int(row[1]) for row in data[1:]}
        # num_data_orig = super().__len__()
        self.orig_to_hq = {int(row[1]): int(row[0]) for row in data[1:]}
        full_num_data_hq = len(self.orig_to_hq)

        # Count occurances of identity
        self.occurences = defaultdict(int)
        for celeb_id in self.identity:
            self.occurences[celeb_id.item()] += 1

        # create HQ version of each data field
        self.attr_hq = [None] * full_num_data_hq
        self.identity_hq = [None] * full_num_data_hq
        self.bbox_hq = [None] * full_num_data_hq
        self.landmarks_align_hq = [None] * full_num_data_hq
        self.filename_hq = [None] * full_num_data_hq
        for idx_orig, idx_hq in self.orig_to_hq.items():
            # skip IDs with occurences less than min_occurences
            if self.occurences[self.identity[idx_orig, 0].item()] < self.min_occurences:
                continue
            # map data from original indices to HQ indices
            self.attr_hq[idx_hq] = self.attr[idx_orig]
            self.identity_hq[idx_hq] = self.identity[idx_orig]
            self.bbox_hq[idx_hq] = self.bbox[idx_orig]
            self.landmarks_align_hq[idx_hq] = self.landmarks_align[idx_orig]
            # remove leading zeros from filename (HQ's dataset standard)
            self.filename_hq[idx_hq] = remove_leading_zeros(self.filename[idx_hq])

        # Filter None out and stack
        self.attr_hq = torch.stack(filter_none(self.attr_hq))
        self.identity_hq = torch.stack(filter_none(self.identity_hq))
        self.bbox_hq = torch.stack(filter_none(self.bbox_hq))
        self.landmarks_align_hq = torch.stack(filter_none(self.landmarks_align_hq))
        self.filename_hq = filter_none(self.filename_hq)

        # Split
        split_idx = round(0.8 * len(self.attr_hq))
        if split_orig == "train":
            mask = slice(split_idx)
        elif split_orig == "test":
            mask = slice(split_idx, -1)
        else:
            mask = slice(None)
        self.attr_hq = self.attr_hq[mask]
        self.identity_hq = self.identity_hq[mask]
        self.bbox_hq = self.bbox_hq[mask]
        self.landmarks_align_hq = self.landmarks_align_hq[mask]
        self.filename_hq = self.filename_hq[mask]

        self.unique_identities = list(set([x.item() for x in self.identity_hq]))
        self.num_ids = len(self.unique_identities)
        self.num_attrs = len(self.attr_hq[0])
        self.reordered_to_orig_identity = {}
        self.orig_to_reordered_identity = {}
        for reordered_id, orig_id in enumerate(sorted(list(map(int, self.unique_identities)))):
            self.orig_to_reordered_identity[orig_id] = reordered_id
        for i in range(len(self.identity_hq)):
            self.identity_hq[i, 0] = self.orig_to_reordered_identity[self.identity_hq[i, 0].item()]

        print(f"[CelebA-HQ]: Length = {self.__len__()}, num_ids = {self.num_ids}.")

    def _check_integrity(self) -> bool:
        return True  # XXX

    def __len__(self) -> int:
        return len(self.attr_hq)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = PIL.Image.open(os.path.join(self.root, self.base_folder, self.img_folder_hq, self.filename_hq[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr_hq[index, :])
            elif t == "identity":
                target.append(F.one_hot(self.identity_hq[index, 0], num_classes=self.num_ids).float())
            elif t == "bbox":
                target.append(self.bbox_hq[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align_hq[index, :])
            else:
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            image = self.transform(image)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return image, target


class PrivacyFaces(Dataset):
    def __init__(self, imgs, origin_idx, neighbor_idx, id_labels, transform=None):
        self.imgs = imgs
        self.origin_idx = origin_idx
        self.neighbor_idx = neighbor_idx
        self.transform = transform
        self.id_labels = id_labels.astype(int)

    def __len__(self):
        return len(self.origin_idx)

    def __getitem__(self, index):
        pair_imgs = self.imgs[index]
        img0 = pair_imgs[0]
        img1 = pair_imgs[1]
        #         img =torch.from_numpy(img).double()
        if self.transform:
            img0 = self.transform(img0).float()
            img1 = self.transform(img1).float()
        label0 = self.id_labels[self.origin_idx[index]]
        label1 = self.id_labels[self.neighbor_idx[index]]
        return img0, img1, label0, label1


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * torch.sum(b,dim=1)
        return torch.mean(b)


def load_data(img_size, max_id):
    root_dir = 'data/'
    imgs = np.load(root_dir + 'imgs%d_id%d.npy' % (img_size, max_id))
    origin_idx = np.load(root_dir + 'origin_idx_id%d.npy' % max_id)
    neighbor_idx = np.load(root_dir + 'neighbor_idx_id%d.npy' % max_id)
    sub_id_data = np.load(root_dir + 'sub_id_data.npy')
    return imgs, origin_idx, neighbor_idx, sub_id_data


@torch.no_grad()
def tensor_to_np(tensor):
    tensor = tensor.cpu().numpy() * 255 + 0.5
    ndarr = tensor.clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    if ndarr.shape[-1] == 1:
        ndarr = ndarr.squeeze(2)
    return ndarr


def D_criterion_hinge(D_real, D_fake):
    return torch.mean(F.relu(1. - D_real) + F.relu(1. + D_fake))


def G_criterion_hinge(D_fake):
    return -torch.mean(D_fake)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def update_G_progress(model, fixed_variables, celeba_hq=False, limit=10, device=None):
    # unpack fixed variables and take first `limit` values
    fixed_latents, fixed_data = fixed_variables
    fixed_zc, fixed_zs = fixed_latents
    fixed_zc = fixed_zc[:limit].to(device)
    fixed_zs = fixed_zs[:limit].to(device)
    # transfer to device
    if celeba_hq:
        fixed_x, (fixed_attr, fixed_y) = fixed_data
        fixed_x = fixed_x[:limit].to(device)
        fixed_y = fixed_y[:limit].to(device).long()
        fixed_attr = fixed_attr[:limit].to(device).float()
        rand_attr = torch.rand_like(fixed_attr).round()
    else:
        fixed_x, _, fixed_y, _ = fixed_data
        fixed_x = fixed_x[:limit].to(device)
        fixed_y = fixed_y[:limit].to(device).long()
        fixed_attr = None
        rand_attr = None
    # generate new output from fixed variables and contrast with random variables
    zc = model.sample_latent(limit, device=device)
    zs = model.sample_latent(limit, device=device)
    # model.eval()
    im_grid = torch.cat([
        model.generate(fixed_zc, fixed_zs, label=fixed_y, cond=fixed_attr, alpha=1.0),
        model.generate(fixed_zc, fixed_zs, label=fixed_y, cond=fixed_attr, alpha=0.5),
        model.generate(fixed_zc, fixed_zs, label=fixed_y, cond=fixed_attr, alpha=0.0),
        model.generate(fixed_zc, zs, label=fixed_y, cond=fixed_attr, alpha=0.0),
        model.generate(zc, fixed_zs, label=fixed_y, cond=fixed_attr, alpha=0.0),
        model.generate(zc, fixed_zs, label=None, cond=fixed_attr, alpha=0.0),
        model.generate(zc, fixed_zs, label=fixed_y, cond=rand_attr, alpha=0.0),
        model.generate(zc, fixed_zs, label=None, cond=rand_attr, alpha=0.0),
    ], dim=0)
    # model.train()
    im_grid = 0.5 * im_grid + 0.5  # inv_normalize to [0,1]
    grid = make_grid(im_grid, nrow=limit, padding=2).cpu()
    return grid


