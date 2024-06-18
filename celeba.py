

import os
import PIL
from typing import Any, Tuple
from collections import defaultdict
import csv

import torch
import torch.nn.functional as F
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
