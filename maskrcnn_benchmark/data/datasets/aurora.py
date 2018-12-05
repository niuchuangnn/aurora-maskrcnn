# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import torch.utils.data as data
from PIL import Image
import os
from pycocotools.coco import COCO

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

class AuroraDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None):

        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs([img_id])[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class AuroraDataset(AuroraDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(AuroraDataset, self).__init__(root, ann_file)

        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)) > 0
            ]

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(AuroraDataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        is_show = False
        if is_show:
            img_show = img.numpy()
            img_show += 36.6475
            img_show = np.transpose(img_show, [1, 2, 0]) / 255.0

            plt.imshow(img_show)
            ax = plt.gca()
            # polygons = target.return_polygons()
            polygons = []
            color = []
            for poly in target.extra_fields['masks'].polygons:
                poly = poly.polygons[0].numpy()
                c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
                poly = np.array(poly).reshape((int(len(poly) / 2), 2))
                polygons.append(Polygon(poly))
                color.append(c)

            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)

            bboxes = target.return_boxes()
            for b in range(bboxes.shape[0]):
                x_min = bboxes[b, 0]
                y_min = bboxes[b, 1]
                x_max = bboxes[b, 2]
                y_max = bboxes[b, 3]
                coords_r = (x_min, y_min), x_max - x_min, y_max - y_min
                ax.add_patch(plt.Rectangle(*coords_r, fill=False, edgecolor='g', linewidth=3))

                ax.text(x_min, y_min, 'Arc',
                         bbox={'facecolor': 'g', 'alpha': 0.5})

            plt.show()

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
