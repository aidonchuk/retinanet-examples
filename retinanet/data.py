import os
import random
from contextlib import redirect_stdout

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from albumentations import CLAHE, IAASharpen, IAAEmboss, RandomBrightnessContrast, RGBShift, ImageCompression, \
    RandomGamma, ChannelShuffle, InvertImg, ToGray, RandomSnow, RandomRain, RandomFog, ChannelDropout, ISONoise, OneOf, \
    IAAAdditiveGaussianNoise, GaussNoise, Blur, MotionBlur, MedianBlur, HueSaturationValue
from pycocotools.coco import COCO
from torch.utils import data


class CocoDataset(data.dataset.Dataset):
    'Dataset looping through a set of images'

    def __init__(self, path, resize, max_size, stride, annotations=None, training=False, crop_number=False):
        super().__init__()

        self.path = os.path.expanduser(path)
        self.resize = resize
        self.max_size = max_size
        self.stride = stride
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.training = training
        self.crop_number = crop_number

        with redirect_stdout(None):
            self.coco = COCO(annotations)
        self.ids = list(self.coco.imgs.keys())
        if 'categories' in self.coco.dataset:
            self.categories_inv = {k: i for i, k in enumerate(self.coco.getCatIds())}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ' Get sample'

        # Load image
        id = self.ids[index]
        if self.coco:
            image = self.coco.loadImgs(id)[0]['file_name']
        im = Image.open('{}/{}'.format(self.path, image)).convert("RGB")

        if self.crop_number:
            boxes, categories = self._get_target(id)
            for i, j in enumerate(boxes):
                if categories[i] == 11:
                    b = np.asarray(j, dtype=np.int)
                    im = np.asarray(im)
                    im = im[b[1]:b[1] + b[3], b[0]:b[0] + b[2], :]
                    im = Image.fromarray(im)

        # Randomly sample scale for resize during training
        resize = self.resize
        if isinstance(resize, list):
            resize = random.randint(self.resize[0], self.resize[-1])

        ratio = resize / min(im.size)
        if ratio * max(im.size) > self.max_size:
            ratio = self.max_size / max(im.size)
        im = im.resize((int(ratio * d) for d in im.size), Image.BILINEAR)
        # im.save(str(id) + '.png', 'PNG')

        if self.training:
            # Get annotations
            boxes, categories = self._get_target(id)
            if self.crop_number:
                boxes, categories = self.new_bbox_coords(boxes, categories)
            boxes *= ratio

            annotations = {'image': np.asarray(im), 'bboxes': np.asarray(boxes),
                           'category_id': np.asarray(categories)}
            # print(image)
            aug = self.get_aug([
                OneOf([
                    CLAHE(),
                    IAASharpen(),
                    IAAEmboss(),
                    RandomBrightnessContrast(),
                    RGBShift(),
                    ImageCompression(),
                    RandomGamma(),
                    ChannelShuffle(),
                    InvertImg(),
                    ToGray(),
                    RandomSnow(),
                    RandomRain(),
                    RandomFog(),
                    ChannelDropout(),
                    ISONoise()
                ], p=0.4),
                OneOf([
                    IAAAdditiveGaussianNoise(),
                    GaussNoise(),
                ], p=0.3),
                OneOf([
                    Blur(),
                    MotionBlur(),
                    MedianBlur(),
                ], p=0.4),
                HueSaturationValue()
            ])

            try:
                augmented = aug(**annotations)
                im, boxes, categories = augmented['image'], torch.tensor(augmented['bboxes']), torch.tensor(
                    augmented['category_id'])
                target = torch.cat([boxes, categories], dim=1)
                im = Image.fromarray(im)
                # im.save(str(id) + '.png', 'PNG')
                # print(str(id) + ' ' + str(boxes))
            except Exception as e:
                print(image)
                print(e)
        # Convert to tensor and normalize
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
        data = data.float().div(255).view(*im.size[::-1], len(im.mode))
        data = data.permute(2, 0, 1)

        for t, mean, std in zip(data, self.mean, self.std):
            t.sub_(mean).div_(std)

        # Apply padding
        pw, ph = ((self.stride - d % self.stride) % self.stride for d in im.size)
        data = F.pad(data, (0, pw, 0, ph))

        if self.training:
            return data, target

        return data, id, ratio

    def new_bbox_coords(self, boxes, categories):
        boxes, categories = np.asarray(boxes), np.asarray(categories)
        parent = None
        for i, j in enumerate(boxes):
            if categories[i] == 11:
                parent = j.copy()
        b = []
        c = []
        for i, j in enumerate(boxes):
            if not categories[i] == 11:
                if parent[0] < j[0] < (parent[0] + parent[2]) and parent[1] < j[1] < (parent[1] + parent[3]):
                    j[0] = j[0] - parent[0]
                    j[1] = j[1] - parent[1]
                    b.append([j[0], j[1], j[2], j[3]])
                    c.append(categories[i])

        return torch.tensor(b), torch.tensor(c)

    def get_aug(self, aug, min_area=0., min_visibility=0.):
        return A.Compose(aug, A.BboxParams(format='coco', min_area=min_area,
                                           min_visibility=min_visibility, label_fields=['category_id']))

    def _get_target(self, id):
        'Get annotations for sample'

        ann_ids = self.coco.getAnnIds(imgIds=[id])
        annotations = self.coco.loadAnns(ann_ids)

        boxes, categories = [], []
        for ann in annotations:
            if ann['bbox'][2] < 1 and ann['bbox'][3] < 1:
                continue
            boxes.append(ann['bbox'])
            cat = ann['category_id']
            # if 'categories' in self.coco.dataset:
            #    cat = self.categories_inv[cat]
            categories.append(cat)

        if boxes:
            target = (torch.FloatTensor(boxes),
                      torch.FloatTensor(categories).unsqueeze(1))
        else:
            target = (torch.ones([1, 4]), torch.ones([1, 1]) * -1)

        return target

    def collate_fn(self, batch):
        'Create batch from multiple samples'

        if self.training:
            data, targets = zip(*batch)
            max_det = max([t.size()[0] for t in targets])
            targets = [torch.cat([t, torch.ones([max_det - t.size()[0], 5]) * -1]) for t in targets]
            targets = torch.stack(targets, 0)
        else:
            data, indices, ratios = zip(*batch)

        # Pad data to match max batch dimensions
        sizes = [d.size()[-2:] for d in data]
        w, h = (max(dim) for dim in zip(*sizes))

        data_stack = []
        for datum in data:
            pw, ph = w - datum.size()[-2], h - datum.size()[-1]
            data_stack.append(
                F.pad(datum, (0, ph, 0, pw)) if max(ph, pw) > 0 else datum)

        data = torch.stack(data_stack)

        if self.training:
            return data, targets

        ratios = torch.FloatTensor(ratios).view(-1, 1, 1)
        return data, torch.IntTensor(indices), ratios


class DataIterator():
    'Data loader for data parallel'

    def __init__(self, path, resize, max_size, batch_size, stride, world, annotations, training=False,
                 crop_number=False):
        self.resize = resize
        self.max_size = max_size

        self.dataset = CocoDataset(path, resize=resize, max_size=max_size,
                                   stride=stride, annotations=annotations, training=training, crop_number=crop_number)
        self.ids = self.dataset.ids
        self.coco = self.dataset.coco

        self.sampler = data.distributed.DistributedSampler(self.dataset) if world > 1 else None
        self.dataloader = data.DataLoader(self.dataset, batch_size=batch_size // world,
                                          sampler=self.sampler, collate_fn=self.dataset.collate_fn, num_workers=2,
                                          pin_memory=True)

    def __repr__(self):
        return '\n'.join([
            '    loader: pytorch',
            '    resize: {}, max: {}'.format(self.resize, self.max_size),
        ])

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for output in self.dataloader:
            if self.dataset.training:
                data, target = output
            else:
                data, ids, ratio = output

            if torch.cuda.is_available():
                data = data.cuda(non_blocking=True)

            if self.dataset.training:
                if torch.cuda.is_available():
                    target = target.cuda(non_blocking=True)
                yield data, target
            else:
                if torch.cuda.is_available():
                    ids = ids.cuda(non_blocking=True)
                    ratio = ratio.cuda(non_blocking=True)
                yield data, ids, ratio
