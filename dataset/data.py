import os
from . import utils
import torch
import torchvision
import numpy as np
import PIL.Image
import zipfile
import tarfile


class Data(torch.utils.data.Dataset):
    def __init__(self, labels, transform=None):
        # e.g., labels = range(0, 50) for using first 50 classes only
        self.labels = labels
        if transform:
            self.transform = transform
        self.ys, self.im_paths = [], []

    @staticmethod
    def factory(type, root, labels, is_extracted=False, transform=None):
        if type == 'Birds':
            return Birds(root, labels, is_extracted, transform)
        if type == 'Food':
            return Food(root, labels, is_extracted, transform)
        assert 0, 'Unknown Dataset type: ' + type

    def nb_classes(self):
        n = len(np.unique(self.ys))
        assert n == len(self.labels)
        return n

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = PIL.Image.open(self.im_paths[index])
        im = self.transform(im)
        return im, self.ys[index]


class Food(Data):
    def __init__(self, root, labels, is_extracted=False, transform=None):
        super().__init__(labels=labels, transform=transform)

        # download UPMC-G20 images, verify md5
        filename = 'Gaze_UPMC_Food20.zip'
        torchvision.datasets.utils.download_url(
            url='http://visiir.lip6.fr/data/public/Gaze_UPMC_Food20.zip',
            root=root,
            filename=filename,
            md5='be5aed840477c8a62b35a9ee1d4f2754'
        )

        # extract zip; if not extracted, then overwrite
        if not is_extracted:
            with zipfile.ZipFile(
                file=os.path.join(root, filename), mode='r'
            ) as zip_ref:
                zip_ref.extractall(path=root)

        self.ys, self.im_paths = [], []
        for i in torchvision.datasets.ImageFolder(
            root=os.path.join(root, 'images')
        ).imgs:
            self.ys.append(i[1])
            self.im_paths.append(i[0])


class Birds(Data):
    def __init__(self, root, labels, is_extracted=False, transform=None):
        super().__init__(labels=labels, transform=transform)

        # download CUB images, verify md5
        filename = 'images.tgz'
        torchvision.datasets.utils.download_url(
            url='http://www.vision.caltech.edu/visipedia-data/' +
            'CUB-200/images.tgz',
            root=root,
            filename=filename,
            md5='2bbe304ef1aa3ddb6094aa8f53487cf2'
        )

        # extract tgz; if not extracted, then overwrite
        if not is_extracted:
            with tarfile.open(os.path.join(root, filename)) as tar:
                tar.extractall(path=root)

        self.ys, self.im_paths = [], []
        for i in torchvision.datasets.ImageFolder(
            root=os.path.join(root, 'images')
        ).imgs:
            # i[1]: label, i[0]: path to file, including root
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.labels and fn[:2] != '._':
                self.ys += [y]
                self.im_paths.append(i[0])
