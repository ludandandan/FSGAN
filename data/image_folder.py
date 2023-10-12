###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path
import json

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, train):# dir是数据库文件的位置,# train是anno_train.json的位置
    images = [] # 这里存放各个照片的绝对路径
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    data = json.load(open(train)) # 载入anno_train.json文件中的数据
    for d in data:#files = os.listdir(dir)
        image_name = d['image_name']
        image_name = image_name.replace('/image', '_image').replace('/photo', '_photo')
        if 'image1136' in image_name:# 为什么要把image1136跳过呢
            continue
        else:
            if is_image_file(image_name):
                path = os.path.join(dir, image_name)
            else:
                path = os.path.join(dir, image_name)
                path = path + '.png'
            if not os.path.exists(path):
                path = path.replace('.png', '.jpg')
            images.append(path)
    return sorted(images)


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
