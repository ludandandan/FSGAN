import importlib
import torch.utils.data
from data.base_data_loader import BaseDataLoader
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    # Given the option --dataset_mode [datasetname], 默认是aligned
    # the file "data/datasetname_dataset.py"
    # will be imported.
    dataset_filename = "data." + dataset_name + "_dataset" # data.aligned_dataset
    datasetlib = importlib.import_module(dataset_filename) # import data.aligned_dataset

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        print("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))
        exit(0)

    return dataset


def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    dataset = find_dataset_using_name(opt.dataset_mode) #<class 'data.aligned_dataset.AlignedDataset'>
    instance = dataset()
    instance.initialize(opt)
    print("dataset [%s] was created" % (instance.name())) # instance的A_paths中存储了训练集照片图像的绝对路径；style_dict存储了训练集和测试集所有图片的风格类型
    return instance


def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader


# Wrapper class of Dataset class that performs
# multi-threaded data loading 多线程数据载入
class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt) # 将参数给到BaseDataLoader
        self.dataset = create_dataset(opt)#创建数据集data.aligned_dataset.AlignedDataset，包括dir_AB数据集的位置，A_paths训练集照片的绝对路径列表，opt参数，style_dict数据集内所有图像的风格
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,#in training, serial_batches by default is false, shuffle=true
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            # print('>>>>>>>>>>>')
            # print(i, type(data), data.keys())
            # for idx, (k, v) in enumerate(data.items()):
            #     print(k, v)
            #     if isinstance(v[0], str) or len(v[0].shape) != 3:
            #         print(k, v[0])
            #         pass
            #     else:
            #         print(k, v[0].shape, v[0].min(), v[0].max(), v[0].mean())
            #         tensor = v[0].permute(1, 2, 0).cpu().numpy()
            #         import cv2
            #         import os
            #         import numpy as np
            #         cv2.imwrite(os.path.join('tmp', k+'.png'), cv2.cvtColor(((tensor * 0.5 + 0.5)*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            # print('<<<<<<<<<<<')
            yield data
