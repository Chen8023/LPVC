"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
#from data.vimeo_train_dataset import VimeotTrainDataSet
#from data.test_classA_E_dataset import TestDataSet
from data.train2frame_dataset import Train2FDataSet

from  data.test_dataset import TestDataSet
#from data.meeting_val_dataset import MeetValDataSet


from torch.utils.data import DataLoader

def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """

    # data_loader = CustomDatasetDataLoader(opt)
    # dataset = data_loader.load_data()
    if opt.isTrain:
        train_dataset = Train2FDataSet(opt)
        shuffle = True
        print('2 frames are used in training......')

            # raise NotImplementedError(
            #     'dataset name [%s] is not found' % opt.dataset_name)
        #shuffle=True洗牌乱序，shuffle=False 按序
        '''
        通常，我们使用的dataloader来读取数据，会发现即使把 shuffle=False 也不会按照1，2，3…读取图片，
        而是1，10，11，12，13,…,100,…这种顺序，不符合我们的预期
        '''
        train_loader = DataLoader(dataset = train_dataset, shuffle=shuffle, num_workers=torch.cuda.device_count(),
                                  batch_size=opt.batch_size, pin_memory=True,drop_last=True)
        return train_loader #dataset
    else:
        test_dataset = TestDataSet(opt)
        shuffle = False
        # if opt.dataset_name == 'val':
        #     test_dataset = MeetValDataSet(opt)
        # elif opt.dataset_name == 'test_class':
        #     test_dataset = TestDataSet(opt)
        # elif opt.dataset_name == 'test_meeting':
        #     test_dataset = MeetTestDataSet(opt)
        # else:
        #     raise NotImplementedError(
        #         'dataset name [%s] is not found' % opt.dataset_name)

        #drop_last (bool, optional) –如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。
        # 如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。(默认: False)
        #https://blog.csdn.net/qq_39852676/article/details/105919169
        test_loader = DataLoader(dataset=test_dataset, shuffle=shuffle, num_workers=torch.cuda.device_count(),
                                  batch_size=opt.batch_size, pin_memory=True,drop_last=True)
        return test_loader  # ,dataset

