from torchvision import datasets, transforms, utils
from base import BaseDataLoader
from scripts_for_datasets import COWCDataset, COWCGANDataset, COWCFRCNNDataset, COWCGANFrcnnDataset, MyDataMaskRCNNDataset

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,
    BboxParams, RandomCrop, Normalize, Resize, VerticalFlip
)

from albumentations.pytorch import ToTensor
from utils import collate_fn
#from detection.utils import collate_fn


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class COWCDataLoader(BaseDataLoader):
    """
    COWC data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        #data transformation
        #According to this link: https://discuss.pytorch.org/t/normalization-of-input-image/34814/8
        #satellite image 0.5 is good otherwise calculate mean and std for the whole dataset.
        #calculted mean and std using method from util
        data_transforms = Compose([
            Resize(256, 256),
            HorizontalFlip(),
            OneOf([
                    IAAAdditiveGaussianNoise(),
                    GaussNoise(),
                ], p=0.2),
            OneOf([
                    CLAHE(clip_limit=2),
                    IAASharpen(),
                    IAAEmboss(),
                    RandomBrightnessContrast(),
                ], p=0.3),
                HueSaturationValue(p=0.3),
            Normalize( #mean std for potsdam dataset from COWC [Calculate also for spot6]
                mean=[0.3442, 0.3708, 0.3476],
                std=[0.1232, 0.1230, 0.1284]
                )
        ],
            bbox_params=BboxParams(
             format='pascal_voc',
             min_area=0,
             min_visibility=0,
             label_fields=['labels'])
        )


        self.data_dir = data_dir
        self.dataset = COWCDataset(self.data_dir, transform=data_transforms)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)

class COWCGANDataLoader(BaseDataLoader):
    """
    COWC data loading using BaseDataLoader
    """
    def __init__(self, data_dir_GT, data_dir_LQ, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        #data transformation
        #According to this link: https://discuss.pytorch.org/t/normalization-of-input-image/34814/8
        #satellite image 0.5 is good otherwise calculate mean and std for the whole dataset.
        #calculted mean and std using method from util
        '''
        Data transform for GAN training
        '''
        data_transforms_train = Compose([
            HorizontalFlip(),
            Normalize( #mean std for potsdam dataset from COWC [Calculate also for spot6]
                mean=[0.3442, 0.3708, 0.3476],
                std=[0.1232, 0.1230, 0.1284]
                )
        ],
            additional_targets={
             'image_lq':'image'
            },
            bbox_params=BboxParams(
             format='pascal_voc',
             min_area=0,
             min_visibility=0,
             label_fields=['labels'])
        )

        data_transforms_test = Compose([
            Normalize( #mean std for potsdam dataset from COWC [Calculate also for spot6]
                mean=[0.3442, 0.3708, 0.3476],
                std=[0.1232, 0.1230, 0.1284]
                )],
            additional_targets={
                 'image_lq':'image'
                })

        self.data_dir_gt = data_dir_GT
        self.data_dir_lq = data_dir_LQ

        if training == True:
            self.dataset = COWCGANDataset(self.data_dir_gt, self.data_dir_lq, transform=data_transforms_train)
        else:
            self.dataset = COWCGANDataset(self.data_dir_gt, self.data_dir_lq, transform=data_transforms_test)
        self.length = len(self.dataset)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)

class COWCGANFrcnnDataLoader(BaseDataLoader):
    """
    COWC data loading using BaseDataLoader
    """
    def __init__(self, data_dir_GT, data_dir_LQ, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        #data transformation
        #According to this link: https://discuss.pytorch.org/t/normalization-of-input-image/34814/8
        #satellite image 0.5 is good otherwise calculate mean and std for the whole dataset.
        #calculted mean and std using method from util
        '''
        Data transform for GAN training
        '''
        # print(data_dir_GT, data_dir_LQ)
        data_transforms_train = Compose([
            HorizontalFlip(),
            Normalize( #mean std for potsdam dataset from COWC [Calculate also for spot6]
                mean=[0.3442, 0.3708, 0.3476],
                std=[0.1232, 0.1230, 0.1284]
                )
        ],
            additional_targets={
             'image_lq':'image'
            },
            bbox_params=BboxParams(
             format='pascal_voc',
             min_area=0,
             min_visibility=0,
             label_fields=['labels'])
        )

        data_transforms_test = Compose([
            Normalize( #mean std for potsdam dataset from COWC [Calculate also for spot6]
                mean=[0.3442, 0.3708, 0.3476],
                std=[0.1232, 0.1230, 0.1284]
                )],
            additional_targets={
                 'image_lq':'image'
                })

        self.data_dir_gt = data_dir_GT
        self.data_dir_lq = data_dir_LQ

        if training == True:
            self.dataset = COWCGANFrcnnDataset(self.data_dir_gt, self.data_dir_lq, transform=data_transforms_train)
        else:
            self.dataset = COWCGANFrcnnDataset(self.data_dir_gt, self.data_dir_lq, transform=data_transforms_test)
        self.length = len(self.dataset)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)

class MyDataMaskRCNNDataloader(BaseDataLoader):
    """
    COWC data loading using BaseDataLoader
    """
    def __init__(self, 
                 data_dir, 
                 meta_data_dir, 
                 mask_dir,
                 batch_size, 
                 shuffle=True, 
                 validation_split=0.0, 
                 num_workers=1, 
                 training=True):
        #data transformation
        #According to this link: https://discuss.pytorch.org/t/normalization-of-input-image/34814/8
        #satellite image 0.5 is good otherwise calculate mean and std for the whole dataset.
        #calculted mean and std using method from util
        '''
        Data transform for GAN training
        '''
        # print(data_dir, meta_data_dir)
        data_transforms_train = Compose([
            # HorizontalFlip(),
            Normalize( #mean std for potsdam dataset from COWC [Calculate also for spot6]
                mean=[0.3442, 0.3708, 0.3476],
                std=[0.1232, 0.1230, 0.1284]
                )
        ],
            additional_targets={
             'image_lq':'image'
            },
            bbox_params=BboxParams(
             format='coco',
             min_area=0,
             min_visibility=0,
             label_fields=['labels'])
        )

        data_transforms_test = Compose([
            Normalize( #mean std for potsdam dataset from COWC [Calculate also for spot6]
                mean=[0.3442, 0.3708, 0.3476],
                std=[0.1232, 0.1230, 0.1284]
                )],
            additional_targets={
                 'image_lq':'image'
                })

        self.data_dir = data_dir
        self.meta_data = meta_data_dir
        self.mask_dir = mask_dir

        if training == True:
            self.dataset = MyDataMaskRCNNDataset(data_dir=self.data_dir, 
                                                 meta_data_dir=self.meta_data,
                                                 mask_dir=self.mask_dir,
                                                 transform=data_transforms_train)
        else:
            self.dataset = MyDataMaskRCNNDataset(data_dir=self.data_dir, 
                                                 meta_data_dir=self.meta_data,
                                                 mask_dir=self.mask_dir, 
                                                 transform=data_transforms_test,
                                                 verbose="val")
        self.length = len(self.dataset)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=my_collate_fn)

def my_collate_fn(batch):
    '''
    Image have a different number of objects, we need a collate function
    (to be passed to the DataLoader).
    '''

    import torch

    target = list()
    image = {}
    image['image'] = list()
    image['image_lq'] = list()
    image['LQ_path'] = list()

    for obj in batch:
        b = obj[0]
        image['image'].append(b['image'])
        image['image_lq'].append(b['image_lq'])
        image['LQ_path'].append(b['LQ_path'])
        target.append(obj[1])

    image['image'] = torch.stack(image['image'], dim=0)
    image['image_lq'] = torch.stack(image['image_lq'], dim=0)

    return image, target