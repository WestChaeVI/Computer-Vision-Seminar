from cv_dataset import Custom_dataset as C
from torch.utils.data import Dataset
import cv2
import os
import torch
import torchvision
from torchvision import transforms # �̹��� ������ augmentation
import os
import glob
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 # albumentations �ټ�ȭ �Լ�

def cv_data_loader():
    root_path = '/content/drive/MyDrive/CV_seminar_project'

    train_transforms = A.Compose([
        A.Resize(224,224), # 
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        A.ChannelShuffle(),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0), # �̹����� �����ͼ� ��谪���� Normalize
        A.CoarseDropout(p=0.5),
        ToTensorV2() # �ټ�ȭ�� ���� , pytorch albumentation�� totensor �� min_max scaling �������
    ])

    test_transforms = A.Compose([
        A.Resize(224,224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0), # �ټ�Ÿ���� ������
        ToTensorV2() # Normalize�� �����ϰ� tensorȭ�� �����ؾ��Ѵ�.
    ])

    ### Pytorch ������ Ŭ���� ����
    train_class = C(root_path=root_path, mode='train', transforms=train_transforms)
    valid_class = C(root_path=root_path, mode='valid', transforms=test_transforms)
    test_class = C(root_path=root_path, mode='test', transforms=test_transforms)

    ### Pytorch BatchLoader ���� (�н��� �̿��� ���� dataloader)
    from torch.utils.data import DataLoader as DataLoader

    train_loader = DataLoader(train_class, batch_size=batch_size, shuffle = True, num_workers=0)
    valid_loader = DataLoader(valid_class, batch_size=batch_size, shuffle = False, num_workers=0)
    test_loader = DataLoader(test_class, batch_size=batch_size, shuffle = False, num_workers=0)

    return train_loader, valid_loader, test_loader