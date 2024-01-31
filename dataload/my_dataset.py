# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import torch
import json
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset

lesion_label_default = ['aneurysm']

BBOXES = 'bboxes'

def load_series_list(series_list_path: str):
    """
    Return:
        series_list: list of tuples (series_folder, file_name)

    """
    with open(series_list_path, 'r') as f:
        lines = f.readlines()
        lines = lines[1:] # Remove the line of description
        
    series_list = []
    for series_info in lines:
        series_info = series_info.strip()
        series_folder, file_name = series_info.split(',')
        series_list.append([series_folder, file_name])
    return series_list

class DetDatasetCSVR(Dataset):
    """Dataset for loading numpy images with dimension order [D, H, W]

    Arguments:
        roots (list): list of dirs of the dataset
        transform_post: transform object after cropping
        crop_fn: cropping function
        lesion_label (list): label names of lesion, such as ['Aneurysm']

    """
    def __init__(self, series_list_path: str, transform_post=None, crop_fn=None):
        self.labels = []
        self.dicom_paths = []
        spacing = np.array([[1.0, 0.8, 0.8]], dtype=np.float32) # (z, y, x)
        self.spacing = spacing
        
        series_infos = load_series_list(series_list_path)
        for folder, series_name in series_infos:
            dicom_path = os.path.join(folder, 'npy', f'{series_name}.npy')
            self.dicom_paths.append(dicom_path)
            label_path = os.path.join(folder, 'mask', f'{series_name}_nodule_count.json')
            
            with open(label_path, 'r') as f:
                info = json.load(f)
                
            bboxes = info[BBOXES]
            bboxes = np.array(bboxes)

            if len(bboxes) == 0:
                continue
            # calculate center of bboxes
            all_loc = ((bboxes[:, 0] + bboxes[:, 1]) / 2).astype(np.float32) # (y, x, z)
            all_rad = (bboxes[:, 1] - bboxes[:, 0]).astype(np.float32) # (y, x, z)

            all_loc = all_loc[:, [2, 0, 1]] # (z, x, y)
            all_rad = all_rad[:, [2, 0, 1]] # (z, x, y)
            all_rad = all_rad * spacing # (z, x, y)
            all_cls = np.zeros((all_loc.shape[0],), dtype=np.int32)
            
            label = {'all_loc': all_loc, 
                    'all_rad': all_rad,
                    'all_cls': all_cls}
            self.labels.append(label)

        self.transform_post = transform_post
        self.crop_fn = crop_fn

    def __len__(self):
        return len(self.labels)
    
    def __norm__(self, data):
        max_value = np.percentile(data, 99)
        min_value = 0.
        data[data>max_value] = max_value
        data[data<min_value] = min_value
        data = data/max_value
        return data

    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        label = self.labels[idx]

        image_spacing = self.spacing.copy() # z, y, x
        image = np.load(dicom_path).astype('float32') # z, y, x
        image = self.__norm__(image) # normalized

        data = {}
        data['image'] = image
        data['all_loc'] = label['all_loc'] # z, y, x
        data['all_rad'] = label['all_rad'] # d, h, w
        data['all_cls'] = label['all_cls']
        data['file_name'] = os.path.basename(dicom_path)
        samples = self.crop_fn(data, image_spacing)
        random_samples = []

        for i in range(len(samples)):
            sample = samples[i]
            if self.transform_post:
                sample = self.transform_post(sample)
            sample['image'] = (sample['image'] * 2.0 - 1.0) # normalized to -1 ~ 1
            random_samples.append(sample)

        return random_samples



class DetDatasetCSVRTest(Dataset):
    """Dataset for loading numpy images with dimension order [D, H, W]
    """

    def __init__(self, series_list_path: str, SplitComb):
        self.labels = []
        self.dicom_paths = []
        spacing = np.array([1.0, 0.8, 0.8], dtype=np.float32) # (z, y, x)
        self.spacing = spacing
        
        series_infos = load_series_list(series_list_path)
        for folder, series_name in series_infos:
            dicom_path = os.path.join(folder, 'npy', f'{series_name}.npy')
            self.dicom_paths.append(dicom_path)
            label_path = os.path.join(folder, 'mask', f'{series_name}_nodule_count.json')
            
            with open(label_path, 'r') as f:
                info = json.load(f)
                
            bboxes = info[BBOXES]
            bboxes = np.array(bboxes)
            
            if len(bboxes) == 0:
                continue
            # calculate center of bboxes
            all_loc = ((bboxes[:, 0] + bboxes[:, 1]) / 2).astype(np.float32) # (y, x, z)
            all_rad = (bboxes[:, 1] - bboxes[:, 0]).astype(np.float32) # (y, x, z)

            all_loc = all_loc[:, [2, 0, 1]] # (z, x, y)
            all_rad = all_rad[:, [2, 0, 1]] # (z, x, y)
            all_rad = all_rad * spacing # (z, x, y)
            all_cls = np.zeros((all_loc.shape[0],), dtype=np.int32)
            
            label = {'all_loc': all_loc, 
                    'all_rad': all_rad,
                    'all_cls': all_cls}
            self.labels.append(label)
        self.splitcomb = SplitComb
    def __len__(self):
        return len(self.labels)
    
    def __norm__(self, data):
        max_value = np.percentile(data, 99)
        min_value = 0.
        data[data>max_value] = max_value
        data[data<min_value] = min_value
        data = data/max_value
        return data

    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        label = self.labels[idx]

        image_spacing = self.spacing.copy() # z, y, x
        image = np.load(dicom_path).astype('float32') # z, y, x
        image = self.__norm__(image) # normalized

        data = {}
        # convert to -1 ~ 1  note ste pad_value to -1 for SplitComb
        image = image * 2.0 - 1.0
        # split_images [N, 1, crop_z, crop_y, crop_x]
        split_images, nzhw = self.splitcomb.split(image)
        data['split_images'] = np.ascontiguousarray(split_images)
        data['nzhw'] = nzhw
        data['spacing'] = image_spacing
        data['file_name'] = os.path.basename(dicom_path)
        return data

def collate_fn_dict(batches):
    batch = []
    [batch.extend(b) for b in batches]
    imgs = [s['image'] for s in batch]
    imgs = np.stack(imgs)
    annots = [s['annot'] for s in batch]
    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        annot_padded = np.ones((len(annots), max_num_annots, 7), dtype='float32') * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = np.ones((len(annots), 1, 7), dtype='float32') * -1

    return {'image': torch.tensor(imgs), 'annot': torch.tensor(annot_padded)}
