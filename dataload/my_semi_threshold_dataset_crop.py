# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import random
import logging
import torch
import json
import numpy as np
from typing import List
from torch.utils.data import Dataset
lesion_label_default = ['aneurysm']

BBOXES = 'bboxes'
USE_BG = False

logger = logging.getLogger(__name__)

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

def normalize(image: np.ndarray) -> np.ndarray: 
    HU_MIN, HU_MAX = -1000, 400
    image = np.clip(image, HU_MIN, HU_MAX)
    image = image - HU_MIN
    image = image.astype(np.float32) / (HU_MAX - HU_MIN)
    return image

def load_image(dicom_path: str) -> np.ndarray:
    """
    Return:
        A 3D numpy array with dimension order [D, H, W] (z, y, x)
    """
    image = np.load(dicom_path)
    image = np.transpose(image, (2, 0, 1))
    image = normalize(image)
    return image

class LabeledDataset(Dataset):
    """Dataset for loading numpy images with dimension order [D, H, W]

    Arguments:
        transform_post: transform object after cropping
        crop_fn: cropping function
    """
    def __init__(self, series_list_path: str, image_spacing: List[float], transform_post=None, crop_fn=None):
        self.labels = []
        self.dicom_paths = []
        self.series_list_path = series_list_path
        self.series_names = []
        self.image_spacing = np.array(image_spacing, dtype=np.float32) # (z, y, x)
        
        series_infos = load_series_list(series_list_path)
        for folder, series_name in series_infos:
            label_path = os.path.join(folder, 'mask', f'{series_name}_nodule_count_crop.json')
            dicom_path = os.path.join(folder, 'npy', f'{series_name}_crop.npy')
            
            with open(label_path, 'r') as f:
                info = json.load(f)
                
            bboxes = info[BBOXES]
            bboxes = np.array(bboxes)

            if len(bboxes) == 0:
                if not USE_BG:
                    continue
                self.dicom_paths.append(dicom_path)
                self.series_names.append(series_name)
                label = {'all_loc': np.zeros((0, 3), dtype=np.float32),
                        'all_rad': np.zeros((0, 3), dtype=np.float32),
                        'all_cls': np.zeros((0,), dtype=np.int32)}
            else:
                self.dicom_paths.append(dicom_path)
                self.series_names.append(series_name)
                # calculate center of bboxes
                all_loc = ((bboxes[:, 0] + bboxes[:, 1] - 1) / 2).astype(np.float32) # (y, x, z)
                all_rad = (bboxes[:, 1] - bboxes[:, 0]).astype(np.float32) # (y, x, z)

                all_loc = all_loc[:, [2, 0, 1]] # (z, y, x)
                all_rad = all_rad[:, [2, 0, 1]] # (z, y, x)
                all_rad = all_rad * self.image_spacing # (z, y, x)
                all_cls = np.zeros((all_loc.shape[0],), dtype=np.int32)
                
                label = {'all_loc': all_loc, 
                        'all_rad': all_rad,
                        'all_cls': all_cls}
            self.labels.append(label)

        self.transform_post = transform_post
        self.crop_fn = crop_fn

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        series_name = self.series_names[idx]
        label = self.labels[idx]

        image_spacing = self.image_spacing.copy() # z, y, x
        image = load_image(dicom_path) # z, y, x
        
        data = {}
        data['image'] = image
        data['all_loc'] = label['all_loc'] # z, y, x
        data['all_rad'] = label['all_rad'] # d, h, w
        data['all_cls'] = label['all_cls']
        data['file_name'] = series_name
        samples = self.crop_fn(data, image_spacing)
        random_samples = []

        for i in range(len(samples)):
            sample = samples[i]
            if self.transform_post:
                sample['ctr_transform'] = []
                sample = self.transform_post(sample)
            sample['image'] = (sample['image'] * 2.0 - 1.0) # normalized to -1 ~ 1
            random_samples.append(sample)
        return random_samples

class UnLabeledTrainDataset(Dataset):
    """Dataset for loading numpy images with dimension order [D, H, W]

    Arguments:
        transform_post: transform object after cropping
        crop_fn: cropping function
    """
    def __init__(self, series_list_path: str, image_spacing: List[float], transform_post=None, crop_fn=None, prob_threshold: float = 0.9):
        self.series_list_path = series_list_path
        self.image_spacing = np.array(image_spacing, dtype=np.float32) # (z, y, x)
        self.prob_threshold = prob_threshold
        
        self.transform_post = transform_post
        self.crop_fn = crop_fn
        self.series_infos = load_series_list(series_list_path)
    
    def __len__(self):
        if hasattr(self, 'dicom_paths'):
            return len(self.dicom_paths)
        else:
            logger.warning('Please call update_labels() before using this dataset')
            return 0
    @staticmethod
    def read_labels(label_path, prob_threshold: float = 0.9):
        with open(label_path, 'r') as f:
            label = json.load(f)
            
        all_prob = label['all_prob']
        
        selected_indices = []
        for i, prob in enumerate(all_prob):
            if prob[0] >= prob_threshold:
                selected_indices.append(i)
        if len(selected_indices) == 0:
            all_loc = np.zeros((0, 3), dtype=np.float32)
            all_rad = np.zeros((0, 3), dtype=np.float32)
            all_cls = np.zeros((0,), dtype=np.int32)
        else:
            all_loc = np.array(label['all_loc'])[selected_indices]
            all_rad = np.array(label['all_rad'])[selected_indices]
            all_cls = np.zeros((all_loc.shape[0],), dtype=np.int32)

        label = {'all_loc': all_loc,
                'all_rad': all_rad,
                'all_cls': all_cls}
        return label
    
    def update_labels(self):
        dicom_paths = []
        series_names = []
        series_folders = []
        labels = []
        for folder, series_name in self.series_infos:
            label_path = os.path.join(folder, 'pseudo_label', f'{series_name}.json')
            label = self.read_labels(label_path, self.prob_threshold)
            if len(label['all_loc']) == 0:
                continue
            
            dicom_path = os.path.join(folder, 'npy', f'{series_name}_crop.npy')
            dicom_paths.append(dicom_path)
            series_names.append(series_name)
            series_folders.append(folder)
            labels.append(label)
        self.dicom_paths = dicom_paths
        self.series_names = series_names
        self.series_folders = series_folders
        self.labels = labels
        
    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        series_name = self.series_names[idx]
        label = self.labels[idx]

        if len(label['all_loc']) == 0:
            return None
    
        image_spacing = self.image_spacing.copy() # z, y, x
        image = load_image(dicom_path) # z, y, x
        
        data = {}
        data['image'] = image
        data['all_loc'] = label['all_loc'] # z, y, x
        data['all_rad'] = label['all_rad'] # d, h, w
        data['all_cls'] = label['all_cls']
        data['file_name'] = series_name
        samples = self.crop_fn(data, image_spacing)
        random_samples = []

        for i in range(len(samples)):
            sample = samples[i]
            if self.transform_post:
                sample['ctr_transform'] = []
                sample = self.transform_post(sample)
            sample['image'] = (sample['image'] * 2.0 - 1.0) # normalized to -1 ~ 1
            random_samples.append(sample)
        return random_samples

# class UnLabeledTrainDataset(Dataset):
#     """Dataset for loading numpy images with dimension order [D, H, W]

#     Arguments:
#         transform_post: transform object after cropping
#         crop_fn: cropping function
#     """
#     def __init__(self, series_list_path: str, image_spacing: List[float], transform_post=None, crop_fn=None, prob_threshold: float = 0.9):
#         self.series_list_path = series_list_path
#         self.image_spacing = np.array(image_spacing, dtype=np.float32) # (z, y, x)
#         self.prob_threshold = prob_threshold
        
#         self.labels = []
#         self.dicom_paths = []
#         self.series_names = []
#         self.series_folders = []
        
#         series_infos = load_series_list(series_list_path)
#         for folder, series_name in series_infos:
#             dicom_path = os.path.join(folder, 'npy', f'{series_name}_crop.npy')
#             self.dicom_paths.append(dicom_path)
#             self.series_names.append(series_name)
#             self.series_folders.append(folder)
            
#         self.transform_post = transform_post
#         self.crop_fn = crop_fn

#         self.selected_indices = list(range(len(self.dicom_paths)))
        
#     def shuffle(self, seed: int):
#         random.seed(seed)
#         random.shuffle(self.selected_indices)
    
#     def __len__(self):
#         return len(self.dicom_paths)
    
#     def read_labels(self, idx):
#         series_folder = self.series_folders[idx]
#         series_name = self.series_names[idx]
        
#         label_path = os.path.join(series_folder, 'pseudo_label', f'{series_name}.json')
        
#         with open(label_path, 'r') as f:
#             label = json.load(f)
            
#         all_prob = label['all_prob']
        
#         selected_indices = []
#         for i, prob in enumerate(all_prob):
#             if prob[0] >= self.prob_threshold:
#                 selected_indices.append(i)
        
#         if len(selected_indices) == 0:
#             all_loc = np.zeros((0, 3), dtype=np.float32)
#             all_rad = np.zeros((0, 3), dtype=np.float32)
#             all_cls = np.zeros((0,), dtype=np.int32)
#         else:
#             all_loc = np.array(label['all_loc'])[selected_indices]
#             all_rad = np.array(label['all_rad'])[selected_indices]
#             all_cls = np.zeros((all_loc.shape[0],), dtype=np.int32)

#         label = {'all_loc': all_loc,
#                 'all_rad': all_rad,
#                 'all_cls': all_cls}
#         return label
    
#     def __getitem__(self, idx):
#         idx = self.selected_indices[idx]
#         dicom_path = self.dicom_paths[idx]
#         series_name = self.series_names[idx]
#         series_folder = self.series_folders[idx]
#         label = self.read_labels(idx)

#         if len(label['all_loc']) == 0:
#             return None
    
#         image_spacing = self.image_spacing.copy() # z, y, x
#         image = load_image(dicom_path) # z, y, x
        
#         data = {}
#         data['image'] = image
#         data['all_loc'] = label['all_loc'] # z, y, x
#         data['all_rad'] = label['all_rad'] # d, h, w
#         data['all_cls'] = label['all_cls']
#         data['file_name'] = series_name
#         samples = self.crop_fn(data, image_spacing)
#         random_samples = []

#         for i in range(len(samples)):
#             sample = samples[i]
#             if self.transform_post:
#                 sample = self.transform_post(sample)
#             sample['image'] = (sample['image'] * 2.0 - 1.0) # normalized to -1 ~ 1
#             random_samples.append(sample)
#         sample['ctr_transform'] = []
#         return random_samples

class UnLabeledInferDataset(Dataset):
    """Dataset for loading numpy images with dimension order [D, H, W]
    """

    def __init__(self, series_list_path: str, image_spacing: List[float], SplitComb):
        self.labels = []
        self.dicom_paths = []
        self.series_list_path = series_list_path
        self.series_names = []
        self.series_folders = []
        self.image_spacing = np.array(image_spacing, dtype=np.float32) # (z, y, x)
        
        series_infos = load_series_list(series_list_path)
        for folder, series_name in series_infos:
            dicom_path = os.path.join(folder, 'npy', f'{series_name}_crop.npy')
            self.dicom_paths.append(dicom_path)
            self.series_names.append(series_name)
            self.series_folders.append(folder)
            
        self.splitcomb = SplitComb
    
        self.selected_indices = list(range(len(self.dicom_paths)))
        
    def shuffle(self, seed: int):
        random.seed(seed)
        random.shuffle(self.selected_indices)
        
    def __len__(self):
        return len(self.dicom_paths)
    
    def __getitem__(self, idx):
        idx = self.selected_indices[idx]
        dicom_path = self.dicom_paths[idx]
        series_name = self.series_names[idx]
        series_folder = self.series_folders[idx]
        image_spacing = self.image_spacing.copy() # z, y, x
        image = load_image(dicom_path) # z, y, x

        data = {}
        # convert to -1 ~ 1  note ste pad_value to -1 for SplitComb
        image = image * 2.0 - 1.0
        # split_images [N, 1, crop_z, crop_y, crop_x]
        split_images, nzhw = self.splitcomb.split(image)
        data['split_images'] = np.ascontiguousarray(split_images)
        data['nzhw'] = nzhw
        data['spacing'] = image_spacing
        data['series_name'] = series_name
        data['series_folder'] = series_folder
        return data

class DetDatasetCSVRTest(Dataset):
    """Dataset for loading numpy images with dimension order [D, H, W]
    """

    def __init__(self, series_list_path: str, image_spacing: List[float], SplitComb):
        self.labels = []
        self.dicom_paths = []
        self.series_list_path = series_list_path
        self.series_names = []
        self.image_spacing = np.array(image_spacing, dtype=np.float32) # (z, y, x)
        
        series_infos = load_series_list(series_list_path)
        for folder, series_name in series_infos:
            dicom_path = os.path.join(folder, 'npy', f'{series_name}_crop.npy')
            self.dicom_paths.append(dicom_path)
            self.series_names.append(series_name)
        self.splitcomb = SplitComb
        
    def __len__(self):
        return len(self.dicom_paths)
    
    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        series_name = self.series_names[idx]
        image_spacing = self.image_spacing.copy() # z, y, x
        image = load_image(dicom_path) # z, y, x

        data = {}
        # convert to -1 ~ 1  note ste pad_value to -1 for SplitComb
        image = image * 2.0 - 1.0
        # split_images [N, 1, crop_z, crop_y, crop_x]
        split_images, nzhw = self.splitcomb.split(image)
        data['split_images'] = np.ascontiguousarray(split_images)
        data['nzhw'] = nzhw
        data['spacing'] = image_spacing
        data['file_name'] = series_name
        return data