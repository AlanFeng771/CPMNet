# -*- coding: utf-8 -*-
from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import random
from itertools import product
from .utils import compute_bbox3d_intersection_volume
from scipy.ndimage import affine_transform
class InstanceCrop(object):
    """Randomly crop the input image (shape [C, D, H, W])

    Args:
        crop_size (list[int]): The size of the patch to be cropped.
        rand_trans (list[int], optional): Random translation. Defaults to None.
        instance_crop (bool, optional): Flag to enable additional sampling with instance around center. Defaults to True.
        overlap_size (list[int], optional): The size of overlap of sliding window. Defaults to [16, 32, 32].
        tp_ratio (float, optional): Sampling rate for a patch containing at least one lesion. Defaults to 0.7.
        sample_num (int, optional): Patch number per CT. Defaults to 2.
        blank_side (int, optional): Labels within blank_side pixels near patch border is set to ignored. Defaults to 0.
        sample_cls (list[int], optional): The class of the sample. Defaults to [0].
        tp_iou (float, optional): IoU threshold to determine the label of the patches. Defaults to 0.5.
    """
    def __init__(self, crop_size, overlap_ratio: float = 0.25, rand_trans=None, rand_rot=None, instance_crop=True, 
                 tp_ratio=0.7, sample_num=2, blank_side=0, sample_cls=[0], tp_iou=0.5):
        """This is crop function with spatial augmentation for training Lesion Detection.
        """
        self.sample_cls = sample_cls
        self.crop_size = np.array(crop_size, dtype=np.int32)
        self.overlap_ratio = overlap_ratio
        self.overlap_size = (self.crop_size * self.overlap_ratio).astype(np.int32)
        self.stride_size = self.crop_size - self.overlap_size
        
        self.tp_ratio = tp_ratio
        self.sample_num = sample_num
        self.blank_side = blank_side
        self.instance_crop = instance_crop
        self.tp_iou = tp_iou

        if rand_trans == None:
            self.rand_trans = None
        else:
            self.rand_trans = np.array(rand_trans)

        if rand_rot == None:
            self.rand_rot = None
        else:
            self.rand_rot = np.array(rand_rot)

    def get_crop_centers(self, shape, dim: int):
        crop = self.crop_size[dim]
        overlap = self.overlap_size[dim]
        stride = self.stride_size[dim]
        shape = shape[dim]
        
        crop_centers = np.arange(0, shape - overlap, stride) + crop / 2
        crop_centers = np.clip(crop_centers, a_max=shape - crop / 2, a_min=None)
        
        # Add final center
        crop_centers = np.append(crop_centers, shape - crop / 2)
        crop_centers = np.unique(crop_centers)
        
        return crop_centers
    
    def rotate_image(self, image, angle_x, angle_y, angle_z):
        # 計算圖像中心
        center = np.array(image.shape) // 2
        # 將角度轉換為弧度
        angle_x = np.radians(angle_x)
        angle_y = np.radians(angle_y)
        angle_z = np.radians(angle_z)
        # 定義旋轉矩陣
        rotation_matrix = np.array([[np.cos(angle_y)*np.cos(angle_z), -np.cos(angle_x)*np.sin(angle_z) + np.sin(angle_x)*np.sin(angle_y)*np.cos(angle_z), np.sin(angle_x)*np.sin(angle_z) + np.cos(angle_x)*np.sin(angle_y)*np.cos(angle_z), 0],
                                    [np.cos(angle_y)*np.sin(angle_z), np.cos(angle_x)*np.cos(angle_z) + np.sin(angle_x)*np.sin(angle_y)*np.sin(angle_z), -np.sin(angle_x)*np.cos(angle_z) + np.cos(angle_x)*np.sin(angle_y)*np.sin(angle_z), 0],
                                    [-np.sin(angle_y), np.sin(angle_x)*np.cos(angle_y), np.cos(angle_x)*np.cos(angle_y), 0],
                                    [0, 0, 0, 1]])
        # 將圖像中心設置為原點
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = -center
        translation_matrix_inv = np.eye(4)
        translation_matrix_inv[:3, 3] = center
        # 將旋轉和平移合併為一個仿射變換矩陣
        affine_matrix = np.dot(np.dot(translation_matrix_inv, rotation_matrix), translation_matrix)
        # 對圖像應用仿射變換
        rotated_image = affine_transform(image, affine_matrix, order=3)
        return rotated_image

    def rotate_bbox(self, centers, bbox_sizes, rotate_center, angle_x, angle_y, angle_z):
        simbol = np.unique(np.array([*product(np.array([-1, 1, 1]), np.array([1, -1, 1]), np.array([1, 1, -1]))]), axis=0)
        # print(simbol*bbox_sizes/2)
        bbox_sizes = np.expand_dims(bbox_sizes, axis=1)
        centers = np.expand_dims(centers, axis=1)
        pos = centers-simbol*bbox_sizes/2

        # fix format [z. y, x, 1]
        pos_ = np.ones([pos.shape[0], pos.shape[1], pos.shape[2]+1])
        pos_[:pos.shape[0], :pos.shape[1], :pos.shape[2]] = pos
        pos_ = pos_.reshape((-1, 4))

        # 計算圖像中心
        center = rotate_center
        # 將角度轉換為弧度
        angle_x = np.radians(angle_x)
        angle_y = np.radians(angle_y)
        angle_z = np.radians(angle_z)
        # 定義旋轉矩陣
        rotation_matrix = np.array([[np.cos(angle_y)*np.cos(angle_z), -np.cos(angle_x)*np.sin(angle_z) + np.sin(angle_x)*np.sin(angle_y)*np.cos(angle_z), np.sin(angle_x)*np.sin(angle_z) + np.cos(angle_x)*np.sin(angle_y)*np.cos(angle_z), 0],
                                    [np.cos(angle_y)*np.sin(angle_z), np.cos(angle_x)*np.cos(angle_z) + np.sin(angle_x)*np.sin(angle_y)*np.sin(angle_z), -np.sin(angle_x)*np.cos(angle_z) + np.cos(angle_x)*np.sin(angle_y)*np.sin(angle_z), 0],
                                    [-np.sin(angle_y), np.sin(angle_x)*np.cos(angle_y), np.cos(angle_x)*np.cos(angle_y), 0],
                                    [0, 0, 0, 1]])
        # 將圖像中心設置為原點
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = -center
        translation_matrix_inv = np.eye(4)
        translation_matrix_inv[:3, 3] = center
        # 將旋轉和平移合併為一個仿射變換矩陣
        affine_matrix = np.dot(np.dot(translation_matrix_inv, rotation_matrix), translation_matrix)
        # 對圖像應用仿射變換
        # rotated_coord = affine_transform(centers, affine_matrix, order=3)
        rotated_coord =  np.dot(affine_matrix, pos_.T)
        rotated_coord = rotated_coord.T[:, :-1]
        rotated_coord = rotated_coord.reshape((-1, 8, 3)) 
        rotated_center = (np.max(rotated_coord, 1) + np.min(rotated_coord, 1))/2
        rotated_shape = np.max(rotated_coord, 1) - np.min(rotated_coord, 1)
        return rotated_center, rotated_shape
    def rand_rotate(self, image, instance_loc, instance_shape, angle_range_d, angle_range_h, angle_range_w, p):
        rot_center = np.array(image.shape) // 2
        angle_x = 0
        angle_y = 0
        angle_z = 0
        if (angle_range_d[1]-angle_range_d[0] > 0) and (random.random() < p):
            angle_x = np.random.uniform(angle_range_d[0], angle_range_d[1])

        if (angle_range_h[1]-angle_range_h[0] > 0) and (random.random() < p):
            angle_y = np.random.uniform(angle_range_h[0], angle_range_h[1])

        if (angle_range_w[1]-angle_range_w[0] > 0) and (random.random() < p):
            angle_z = np.random.uniform(angle_range_w[0], angle_range_w[1])
        print(angle_x, angle_y, angle_z)
        rotated_image = self.rotate_image(image, angle_x, angle_y, angle_z)
        rotated_loc, rotated_shape = self.rotate_bbox(instance_loc, instance_shape, rot_center,  -angle_x, -angle_y, -angle_z)

        return rotated_image, rotated_loc, rotated_shape
    def __call__(self, sample, image_spacing: np.ndarray):
        image = sample['image']
        all_loc = sample['all_loc']
        all_rad = sample['all_rad']
        all_cls = sample['all_cls']
        all_rad_pixel = all_rad / image_spacing
        if self.rand_rot is not None:
            image, all_loc, all_rad_pixel = self.rand_rotate(image, all_loc , all_rad_pixel, [-self.rand_rot[0], self.rand_rot[0]],
                                    [-self.rand_rot[1], self.rand_rot[1]],
                                    [-self.rand_rot[2], self.rand_rot[2]], p=0.8)
        all_nodule_bb_min = all_loc - all_rad_pixel / 2
        all_nodule_bb_max = all_loc + all_rad_pixel / 2
        nodule_bboxes = np.stack([all_nodule_bb_min, all_nodule_bb_max], axis=1) # [N, 2, 3]
        nodule_volumes = np.prod(all_rad_pixel, axis=1) # [N]
        
        instance_loc = all_loc[np.sum([all_cls == cls for cls in self.sample_cls], axis=0, dtype='bool')]
        shape = image.shape
        crop_size = np.array(self.crop_size)

        z_crop_centers = self.get_crop_centers(shape, 0)
        y_crop_centers = self.get_crop_centers(shape, 1)
        x_crop_centers = self.get_crop_centers(shape, 2)
        
        # Generate crop centers
        crop_centers = [*product(z_crop_centers, y_crop_centers, x_crop_centers)]
        crop_centers = np.array(crop_centers)
        
        if self.instance_crop and len(instance_loc) > 0:
            ## why random twice
            # if self.rand_trans is not None:
            #     instance_crop = instance_loc + np.random.randint(low=-self.rand_trans, high=self.rand_trans, size=(len(instance_loc), 3))
            # else:
            #     instance_crop = instance_loc
            instance_crop = instance_loc
            crop_centers = np.append(crop_centers, instance_crop, axis=0)

        if self.rand_trans is not None:
            crop_centers = crop_centers + np.random.randint(low=-self.rand_trans, high=self.rand_trans, size=(len(crop_centers), 3))
        
        all_crop_bb_min = crop_centers - crop_size / 2
        all_crop_bb_min = np.clip(all_crop_bb_min, a_min=0, a_max=shape - crop_size)
        all_crop_bb_min = np.unique(all_crop_bb_min, axis=0)
        
        all_crop_bb_max = all_crop_bb_min + crop_size
        all_crop_bboxes = np.stack([all_crop_bb_min, all_crop_bb_max], axis=1) # [M, 2, 3]
        
        # Compute IoU to determine the label of the patches
        inter_volumes = compute_bbox3d_intersection_volume(all_crop_bboxes, nodule_bboxes) # [M, N]
        all_ious = inter_volumes / nodule_volumes[np.newaxis, :] # [M, N]
        max_ious = np.max(all_ious, axis=1) # [M]
        
        tp_indices = max_ious > self.tp_iou
        neg_indices = ~tp_indices

        # Sample patches
        tp_prob = self.tp_ratio / tp_indices.sum() if tp_indices.sum() > 0 else 0
        probs = np.zeros(shape=len(max_ious))
        probs[tp_indices] = tp_prob
        probs[neg_indices] = (1. - probs.sum()) / neg_indices.sum() if neg_indices.sum() > 0 else 0
        probs = probs / probs.sum() # normalize
        sample_indices = np.random.choice(np.arange(len(all_crop_bboxes)), size=self.sample_num, p=probs, replace=False)
        
        # Crop patches
        samples = []
        for sample_i in sample_indices:
        # for sample_i in tp_indices:
            crop_bb_min = all_crop_bb_min[sample_i].astype(np.int32)
            crop_bb_max = crop_bb_min + crop_size
            image_crop = image[crop_bb_min[0]: crop_bb_max[0], 
                               crop_bb_min[1]: crop_bb_max[1], 
                               crop_bb_min[2]: crop_bb_max[2]]
            image_crop = np.expand_dims(image_crop, axis=0)
            
            ious = all_ious[sample_i] # [N]
            in_idx = np.where(ious > self.tp_iou)[0]
            if in_idx.size > 0:
                # Compute new ctr and rad because of the crop
                all_nodule_bb_min_crop = all_nodule_bb_min - crop_bb_min
                all_nodule_bb_max_crop = all_nodule_bb_max - crop_bb_min
                
                nodule_bb_min_crop = all_nodule_bb_min_crop[in_idx]
                nodule_bb_max_crop = all_nodule_bb_max_crop[in_idx]
                
                nodule_bb_min_crop = np.clip(nodule_bb_min_crop, a_min=0, a_max=None)
                nodule_bb_max_crop = np.clip(nodule_bb_max_crop, a_min=None, a_max=crop_size)
                
                ctr = (nodule_bb_min_crop + nodule_bb_max_crop) / 2
                rad = nodule_bb_max_crop - nodule_bb_min_crop
                cls = all_cls[in_idx]
            else:
                ctr = np.array([]).reshape(-1, 3)
                rad = np.array([])
                cls = np.array([])

            sample = dict()
            sample['image'] = image_crop
            sample['ctr'] = ctr
            sample['rad'] = rad
            sample['cls'] = cls
            sample['spacing'] = image_spacing
            samples.append(sample)
        return samples