# -*- coding: utf-8 -*-
from __future__ import print_function, division

import json
from .abstract_transform import AbstractTransform
from .image_process import *
from .ctr_transform import EmptyTransform, TransposeCTR, RotateCTR
import random
import math
from typing import Tuple

def rotate_vecs_3d(vec, angle, axis):
    rad = np.deg2rad(angle)
    rotated_vec = vec.copy()
    rotated_vec[::, axis[0]] = vec[::, axis[0]] * np.cos(rad) - vec[::, axis[1]] * np.sin(rad)
    rotated_vec[::, axis[1]] = vec[::, axis[0]] * np.sin(rad) + vec[::, axis[1]] * np.cos(rad)
    return rotated_vec

class RandomRotate90(AbstractTransform):
    def __init__(self, p=0.5, rot_xy: bool = True, rot_xz: bool = False, rot_yz: bool = False):
        self.p = p
        self.rot_xy = rot_xy
        self.rot_xz = rot_xz
        self.rot_yz = rot_yz
        self.rot_angles = np.array([90, 180, 270], dtype=np.int32)
    
    def __call__(self, sample):
        image = sample['image']
        image_shape = image.shape[1:] # remove channel dimension
        
        all_rot_axes = []
        rot_angles = []
        if random.random() < self.p and self.rot_xy:
            all_rot_axes.append((2, 1))
            rot_angles.append(np.random.choice(self.rot_angles, 1)[0])
        
        if random.random() < self.p and self.rot_xz:
            all_rot_axes.append((2, 0))
            rot_angles.append(np.random.choice(self.rot_angles, 1)[0])
        
        if random.random() < self.p and self.rot_yz:
            all_rot_axes.append((1, 0))
            rot_angles.append(np.random.choice(self.rot_angles, 1)[0])
        
        if len(all_rot_axes) > 0:
            rot_image = sample['image']
            rot_ctr = sample['ctr']
            rot_rad = sample['rad']
            rot_spacing = sample['spacing']
            for rot_axes, rot_angle in zip(all_rot_axes, rot_angles):
                rot_image = self.rotate_3d_image(rot_image, rot_axes, rot_angle)
                if len(rot_ctr) != 0:
                    rot_ctr, rot_rad, rot_spacing = self.rotate_3d_bbox(rot_ctr, rot_rad, rot_spacing, image_shape, rot_axes, rot_angle)
                sample['ctr_transform'].append(RotateCTR(rot_angle, rot_axes, image_shape))
            sample['image'] = rot_image
            sample['ctr'] = rot_ctr
            sample['rad'] = rot_rad
            sample['spacing'] = rot_spacing
        return sample
    
    @staticmethod
    def rotate_3d_image(data: np.ndarray, rot_axes: Tuple[int], rot_angle: int):
        """
        Args:
            data: 3D image data with shape (D, H, W).
            rot_axes: rotation axes.
            rot_angle: rotation angle. One of 90, 180, or 270.
        """
        rotated_data = data.copy()
        rot_axes = np.array(rot_axes) + 1 # Image has channel dimension in the first dimension 
        rotated_data = np.rot90(rotated_data, k=rot_angle // 90, axes=rot_axes)
        return rotated_data

    @staticmethod
    def rotate_3d_bbox(ctrs: np.ndarray, bbox_shapes: np.ndarray, image_spacing: np.ndarray, image_shape: np.ndarray, rot_axes: Tuple[int], angle: int):
        """
        Args:
            ctrs: 3D bounding box centers with shape (N, 3).
            bbox_shapes: 3D bounding box shapes with shape (N, 3).
            image_shape: 3D image shape with shape (3,).
            angle: rotation angle. One of 90, 180, or 270.
            plane: rotation plane. One of 'xy', 'xz', or 'yz'.
        """
        radian = math.radians(angle)
        cos = np.cos(radian)
        sin = np.sin(radian)
        img_center = np.array(image_shape) / 2
        new_ctr_zyx = ctrs.copy()
        new_ctr_zyx[:, rot_axes[0]] = (ctrs[:, rot_axes[0]] - img_center[rot_axes[0]]) * cos - (ctrs[:, rot_axes[1]] - img_center[rot_axes[1]]) * sin + img_center[rot_axes[0]]
        new_ctr_zyx[:, rot_axes[1]] = (ctrs[:, rot_axes[0]] - img_center[rot_axes[0]]) * sin + (ctrs[:, rot_axes[1]] - img_center[rot_axes[1]]) * cos + img_center[rot_axes[1]]
        
        new_shape_dhw = bbox_shapes.copy()
        new_image_spacing = image_spacing.copy()
        if angle == 90 or angle == 270:
            new_shape_dhw[:, rot_axes[0]] = bbox_shapes[:, rot_axes[1]] 
            new_shape_dhw[:, rot_axes[1]] = bbox_shapes[:, rot_axes[0]]
            new_image_spacing[rot_axes[0]] = image_spacing[rot_axes[1]]
            new_image_spacing[rot_axes[1]] = image_spacing[rot_axes[0]]
        return new_ctr_zyx, new_shape_dhw, new_image_spacing

class RandomRotate(AbstractTransform):
    """
    random rotate the image (shape [C, D, H, W] or [C, H, W]) 
    """

    def __init__(self, angle_range_d, angle_range_h, angle_range_w, only_one=True, reshape=True, p=0.3):
        """
        angle_range_d (tuple/list/None) : rorate angle range along depth axis (degree),
               only used for 3D images
        angle_range_h (tuple/list/None) : rorate angle range along height axis (degree)
        angle_range_w (tuple/list/None) : rorate angle range along width axis (degree)
        """
        self.angle_range_d = angle_range_d
        self.angle_range_h = angle_range_h
        self.angle_range_w = angle_range_w
        self.only_one = only_one
        self.reshape = reshape
        self.p = p

    def __apply_transformation(self, image, transform_param_list, order=1, cval=0):
        """
        apply rotation transformation to an ND image
        Args:
            image (nd array): the input nd image
            transform_param_list (list): a list of roration angle and axes
            order (int): interpolation order
        """
        for angle, axes in transform_param_list:
            image = ndimage.rotate(image, angle, axes, reshape=self.reshape, order=order, cval=cval)

        return image

    def __apply_transformation_coord(self, image, coord, transform_param_list, order=1, cval=0):
        """
        apply rotation transformation to an ND image
        Args:
            image (nd array): the input nd image
            transform_param_list (list): a list of roration angle and axes
            order (int): interpolation order
        """
        for angle, axes in transform_param_list:
            org_center = (np.array(image.shape[-3:]) - 1) / 2.
            image = ndimage.rotate(image, angle, axes, reshape=self.reshape, order=order, cval=cval)
            rot_center = (np.array(image.shape[-3:]) - 1) / 2.

            org = coord - org_center
            new = rotate_vecs_3d(org, angle, axes)
            coord = new + rot_center

        return image, coord

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1

        transform_param_list = []

        if (self.angle_range_d is not None) and random.random() < self.p:
            angle_d = np.random.uniform(self.angle_range_d[0], self.angle_range_d[1])
            transform_param_list.append([angle_d, (-2, -1)])
        if (self.angle_range_h is not None) and random.random() < self.p:
            angle_h = np.random.uniform(self.angle_range_h[0], self.angle_range_h[1])
            transform_param_list.append([angle_h, (-3, -1)])
        if (self.angle_range_w is not None) and random.random() < self.p:
            angle_w = np.random.uniform(self.angle_range_w[0], self.angle_range_w[1])
            transform_param_list.append([angle_w, (-3, -2)])

        if self.only_one and len(transform_param_list) > 0:
            transform_param_list = random.sample(transform_param_list, 1)

        if len(transform_param_list) > 0:
            if 'ctr' in sample:
                image_t, coord = self.__apply_transformation_coord(image, sample['ctr'].copy(), transform_param_list,
                                                                   1)
                sample['ctr'] = coord
            else:
                image_t = self.__apply_transformation(image, transform_param_list, 1)
            sample['image'] = image_t

        return sample

class RandomTranspose(AbstractTransform):
    """
    random rotate the image (shape [C, D, H, W] or [C, H, W])
    """

    def __init__(self, trans_xy=True, trans_zx=False, trans_zy=False, p=0.5):
        self.trans_xy = trans_xy
        self.trans_zx = trans_zx
        self.trans_zy = trans_zy
        self.p = p

    def __call__(self, sample):
        transpose_list = []

        if self.trans_zy and random.random() < self.p:
            transpose_list.append(np.array([0, 2, 1, 3]))
        if self.trans_xy and random.random() < self.p:
            transpose_list.append(np.array([0, 1, 3, 2]))
        if self.trans_zx and random.random() < self.p:
            transpose_list.append(np.array([0, 3, 2, 1]))

        if len(transpose_list) > 0:
            transpose_order = np.array([0, 1, 2, 3])
            for transpose in transpose_list:
                transpose_order = transpose_order[transpose]
            sample['image'] = np.transpose(sample['image'], transpose_order)
            sample['ctr'] = sample['ctr'][:, transpose_order[1:] - 1]
            sample['rad'] = sample['rad'][:, transpose_order[1:] - 1]
            sample['spacing'] = sample['spacing'][transpose_order[1:] - 1]
            sample['ctr_transform'].append(TransposeCTR(transpose_order))
        return sample

class RandomMaskTranspose(AbstractTransform):
    """
    random rotate the image (shape [C, D, H, W] or [C, H, W])
    """

    def __init__(self, trans_xy=True, trans_zx=False, trans_zy=False, p=0.5):
        self.trans_xy = trans_xy
        self.trans_zx = trans_zx
        self.trans_zy = trans_zy
        self.p = p

    def __call__(self, sample):
        transpose_list = []

        if self.trans_zy and random.random() < self.p:
            transpose_list.append((0, 2, 1, 3))
        if self.trans_xy and random.random() < self.p:
            transpose_list.append((0, 1, 3, 2))
        if self.trans_zx and random.random() < self.p:
            transpose_list.append((0, 3, 2, 1))

        if len(transpose_list) > 0:
            image_t = sample['image']
            mask_t = sample['mask']
            for transpose in transpose_list:
                image_t = np.transpose(image_t, transpose)
                mask_t = np.transpose(mask_t, transpose)

            sample['image'] = image_t
            sample['mask'] = mask_t

        return sample

class RandomMaskRotate(AbstractTransform):
    """
    random rotate the image (shape [C, D, H, W] or [C, H, W]) 
    """

    def __init__(self, angle_range_d, angle_range_h, angle_range_w, reshape=True, p=0.3):
        """
        angle_range_d (tuple/list/None) : rorate angle range along depth axis (degree),
               only used for 3D images
        angle_range_h (tuple/list/None) : rorate angle range along height axis (degree)
        angle_range_w (tuple/list/None) : rorate angle range along width axis (degree)
        """
        self.angle_range_d = angle_range_d
        self.angle_range_h = angle_range_h
        self.angle_range_w = angle_range_w
        self.reshape = reshape
        self.p = p

    def __apply_transformation(self, image, transform_param_list, order=1, cval=0):
        """
        apply rotation transformation to an ND image
        Args:
            image (nd array): the input nd image
            transform_param_list (list): a list of roration angle and axes
            order (int): interpolation order
        """
        for angle, axes in transform_param_list:
            image = ndimage.rotate(image, angle, axes, reshape=self.reshape, order=order, cval=cval)
        return image

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']

        transform_param_list = []

        if (self.angle_range_d is not None) and random.random() < self.p:
            angle_d = np.random.uniform(self.angle_range_d[0], self.angle_range_d[1])
            transform_param_list.append([angle_d, (-2, -1)])
        if (self.angle_range_h is not None) and random.random() < self.p:
            angle_h = np.random.uniform(self.angle_range_h[0], self.angle_range_h[1])
            transform_param_list.append([angle_h, (-3, -1)])
        if (self.angle_range_w is not None) and random.random() < self.p:
            angle_w = np.random.uniform(self.angle_range_w[0], self.angle_range_w[1])
            transform_param_list.append([angle_w, (-3, -2)])

        if len(transform_param_list) > 0:
            image_t = self.__apply_transformation(image, transform_param_list, 1)
            mask_t = self.__apply_transformation(mask, transform_param_list, 0)
            sample['image'] = image_t
            sample['mask'] = mask_t

        return sample