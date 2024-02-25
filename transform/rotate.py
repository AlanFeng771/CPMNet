# -*- coding: utf-8 -*-
from __future__ import print_function, division

import json
from .abstract_transform import AbstractTransform
from .image_process import *
from .ctr_transform import EmptyTransform, TransposeCTR, RotateCTR
import random


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
        
        rot_planes = []
        rot_angles = []
        if random.random() < self.p and self.rot_xy:
            rot_planes.append('xy')
            rot_angles.append(90)
            # rot_angles.append(np.random.choice(self.rot_angles, 1)[0])
        
        if random.random() < self.p and self.rot_xz:
            rot_planes.append('xz')
            rot_angles.append(np.random.choice(self.rot_angles, 1)[0])
        
        if random.random() < self.p and self.rot_yz:
            rot_planes.append('yz')
            rot_angles.append(np.random.choice(self.rot_angles, 1)[0])
        
        if len(rot_planes) > 0:
            rot_image = sample['image']
            rot_ctr = sample['ctr']
            rot_rad = sample['rad']
            for rot_plane, rot_angle in zip(rot_planes, rot_angles):
                rot_image = self.rotate_3d_image(rot_image, rot_plane, rot_angle)
                if len(rot_ctr) != 0:
                    rot_ctr, rot_rad = self.rotate_3d_bbox(rot_ctr, rot_rad, image_shape, rot_angle, rot_plane)
                sample['ctr_transform'].append(RotateCTR(rot_angle, rot_plane, image_shape))
            sample['image'] = rot_image
            sample['ctr'] = rot_ctr       
            sample['rad'] = rot_rad
        return sample
    
    @staticmethod
    def rotate_3d_image(data: np.ndarray, rot_plane: str, rot_angle: int):
        """
        Args:
            data: 3D image data with shape (D, H, W).
            rot_plane: rotation plane. One of 'xy', 'xz', or 'yz'.
            rot_angle: rotation angle. One of 90, 180, or 270.
        """
        rot_angle = rot_angle * -1 # from counter-clockwise to clockwise
        rotated_data = data.copy()
        if rot_plane == 'yz':
            rotated_data = np.rot90(rotated_data, k=rot_angle // 90, axes=(1, 0))
        elif rot_plane == 'xz':
            rotated_data = np.rot90(rotated_data, k=rot_angle // 90, axes=(2, 0))
        elif rot_plane == 'xy':
            rotated_data = np.rot90(rotated_data, k=rot_angle // 90, axes=(2, 1))
        else:
            raise ValueError("Invalid rotation_plane. Please choose from 'xy', 'xz', or 'yz'.")
        return rotated_data

    @staticmethod
    def rotate_3d_bbox(ctrs: np.ndarray, bbox_shapes: np.ndarray, image_shape: np.ndarray, angle: int, plane: str):
        """
        Args:
            ctrs: 3D bounding box centers with shape (N, 3).
            bbox_shapes: 3D bounding box shapes with shape (N, 3).
            image_shape: 3D image shape with shape (3,).
            angle: rotation angle. One of 90, 180, or 270.
            plane: rotation plane. One of 'xy', 'xz', or 'yz'.
        """
        # ctrs = np.array(ctrs)
        # bbox_shapes = np.array(bbox_shapes)
        # image_shape = np.array(image_shape)
        if plane == 'xy':
            axes = (2, 1)
        elif plane == 'yz':
            axes = (1, 0)
        elif plane == 'xz':
            axes = (2, 0)
        
        radian = np.deg2rad(angle)
        cos = np.cos(radian)
        sin = np.sin(radian)
        img_center = np.array(image_shape) / 2
        
        new_ctr_zyx = ctrs.copy()
        new_ctr_zyx[:, axes[0]] = (ctrs[:, axes[0]] - img_center[axes[0]]) * cos + (ctrs[:, axes[1]] - img_center[axes[1]]) * sin + img_center[axes[0]]
        new_ctr_zyx[:, axes[1]] = (ctrs[:, axes[0]] - img_center[axes[0]]) * -sin + (ctrs[:, axes[1]] - img_center[axes[1]]) * cos + img_center[axes[1]]
        
        new_shape_dhw = bbox_shapes.copy()
        if angle == 90 or angle == 270:
            new_shape_dhw[:, axes[0]] = bbox_shapes[:, axes[1]] 
            new_shape_dhw[:, axes[1]] = bbox_shapes[:, axes[0]]
        return new_ctr_zyx, new_shape_dhw

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