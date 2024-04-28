# -*- coding: utf-8 -*-
from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import random
from itertools import product
from .utils import compute_bbox3d_intersection_volume
class InstanceCrop(object):
    """Randomly crop the input image (shape [C, D, H, W])

    Args:
        crop_size (list[int]): The size of the patch to be cropped.
        rand_trans (list[int], optional): The range of random translation. Defaults to None.
        rand_rot (list[int], optional): The range of random rotation. Defaults to None.
        instance_crop (bool, optional): Whether to perform additional sampling with instance around the center. Defaults to True.
        overlap_size (list[int], optional): The size of overlap of sliding window. Defaults to [16, 32, 32].
        tp_ratio (float, optional): The sampling rate for a patch containing at least one lesion. Defaults to 0.7.
        sample_num (int, optional): The number of patches per CT. Defaults to 2.
        blank_side (int, optional): The number of pixels near the patch border where labels are set to ignored. Defaults to 0.
        sample_cls (list[int], optional): The list of classes to sample patches from. Defaults to [0].
    """

    def __init__(self, crop_size, overlap_ratio: float = 0.25, rand_trans=None, rand_rot=None, instance_crop=True, 
                 tp_ratio=0.7, sample_num=2, blank_side=0, sample_cls=[0], tp_iou=0.5):
        """This is crop function with spatial augmentation for training Lesion Detection.

        Arguments:
            crop_size: patch size
            rand_trans: random translation
            rand_rot: random rotation
            instance_crop: additional sampling with instance around center
            spacing: output patch spacing, [z,y,x]
            base_spacing: spacing of the numpy image.
            overlap_size: the size of overlap  of sliding window
            tp_ratio: sampling rate for a patch containing at least one leision
            sample_num: patch number per CT
            blank_side:  labels within blank_side pixels near patch border is set to ignored.
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
    
    def __call__(self, sample, image_spacing: np.ndarray):
        image = sample['image'].astype('float32')
        all_loc = sample['all_loc']
        all_rad = sample['all_rad']
        all_cls = sample['all_cls']
        
        all_rad_pixel = all_rad / image_spacing
        nodule_bb_min = all_loc - all_rad_pixel / 2
        nodule_bb_max = all_loc + all_rad_pixel / 2
        nodule_bboxes = np.stack([nodule_bb_min, nodule_bb_max], axis=1) # [N, 2, 3]
        nodule_volumes = np.prod(all_rad_pixel, axis=1) # [N]
        instance_loc = all_loc[np.sum([all_cls == cls for cls in self.sample_cls], axis=0, dtype='bool')]
        

        image_itk = sitk.GetImageFromArray(image)
        shape = image.shape
        crop_size = np.array(self.crop_size)

        z_crop_centers = self.get_crop_centers(shape, 0)
        y_crop_centers = self.get_crop_centers(shape, 1)
        x_crop_centers = self.get_crop_centers(shape, 2)
        
        crop_centers = [*product(z_crop_centers, y_crop_centers, x_crop_centers)]
        crop_centers = np.array(crop_centers)
        
        if self.instance_crop and len(instance_loc) > 0:
            crop_centers = np.append(crop_centers, instance_loc, axis=0)
        
        if self.rand_trans is not None:
            crop_centers = crop_centers + np.random.randint(low=-self.rand_trans, high=self.rand_trans, size=(len(crop_centers), 3))

        matrixs = []
        all_crop_bboxes = None
        all_crop_bb_min = None
        all_crop_bb_max = None
        for i in range(len(crop_centers)):
            C = crop_centers[i]

            O = C - np.array(crop_size) / 2
            Z = O + np.array([crop_size[0] - 1, 0, 0])
            Y = O + np.array([0, crop_size[1] - 1, 0])
            X = O + np.array([0, 0, crop_size[2] - 1])
            matrix = np.array([O, X, Y, Z])
            if self.rand_rot is not None:
                matrix = rand_rot_coord(matrix, [-self.rand_rot[0], self.rand_rot[0]],
                                        [-self.rand_rot[1], self.rand_rot[1]],
                                        [-self.rand_rot[2], self.rand_rot[2]], rot_center=C, p=0.8)
            matrixs.append(matrix)
            # According to the matrixs, we can decide if the crop is foreground or background
            crop_bb_min = matrix[0]
            crop_bb_min = np.clip(crop_bb_min, a_min=0, a_max=shape - crop_size)
            crop_bb_max = crop_bb_min + crop_size
            crop_bbox = np.stack([crop_bb_min, crop_bb_max], axis=0) # [2, 3]
            crop_bbox = np.expand_dims(crop_bbox, axis=0) # [1, 2, 3]
            crop_bb_min = np.expand_dims(crop_bb_min, axis=0) # [1, 2, 3]
            crop_bb_max = np.expand_dims(crop_bb_max, axis=0) # [1, 2, 3]
            if all_crop_bboxes is not None:
                all_crop_bboxes = np.concatenate([all_crop_bboxes, crop_bbox], axis=0)
                all_crop_bb_min = np.concatenate([all_crop_bb_min, crop_bb_min], axis=0)
                all_crop_bb_max = np.concatenate([all_crop_bb_max, crop_bb_max], axis=0)
            else:
                all_crop_bboxes = crop_bbox
                all_crop_bb_min = crop_bb_min
                all_crop_bb_max = crop_bb_max
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
            space = np.array([1.0, 1.0, 1.0], dtype=np.float64)
            matrix = matrixs[sample_i]
            matrix = matrix[:, ::-1]  # in itk axis
            crop_bb_min = all_crop_bb_min[sample_i].astype(np.int32)
            crop_bb_max = crop_bb_min + crop_size
            image_itk_crop = reorient(image_itk, matrix, spacing=list(space), interp1=sitk.sitkLinear)
            all_loc_crop = [image_itk_crop.TransformPhysicalPointToContinuousIndex(c.tolist()[::-1])[::-1] for c in
                            all_loc]
            all_loc_crop = np.array(all_loc_crop)
            in_idx = []
            for j in range(all_loc_crop.shape[0]):
                if (all_loc_crop[j] <= np.array(image_itk_crop.GetSize()[::-1])).all() and (
                        all_loc_crop[j] >= np.zeros([3])).all():
                    in_idx.append(True)
                else:
                    in_idx.append(False)
            in_idx = np.array(in_idx)

            if in_idx.size > 0:
                ctr = all_loc_crop[in_idx]
                rad = all_rad[in_idx]
                cls = all_cls[in_idx]
            else:
                ctr = np.array([]).reshape(-1, 3)
                rad = np.array([])
                cls = np.array([])
           

            image_crop = sitk.GetArrayFromImage(image_itk_crop)
            CT_crop = np.expand_dims(image_crop, axis=0)
            shape = np.array(CT_crop.shape[1:])
            if len(rad) > 0:
                rad = rad / image_spacing  # convert pixel coord
            sample = dict()
            sample['image'] = CT_crop
            sample['ctr'] = ctr
            sample['rad'] = rad
            sample['cls'] = cls
            sample['spacing'] = image_spacing
            samples.append(sample)
        return samples

def rotate_vecs_3d(vec, angle, axis):
    rad = np.deg2rad(angle)
    rotated_vec = vec.copy()
    rotated_vec[::, axis[0]] = vec[::, axis[0]] * np.cos(rad) - vec[::, axis[1]] * np.sin(rad)
    rotated_vec[::, axis[1]] = vec[::, axis[0]] * np.sin(rad) + vec[::, axis[1]] * np.cos(rad)
    return rotated_vec

def apply_transformation_coord(coord, transform_param_list, rot_center):
    """
    apply rotation transformation to an ND image
    Args:
        image (nd array): the input nd image
        transform_param_list (list): a list of roration angle and axes
        order (int): interpolation order
    """
    for angle, axes in transform_param_list:
        # rot_center = np.random.uniform(low=np.min(coord, axis=0), high=np.max(coord, axis=0), size=3)
        org = coord - rot_center
        new = rotate_vecs_3d(org, angle, axes)
        coord = new + rot_center

    return coord

def rand_rot_coord(coord, angle_range_d, angle_range_h, angle_range_w, rot_center, p):
    transform_param_list = []

    if (angle_range_d[1]-angle_range_d[0] > 0) and (random.random() < p):
        angle_d = np.random.uniform(angle_range_d[0], angle_range_d[1])
        transform_param_list.append([angle_d, (-2, -1)])
    if (angle_range_h[1]-angle_range_h[0] > 0) and (random.random() < p):
        angle_h = np.random.uniform(angle_range_h[0], angle_range_h[1])
        transform_param_list.append([angle_h, (-3, -1)])
    if (angle_range_w[1]-angle_range_w[0] > 0) and (random.random() < p):
        angle_w = np.random.uniform(angle_range_w[0], angle_range_w[1])
        transform_param_list.append([angle_w, (-3, -2)])

    if len(transform_param_list) > 0:
        coord = apply_transformation_coord(coord, transform_param_list, rot_center)

    return coord

def reorient(itk_img, mark_matrix, spacing=[1., 1., 1.], interp1=sitk.sitkLinear):
    '''
    itk_img: image to reorient
    mark_matric: physical mark point
    '''
    spacing = spacing[::-1]
    origin, x_mark, y_mark, z_mark = np.array(mark_matrix[0]), np.array(mark_matrix[1]), np.array(
        mark_matrix[2]), np.array(mark_matrix[3])

    # centroid_world = itk_img.TransformContinuousIndexToPhysicalPoint(centroid)
    filter_resample = sitk.ResampleImageFilter()
    filter_resample.SetInterpolator(interp1)
    filter_resample.SetOutputSpacing(spacing)

    # set origin
    origin_reorient = mark_matrix[0]
    # set direction
    # !!! note: column wise
    x_base = (x_mark - origin) / np.linalg.norm(x_mark - origin)
    y_base = (y_mark - origin) / np.linalg.norm(y_mark - origin)
    z_base = (z_mark - origin) / np.linalg.norm(z_mark - origin)
    direction_reorient = np.stack([x_base, y_base, z_base]).transpose().reshape(-1).tolist()

    # set size
    x, y, z = np.linalg.norm(x_mark - origin) / spacing[0], np.linalg.norm(y_mark - origin) / spacing[
        1], np.linalg.norm(z_mark - origin) / spacing[2]
    size_reorient = (int(np.ceil(x + 0.5)), int(np.ceil(y + 0.5)), int(np.ceil(z + 0.5)))

    filter_resample.SetOutputOrigin(origin_reorient)
    filter_resample.SetOutputDirection(direction_reorient)
    filter_resample.SetSize(size_reorient)
    # filter_resample.SetSpacing([sp]*3)

    filter_resample.SetOutputPixelType(itk_img.GetPixelID())
    itk_out = filter_resample.Execute(itk_img)

    return itk_out