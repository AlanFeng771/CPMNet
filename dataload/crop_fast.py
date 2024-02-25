# -*- coding: utf-8 -*-
from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import random

class InstanceCrop(object):
    """Randomly crop the input image (shape [C, D, H, W]
    """

    def __init__(self, crop_size, rand_trans=None, rand_rot=None, rand_space=None, instance_crop=True, overlap_size=[16, 32, 32], tp_ratio=0.7, sample_num=2, blank_side=0, sample_cls=[0]):
        """This is crop function with spatial augmentation for training Lesion Detection.

        Arguments:
            crop_size: patch size
            rand_trans: random translation
            rand_rot: random rotation
            rand_space: random spacing
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
        self.overlap_size = np.array(overlap_size, dtype=np.int32)
        self.stride_size = self.crop_size - self.overlap_size
        
        self.tp_ratio = tp_ratio
        self.sample_num = sample_num
        self.blank_side = blank_side
        self.instance_crop = instance_crop

        if rand_trans == None:
            self.rand_trans = None
        else:
            self.rand_trans = np.array(rand_trans)

        if rand_rot == None:
            self.rand_rot = None
        else:
            self.rand_rot = np.array(rand_rot)

        if rand_space == None:
            self.rand_space = None
        else:
            self.rand_space = np.array(rand_space)

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
        instance_loc = all_loc[np.sum([all_cls == cls for cls in self.sample_cls], axis=0, dtype='bool')]

        image_itk = sitk.GetImageFromArray(image)
        shape = image.shape
        crop_size = np.array(self.crop_size)
        overlap_size = self.overlap_size

        z_crop_centers = self.get_crop_centers(shape, 0)
        y_crop_centers = self.get_crop_centers(shape, 1)
        x_crop_centers = self.get_crop_centers(shape, 2)
        
        crop_centers = []
        for z in z_crop_centers:
            for y in y_crop_centers:
                for x in x_crop_centers:
                    crop_centers.append(np.array([z, y, x]))

        if self.instance_crop and len(instance_loc) > 0:
            if self.rand_trans is not None:
                instance_crop = instance_loc + np.random.randint(low=-self.rand_trans, high=self.rand_trans,size=3)
            else:
                instance_crop = instance_loc
            crop_centers = np.append(crop_centers, instance_crop, axis=0)

        matrixs = []
        tp_nums = []
        for i in range(len(crop_centers)):
            C = crop_centers[i]
            if self.rand_trans is not None:
                C = C + np.random.randint(low=-self.rand_trans, high=self.rand_trans, size=3)

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
            bb_min = np.maximum(matrix[0] - 10, 0)
            bb_max = bb_min + crop_size + 10
            if len(all_loc) == 0:
                tp_num = 0
            else:
                tp_num = np.sum((nodule_bb_min > bb_min).all(axis=1) & (nodule_bb_max < bb_max).all(axis=1))
            tp_nums.append(tp_num)
        
        # Sample patches
        tp_nums = np.array(tp_nums)
        tp_idx = tp_nums > 0
        neg_idx = tp_nums == 0

        if tp_idx.sum() > 0:
            tp_pos = self.tp_ratio / tp_idx.sum()
        else:
            tp_pos = 0

        p = np.zeros(shape=tp_nums.shape)
        p[tp_idx] = tp_pos
        p[neg_idx] = (1. - p.sum()) / neg_idx.sum() if neg_idx.sum() > 0 else 0
        p = p * 1 / p.sum()

        sample_indices = np.random.choice(np.arange(len(crop_centers)), size=self.sample_num, p=p, replace=False)
        
        # Crop patches
        samples = []
        for sample_i in sample_indices:
            space = np.array([1.0, 1.0, 1.0], dtype=np.float64)
            matrix = matrixs[sample_i]
            matrix = matrix[:, ::-1]  # in itk axis
            
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


def convert_to_one_hot(label, class_num):
    label_prob = []
    for i in range(class_num):
        temp_prob = label == i * np.ones_like(label)
        label_prob.append(temp_prob)
    label_prob = np.asarray(label_prob, dtype='float32')
    return label_prob


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


def resample_simg(simg, interp=sitk.sitkBSpline, spacing=[1., 0.7, 0.7]):
    identity1 = sitk.Transform(3, sitk.sitkIdentity)
    new_spacing = spacing[::-1]

    sp1 = simg.GetSpacing()
    sz1 = simg.GetSize()
    sz2 = (int(round(sz1[0] * sp1[0] / new_spacing[0])), int(round(sz1[1] * sp1[1] / new_spacing[1])),
           int(round(sz1[2] * sp1[2] / new_spacing[2])))

    new_origin = simg.GetOrigin()
    new_origin = (new_origin[0] - sp1[0] / 2 + new_spacing[0] / 2, new_origin[1] - sp1[1] / 2 + new_spacing[1] / 2,
                  new_origin[2] - sp1[2] / 2 + new_spacing[2] / 2)
    imRefImage = sitk.Image(sz2, simg.GetPixelIDValue())
    imRefImage.SetSpacing(new_spacing)
    imRefImage.SetOrigin(new_origin)
    imRefImage.SetDirection(simg.GetDirection())
    resampled_image = sitk.Resample(simg, imRefImage, identity1, interp)
    return resampled_image