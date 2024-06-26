from typing import List, Tuple, Dict, Any, Union
import numpy as np
import torch

def train_collate_fn(batches) -> Dict[str, torch.Tensor]:
    batch = []
    for b in batches:
        batch.extend(b)
        
    imgs = []
    annots = []
    for b in batch:
        imgs.append(b['image'])
        annots.append(b['annot'])
    imgs = np.stack(imgs)
    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        annot_padded = np.ones((len(annots), max_num_annots, 10), dtype='float32') * -1
        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = np.ones((len(annots), 1, 10), dtype='float32') * -1

    return {'image': torch.from_numpy(imgs), 'annot': torch.from_numpy(annot_padded)}

def validation_collate_fn(batches) -> Dict[str, torch.Tensor]:
    batch = []
    for b in batches:
        batch.extend(b)
        
    imgs = []
    for b in batch:
        imgs.append(b['image'])
        
    imgs = np.stack(imgs)
   
    return {'image': torch.from_numpy(imgs)}

def infer_collate_fn(batches) -> Dict[str, torch.Tensor]:
    num_splits = []
    imgs = []
    nzhws = []
    spacings = []
    series_names = []
    series_folders = []

    for b in batches:
        imgs.append(b['split_images'])
        num_splits.append(b['split_images'].shape[0])
        nzhws.append(b['nzhw'])
        spacings.append(b['spacing'])
        series_names.append(b['series_name'])
        series_folders.append(b['series_folder'])
        
    imgs = np.concatenate(imgs, axis=0)
    nzhws = np.stack(nzhws)
    num_splits = np.array(num_splits)
    
    return {'split_images': torch.from_numpy(imgs),
            'nzhws': torch.from_numpy(nzhws), 
            'num_splits': num_splits, 
            'spacings': spacings, 
            'series_names': series_names,
            'series_folders': series_folders}

def infer_refined_collate_fn(batches) -> Dict[str, torch.Tensor]:
    num_splits = []
    crop_images = []
    nodule_centers = []
    nodule_shapes = []
    crop_bb_mins = []
    series_names = []
    series_paths = []

    for b in batches:
        crop_images.append(b['crop_images'])
        num_splits.append(b['crop_images'].shape[0])
        nodule_centers.append(b['nodule_centers'])
        nodule_shapes.append(b['nodule_shapes'])
        crop_bb_mins.append(b['crop_bb_mins'])
        series_names.append(b['series_name'])
        series_paths.append(b['series_path'])
        
    crop_images = np.concatenate(crop_images, axis=0)
    
    return {'crop_images': torch.from_numpy(crop_images),
            'num_splits': np.array(num_splits),
            'nodule_centers': nodule_centers,
            'nodule_shapes': nodule_shapes,
            'crop_bb_mins': crop_bb_mins,
            'series_names': series_names,
            'series_paths': series_paths}

def unlabeled_train_collate_fn(batches: Dict[str, List[Dict[str, any]]]):
    """
    Args:
        batches: Dict[str, List[Dict[str, any]]]
            A dictionary with keys 'weak' and 'strong', each of which is a list of dictionaries.
            Each dictionary contains keys 'image', 'annot', 'ctr_transform'.
    """
    
    # Prepare weak and strong batches
    weak_batch = []
    strong_batch = []
    for b in batches['weak']:
        weak_batch.extend(b)
    for b in batches['strong']:
        strong_batch.extend(b)
    
    # Prepare weak and strong images and annotations
    weak_imgs = []
    weak_annots = []
    strong_imgs = []
    strong_annots = []
    for b in weak_batch:
        weak_imgs.append(b['image'])
        weak_annots.append(b['annot'])
    for b in strong_batch:
        strong_imgs.append(b['image'])
        strong_annots.append(b['annot'])
    
    weak_imgs = np.stack(weak_imgs)
    strong_imgs = np.stack(strong_imgs)
    
    weak_max_num_annots = max(annot.shape[0] for annot in weak_annots)
    strong_max_num_annots = max(annot.shape[0] for annot in strong_annots)
    
    # Prepare weak and strong center transforms
    weak_ctr_transforms = [s['ctr_transform'] for s in weak_batch]
    strong_ctr_transforms = [s['ctr_transform'] for s in strong_batch]
    
    # Pad weak and strong annotations
    if weak_max_num_annots > 0:
        weak_annot_padded = np.ones((len(weak_annots), weak_max_num_annots, 10), dtype='float32') * -1
        for idx, annot in enumerate(weak_annots):
            if annot.shape[0] > 0:
                weak_annot_padded[idx, :annot.shape[0], :] = annot
    else:
        weak_annot_padded = np.ones((len(weak_annots), 1, 10), dtype='float32') * -1
        
    if strong_max_num_annots > 0:
        strong_annot_padded = np.ones((len(strong_annots), strong_max_num_annots, 10), dtype='float32') * -1
        for idx, annot in enumerate(strong_annots):
            if annot.shape[0] > 0:
                strong_annot_padded[idx, :annot.shape[0], :] = annot
    else:
        strong_annot_padded = np.ones((len(strong_annots), 1, 10), dtype='float32') * -1
    
    # Return the samples
    weak_samples = {'image': torch.from_numpy(weak_imgs), 
                    'annot': torch.from_numpy(weak_annot_padded), 
                    'ctr_transform': weak_ctr_transforms}
    
    strong_samples = {'image': torch.from_numpy(strong_imgs),
                    'annot': torch.from_numpy(strong_annot_padded),
                    'ctr_transform': strong_ctr_transforms}
    
    samples = {'weak': weak_samples, 
               'strong': strong_samples}
    return samples