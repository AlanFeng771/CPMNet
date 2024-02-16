import os
import math
import logging
import numpy as np
import pandas as pd
from typing import List, Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.box_utils import nms_3D
from utils.generate_annot_csv_from_series_list import generate_annot_csv
from evaluationScript.detectionCADEvalutionIOU import nodule_evaluation

logger = logging.getLogger(__name__)


def convert_to_standard_csv(csv_path: str, annot_save_path: str, series_uids_save_path: str, spacing):
    '''
    convert [seriesuid	coordX	coordY	coordZ	w	h	d] to 
    'seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm'
    spacing:[z, y, x]
    '''
    column_order = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'w', 'h', 'd']
    gt_list = []
    csv_file = pd.read_csv(csv_path)
    seriesuid = csv_file['seriesuid']
    coordX, coordY, coordZ = csv_file['coordX'], csv_file['coordY'], csv_file['coordZ']
    w, h, d = csv_file['w'], csv_file['h'], csv_file['d']
    
    clean_seriesuid = []
    for j in range(seriesuid.shape[0]):
        if seriesuid[j] not in clean_seriesuid: 
            clean_seriesuid.append(seriesuid[j])
        gt_list.append([seriesuid[j], coordX[j], coordY[j], coordZ[j], w[j]/spacing[2], h[j]/spacing[1], d[j]/spacing[0]])
    df = pd.DataFrame(gt_list, columns=column_order)
    df.to_csv(annot_save_path, index=False)
    df = pd.DataFrame(clean_seriesuid)
    df.to_csv(series_uids_save_path, index=False, header=None)


def convert_to_standard_output(output: np.ndarray, spacing: torch.Tensor, name: str) -> List[List[Any]]:
    """
    convert [id, prob, ctr_z, ctr_y, ctr_x, d, h, w] to
    ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability', 'w', 'h', 'd']
    """
    preds = []
    spacing = np.array([spacing[0].numpy(), spacing[1].numpy(), spacing[2].numpy()]).reshape(-1, 3)
    for j in range(output.shape[0]):
        preds.append([name, output[j, 4], output[j, 3], output[j, 2], output[j, 1], output[j, 7], output[j, 6], output[j, 5]])
    return preds

def val(args,
        model: nn.Module,
        detection_postprocess,
        val_loader: DataLoader,
        device: torch.device,
        image_spacing: List[float],
        series_list_path: str,
        exp_folder: str,
        epoch: int = 0,
        nms_keep_top_k: int = 40) -> Dict[str, float]:
    
    annot_dir = os.path.join(exp_folder, 'annotation')
    os.makedirs(annot_dir, exist_ok=True)
    state = 'val'
    origin_annot_path = os.path.join(annot_dir, 'origin_annotation_{}.csv'.format(state))
    annot_path = os.path.join(annot_dir, 'annotation_{}.csv'.format(state))
    series_uids_path = os.path.join(annot_dir, 'seriesuid_{}.csv'.format(state))
    generate_annot_csv(series_list_path, origin_annot_path, spacing=image_spacing)
    convert_to_standard_csv(csv_path = origin_annot_path, 
                            annot_save_path=annot_path,
                            series_uids_save_path=series_uids_path,
                            spacing = image_spacing)
    
    model.eval()
    split_comber = val_loader.dataset.splitcomb
    batch_size = args.unlabeled_batch_size * args.num_samples
    all_preds = []
    for sample in val_loader:
        data = sample['split_images'][0].to(device, non_blocking=True)
        nzhw = sample['nzhw']
        name = sample['file_name'][0]
        spacing = sample['spacing'][0]
        outputlist = []
        
        for i in range(int(math.ceil(data.size(0) / batch_size))):
            end = (i + 1) * batch_size
            if end > data.size(0):
                end = data.size(0)
            input = data[i * batch_size:end]
            with torch.no_grad():
                output = model(input)
                output = detection_postprocess(output, device=device) #1, prob, ctr_z, ctr_y, ctr_x, d, h, w
            outputlist.append(output.data.cpu().numpy())
            
        output = np.concatenate(outputlist, 0)
        output = split_comber.combine(output, nzhw=nzhw)
        output = torch.from_numpy(output).view(-1, 8)
        
        # Remove the padding
        object_ids = output[:, 0] != -1.0
        output = output[object_ids]
        
        # NMS
        if len(output) > 0:
            keep = nms_3D(output[:, 1:], overlap=0.05, top_k=nms_keep_top_k)
            output = output[keep]
        output = output.numpy()
        
        preds = convert_to_standard_output(output, spacing, name) # convert to ['seriesuid', 'coordX', 'coordY', 'coordZ', 'radius', 'probability']
        all_preds.extend(preds)
        
    # Save the results to csv
    header = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability', 'w', 'h', 'd']
    df = pd.DataFrame(all_preds, columns=header)
    pred_results_path = os.path.join(annot_dir, 'predict_epoch_{}.csv'.format(epoch))
    df.to_csv(pred_results_path, index=False)
    
    output_dir = os.path.join(annot_dir, f'epoch_{epoch}')
    os.makedirs(output_dir, exist_ok=True)
    
    FP_ratios = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    froc_out, fixed_out = nodule_evaluation(annot_path = annot_path,
                                            series_uids_path = series_uids_path, 
                                            pred_results_path = pred_results_path,
                                            output_dir = output_dir,
                                            iou_threshold = args.val_iou_threshold,
                                            fixed_prob_threshold=args.val_fixed_prob_threshold)
    fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up, sens_points = froc_out
    
    logger.info('====> Epoch: {}'.format(epoch))
    for i in range(len(sens_points)):
        logger.info('====> fps:{:.4f} iou 0.1 frocs:{:.4f}'.format(FP_ratios[i], sens_points[i]))
    logger.info('====> mean frocs:{:.4f}'.format(np.mean(np.array(sens_points))))
    
    fixed_tp, fixed_fp, fixed_fn, fixed_recall, fixed_precision, fixed_f1_score = fixed_out
    metrics = {'tp': fixed_tp,
                'fp': fixed_fp,
                'fn': fixed_fn,
                'recall': fixed_recall,
                'precision': fixed_precision,
                'f1_score': fixed_f1_score}
    for fp_ratio, sens_p in zip(FP_ratios, sens_points):
        metrics['froc_{}_recall'.format(str(fp_ratio))] = sens_p
    
    return metrics