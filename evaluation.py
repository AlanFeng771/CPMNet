from evaluationScript.eval import nodule_evaluation
import numpy as np
import argparse
import os
import logging
from utils.logs import setup_logging
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    # Rraining settings
    parser.add_argument('--annot_path', type=str, required=True, help='annotation_val.csv')
    parser.add_argument('--series_uids_path', type=str, required=True, help='seriesuid_val.csv')
    parser.add_argument('--pred_results_path', type=str, required=True, help='predict.csv')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--fixed_prob_threshold', type=float, default=0.65)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    FP_ratios = [0.125, 0.25, 0.5, 1, 2, 4, 8.]
    annot_path = args.annot_path
    series_uids_path = args.series_uids_path
    pred_results_path = args.pred_results_path
    output_dir = args.output_dir
    setup_logging(level='info', log_file=os.path.join(output_dir, 'log.txt'))
    
    froc_out, fixed_out, (best_f1_score, best_f1_threshold) = nodule_evaluation(annot_path = annot_path,
                                                                                series_uids_path = series_uids_path, 
                                                                                pred_results_path = pred_results_path,
                                                                                output_dir = output_dir,
                                                                                iou_threshold = 0.1,
                                                                                fixed_prob_threshold=args.fixed_prob_threshold)
    fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up, sens_points = froc_out



    fixed_tp, fixed_fp, fixed_fn, fixed_recall, fixed_precision, fixed_f1_score = fixed_out
    metrics = {'tp': fixed_tp,
                'fp': fixed_fp,
                'fn': fixed_fn,
                'recall': fixed_recall,
                'precision': fixed_precision,
                'f1_score': fixed_f1_score,
                'best_f1_score': best_f1_score,
                'best_f1_threshold': best_f1_threshold}
    mean_recall = np.mean(np.array(sens_points))
    metrics['froc_mean_recall'] = float(mean_recall)
    for fp_ratio, sens_p in zip(FP_ratios, sens_points):
        metrics['froc_{}_recall'.format(str(fp_ratio))] = sens_p
