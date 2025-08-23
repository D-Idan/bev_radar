"""
Tracking evaluation with HOTA, MOTA, IoU, DetA, and Precision metrics.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import motmetrics as mm
import json
import csv
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from utils.metrics.camera_iou_metrics import CameraIoUCalculator


class TrackingDetectionEvaluator:
    """
    Tracking evaluation with HOTA, MOTA, IoU, DetA, and Precision metrics.

    Metric Categories:
    - Precision: Range-azimuth based (uses radar coordinates)
    - DetA, IoU: Camera bounding box based (uses pixel coordinates)
    - MOTA, HOTA: Identity tracking (uses camera bounding boxes + track IDs)
    - Camera IoU: Bounding box overlap (uses pixel coordinates)
    """

    def __init__(self, distance_threshold: float = 5.0, iou_thresholds: List[float] = None,
                 use_cvpr_labels_only: bool = True):
        """
        Initialize evaluator.

        Args:
            distance_threshold: Max distance for valid associations (meters)
            iou_thresholds: List of IoU thresholds for evaluation
            use_cvpr_labels_only: If True, only use labels with cvpr_updated=True for range-azimuth metrics
        """
        self.distance_threshold = distance_threshold
        self.iou_thresholds = iou_thresholds if iou_thresholds is not None else [0.3, 0.5]
        self.use_cvpr_labels_only = use_cvpr_labels_only
        self.camera_iou_calculator = CameraIoUCalculator(iou_thresholds=self.iou_thresholds)

        # Initialize motmetrics accumulator for MOTA calculation
        self.acc = mm.MOTAccumulator(auto_id=False)

        # Store data for HOTA calculation
        self.hota_data = {
            'gt_ids_per_frame': {},
            'pred_ids_per_frame': {},
            'similarity_scores': {},
            'iou_distances': {}  # Store IoU-based distances for consistency
        }

        self.frame_results = []

    def evaluate_frame(self, predictions: pd.DataFrame, ground_truth: pd.DataFrame,
                       tracks: pd.DataFrame, frame_id: int) -> Dict[str, Any]:
        """Evaluate single frame."""

        # For precision calculations (using range-azimuth maps), filter if needed
        if self.use_cvpr_labels_only:
            gt_ra = ground_truth[ground_truth['cvpr_updated'] == True].copy()
        else:
            gt_ra = ground_truth.copy()

        # For all other metrics, always use all labels
        gt_all = ground_truth.copy()

        # Extract positions - for precision we use filtered (gt_ra), for others we use all (gt_all)
        pred_positions = self._extract_positions(predictions, 'predictions')
        track_positions = self._extract_positions(tracks, 'tracks')
        gt_positions_ra = self._extract_positions(gt_ra, 'ground_truth')  # For precision
        gt_positions_all = self._extract_positions(gt_all, 'ground_truth')  # For other metrics

        # Get IDs - use all labels for tracking metrics
        if 'ID' in gt_all.columns:
            # Filter out any NaN or invalid IDs
            valid_gt = gt_all[gt_all['ID'].notna()]
            gt_ids = valid_gt['ID'].values.astype(int)
            gt_all = valid_gt  # Update gt_all to only include valid IDs
        else:
            gt_ids = np.arange(len(gt_all))

        # Ensure we have valid track IDs
        if 'track_id' in tracks.columns:
            track_ids = tracks['track_id'].values.astype(int)
        else:
            track_ids = np.arange(len(tracks))

        # Calculate distance matrix for MOTA (using all labels)
        # IMPORTANT: For camera-based tracking, use IoU distance (1 - IoU) instead of Euclidean
        if len(tracks) > 0 and len(gt_all) > 0:
            # Build IoU-based distance matrix
            distances = np.ones((len(tracks), len(gt_all))) * np.inf

            for i, (track_idx, track_row) in enumerate(tracks.iterrows()):
                track_bbox = self.camera_iou_calculator.range_azimuth_to_camera_bbox_consistent(
                    track_row['range_m'], track_row['azimuth_deg'], (540, 960)
                )

                for j, (_, gt_row) in enumerate(gt_all.iterrows()):
                    gt_bbox = self.camera_iou_calculator.get_label_bbox_consistent(gt_row, (540, 960))
                    iou = self.camera_iou_calculator.calculate_bbox_iou(track_bbox, gt_bbox)

                    # Convert IoU to distance (1 - IoU)
                    distances[i, j] = 1.0 - iou

            # Apply threshold - for IoU distance, threshold should be (1 - min_iou)
            filtered_distances = distances.copy()
            min_threshold = min(self.iou_thresholds)
            iou_distance_threshold = 1.0 - min_threshold
            filtered_distances[filtered_distances > iou_distance_threshold] = np.nan

            # Ensure we're using the same ground truth for all metrics
            # Store the IoU-based matches for consistency
            self.hota_data['iou_distances'][frame_id] = distances
        else:
            distances = np.empty((0, 0))
            filtered_distances = distances

        # Update motmetrics accumulator for MOTA (using all labels)
        # Only update if we have both tracks and ground truth
        if len(track_ids) > 0 or len(gt_ids) > 0:
            self.acc.update(
                gt_ids,
                track_ids,
                filtered_distances,
                frameid=frame_id
            )

        # Store data for HOTA calculation (using all labels)
        self.hota_data['gt_ids_per_frame'][frame_id] = gt_ids
        self.hota_data['pred_ids_per_frame'][frame_id] = track_ids
        self.hota_data['similarity_scores'][frame_id] = distances

        # Calculate frame metrics (precision uses gt_ra, others use gt_all)
        frame_metrics = self._calculate_frame_metrics(
            pred_positions, track_positions, gt_positions_ra, gt_all
        )

        # Camera IoU evaluation (using ALL labels)
        camera_iou_result = self.camera_iou_calculator.evaluate_camera_iou_single_frame(
            predictions, gt_all, tracks, frame_id, image_shape=(540, 960)
        )
        # Calculate DetA from camera results
        camera_det_a = self.camera_iou_calculator.calculate_det_a_from_camera_results(camera_iou_result)

        # Override the placeholder values in frame_metrics with camera-based calculations
        # Get metrics from the minimum threshold (most permissive)
        min_threshold = min(self.iou_thresholds)
        threshold_metrics = camera_iou_result.get('threshold_metrics', {}).get(min_threshold, {})

        frame_metrics.update({
            'detection_det_a': camera_det_a['detection_det_a'],
            'tracking_det_a': camera_det_a['tracking_det_a'],
            # Add new metrics from threshold-specific results
            'detection_tp': threshold_metrics.get('detection_tp', 0),
            'detection_fp': threshold_metrics.get('detection_fp', 0),
            'tracking_tp': threshold_metrics.get('tracking_tp', 0),
            'tracking_fp': threshold_metrics.get('tracking_fp', 0),
        })

        frame_result = {
            'frame_id': frame_id,
            'metrics': frame_metrics,
            'camera_iou_results': camera_iou_result,
            'num_predictions': len(predictions),
            'num_ground_truth_total': len(ground_truth),
            'num_ground_truth_cvpr': len(gt_ra),
            'num_tracks': len(tracks)
        }

        self.frame_results.append(frame_result)
        return frame_result

    def _extract_positions(self, df: pd.DataFrame, data_type: str) -> np.ndarray:
        """Extract positions from dataframe."""
        if df.empty:
            return np.zeros((0, 2))

        if data_type == 'ground_truth':
            # Use radar coordinates from labels
            if 'radar_X_m' in df.columns and 'radar_Y_m' in df.columns:
                x = df['radar_X_m'].values
                y = df['radar_Y_m'].values
            elif 'radar_R_m' in df.columns and 'radar_A_deg' in df.columns:
                # Fallback to polar coordinates
                ranges = df['radar_R_m'].values
                azimuths = np.deg2rad(df['radar_A_deg'].values)
                x = ranges * np.sin(azimuths)
                y = ranges * np.cos(azimuths)
            else:
                raise ValueError(f"Missing required radar coordinate columns for {data_type}")
        else:
            # Predictions and tracks
            if 'range_m' in df.columns and 'azimuth_deg' in df.columns:
                ranges = df['range_m'].values
                azimuths = np.deg2rad(df['azimuth_deg'].values)
                x = ranges * np.sin(azimuths)
                y = ranges * np.cos(azimuths)
            else:
                return np.zeros((0, 2))

        return np.column_stack([x, y])

    def _calculate_frame_metrics(self, pred_pos: np.ndarray, track_pos: np.ndarray,
                                 gt_pos_ra: np.ndarray, ground_truth_all: pd.DataFrame) -> Dict:
        """Calculate Precision (RA), DetA, and IoU (camera) for the frame.

        IMPORTANT:
        - Precision = average distance to matched ground truth in meters (filtered by use_cvpr_labels_only)
        - DetA & IoU always use ALL labels regardless of use_cvpr_labels_only flag

        Args:
            pred_pos: Prediction positions in RA space (for precision only)
            track_pos: Track positions in RA space (for precision only)
            gt_pos_ra: Ground truth positions for precision (may be filtered by cvpr_updated)
            ground_truth_all: Full unfiltered ground truth DataFrame (for camera-based DetA, IoU)

        Returns:
            Dictionary containing detection and tracking metrics
        """
        metrics = {}

        # === PRECISION: Range-Azimuth based (can use filtered labels) ===
        num_gt_ra = len(gt_pos_ra)

        # Detection precision (RA-based) - average distance to matched ground truth in meters
        if len(pred_pos) > 0 and len(gt_pos_ra) > 0:
            distances_ra = cdist(pred_pos, gt_pos_ra)
            pred_indices, gt_indices = linear_sum_assignment(distances_ra)
            matched_distances = distances_ra[pred_indices, gt_indices]
            valid_matches = matched_distances <= self.distance_threshold

            if np.sum(valid_matches) > 0:
                detection_precision = float(np.mean(matched_distances[valid_matches]))
            else:
                detection_precision = self.distance_threshold  # No valid matches
        else:
            detection_precision = 0.0 if len(gt_pos_ra) == 0 else self.distance_threshold

        # Tracking precision (RA-based) - average distance to matched ground truth in meters
        if len(track_pos) > 0 and len(gt_pos_ra) > 0:
            distances_ra = cdist(track_pos, gt_pos_ra)
            track_indices, gt_indices = linear_sum_assignment(distances_ra)
            matched_distances = distances_ra[track_indices, gt_indices]
            valid_matches = matched_distances <= self.distance_threshold

            if np.sum(valid_matches) > 0:
                tracking_precision = float(np.mean(matched_distances[valid_matches]))
            else:
                tracking_precision = self.distance_threshold  # No valid matches
        else:
            tracking_precision = 0.0 if len(gt_pos_ra) == 0 else self.distance_threshold

        metrics.update({
            'detection_accuracy': detection_precision,  # Distance in meters
            'tracking_accuracy': tracking_precision  # Distance in meters
        })

        # === DetA: Camera-based (always use ALL labels) ===
        # These will be calculated by camera IoU module and combined later
        # For now, set placeholder values - they'll be overridden by camera IoU results
        metrics.update({
            'detection_det_a': 0.0,  # Will be calculated from camera IoU
            'tracking_det_a': 0.0,  # Will be calculated from camera IoU
        })

        return metrics

    def _calculate_hota(self) -> float:
        """Calculate HOTA (Higher Order Tracking Accuracy) metric."""
        # Calculate proper association accuracy from MOTA components
        summary = mm.metrics.create().compute(self.acc, metrics=['num_matches', 'num_false_positives',
                                                                 'num_misses', 'num_switches'], name='acc')

        num_matches = float(summary['num_matches'].values[0])
        num_fp = float(summary['num_false_positives'].values[0])
        num_fn = float(summary['num_misses'].values[0])
        num_switches = float(summary['num_switches'].values[0])

        # Calculate detection accuracy (DetA)
        total_det_a = 0
        total_gt = 0

        for frame in self.frame_results:
            metrics = frame['metrics']
            det_a_value = metrics.get('tracking_det_a', 0)
            num_gt = frame.get('num_ground_truth_total', 0)

            if num_gt > 0:
                total_det_a += det_a_value * num_gt
                total_gt += num_gt

        avg_det_a = total_det_a / total_gt if total_gt > 0 else 0.0

        # Calculate association accuracy (AssA)
        # AssA = TP / (TP + FN + IDSW) where TP = num_matches
        if num_matches > 0:
            ass_a = num_matches / (num_matches + num_fn + num_switches)
        else:
            ass_a = 0.0

        # HOTA is geometric mean of detection and association accuracy
        hota = np.sqrt(avg_det_a * ass_a) if avg_det_a > 0 and ass_a > 0 else 0.0

        return hota

    def _calculate_threshold_metrics(self, threshold_results: Dict) -> Dict:
        """Calculate Precision, Recall, F1, AP for a specific IoU threshold."""
        detection_tp = threshold_results.get('detection_tp', 0)
        detection_fp = threshold_results.get('detection_fp', 0)
        detection_fn = threshold_results.get('detection_fn', 0)

        tracking_tp = threshold_results.get('tracking_tp', 0)
        tracking_fp = threshold_results.get('tracking_fp', 0)
        tracking_fn = threshold_results.get('tracking_fn', 0)

        # Detection metrics
        det_precision = detection_tp / (detection_tp + detection_fp) if (detection_tp + detection_fp) > 0 else 0.0
        det_recall = detection_tp / (detection_tp + detection_fn) if (detection_tp + detection_fn) > 0 else 0.0
        det_f1 = 2 * (det_precision * det_recall) / (det_precision + det_recall) if (
                                                                                                det_precision + det_recall) > 0 else 0.0

        # Tracking metrics
        track_precision = tracking_tp / (tracking_tp + tracking_fp) if (tracking_tp + tracking_fp) > 0 else 0.0
        track_recall = tracking_tp / (tracking_tp + tracking_fn) if (tracking_tp + tracking_fn) > 0 else 0.0
        track_f1 = 2 * (track_precision * track_recall) / (track_precision + track_recall) if (
                                                                                                          track_precision + track_recall) > 0 else 0.0

        return {
            'detection': {
                'precision': det_precision,
                'recall': det_recall,
                'f1_score': det_f1,
                'tp': detection_tp,
                'fp': detection_fp,
                'fn': detection_fn
            },
            'tracking': {
                'precision': track_precision,
                'recall': track_recall,
                'f1_score': track_f1,
                'tp': tracking_tp,
                'fp': tracking_fp,
                'fn': tracking_fn
            }
        }

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate final report with HOTA, MOTA, IoU, DetA, and Precision."""

        # Calculate motmetrics summary for MOTA
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=['mota', 'motp', 'precision', 'recall',
                                               'num_switches'],
                            name='acc')

        # Calculate aggregate metrics
        detection_metrics = self._aggregate_detection_metrics()
        tracking_metrics = self._aggregate_tracking_metrics()
        camera_iou_summary = self._calculate_camera_iou_summary()

        # Calculate HOTA
        hota = self._calculate_hota()

        # Calculate metrics for each IoU threshold
        threshold_metrics = {}
        for threshold in self.iou_thresholds:
            threshold_tp = 0
            threshold_fp = 0
            threshold_fn = 0
            tracking_tp = 0
            tracking_fp = 0
            tracking_fn = 0

            for frame in self.frame_results:
                frame_threshold_metrics = frame['camera_iou_results'].get('threshold_metrics', {}).get(threshold, {})
                threshold_tp += frame_threshold_metrics.get('detection_tp', 0)
                threshold_fp += frame_threshold_metrics.get('detection_fp', 0)
                threshold_fn += frame_threshold_metrics.get('detection_fn', 0)
                tracking_tp += frame_threshold_metrics.get('tracking_tp', 0)
                tracking_fp += frame_threshold_metrics.get('tracking_fp', 0)
                tracking_fn += frame_threshold_metrics.get('tracking_fn', 0)

            threshold_metrics[threshold] = self._calculate_threshold_metrics({
                'detection_tp': threshold_tp,
                'detection_fp': threshold_fp,
                'detection_fn': threshold_fn,
                'tracking_tp': tracking_tp,
                'tracking_fp': tracking_fp,
                'tracking_fn': tracking_fn
            })

        # Calculate average metrics across thresholds
        avg_metrics = {
            'detection': {},
            'tracking': {}
        }

        for phase in ['detection', 'tracking']:
            for metric in ['precision', 'recall', 'f1_score']:
                values = [threshold_metrics[t][phase][metric] for t in self.iou_thresholds]
                avg_metrics[phase][metric] = np.mean(values) if values else 0.0

        # Get TP/FP from minimum threshold for primary metrics
        min_threshold = min(self.iou_thresholds)
        min_threshold_metrics = threshold_metrics[min_threshold]
        detection_tp = min_threshold_metrics['detection']['tp']
        detection_fp = min_threshold_metrics['detection']['fp']
        tracking_tp = min_threshold_metrics['tracking']['tp']
        tracking_fp = min_threshold_metrics['tracking']['fp']

        report = {
            'evaluation_summary': {
                'frames_evaluated': len(self.frame_results),
                'distance_threshold_m': self.distance_threshold,
                'iou_thresholds': self.iou_thresholds,
                'primary_metrics_threshold': min(self.iou_thresholds),
                'use_cvpr_labels_only': self.use_cvpr_labels_only
            },
            'threshold_specific_metrics': threshold_metrics,
            'average_metrics': avg_metrics,
            'primary_metrics': {
                'hota': float(hota),
                'mota': float(summary['mota'].values[0]) if not np.isnan(summary['mota'].values[0]) else 0.0,
                'detection_accuracy': detection_metrics['accuracy'],  # Distance in meters
                'detection_det_a': detection_metrics['det_a'],
                'tracking_accuracy': tracking_metrics['accuracy'],  # Distance in meters
                'tracking_det_a': tracking_metrics['det_a'],
                'camera_iou_mean': camera_iou_summary['tracking_camera_iou']['mean'],
                'detection_ncle': camera_iou_summary['detection_ncle']['mean'],
                'tracking_ncle': camera_iou_summary['tracking_ncle']['mean'],
                'detection_tp': detection_tp,
                'detection_fp': detection_fp,
                'tracking_tp': tracking_tp,
                'tracking_fp': tracking_fp,
                'detection_precision': detection_tp / (detection_tp + detection_fp) if
                (detection_tp + detection_fp) > 0 else 0.0,  # TP/(TP+FP) ratio
                'tracking_precision': tracking_tp / (tracking_tp + tracking_fp) if
                (tracking_tp + tracking_fp) > 0 else 0.0  # TP/(TP+FP) ratio
            },
            'detailed_tracking_metrics': {
                'motp': float(summary['motp'].values[0]) if not np.isnan(summary['motp'].values[0]) else None,
                'num_switches': int(summary['num_switches'].values[0]),
                'recall': float(summary['recall'].values[0]) if not np.isnan(summary['recall'].values[0]) else 0.0
            },
            'detection_performance': {**detection_metrics,
                                      'avg_precision': avg_metrics['detection']['precision'],
                                      'avg_recall': avg_metrics['detection']['recall'],
                                      'avg_f1_score': avg_metrics['detection']['f1_score']},
            'tracking_performance': {**tracking_metrics,
                                     'hota': float(hota),
                                     'mota': float(summary['mota'].values[0]) if not
                                     np.isnan(summary['mota'].values[0]) else 0.0,
                                     'precision_ratio': tracking_tp / (tracking_tp + tracking_fp) if
                                     (tracking_tp + tracking_fp) > 0 else 0.0,
                                     'avg_precision': avg_metrics['tracking']['precision'],
                                     'avg_recall': avg_metrics['tracking']['recall'],
                                     'avg_f1_score': avg_metrics['tracking']['f1_score']},
            'camera_iou_performance': camera_iou_summary,
            'frame_by_frame_results': self.frame_results
        }

        return report

    def _aggregate_detection_metrics(self) -> Dict:
        """Aggregate detection-level metrics."""
        all_accuracy = []
        all_det_a = []

        for frame in self.frame_results:
            metrics = frame['metrics']
            all_accuracy.append(metrics['detection_accuracy'])
            all_det_a.append(metrics['detection_det_a'])

        # Calculate precision_ratio from frame results
        all_detection_tp = sum(frame['metrics']['detection_tp'] for frame in self.frame_results)
        all_detection_fp = sum(frame['metrics']['detection_fp'] for frame in self.frame_results)
        detection_precision_ratio = all_detection_tp / (all_detection_tp + all_detection_fp) if (
                (all_detection_tp + all_detection_fp) > 0) else 0.0

        return {
            'accuracy': np.mean(all_accuracy) if all_accuracy else 0.0,
            'det_a': np.mean(all_det_a) if all_det_a else 0.0,
            'precision': detection_precision_ratio  # TP/(TP+FP)
        }

    def _aggregate_tracking_metrics(self) -> Dict:
        """Aggregate tracking-level metrics."""
        all_accuracy = []
        all_det_a = []

        for frame in self.frame_results:
            metrics = frame['metrics']
            all_accuracy.append(metrics['tracking_accuracy'])
            all_det_a.append(metrics['tracking_det_a'])

        # Calculate precision_ratio from frame results
        all_tracking_tp = sum(frame['metrics']['tracking_tp'] for frame in self.frame_results)
        all_tracking_fp = sum(frame['metrics']['tracking_fp'] for frame in self.frame_results)
        tracking_precision_ratio = all_tracking_tp / (all_tracking_tp + all_tracking_fp) if (
                (all_tracking_tp + all_tracking_fp) > 0) else 0.0

        return {
            'accuracy': np.mean(all_accuracy) if all_accuracy else 0.0,
            'det_a': np.mean(all_det_a) if all_det_a else 0.0,
            'precision': tracking_precision_ratio  # TP/(TP+FP)
        }

    def _calculate_camera_iou_summary(self) -> Dict[str, Any]:
        """Calculate camera IoU summary statistics."""
        all_detection_ious = []
        all_tracking_ious = []
        all_detection_ncles = []
        all_tracking_ncles = []
        total_detection_tp = 0
        total_detection_fp = 0
        total_tracking_tp = 0
        total_tracking_fp = 0

        # Initialize per-threshold accumulators
        threshold_tp_fp = {t: {'detection_tp': 0, 'detection_fp': 0, 'tracking_tp': 0, 'tracking_fp': 0}
                           for t in self.iou_thresholds}

        for result in self.frame_results:
            camera_result = result['camera_iou_results']
            all_detection_ious.extend(camera_result.get('detection_vs_labels_ious', []))
            all_detection_ncles.extend(camera_result.get('detection_ncles', []))

            if camera_result.get('tracking_vs_labels_ious'):
                all_tracking_ious.extend(camera_result['tracking_vs_labels_ious'])
                all_tracking_ncles.extend(camera_result.get('tracking_ncles', []))

            # Accumulate TP/FP per threshold
            for threshold in self.iou_thresholds:
                threshold_data = camera_result.get('threshold_metrics', {}).get(threshold, {})
                threshold_tp_fp[threshold]['detection_tp'] += threshold_data.get('detection_tp', 0)
                threshold_tp_fp[threshold]['detection_fp'] += threshold_data.get('detection_fp', 0)
                threshold_tp_fp[threshold]['tracking_tp'] += threshold_data.get('tracking_tp', 0)
                threshold_tp_fp[threshold]['tracking_fp'] += threshold_data.get('tracking_fp', 0)

        # Use minimum threshold for summary
        min_threshold = min(self.iou_thresholds)
        total_detection_tp = threshold_tp_fp[min_threshold]['detection_tp']
        total_detection_fp = threshold_tp_fp[min_threshold]['detection_fp']
        total_tracking_tp = threshold_tp_fp[min_threshold]['tracking_tp']
        total_tracking_fp = threshold_tp_fp[min_threshold]['tracking_fp']

        return {
            'detection_camera_iou': {
                'mean': float(np.mean(all_detection_ious)) if all_detection_ious else 0.0,
                'count': len(all_detection_ious)
            },
            'tracking_camera_iou': {
                'mean': float(np.mean(all_tracking_ious)) if all_tracking_ious else 0.0,
                'count': len(all_tracking_ious)
            },
            'detection_ncle': {
                'mean': float(np.mean(all_detection_ncles)) if all_detection_ncles else 0.0,
                'std': float(np.std(all_detection_ncles)) if all_detection_ncles else 0.0,
                'count': len(all_detection_ncles)
            },
            'tracking_ncle': {
                'mean': float(np.mean(all_tracking_ncles)) if all_tracking_ncles else 0.0,
                'std': float(np.std(all_tracking_ncles)) if all_tracking_ncles else 0.0,
                'count': len(all_tracking_ncles)
            },
            'detection_tp_fp': {
                'true_positives': total_detection_tp,
                'false_positives': total_detection_fp,
                'total': total_detection_tp + total_detection_fp
            },
            'tracking_tp_fp': {
                'true_positives': total_tracking_tp,
                'false_positives': total_tracking_fp,
                'total': total_tracking_tp + total_tracking_fp
            }
        }

    def save_reports(self, output_path: str):
        """Save evaluation reports."""
        report = self.generate_comprehensive_report()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = output_path.parent / f"{output_path.stem}_summary{output_path.suffix}"
        summary_report = {k: v for k, v in report.items() if k != 'frame_by_frame_results'}
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)

        # Save frame details
        details_path = output_path.parent / f"{output_path.stem}_frame_details{output_path.suffix}"
        with open(details_path, 'w') as f:
            json.dump({'frame_by_frame_results': report['frame_by_frame_results']}, f, indent=2, default=str)

        # Save CSV file
        csv_path = output_path.parent / f"{output_path.stem}_metrics.csv"

        # Find tracking.csv in parent directories
        tracking_csv_path = output_path.parent.parent / 'tracks' / 'tracking.csv'
        tracking_csv_path = str(tracking_csv_path)

        csv_data = self._generate_csv_data(report, tracking_csv_path=tracking_csv_path)

        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['Sample', 'Type', 'TP', 'TN', 'FP', 'FN', 'IoU', 'timestamp', 'T']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(csv_data)

    def save_text_report(self, output_path: str):
        """Save human-readable text report."""
        report = self.generate_comprehensive_report()

        output_path = Path(output_path)

        with (open(output_path, 'w') as f):
            f.write("TRACKING EVALUATION REPORT - TABLE FORMAT\n")
            f.write("=" * 80 + "\n\n")

            # Configuration table
            f.write("CONFIGURATION\n")
            f.write("-" * 50 + "\n")
            summary = report['evaluation_summary']
            f.write(f"{'Frames Evaluated':<30} {summary['frames_evaluated']:>10}\n")
            f.write(f"{'Distance Threshold (m)':<30} {summary['distance_threshold_m']:>10.1f}\n")
            f.write(f"{'IoU Thresholds':<30} {str(summary['iou_thresholds']):>10}\n")
            f.write(
                f"{'Primary Metrics IoU':<30} {summary.get('primary_metrics_threshold', min(summary['iou_thresholds'])):>10.1f}\n")
            f.write(f"{'Use CVPR Labels Only':<30} {str(summary['use_cvpr_labels_only']):>10}\n")
            f.write(f"{'RA Map Filtering':<30} {'Precision Only':>10}\n")
            f.write(f"{'Bbox Metrics Filtering':<30} {'All Labels':>10}\n\n")

            # IoU Threshold-specific metrics table
            f.write("\nMETRICS BY IoU THRESHOLD\n")
            f.write("=" * 100 + "\n")

            for threshold in report['evaluation_summary']['iou_thresholds']:
                f.write(f"\nIoU Threshold: {threshold}\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Metric':<20} {'Detection':<15} {'Tracking':<15} {'Improvement':<15}\n")
                f.write("-" * 100 + "\n")

                threshold_data = report['threshold_specific_metrics'][threshold]

                # Precision
                det_prec = threshold_data['detection']['precision']
                track_prec = threshold_data['tracking']['precision']
                prec_imp = ((track_prec - det_prec) / det_prec * 100) if det_prec > 0 else 0
                f.write(f"{'Precision':<20} {det_prec:<15.4f} {track_prec:<15.4f} {prec_imp:>+13.1f}%\n")

                # Recall
                det_rec = threshold_data['detection']['recall']
                track_rec = threshold_data['tracking']['recall']
                rec_imp = ((track_rec - det_rec) / det_rec * 100) if det_rec > 0 else 0
                f.write(f"{'Recall':<20} {det_rec:<15.4f} {track_rec:<15.4f} {rec_imp:>+13.1f}%\n")

                # F1 Score
                det_f1 = threshold_data['detection']['f1_score']
                track_f1 = threshold_data['tracking']['f1_score']
                f1_imp = ((track_f1 - det_f1) / det_f1 * 100) if det_f1 > 0 else 0
                f.write(f"{'F1 Score':<20} {det_f1:<15.4f} {track_f1:<15.4f} {f1_imp:>+13.1f}%\n")

                # TP/FP/FN counts
                f.write(
                    f"{'True Positives':<20} {threshold_data['detection']['tp']:<15d} {threshold_data['tracking']['tp']:<15d}\n")
                f.write(
                    f"{'False Positives':<20} {threshold_data['detection']['fp']:<15d} {threshold_data['tracking']['fp']:<15d}\n")
                f.write(
                    f"{'False Negatives':<20} {threshold_data['detection']['fn']:<15d} {threshold_data['tracking']['fn']:<15d}\n")

            # Average metrics across thresholds
            f.write("\nAVERAGE METRICS ACROSS ALL THRESHOLDS\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Metric':<20} {'Detection':<15} {'Tracking':<15} {'Improvement':<15}\n")
            f.write("-" * 100 + "\n")

            avg_data = report['average_metrics']

            # Average Precision (AP)
            det_ap = avg_data['detection']['precision']
            track_ap = avg_data['tracking']['precision']
            ap_imp = ((track_ap - det_ap) / det_ap * 100) if det_ap > 0 else 0
            f.write(f"{'Avg Precision (AP)':<20} {det_ap:<15.4f} {track_ap:<15.4f} {ap_imp:>+13.1f}%\n")

            # Average Recall (AR)
            det_ar = avg_data['detection']['recall']
            track_ar = avg_data['tracking']['recall']
            ar_imp = ((track_ar - det_ar) / det_ar * 100) if det_ar > 0 else 0
            f.write(f"{'Avg Recall (AR)':<20} {det_ar:<15.4f} {track_ar:<15.4f} {ar_imp:>+13.1f}%\n")

            # Average F1
            det_avg_f1 = avg_data['detection']['f1_score']
            track_avg_f1 = avg_data['tracking']['f1_score']
            avg_f1_imp = ((track_avg_f1 - det_avg_f1) / det_avg_f1 * 100) if det_avg_f1 > 0 else 0
            f.write(f"{'Avg F1 Score':<20} {det_avg_f1:<15.4f} {track_avg_f1:<15.4f} {avg_f1_imp:>+13.1f}%\n")
            f.write("\n")

            # Primary metrics table
            f.write("PRIMARY METRICS COMPARISON\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Metric':<20} {'Detection':<15} {'Tracking':<15} {'Improvement':<15} {'Notes':<15}\n")
            f.write("-" * 80 + "\n")

            metrics = report['primary_metrics']
            det_accuracy = metrics['detection_accuracy']
            track_accuracy = metrics['tracking_accuracy']
            accuracy_improvement = (
                    (det_accuracy - track_accuracy) / det_accuracy * 100) if det_accuracy > 0 else 0

            det_det_a = metrics['detection_det_a']
            track_det_a = metrics['tracking_det_a']
            det_a_improvement = ((track_det_a - det_det_a) / det_det_a * 100) if det_det_a > 0 else 0

            camera_iou = metrics['camera_iou_mean']

            f.write(
                f"{'Accuracy (RA) [m]':<20} {det_accuracy:<15.4f} {track_accuracy:<15.4f} "
                f"{accuracy_improvement:>+13.1f}% {'Filtered' if summary['use_cvpr_labels_only'] else 'All':<15}\n")
            f.write(
                f"{'DetA (Camera)':<20} {det_det_a:<15.4f} {track_det_a:<15.4f} {det_a_improvement:>+13.1f}% {'All Labels':<15}\n")

            # Get detection camera IoU from the report
            det_camera_iou = report['camera_iou_performance']['detection_camera_iou']['mean']
            track_camera_iou = report['camera_iou_performance']['tracking_camera_iou']['mean']
            camera_iou_improvement = (
                        (track_camera_iou - det_camera_iou) / det_camera_iou * 100) if det_camera_iou > 0 else 0

            f.write(
                f"{'Camera IoU Mean':<20} {det_camera_iou:<15.4f} {track_camera_iou:<15.4f} {camera_iou_improvement:>+13.1f}% {'All Labels':<15}\n")

            f.write(f"{'HOTA':<20} {'-':<15} {metrics['hota']:<15.4f} {'-':<15} {'All Labels':<15}\n")
            f.write(f"{'MOTA':<20} {'-':<15} {metrics['mota']:<15.4f} {'-':<15} {'All Labels':<15}\n")

            # Calculate NCLE improvement (lower is better)
            det_ncle = metrics['detection_ncle']
            track_ncle = metrics['tracking_ncle']
            ncle_improvement = ((det_ncle - track_ncle) / det_ncle * 100) if det_ncle > 0 else 0

            f.write(
                f"{'NCLE':<20} {det_ncle:<15.4f} {track_ncle:<15.4f} {ncle_improvement:>+13.1f}% {'All Labels':<15}\n")

            # Calculate TP improvement (higher is better)
            det_tp = metrics['detection_tp']
            track_tp = metrics['tracking_tp']
            tp_improvement = ((track_tp - det_tp) / det_tp * 100) if det_tp > 0 else 0

            f.write(
                f"{'True Positives':<20} {det_tp:<15d} {track_tp:<15d} {tp_improvement:>+13.1f}% {'All Labels':<15}\n")

            # Calculate FP improvement (lower is better, so invert calculation)
            det_fp = metrics['detection_fp']
            track_fp = metrics['tracking_fp']
            fp_improvement = ((det_fp - track_fp) / det_fp * 100) if det_fp > 0 else 0
            # Calculate Precision Ratio improvement (higher is better)
            det_precision = metrics['detection_precision']
            track_precision = metrics['tracking_precision']

            f.write(
                f"{'False Positives':<20} {det_fp:<15d} {track_fp:<15d} {fp_improvement:>+13.1f}% {'All Labels':<15}\n")

            precision_improvement = ((track_precision - det_precision) / det_precision * 100) if det_precision > 0 else 0
            f.write(
                f"{'Precision':<20} {det_precision:<15.4f} {track_precision:<15.4f} {precision_improvement:>+13.1f}% {'All Labels':<15}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("LEGEND:\n")
            f.write("- Detection: Raw network predictions vs ground truth\n")
            f.write("- Tracking: Track outputs vs ground truth\n")
            f.write("- RA: Range-Azimuth based (uses distance threshold)\n")
            f.write("- Camera: Camera image IoU based (uses bounding boxes)\n")
            f.write("- Filtered: Only CVPR labels (cvpr_updated=True)\n")
            f.write("- All Labels: All labels regardless of cvpr_updated flag\n")

    def _generate_csv_data(self, report, include_network_detection: bool = False, threshold: float = 0.3,
                           tracking_csv_path: str = None):
        """Generate CSV data from frame-by-frame results.

        Args:
            report: Comprehensive report dictionary
            include_network_detection: Whether to include network detection rows
            threshold: IoU threshold for TP/FP classification (default: 0.3)
            tracking_csv_path: Path to tracking.csv file for timestamp and track_id data

        Returns:
            List of dictionaries for CSV rows
        """
        csv_rows = []

        # Load tracking data if path provided
        tracking_data = {}
        if tracking_csv_path and Path(tracking_csv_path).exists():
            tracking_df = pd.read_csv(tracking_csv_path)
            # Group by frame_id for easy lookup
            for frame_id in tracking_df['frame_id'].unique():
                frame_tracks = tracking_df[tracking_df['frame_id'] == frame_id]
                tracking_data[frame_id] = {
                    'timestamp': frame_tracks['timestamp'].iloc[0] if not frame_tracks.empty else None,
                    'tracks': frame_tracks[['track_id', 'range_m', 'azimuth_deg']].to_dict('records')
                }

        for frame_data in report.get('frame_by_frame_results', []):
            frame_id = frame_data['frame_id']
            threshold_metrics = frame_data.get('camera_iou_results', {}).get('threshold_metrics', {})

            # Get metrics for specified threshold
            metrics = threshold_metrics.get(threshold, {})
            if not metrics:
                # Fallback to closest available threshold
                available_thresholds = sorted(threshold_metrics.keys())
                if available_thresholds:
                    closest_threshold = min(available_thresholds, key=lambda x: abs(x - threshold))
                    metrics = threshold_metrics[closest_threshold]

            # Process network detections if enabled
            if include_network_detection:
                self._add_detection_rows(
                    csv_rows,
                    frame_data,
                    frame_id,
                    metrics,
                    threshold,
                    detection_type='Network Detection',
                    tracking_data=tracking_data
                )

            # Process tracking/corrector data
            self._add_tracking_rows(
                csv_rows,
                frame_data,
                frame_id,
                metrics,
                threshold,
                detection_type='Corrector',
                tracking_data=tracking_data
            )

        return csv_rows

    def _add_detection_rows(self, csv_rows, frame_data, frame_id, metrics, threshold, detection_type,
                            tracking_data=None):
        """Add detection rows to CSV data."""
        detection_ious = frame_data.get('camera_iou_results', {}).get('detection_vs_labels_ious', [])
        detection_fn = metrics.get('detection_fn', 0)

        # Get timestamp from tracking data
        timestamp = tracking_data.get(frame_id, {}).get('timestamp', None) if tracking_data else None

        # Add row for each detection
        for iou in detection_ious:
            csv_rows.append({
                'Sample': frame_id,
                'Type': detection_type,
                'TP': 1 if iou >= threshold else 0,
                'TN': 0,
                'FP': 1 if iou < threshold else 0,
                'FN': 0,
                'IoU': round(iou, 6),
                'timestamp': timestamp,
                'T': None  # Detections don't have track_ids
            })

        # Add FN row if there are unmatched ground truth objects
        if detection_fn > 0:
            csv_rows.append({
                'Sample': frame_id,
                'Type': detection_type,
                'TP': 0,
                'TN': 0,
                'FP': 0,
                'FN': detection_fn,
                'IoU': 0.0,
                'timestamp': timestamp,
                'T': None
            })

    def _add_tracking_rows(self, csv_rows, frame_data, frame_id, metrics, threshold, detection_type,
                           tracking_data=None):
        """Add tracking rows to CSV data."""
        tracking_ious = frame_data.get('camera_iou_results', {}).get('tracking_vs_labels_ious', [])
        tracking_fn = metrics.get('tracking_fn', 0)

        # Get frame tracking info
        frame_tracking_info = tracking_data.get(frame_id, {}) if tracking_data else {}
        timestamp = frame_tracking_info.get('timestamp', None)
        frame_tracks = frame_tracking_info.get('tracks', [])

        # Add row for each track with IoU
        for idx, iou in enumerate(tracking_ious):
            # Match track by index (assuming order is preserved)
            track_id = frame_tracks[idx]['track_id'] if idx < len(frame_tracks) else None

            csv_rows.append({
                'Sample': frame_id,
                'Type': detection_type,
                'TP': 1 if iou >= threshold else 0,
                'TN': 0,
                'FP': 1 if iou < threshold else 0,
                'FN': 0,
                'IoU': round(iou, 6),
                'timestamp': timestamp,
                'T': track_id
            })

        # Add FN row if there are unmatched ground truth objects
        if tracking_fn > 0:
            csv_rows.append({
                'Sample': frame_id,
                'Type': detection_type,
                'TP': 0,
                'TN': 0,
                'FP': 0,
                'FN': tracking_fn,
                'IoU': 0.0,
                'timestamp': timestamp,
                'T': None
            })

def evaluate_tracking_sequence(predictions_csv: str, ground_truth_csv: str,
                             tracking_csv: str, output_dir: str,
                             distance_threshold: float = 5.0,
                             iou_thresholds: List[float] = None,
                             use_cvpr_labels_only: bool = True,
                             **kwargs) -> Tuple[Path, Dict[str, Any]]:
    """Evaluate tracking sequence with HOTA, MOTA, IoU, DetA, and Precision metrics."""

    evaluator = TrackingDetectionEvaluator(
        distance_threshold=distance_threshold,
        iou_thresholds=iou_thresholds,
        use_cvpr_labels_only=use_cvpr_labels_only
    )

    # Load data
    predictions_df = pd.read_csv(predictions_csv)
    ground_truth_df = pd.read_csv(ground_truth_csv, sep='\t|,', engine='python')
    tracking_df = pd.read_csv(tracking_csv)

    # Get unique frames
    gt_frames = sorted(ground_truth_df['numSample'].unique())

    # Evaluate each frame
    for frame_id in gt_frames:
        pred_frame = predictions_df[predictions_df['sample_id'] == frame_id]
        gt_frame = ground_truth_df[ground_truth_df['numSample'] == frame_id]
        track_frame = tracking_df[tracking_df['sample_id'] == frame_id]

        evaluator.evaluate_frame(pred_frame, gt_frame, track_frame, frame_id)

    # Save reports
    output_path = Path(output_dir) / 'evaluation_metrics.json'
    evaluator.save_reports(str(output_path))
    evaluator.save_text_report(str(Path(output_dir) / 'evaluation_summary.txt'))

    return output_path, evaluator.generate_comprehensive_report()