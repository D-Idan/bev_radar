"""
Tracking evaluation with HOTA, MOTA, IoU, DetA, and Precision metrics.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import motmetrics as mm
import json
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

    def __init__(self, distance_threshold: float = 5.0, iou_threshold: float = 0.2,
                 use_cvpr_labels_only: bool = True):
        """
        Initialize evaluator.

        Args:
            distance_threshold: Max distance for valid associations (meters)
            iou_threshold: IoU threshold for evaluation
            use_cvpr_labels_only: If True, only use labels with cvpr_updated=True for range-azimuth metrics
        """
        self.distance_threshold = distance_threshold
        self.iou_threshold = iou_threshold
        self.use_cvpr_labels_only = use_cvpr_labels_only
        self.camera_iou_calculator = CameraIoUCalculator(iou_threshold=iou_threshold)

        # Initialize motmetrics accumulator for MOTA calculation
        self.acc = mm.MOTAccumulator(auto_id=False)

        # Store data for HOTA calculation
        self.hota_data = {
            'gt_ids_per_frame': {},
            'pred_ids_per_frame': {},
            'similarity_scores': {}
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
            gt_ids = gt_all['ID'].values.astype(int)
        else:
            gt_ids = np.arange(len(gt_all))

        track_ids = tracks['track_id'].values.astype(int) if 'track_id' in tracks.columns else np.arange(len(tracks))

        # Calculate distance matrix for MOTA (using all labels)
        if len(track_positions) > 0 and len(gt_positions_all) > 0:
            distances = cdist(track_positions, gt_positions_all)
        else:
            distances = np.empty((0, 0))

        # Apply distance threshold for motmetrics
        if len(track_positions) > 0 and len(gt_positions_all) > 0:
            filtered_distances = distances.copy()
            filtered_distances[filtered_distances > self.distance_threshold] = np.nan
        else:
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
        frame_metrics.update({
            'detection_det_a': camera_det_a['detection_det_a'],
            'tracking_det_a': camera_det_a['tracking_det_a'],
            # Add new metrics
            'detection_tp': camera_iou_result.get('detection_tp', 0),
            'detection_fp': camera_iou_result.get('detection_fp', 0),
            'tracking_tp': camera_iou_result.get('tracking_tp', 0),
            'tracking_fp': camera_iou_result.get('tracking_fp', 0),
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
        - Precision uses gt_pos_ra (may be filtered by use_cvpr_labels_only for RA maps)
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

        # Detection precision (RA-based)
        if len(pred_pos) > 0 and len(gt_pos_ra) > 0:
            distances_ra = cdist(pred_pos, gt_pos_ra)
            pred_indices, gt_indices = linear_sum_assignment(distances_ra)
            valid_matches = distances_ra[pred_indices, gt_indices] <= self.distance_threshold
            tp_precision = np.sum(valid_matches)
            fp_precision = len(pred_pos) - tp_precision
            detection_precision = tp_precision / (tp_precision + fp_precision) if (
                                                                                              tp_precision + fp_precision) > 0 else 0.0
        else:
            detection_precision = 0.0

        # Tracking precision (RA-based)
        if len(track_pos) > 0 and len(gt_pos_ra) > 0:
            distances_ra = cdist(track_pos, gt_pos_ra)
            track_indices, gt_indices = linear_sum_assignment(distances_ra)
            valid_matches = distances_ra[track_indices, gt_indices] <= self.distance_threshold
            tp_precision = np.sum(valid_matches)
            fp_precision = len(track_pos) - tp_precision
            tracking_precision = tp_precision / (tp_precision + fp_precision) if (
                                                                                             tp_precision + fp_precision) > 0 else 0.0
        else:
            tracking_precision = 0.0

        metrics.update({
            'detection_precision': detection_precision,
            'tracking_precision': tracking_precision
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
        # Simplified HOTA calculation
        # Full HOTA requires complex association calculation across all frames
        # This is a simplified version focusing on detection and association accuracy

        total_det_a = 0
        total_ass_a = 0
        valid_frames = 0

        for frame in self.frame_results:
            metrics = frame['metrics']
            # Use tracking DetA as proxy for detection accuracy
            det_a_value = metrics.get('tracking_det_a', 0)
            # Use tracking precision as proxy for association accuracy
            ass_a_value = metrics.get('tracking_precision', 0)

            # Only count frames with valid metrics
            if det_a_value > 0 or ass_a_value > 0:
                total_det_a += det_a_value
                total_ass_a += ass_a_value
                valid_frames += 1

        if valid_frames > 0:
            avg_det_a = total_det_a / valid_frames
            avg_ass_a = total_ass_a / valid_frames
            # HOTA is geometric mean of detection and association accuracy
            hota = np.sqrt(avg_det_a * avg_ass_a) if avg_det_a > 0 and avg_ass_a > 0 else 0.0
        else:
            hota = 0.0

        return hota

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

        report = {
            'evaluation_summary': {
                'frames_evaluated': len(self.frame_results),
                'distance_threshold_m': self.distance_threshold,
                'use_cvpr_labels_only': self.use_cvpr_labels_only
            },
            'primary_metrics': {
                'hota': float(hota),
                'mota': float(summary['mota'].values[0]) if not np.isnan(summary['mota'].values[0]) else 0.0,
                'detection_precision': detection_metrics['precision'],
                'detection_det_a': detection_metrics['det_a'],
                'tracking_precision': tracking_metrics['precision'],
                'tracking_det_a': tracking_metrics['det_a'],
                'camera_iou_mean': camera_iou_summary['tracking_camera_iou']['mean'],
                'detection_ncle': camera_iou_summary['detection_ncle']['mean'],
                'tracking_ncle': camera_iou_summary['tracking_ncle']['mean'],
                'detection_tp': camera_iou_summary['detection_tp_fp']['true_positives'],
                'detection_fp': camera_iou_summary['detection_tp_fp']['false_positives'],
                'tracking_tp': camera_iou_summary['tracking_tp_fp']['true_positives'],
                'tracking_fp': camera_iou_summary['tracking_tp_fp']['false_positives']
            },
            'detailed_tracking_metrics': {
                'motp': float(summary['motp'].values[0]) if not np.isnan(summary['motp'].values[0]) else None,
                'num_switches': int(summary['num_switches'].values[0]),
                'recall': float(summary['recall'].values[0]) if not np.isnan(summary['recall'].values[0]) else 0.0
            },
            'detection_performance': detection_metrics,
            'tracking_performance': {**tracking_metrics,
                                     'hota': float(hota),
                                     'mota': float(summary['mota'].values[0]) if not
                                     np.isnan(summary['mota'].values[0]) else 0.0},
            'camera_iou_performance': camera_iou_summary,
            'frame_by_frame_results': self.frame_results
        }

        return report

    def _aggregate_detection_metrics(self) -> Dict:
        """Aggregate detection-level metrics."""
        all_precision = []
        all_det_a = []

        for frame in self.frame_results:
            metrics = frame['metrics']
            all_precision.append(metrics['detection_precision'])
            all_det_a.append(metrics['detection_det_a'])

        return {
            'precision': np.mean(all_precision) if all_precision else 0.0,
            'det_a': np.mean(all_det_a) if all_det_a else 0.0
        }

    def _aggregate_tracking_metrics(self) -> Dict:
        """Aggregate tracking-level metrics."""
        all_precision = []
        all_det_a = []

        for frame in self.frame_results:
            metrics = frame['metrics']
            all_precision.append(metrics['tracking_precision'])
            all_det_a.append(metrics['tracking_det_a'])

        return {
            'precision': np.mean(all_precision) if all_precision else 0.0,
            'det_a': np.mean(all_det_a) if all_det_a else 0.0
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

        for result in self.frame_results:
            camera_result = result['camera_iou_results']
            all_detection_ious.extend(camera_result.get('detection_vs_labels_ious', []))
            all_detection_ncles.extend(camera_result.get('detection_ncles', []))

            if camera_result.get('tracking_vs_labels_ious'):
                all_tracking_ious.extend(camera_result['tracking_vs_labels_ious'])
                all_tracking_ncles.extend(camera_result.get('tracking_ncles', []))

            # Accumulate TP/FP
            total_detection_tp += camera_result.get('detection_tp', 0)
            total_detection_fp += camera_result.get('detection_fp', 0)
            if camera_result.get('tracking_tp') is not None:
                total_tracking_tp += camera_result.get('tracking_tp', 0)
                total_tracking_fp += camera_result.get('tracking_fp', 0)

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

    def save_text_report(self, output_path: str):
        """Save human-readable text report."""
        report = self.generate_comprehensive_report()

        output_path = Path(output_path)

        with open(output_path, 'w') as f:
            f.write("TRACKING EVALUATION REPORT - TABLE FORMAT\n")
            f.write("=" * 80 + "\n\n")

            # Configuration table
            f.write("CONFIGURATION\n")
            f.write("-" * 50 + "\n")
            summary = report['evaluation_summary']
            f.write(f"{'Frames Evaluated':<30} {summary['frames_evaluated']:>10}\n")
            f.write(f"{'Distance Threshold (m)':<30} {summary['distance_threshold_m']:>10.1f}\n")
            f.write(f"{'Use CVPR Labels Only':<30} {str(summary['use_cvpr_labels_only']):>10}\n")
            f.write(f"{'RA Map Filtering':<30} {'Precision Only':>10}\n")
            f.write(f"{'Bbox Metrics Filtering':<30} {'All Labels':>10}\n\n")

            # Primary metrics table
            f.write("PRIMARY METRICS COMPARISON\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Metric':<20} {'Detection':<15} {'Tracking':<15} {'Improvement':<15} {'Notes':<15}\n")
            f.write("-" * 80 + "\n")

            metrics = report['primary_metrics']
            det_precision = metrics['detection_precision']
            track_precision = metrics['tracking_precision']
            precision_improvement = (
                        (track_precision - det_precision) / det_precision * 100) if det_precision > 0 else 0

            det_det_a = metrics['detection_det_a']
            track_det_a = metrics['tracking_det_a']
            det_a_improvement = ((track_det_a - det_det_a) / det_det_a * 100) if det_det_a > 0 else 0

            camera_iou = metrics['camera_iou_mean']

            f.write(
                f"{'Precision (RA)':<20} {det_precision:<15.4f} {track_precision:<15.4f} {precision_improvement:>+13.1f}% {'Filtered' if summary['use_cvpr_labels_only'] else 'All':<15}\n")
            f.write(
                f"{'DetA (Camera)':<20} {det_det_a:<15.4f} {track_det_a:<15.4f} {det_a_improvement:>+13.1f}% {'All Labels':<15}\n")
            f.write(f"{'Camera IoU Mean':<20} {'-':<15} {camera_iou:<15.4f} {'-':<15} {'All Labels':<15}\n")
            f.write(f"{'HOTA':<20} {'-':<15} {metrics['hota']:<15.4f} {'-':<15} {'All Labels':<15}\n")
            f.write(f"{'MOTA':<20} {'-':<15} {metrics['mota']:<15.4f} {'-':<15} {'All Labels':<15}\n")
            f.write(
                f"{'NCLE':<20} {metrics['detection_ncle']:<15.4f} {metrics['tracking_ncle']:<15.4f} {'-':<15} {'All Labels':<15}\n")
            f.write(
                f"{'True Positives':<20} {metrics['detection_tp']:<15d} {metrics['tracking_tp']:<15d} {'-':<15} {'All Labels':<15}\n")
            f.write(
                f"{'False Positives':<20} {metrics['detection_fp']:<15d} {metrics['tracking_fp']:<15d} {'-':<15} {'All Labels':<15}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("LEGEND:\n")
            f.write("- Detection: Raw network predictions vs ground truth\n")
            f.write("- Tracking: Track outputs vs ground truth\n")
            f.write("- RA: Range-Azimuth based (uses distance threshold)\n")
            f.write("- Camera: Camera image IoU based (uses bounding boxes)\n")
            f.write("- Filtered: Only CVPR labels (cvpr_updated=True)\n")
            f.write("- All Labels: All labels regardless of cvpr_updated flag\n")


def evaluate_tracking_sequence(predictions_csv: str, ground_truth_csv: str,
                             tracking_csv: str, output_dir: str,
                             distance_threshold: float = 5.0,
                             use_cvpr_labels_only: bool = True,
                             **kwargs) -> Tuple[Path, Dict[str, Any]]:
    """Evaluate tracking sequence with HOTA, MOTA, IoU, DetA, and Precision metrics."""

    evaluator = TrackingDetectionEvaluator(
        distance_threshold=distance_threshold,
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