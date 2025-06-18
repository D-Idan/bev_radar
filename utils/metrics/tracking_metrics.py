"""
Comprehensive tracking and detection evaluation metrics.
Implements precision, DetA, distance-based metrics, and general statistics.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import json

from utils.metrics.camera_iou_metrics import CameraIoUCalculator, calculate_camera_iou_metrics


@dataclass
class DetectionMetrics:
    """Container for detection performance metrics."""
    # Core metrics
    precision: float
    recall: float
    f1_score: float
    det_a: float  # Detection Accuracy

    # Distance-based metrics
    mean_euclidean_distance: float
    std_euclidean_distance: float
    min_distance: float
    max_distance: float
    mean_iou: float
    motp: float  # Multiple Object Tracking Precision

    # General statistics
    total_associations: int
    true_positives: int
    false_positives: int
    false_negatives: int

    # Frame statistics
    frames_evaluated: int
    avg_detections_per_frame: float
    avg_ground_truth_per_frame: float


class TrackingDetectionEvaluator:
    """Comprehensive evaluation for both detection and tracking performance."""

    def __init__(self, distance_threshold: float = 5.0, iou_threshold: float = 0.3):
        """
        Initialize evaluator.

        Args:
            distance_threshold: Maximum distance for valid association (meters)
            iou_threshold: IoU threshold for bounding box overlap
        """
        self.distance_threshold = distance_threshold
        self.iou_threshold = iou_threshold

        # Add camera IoU calculator
        self.camera_iou_calculator = CameraIoUCalculator()

        # Storage for evaluation data
        self.frame_results = []
        self.detection_frame_results = []
        self.tracking_frame_results = []
        self.camera_iou_results = []

    def evaluate_frame(self, predictions: pd.DataFrame, ground_truth: pd.DataFrame,
                       tracks: pd.DataFrame, frame_id: int) -> Dict[str, Any]:
        """
        Evaluate single frame performance for both detection and tracking.

        Args:
            predictions: Network predictions for frame
            ground_truth: Ground truth detections for frame
            tracks: Tracker outputs for frame
            frame_id: Frame identifier

        Returns:
            Dictionary with frame-level metrics
        """
        # Extract data for evaluation
        pred_data = self._extract_detection_data(predictions)
        gt_data = self._extract_ground_truth_data(ground_truth)
        track_data = self._extract_tracking_data(tracks)

        # Evaluate detection performance (predictions vs ground truth)
        detection_metrics = self._evaluate_associations(
            pred_data, gt_data, "detection", frame_id
        )

        # Evaluate tracking performance (tracks vs ground truth)
        tracking_metrics = self._evaluate_associations(
            track_data, gt_data, "tracking", frame_id
        )

        # Add camera IoU evaluation with consistent image shape
        camera_iou_result = self.camera_iou_calculator.evaluate_camera_iou_single_frame(
            predictions, ground_truth, tracks, frame_id,
            image_shape=(540, 960)  # Use the actual image dimensions from your visualization
        )
        self.camera_iou_results.append(camera_iou_result)

        frame_result = {
            'frame_id': frame_id,
            'detection_results': detection_metrics,
            'tracking_results': tracking_metrics,
            'camera_iou_results': camera_iou_result,
            'num_predictions': len(predictions),
            'num_ground_truth': len(ground_truth),
            'num_tracks': len(tracks)
        }

        self.frame_results.append(frame_result)
        self.detection_frame_results.append(detection_metrics)
        self.tracking_frame_results.append(tracking_metrics)

        return frame_result

    def _extract_detection_data(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract detection data (predictions) from dataframe."""
        if df.empty:
            return {
                'positions': np.zeros((0, 2)),
                'boxes': np.zeros((0, 8)),
                'confidences': np.zeros(0)
            }

        # Extract positions (range, azimuth -> x, y)
        ranges = df['range_m'].values
        azimuths = np.deg2rad(df['azimuth_deg'].values)
        x = ranges * np.sin(azimuths)
        y = ranges * np.cos(azimuths)
        positions = np.column_stack([x, y])

        # Extract bounding boxes (x1, y1, x2, y2, x3, y3, x4, y4)
        box_cols = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
        boxes = df[box_cols].values if all(col in df.columns for col in box_cols) else np.zeros((len(df), 8))

        # Extract confidences
        confidences = df['confidence'].values if 'confidence' in df.columns else np.ones(len(df))

        return {
            'positions': positions,
            'boxes': boxes,
            'confidences': confidences
        }

    def _extract_ground_truth_data(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract ground truth data from dataframe."""
        if df.empty:
            return {
                'positions': np.zeros((0, 2)),
                'boxes': np.zeros((0, 8))
            }

        # Extract positions using radar coordinates
        if 'radar_R_m' in df.columns and 'radar_A_deg' in df.columns:
            ranges = df['radar_R_m'].values
            azimuths = np.deg2rad(df['radar_A_deg'].values)
            x = ranges * np.sin(azimuths)
            y = ranges * np.cos(azimuths)
            positions = np.column_stack([x, y])
        else:
            positions = np.zeros((0, 2))

        # Create bounding boxes from pixel coordinates if available
        if all(col in df.columns for col in ['x1_pix', 'y1_pix', 'x2_pix', 'y2_pix']):
            # For ground truth, we'll create a simple rectangular box
            # This is a simplified approach - you might need to adjust based on your specific needs
            x1, y1 = df['x1_pix'].values, df['y1_pix'].values
            x2, y2 = df['x2_pix'].values, df['y2_pix'].values
            boxes = np.column_stack([x1, y1, x2, y1, x2, y2, x1, y2])
        else:
            boxes = np.zeros((len(df), 8))

        return {
            'positions': positions,
            'boxes': boxes
        }

    def _extract_tracking_data(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract tracking data from dataframe."""
        if df.empty:
            return {
                'positions': np.zeros((0, 2)),
                'boxes': np.zeros((0, 8)),
                'confidences': np.zeros(0),
                'track_ids': np.zeros(0)
            }

        # Extract positions
        ranges = df['range_m'].values
        azimuths = np.deg2rad(df['azimuth_deg'].values)
        x = ranges * np.sin(azimuths)
        y = ranges * np.cos(azimuths)
        positions = np.column_stack([x, y])

        # Extract bounding boxes if available
        box_cols = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
        boxes = df[box_cols].values if all(col in df.columns for col in box_cols) else np.zeros((len(df), 8))

        # Extract confidences and track IDs
        confidences = df['confidence'].values if 'confidence' in df.columns else np.ones(len(df))
        track_ids = df['track_id'].values if 'track_id' in df.columns else np.arange(len(df))

        return {
            'positions': positions,
            'boxes': boxes,
            'confidences': confidences,
            'track_ids': track_ids
        }

    def _evaluate_associations(self, pred_data: Dict, gt_data: Dict,
                               eval_type: str, frame_id: int) -> Dict[str, Any]:
        """Evaluate associations between predictions/tracks and ground truth."""
        pred_positions = pred_data['positions']
        gt_positions = gt_data['positions']
        pred_boxes = pred_data['boxes']
        gt_boxes = gt_data['boxes']

        if len(pred_positions) == 0 or len(gt_positions) == 0:
            return {
                'frame_id': frame_id,
                'eval_type': eval_type,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'det_a': 0.0,
                'mean_euclidean_distance': float('inf'),
                'std_euclidean_distance': 0.0,
                'min_distance': float('inf'),
                'max_distance': 0.0,
                'mean_iou': 0.0,
                'motp': float('inf'),
                'true_positives': 0,
                'false_positives': len(pred_positions),
                'false_negatives': len(gt_positions),
                'total_associations': 0
            }

        # Calculate distance matrix
        distances = cdist(pred_positions, gt_positions)

        # Apply Hungarian algorithm for optimal assignment
        pred_indices, gt_indices = linear_sum_assignment(distances)

        # Determine valid matches based on distance threshold
        valid_matches_mask = distances[pred_indices, gt_indices] <= self.distance_threshold
        valid_pred_indices = pred_indices[valid_matches_mask]
        valid_gt_indices = gt_indices[valid_matches_mask]
        valid_distances = distances[pred_indices, gt_indices][valid_matches_mask]

        # Count metrics
        true_positives = len(valid_distances)
        false_positives = len(pred_positions) - true_positives
        false_negatives = len(gt_positions) - true_positives

        # Calculate basic metrics
        precision = true_positives / len(pred_positions) if len(pred_positions) > 0 else 0.0
        recall = true_positives / len(gt_positions) if len(gt_positions) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # DetA (Detection Accuracy) = TP / (TP + FP + FN)
        det_a = true_positives / (true_positives + false_positives + false_negatives) if (
                                                                                                     true_positives + false_positives + false_negatives) > 0 else 0.0

        # Distance-based metrics
        if len(valid_distances) > 0:
            mean_euclidean_distance = float(np.mean(valid_distances))
            std_euclidean_distance = float(np.std(valid_distances))
            min_distance = float(np.min(valid_distances))
            max_distance = float(np.max(valid_distances))
            motp = mean_euclidean_distance  # MOTP is the mean distance of correct associations
        else:
            mean_euclidean_distance = float('inf')
            std_euclidean_distance = 0.0
            min_distance = float('inf')
            max_distance = 0.0
            motp = float('inf')

        # IoU calculation for bounding boxes
        mean_iou = self._calculate_mean_iou(
            pred_boxes, gt_boxes, valid_pred_indices, valid_gt_indices
        )

        return {
            'frame_id': frame_id,
            'eval_type': eval_type,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'det_a': det_a,
            'mean_euclidean_distance': mean_euclidean_distance,
            'std_euclidean_distance': std_euclidean_distance,
            'min_distance': min_distance,
            'max_distance': max_distance,
            'mean_iou': mean_iou,
            'motp': motp,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_associations': true_positives
        }

    def _calculate_mean_iou(self, pred_boxes: np.ndarray, gt_boxes: np.ndarray,
                            valid_pred_indices: np.ndarray, valid_gt_indices: np.ndarray) -> float:
        """Calculate mean IoU for valid associations."""
        if len(valid_pred_indices) == 0 or pred_boxes.shape[1] != 8 or gt_boxes.shape[1] != 8:
            return 0.0

        ious = []
        for pred_idx, gt_idx in zip(valid_pred_indices, valid_gt_indices):
            try:
                pred_box = pred_boxes[pred_idx].reshape((4, 2))
                gt_box = gt_boxes[gt_idx].reshape((4, 2))

                # Create polygons
                pred_poly = Polygon([(pred_box[i, 0], pred_box[i, 1]) for i in range(4)])
                gt_poly = Polygon([(gt_box[i, 0], gt_box[i, 1]) for i in range(4)])

                # Calculate IoU
                if pred_poly.is_valid and gt_poly.is_valid:
                    intersection = pred_poly.intersection(gt_poly).area
                    union = pred_poly.union(gt_poly).area
                    iou = intersection / union if union > 0 else 0.0
                    ious.append(iou)
            except:
                # Skip invalid polygons
                continue

        return float(np.mean(ious)) if ious else 0.0

    def _aggregate_metrics(self, frame_results: List[Dict], eval_type: str) -> DetectionMetrics:
        """Aggregate metrics across all frames."""
        if not frame_results:
            return DetectionMetrics(
                precision=0.0, recall=0.0, f1_score=0.0, det_a=0.0,
                mean_euclidean_distance=float('inf'), std_euclidean_distance=0.0,
                min_distance=float('inf'), max_distance=0.0, mean_iou=0.0, motp=float('inf'),
                total_associations=0, true_positives=0, false_positives=0, false_negatives=0,
                frames_evaluated=0, avg_detections_per_frame=0.0, avg_ground_truth_per_frame=0.0
            )

        # Sum up counts
        total_tp = sum(result['true_positives'] for result in frame_results)
        total_fp = sum(result['false_positives'] for result in frame_results)
        total_fn = sum(result['false_negatives'] for result in frame_results)

        # Calculate overall metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        det_a = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

        # Distance metrics (only for valid associations)
        valid_distances = []
        valid_ious = []
        for result in frame_results:
            if result['mean_euclidean_distance'] != float('inf'):
                valid_distances.append(result['mean_euclidean_distance'])
            if result['mean_iou'] > 0:
                valid_ious.append(result['mean_iou'])

        if valid_distances:
            mean_euclidean_distance = float(np.mean(valid_distances))
            std_euclidean_distance = float(np.std(valid_distances))
            min_distance = float(np.min(valid_distances))
            max_distance = float(np.max(valid_distances))
            motp = mean_euclidean_distance
        else:
            mean_euclidean_distance = float('inf')
            std_euclidean_distance = 0.0
            min_distance = float('inf')
            max_distance = 0.0
            motp = float('inf')

        mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0

        # Frame statistics
        frames_evaluated = len(frame_results)

        # Calculate averages from main frame results
        if eval_type == "detection":
            avg_detections_per_frame = np.mean([r['num_predictions'] for r in self.frame_results])
        else:
            avg_detections_per_frame = np.mean([r['num_tracks'] for r in self.frame_results])

        avg_ground_truth_per_frame = np.mean([r['num_ground_truth'] for r in self.frame_results])

        return DetectionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            det_a=det_a,
            mean_euclidean_distance=mean_euclidean_distance,
            std_euclidean_distance=std_euclidean_distance,
            min_distance=min_distance,
            max_distance=max_distance,
            mean_iou=mean_iou,
            motp=motp,
            total_associations=total_tp,
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
            frames_evaluated=frames_evaluated,
            avg_detections_per_frame=avg_detections_per_frame,
            avg_ground_truth_per_frame=avg_ground_truth_per_frame
        )

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        if not self.frame_results:
            return {'error': 'No evaluation data available'}

        # Aggregate detection and tracking metrics
        detection_metrics = self._aggregate_metrics(self.detection_frame_results, "detection")
        tracking_metrics = self._aggregate_metrics(self.tracking_frame_results, "tracking")

        # Create comparison table
        comparison_data = self._create_comparison_table(detection_metrics, tracking_metrics)

        # Calculate camera IoU summary
        all_detection_camera_ious = []
        all_tracking_camera_ious = []

        for result in self.camera_iou_results:
            all_detection_camera_ious.extend(result['detection_vs_labels_ious'])
            if result['tracking_vs_labels_ious'] is not None:
                all_tracking_camera_ious.extend(result['tracking_vs_labels_ious'])

        camera_iou_summary = {
            'detection_camera_iou': {
                'mean': float(np.mean(all_detection_camera_ious)) if all_detection_camera_ious else 0.0,
                'std': float(np.std(all_detection_camera_ious)) if all_detection_camera_ious else 0.0,
                'median': float(np.median(all_detection_camera_ious)) if all_detection_camera_ious else 0.0,
                'count': len(all_detection_camera_ious)
            },
            'tracking_camera_iou': {
                'mean': float(np.mean(all_tracking_camera_ious)) if all_tracking_camera_ious else 0.0,
                'std': float(np.std(all_tracking_camera_ious)) if all_tracking_camera_ious else 0.0,
                'median': float(np.median(all_tracking_camera_ious)) if all_tracking_camera_ious else 0.0,
                'count': len(all_tracking_camera_ious)
            }
        }

        report = {
            'evaluation_summary': {
                'frames_evaluated': len(self.frame_results),
                'distance_threshold_m': self.distance_threshold,
                'iou_threshold': self.iou_threshold
            },
            'detection_performance': {
                'precision': detection_metrics.precision,
                'recall': detection_metrics.recall,
                'f1_score': detection_metrics.f1_score,
                'det_a': detection_metrics.det_a,
                'mean_euclidean_distance_m': detection_metrics.mean_euclidean_distance,
                'std_euclidean_distance_m': detection_metrics.std_euclidean_distance,
                'min_distance_m': detection_metrics.min_distance,
                'max_distance_m': detection_metrics.max_distance,
                'mean_iou': detection_metrics.mean_iou,
                'motp_m': detection_metrics.motp,
                'true_positives': detection_metrics.true_positives,
                'false_positives': detection_metrics.false_positives,
                'false_negatives': detection_metrics.false_negatives,
                'total_associations': detection_metrics.total_associations,
                'avg_detections_per_frame': detection_metrics.avg_detections_per_frame,
                'avg_ground_truth_per_frame': detection_metrics.avg_ground_truth_per_frame
            },
            'tracking_performance': {
                'precision': tracking_metrics.precision,
                'recall': tracking_metrics.recall,
                'f1_score': tracking_metrics.f1_score,
                'det_a': tracking_metrics.det_a,
                'mean_euclidean_distance_m': tracking_metrics.mean_euclidean_distance,
                'std_euclidean_distance_m': tracking_metrics.std_euclidean_distance,
                'min_distance_m': tracking_metrics.min_distance,
                'max_distance_m': tracking_metrics.max_distance,
                'mean_iou': tracking_metrics.mean_iou,
                'motp_m': tracking_metrics.motp,
                'true_positives': tracking_metrics.true_positives,
                'false_positives': tracking_metrics.false_positives,
                'false_negatives': tracking_metrics.false_negatives,
                'total_associations': tracking_metrics.total_associations,
                'avg_detections_per_frame': tracking_metrics.avg_detections_per_frame,
                'avg_ground_truth_per_frame': tracking_metrics.avg_ground_truth_per_frame
            },
            'performance_comparison': comparison_data,
            'camera_iou_performance': camera_iou_summary,
            'frame_by_frame_results': self.frame_results
        }

        return report

    def _create_comparison_table(self, det_metrics: DetectionMetrics,
                                 track_metrics: DetectionMetrics) -> Dict[str, Any]:
        """Create a comparison table between detection and tracking performance."""

        def calculate_improvement(tracking_val, detection_val):
            if detection_val == 0:
                return float('inf') if tracking_val > 0 else 0.0
            if detection_val == float('inf'):
                return -100.0 if tracking_val != float('inf') else 0.0
            return ((tracking_val - detection_val) / detection_val) * 100

        metrics_comparison = {
            'precision': {
                'detection': det_metrics.precision,
                'tracking': track_metrics.precision,
                'improvement_percent': calculate_improvement(track_metrics.precision, det_metrics.precision)
            },
            'recall': {
                'detection': det_metrics.recall,
                'tracking': track_metrics.recall,
                'improvement_percent': calculate_improvement(track_metrics.recall, det_metrics.recall)
            },
            'f1_score': {
                'detection': det_metrics.f1_score,
                'tracking': track_metrics.f1_score,
                'improvement_percent': calculate_improvement(track_metrics.f1_score, det_metrics.f1_score)
            },
            'det_a': {
                'detection': det_metrics.det_a,
                'tracking': track_metrics.det_a,
                'improvement_percent': calculate_improvement(track_metrics.det_a, det_metrics.det_a)
            },
            'mean_euclidean_distance_m': {
                'detection': det_metrics.mean_euclidean_distance,
                'tracking': track_metrics.mean_euclidean_distance,
                'improvement_percent': -calculate_improvement(track_metrics.mean_euclidean_distance,
                                                              det_metrics.mean_euclidean_distance)
                # Negative because lower is better
            },
            'motp_m': {
                'detection': det_metrics.motp,
                'tracking': track_metrics.motp,
                'improvement_percent': -calculate_improvement(track_metrics.motp, det_metrics.motp)
                # Negative because lower is better
            }
        }

        return metrics_comparison

    def save_text_report(self, output_path: str) -> None:
        """Save evaluation report to a readable text file."""
        report = self.generate_comprehensive_report()

        if 'error' in report:
            with open(output_path, 'w') as f:
                f.write(f"Error: {report['error']}\n")
            return

        with open(output_path, 'w') as f:
            # Header
            f.write("═" * 100 + "\n")
            f.write("                        TRACKING AND DETECTION EVALUATION REPORT\n")
            f.write("═" * 100 + "\n\n")

            # Executive Summary
            summary = report['evaluation_summary']
            det = report['detection_performance']
            track = report['tracking_performance']
            comp = report['performance_comparison']
            camera_iou = report.get('camera_iou_performance', {})

            f.write("📋 EXECUTIVE SUMMARY\n")
            f.write("─" * 50 + "\n")
            f.write(
                f"Frames: {summary['frames_evaluated']:,} | Distance Threshold: {summary['distance_threshold_m']:.1f}m | IoU Threshold: {summary['iou_threshold']:.2f} | Avg GT/Frame: {det['avg_ground_truth_per_frame']:.1f}\n\n")

            # Performance Summary Table
            f.write("🎯 PERFORMANCE SUMMARY\n")
            f.write("─" * 50 + "\n")
            f.write(f"{'Metric':<25} {'Detection':<12} {'Tracking':<12} {'Improvement':<12}\n")
            f.write("─" * 80 + "\n")

            # Core metrics
            f.write(
                f"{'Precision':<25} {det['precision']:<12.4f} {track['precision']:<12.4f} {comp['precision']['improvement_percent']:>+10.1f}%\n")
            f.write(
                f"{'Recall':<25} {det['recall']:<12.4f} {track['recall']:<12.4f} {comp['recall']['improvement_percent']:>+10.1f}%\n")
            f.write(
                f"{'F1-Score':<25} {det['f1_score']:<12.4f} {track['f1_score']:<12.4f} {comp['f1_score']['improvement_percent']:>+10.1f}%\n")
            f.write(
                f"{'DetA':<25} {det['det_a']:<12.4f} {track['det_a']:<12.4f} {comp['det_a']['improvement_percent']:>+10.1f}%\n")

            # Distance metrics
            det_dist = "N/A" if det['mean_euclidean_distance_m'] == float(
                'inf') else f"{det['mean_euclidean_distance_m']:.2f}"
            track_dist = "N/A" if track['mean_euclidean_distance_m'] == float(
                'inf') else f"{track['mean_euclidean_distance_m']:.2f}"
            dist_improvement = "N/A" if comp['mean_euclidean_distance_m']['improvement_percent'] == float(
                'inf') else f"{-comp['mean_euclidean_distance_m']['improvement_percent']:+10.1f}%"
            f.write(f"{'Mean Distance (m)':<25} {det_dist:<12} {track_dist:<12} {dist_improvement:>11}\n")

            # Camera IoU metrics
            if camera_iou:
                det_cam_iou = camera_iou.get('detection_camera_iou', {})
                track_cam_iou = camera_iou.get('tracking_camera_iou', {})

                det_cam_mean = det_cam_iou.get('mean', 0.0)
                track_cam_mean = track_cam_iou.get('mean', 0.0)
                cam_improvement = "N/A"
                if det_cam_mean > 0 and track_cam_mean > 0:
                    cam_improvement = f"{((track_cam_mean - det_cam_mean) / det_cam_mean) * 100:+10.1f}%"

                f.write(f"{'Camera IoU':<25} {det_cam_mean:<12.4f} {track_cam_mean:<12.4f} {cam_improvement:>11}\n")

            f.write("\n")

            # Detection Details
            f.write("🔍 DETECTION DETAILS\n")
            f.write("─" * 50 + "\n")
            f.write(
                f"TP/FP/FN: {det['true_positives']}/{det['false_positives']}/{det['false_negatives']} | Associations: {det['total_associations']} | Avg Det/Frame: {det['avg_detections_per_frame']:.1f}\n")
            if det['mean_euclidean_distance_m'] != float('inf'):
                f.write(
                    f"Distance Stats: Mean={det['mean_euclidean_distance_m']:.2f}m, Std={det['std_euclidean_distance_m']:.2f}m, Range=[{det['min_distance_m']:.2f}, {det['max_distance_m']:.2f}]m\n")
            else:
                f.write("Distance Stats: No valid associations\n")
            f.write("\n")

            # Tracking Details
            f.write("🔄 TRACKING DETAILS\n")
            f.write("─" * 50 + "\n")
            f.write(
                f"TP/FP/FN: {track['true_positives']}/{track['false_positives']}/{track['false_negatives']} | Associations: {track['total_associations']} | Avg Tracks/Frame: {track['avg_detections_per_frame']:.1f}\n")
            if track['mean_euclidean_distance_m'] != float('inf'):
                f.write(
                    f"Distance Stats: Mean={track['mean_euclidean_distance_m']:.2f}m, Std={track['std_euclidean_distance_m']:.2f}m, Range=[{track['min_distance_m']:.2f}, {track['max_distance_m']:.2f}]m\n")
            else:
                f.write("Distance Stats: No valid associations\n")
            f.write("\n")

            # Camera IoU Details
            if camera_iou:
                f.write("📷 CAMERA IoU DETAILS\n")
                f.write("─" * 50 + "\n")
                det_cam_iou = camera_iou.get('detection_camera_iou', {})
                track_cam_iou = camera_iou.get('tracking_camera_iou', {})

                f.write(
                    f"Detection: Mean={det_cam_iou.get('mean', 0.0):.4f}, Std={det_cam_iou.get('std', 0.0):.4f}, Median={det_cam_iou.get('median', 0.0):.4f}, Count={det_cam_iou.get('count', 0)}\n")
                if track_cam_iou.get('count', 0) > 0:
                    f.write(
                        f"Tracking: Mean={track_cam_iou.get('mean', 0.0):.4f}, Std={track_cam_iou.get('std', 0.0):.4f}, Median={track_cam_iou.get('median', 0.0):.4f}, Count={track_cam_iou.get('count', 0)}\n")
                f.write("\n")

            # Performance Analysis
            f.write("📊 PERFORMANCE ANALYSIS\n")
            f.write("─" * 50 + "\n")

            # Overall assessment
            avg_improvement = (comp['precision']['improvement_percent'] +
                               comp['recall']['improvement_percent'] +
                               comp['f1_score']['improvement_percent']) / 3

            if avg_improvement > 5:
                assessment = "✅ TRACKING SIGNIFICANTLY IMPROVES PERFORMANCE"
            elif avg_improvement > 1:
                assessment = "✅ Tracking provides modest improvement"
            elif avg_improvement > -1:
                assessment = "⚖️ Tracking performance is comparable to detection"
            else:
                assessment = "⚠️ Tracking underperforms compared to detection"

            f.write(f"{assessment} (Avg Improvement: {avg_improvement:+.1f}%)\n")

            # Key insights in one line
            insights = []
            if comp['precision']['improvement_percent'] > 5:
                insights.append("Reduces false positives")
            elif comp['precision']['improvement_percent'] < -5:
                insights.append("Increases false positives")

            if comp['recall']['improvement_percent'] > 5:
                insights.append("Reduces missed detections")
            elif comp['recall']['improvement_percent'] < -5:
                insights.append("Increases missed detections")

            if abs(comp['mean_euclidean_distance_m']['improvement_percent']) > 10:
                if comp['mean_euclidean_distance_m']['improvement_percent'] > 0:
                    insights.append("Improves spatial accuracy")
                else:
                    insights.append("Reduces spatial accuracy")

            if insights:
                f.write(f"Key Insights: {' | '.join(insights)}\n")
            f.write("\n")

            # Footer
            f.write("═" * 100 + "\n")
            f.write("Report generated by Tracking & Detection Evaluator\n")
            f.write("═" * 100 + "\n")

    def print_summary_report(self) -> None:
        """Print a formatted summary report."""
        report = self.generate_comprehensive_report()

        if 'error' in report:
            print(f"Error: {report['error']}")
            return

        print("\n" + "=" * 80)
        print("           TRACKING AND DETECTION EVALUATION REPORT")
        print("=" * 80)

        summary = report['evaluation_summary']
        print(f"\n📊 EVALUATION SUMMARY:")
        print(f"   • Frames Evaluated: {summary['frames_evaluated']}")
        print(f"   • Distance Threshold: {summary['distance_threshold_m']:.1f}m")
        print(f"   • IoU Threshold: {summary['iou_threshold']:.2f}")

        # Detection Performance
        det = report['detection_performance']
        print(f"\n🎯 DETECTION PERFORMANCE:")
        print(f"   • Precision:           {det['precision']:.3f}")
        print(f"   • Recall:              {det['recall']:.3f}")
        print(f"   • F1-Score:            {det['f1_score']:.3f}")
        print(f"   • DetA:                {det['det_a']:.3f}")
        print(f"   • Mean Distance:       {det['mean_euclidean_distance_m']:.2f}m")
        print(f"   • MOTP:                {det['motp_m']:.2f}m")
        print(f"   • Mean IoU:            {det['mean_iou']:.3f}")
        print(
            f"   • TP/FP/FN:            {det['true_positives']}/{det['false_positives']}/{det['false_negatives']}")

        # Tracking Performance
        track = report['tracking_performance']
        print(f"\n🔄 TRACKING PERFORMANCE:")
        print(f"   • Precision:           {track['precision']:.3f}")
        print(f"   • Recall:              {track['recall']:.3f}")
        print(f"   • F1-Score:            {track['f1_score']:.3f}")
        print(f"   • DetA:                {track['det_a']:.3f}")
        print(f"   • Mean Distance:       {track['mean_euclidean_distance_m']:.2f}m")
        print(f"   • MOTP:                {track['motp_m']:.2f}m")
        print(f"   • Mean IoU:            {track['mean_iou']:.3f}")
        print(
            f"   • TP/FP/FN:            {track['true_positives']}/{track['false_positives']}/{track['false_negatives']}")

        # Performance Comparison
        comp = report['performance_comparison']
        print(f"\n📈 PERFORMANCE COMPARISON (Tracking vs Detection):")
        print(f"   • Precision Improvement:     {comp['precision']['improvement_percent']:+.1f}%")
        print(f"   • Recall Improvement:        {comp['recall']['improvement_percent']:+.1f}%")
        print(f"   • F1-Score Improvement:      {comp['f1_score']['improvement_percent']:+.1f}%")
        print(f"   • DetA Improvement:          {comp['det_a']['improvement_percent']:+.1f}%")
        print(
            f"   • Distance Improvement:      {comp['mean_euclidean_distance_m']['improvement_percent']:+.1f}%")
        print(f"   • MOTP Improvement:          {comp['motp_m']['improvement_percent']:+.1f}%")

        print("\n" + "=" * 80)

    def save_report(self, output_path: str) -> None:
        """Save evaluation report to JSON file."""
        report = self.generate_comprehensive_report()

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        report = convert_numpy_types(report)

        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n💾 Evaluation report saved to: {output_path}")

def evaluate_tracking_sequence(predictions_csv: str, ground_truth_csv: str,
                               tracking_csv: str, output_dir: str,
                               distance_threshold: float = 5.0,
                               iou_threshold: float = 0.3,
                               max_frames: Optional[int] = None) -> Tuple[Path, Dict[str, Any]]:
    """
    Evaluate complete tracking sequence.

    Args:
        predictions_csv: Path to network predictions
        ground_truth_csv: Path to ground truth labels
        tracking_csv: Path to tracking results
        output_dir: Directory to save evaluation results
        distance_threshold: Maximum distance for valid association (meters)
        iou_threshold: IoU threshold for bounding box overlap
        max_frames: Maximum number of frames to evaluate (None for all)

    Returns:
        Tuple of (output_path, comprehensive evaluation report)
    """
    evaluator = TrackingDetectionEvaluator(
        distance_threshold=distance_threshold,
        iou_threshold=iou_threshold
    )

    # Load data
    predictions_df = pd.read_csv(predictions_csv)
    ground_truth_df = pd.read_csv(ground_truth_csv, sep='\t|,', engine='python')
    tracking_df = pd.read_csv(tracking_csv)

    # Get ALL frames that have ground truth (we need GT to evaluate)
    gt_frames = set(ground_truth_df['numSample'].unique())
    frame_ids = sorted(gt_frames)
    if max_frames:
        frame_ids = frame_ids[:max_frames]  # Limit to max_frames if specified

    # Evaluate each frame
    for frame_id in frame_ids:
        # Get data for this frame (empty DataFrames if no data exists)
        pred_frame = predictions_df[predictions_df['sample_id'] == frame_id]
        gt_frame = ground_truth_df[ground_truth_df['numSample'] == frame_id]
        track_frame = tracking_df[tracking_df['sample_id'] == frame_id]

        # Always evaluate - missing predictions/tracks will be counted as FN
        evaluator.evaluate_frame(pred_frame, gt_frame, track_frame, frame_id)

    # Make it a Path object for consistency
    output_path = Path(output_dir)

    # Save detailed JSON report
    json_path = output_path / 'comprehensive_evaluation.json'
    evaluator.save_report(str(json_path))

    # Save readable text report
    text_path = output_path / 'evaluation_summary.txt'
    evaluator.save_text_report(str(text_path))

    return output_path, evaluator.generate_comprehensive_report()

# Example usage
if __name__ == "__main__":
    # Example usage
    predictions_csv = "predictions.csv"
    ground_truth_csv = "ground_truth.csv"
    tracking_csv = "tracking_results.csv"
    output_dir = "evaluation_results"

    report = evaluate_tracking_sequence(
        predictions_csv=predictions_csv,
        ground_truth_csv=ground_truth_csv,
        tracking_csv=tracking_csv,
        output_dir=output_dir,
        distance_threshold=5.0,
        iou_threshold=0.3
    )