"""
Comprehensive tracking and association quality metrics.
Implements various evaluation metrics for radar object tracking performance.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import json


@dataclass
class AssociationMetrics:
    """Container for association quality metrics."""
    # Detection-level metrics
    precision: float
    recall: float
    f1_score: float

    # Distance-based metrics
    mean_euclidean_distance: float
    std_euclidean_distance: float
    mean_mahalanobis_distance: Optional[float]

    # Track-level metrics
    track_purity: float
    track_completeness: float

    # Association statistics
    total_associations: int
    correct_associations: int
    false_positives: int
    false_negatives: int

    # Per-frame statistics
    frames_evaluated: int
    avg_detections_per_frame: float
    avg_tracks_per_frame: float


@dataclass
class TrackingQualityMetrics:
    """Container for overall tracking quality metrics."""
    # Accuracy metrics
    mota: float  # Multiple Object Tracking Accuracy
    motp: float  # Multiple Object Tracking Precision

    # Track lifecycle metrics
    track_birth_accuracy: float
    track_death_accuracy: float
    track_maintenance_ratio: float

    # Fragmentation metrics
    fragmentation_rate: float
    id_switches: int

    # Temporal consistency
    temporal_consistency_score: float


class TrackingEvaluator:
    """Comprehensive tracking evaluation with multiple metrics."""

    def __init__(self, distance_threshold: float = 5.0, iou_threshold: float = 0.5):
        """
        Initialize evaluator.

        Args:
            distance_threshold: Maximum distance for valid association (meters)
            iou_threshold: IoU threshold for bounding box overlap (if applicable)
        """
        self.distance_threshold = distance_threshold
        self.iou_threshold = iou_threshold

        # Storage for evaluation data
        self.frame_results = []
        self.track_histories = {}
        self.gt_track_histories = {}

    def evaluate_frame(self, predictions: pd.DataFrame, ground_truth: pd.DataFrame,
                       tracks: pd.DataFrame, frame_id: int) -> Dict[str, Any]:
        """
        Evaluate single frame performance.

        Args:
            predictions: Network predictions for frame
            ground_truth: Ground truth detections for frame
            tracks: Tracker outputs for frame
            frame_id: Frame identifier

        Returns:
            Dictionary with frame-level metrics
        """
        # Extract positions
        pred_positions = self._extract_positions(predictions)
        gt_positions = self._extract_positions(ground_truth, is_gt=True)
        track_positions = self._extract_positions(tracks)

        # Evaluate detection performance (predictions vs ground truth)
        det_metrics = self._evaluate_detection_association(pred_positions, gt_positions)

        # Evaluate tracking performance (tracks vs ground truth)
        track_metrics = self._evaluate_tracking_association(track_positions, gt_positions)

        # Calculate distance-based metrics
        distance_metrics = self._calculate_distance_metrics(track_positions, gt_positions)

        frame_result = {
            'frame_id': frame_id,
            'detection_metrics': det_metrics,
            'tracking_metrics': track_metrics,
            'distance_metrics': distance_metrics,
            'num_predictions': len(predictions),
            'num_ground_truth': len(ground_truth),
            'num_tracks': len(tracks)
        }

        self.frame_results.append(frame_result)
        return frame_result

    def _extract_positions(self, df: pd.DataFrame, is_gt: bool = False) -> np.ndarray:
        """Extract position data from dataframe."""
        if df.empty:
            return np.zeros((0, 2))

        if is_gt:
            # Ground truth uses different column names
            if 'radar_R_m' in df.columns and 'radar_A_deg' in df.columns:
                ranges = df['radar_R_m'].values
                azimuths = np.deg2rad(df['radar_A_deg'].values)
            else:
                return np.zeros((0, 2))
        else:
            # Predictions and tracks
            if 'range_m' in df.columns and 'azimuth_deg' in df.columns:
                ranges = df['range_m'].values
                azimuths = np.deg2rad(df['azimuth_deg'].values)
            else:
                return np.zeros((0, 2))

        # Convert to Cartesian coordinates
        x = ranges * np.sin(azimuths)
        y = ranges * np.cos(azimuths)

        return np.column_stack([x, y])

    def _evaluate_detection_association(self, predictions: np.ndarray,
                                        ground_truth: np.ndarray) -> Dict[str, float]:
        """Evaluate detection-level association quality."""
        if len(predictions) == 0 or len(ground_truth) == 0:
            return {
                'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                'true_positives': 0, 'false_positives': len(predictions),
                'false_negatives': len(ground_truth)
            }

        # Calculate distance matrix
        distances = cdist(predictions, ground_truth)

        # Apply Hungarian algorithm
        pred_indices, gt_indices = linear_sum_assignment(distances)

        # Count associations within threshold
        valid_matches = distances[pred_indices, gt_indices] <= self.distance_threshold
        true_positives = np.sum(valid_matches)
        false_positives = len(predictions) - true_positives
        false_negatives = len(ground_truth) - true_positives

        # Calculate metrics
        precision = true_positives / len(predictions) if len(predictions) > 0 else 0.0
        recall = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

    def _evaluate_tracking_association(self, tracks: np.ndarray,
                                       ground_truth: np.ndarray) -> Dict[str, float]:
        """Evaluate tracking-level association quality."""
        return self._evaluate_detection_association(tracks, ground_truth)

    def _calculate_distance_metrics(self, tracks: np.ndarray,
                                    ground_truth: np.ndarray) -> Dict[str, float]:
        """Calculate distance-based evaluation metrics."""
        if len(tracks) == 0 or len(ground_truth) == 0:
            return {
                'mean_euclidean_distance': float('inf'),
                'std_euclidean_distance': 0.0,
                'min_distance': float('inf'),
                'max_distance': 0.0
            }

        # Calculate distance matrix
        distances = cdist(tracks, ground_truth)

        # Find optimal assignment
        track_indices, gt_indices = linear_sum_assignment(distances)

        # Get distances for valid assignments
        assigned_distances = distances[track_indices, gt_indices]
        valid_distances = assigned_distances[assigned_distances <= self.distance_threshold]

        if len(valid_distances) == 0:
            return {
                'mean_euclidean_distance': float('inf'),
                'std_euclidean_distance': 0.0,
                'min_distance': float('inf'),
                'max_distance': 0.0
            }

        return {
            'mean_euclidean_distance': float(np.mean(valid_distances)),
            'std_euclidean_distance': float(np.std(valid_distances)),
            'min_distance': float(np.min(valid_distances)),
            'max_distance': float(np.max(valid_distances))
        }

    def calculate_mota_motp(self) -> Tuple[float, float]:
        """Calculate MOTA and MOTP metrics across all frames."""
        total_gt = sum(result['num_ground_truth'] for result in self.frame_results)
        total_fp = sum(result['tracking_metrics']['false_positives'] for result in self.frame_results)
        total_fn = sum(result['tracking_metrics']['false_negatives'] for result in self.frame_results)

        # MOTA = 1 - (FN + FP) / GT
        mota = 1.0 - (total_fn + total_fp) / total_gt if total_gt > 0 else 0.0

        # MOTP = average distance of correctly associated tracks
        valid_distances = []
        for result in self.frame_results:
            dist_metrics = result['distance_metrics']
            if dist_metrics['mean_euclidean_distance'] != float('inf'):
                valid_distances.append(dist_metrics['mean_euclidean_distance'])

        motp = np.mean(valid_distances) if valid_distances else float('inf')

        return mota, motp

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        if not self.frame_results:
            return {'error': 'No evaluation data available'}

        # Aggregate frame-level metrics
        detection_metrics = self._aggregate_detection_metrics()
        tracking_metrics = self._aggregate_tracking_metrics()
        distance_metrics = self._aggregate_distance_metrics()

        # Calculate high-level metrics
        mota, motp = self.calculate_mota_motp()

        # Track-level analysis
        track_analysis = self._analyze_track_quality()

        report = {
            'summary': {
                'frames_evaluated': len(self.frame_results),
                'mota': mota,
                'motp': motp
            },
            'detection_performance': detection_metrics,
            'tracking_performance': tracking_metrics,
            'distance_analysis': distance_metrics,
            'track_analysis': track_analysis,
            'frame_by_frame': self.frame_results
        }

        return report

    def _aggregate_detection_metrics(self) -> Dict[str, float]:
        """Aggregate detection metrics across all frames."""
        total_tp = sum(result['detection_metrics']['true_positives'] for result in self.frame_results)
        total_fp = sum(result['detection_metrics']['false_positives'] for result in self.frame_results)
        total_fn = sum(result['detection_metrics']['false_negatives'] for result in self.frame_results)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'overall_precision': precision,
            'overall_recall': recall,
            'overall_f1_score': f1_score,
            'total_true_positives': total_tp,
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn
        }

    def _aggregate_tracking_metrics(self) -> Dict[str, float]:
        """Aggregate tracking metrics across all frames."""
        total_tp = sum(result['tracking_metrics']['true_positives'] for result in self.frame_results)
        total_fp = sum(result['tracking_metrics']['false_positives'] for result in self.frame_results)
        total_fn = sum(result['tracking_metrics']['false_negatives'] for result in self.frame_results)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'overall_precision': precision,
            'overall_recall': recall,
            'overall_f1_score': f1_score,
            'total_true_positives': total_tp,
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn
        }

    def _aggregate_distance_metrics(self) -> Dict[str, float]:
        """Aggregate distance metrics across all frames."""
        valid_distances = []
        for result in self.frame_results:
            dist_metrics = result['distance_metrics']
            if dist_metrics['mean_euclidean_distance'] != float('inf'):
                valid_distances.append(dist_metrics['mean_euclidean_distance'])

        if not valid_distances:
            return {
                'overall_mean_distance': float('inf'),
                'overall_std_distance': 0.0,
                'frames_with_valid_associations': 0
            }

        return {
            'overall_mean_distance': float(np.mean(valid_distances)),
            'overall_std_distance': float(np.std(valid_distances)),
            'frames_with_valid_associations': len(valid_distances)
        }

    def _analyze_track_quality(self) -> Dict[str, Any]:
        """Analyze track-level quality metrics."""
        track_lengths = []
        total_tracks = set()

        for result in self.frame_results:
            # This would need track ID information to fully implement
            # For now, provide basic statistics
            total_tracks.add(result['frame_id'])  # Placeholder

        return {
            'unique_tracks_seen': len(total_tracks),
            'average_track_length': np.mean(track_lengths) if track_lengths else 0.0,
            'track_fragmentation_analysis': 'Not implemented - requires track ID data'
        }

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

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Evaluation report saved to: {output_path}")


def evaluate_tracking_sequence(predictions_csv: str, ground_truth_csv: str,
                               tracking_csv: str, output_dir: str) -> Dict[str, Any]:
    """
    Evaluate complete tracking sequence.

    Args:
        predictions_csv: Path to network predictions
        ground_truth_csv: Path to ground truth labels
        tracking_csv: Path to tracking results
        output_dir: Directory to save evaluation results

    Returns:
        Comprehensive evaluation report
    """
    evaluator = TrackingEvaluator()

    # Load data
    predictions_df = pd.read_csv(predictions_csv)
    ground_truth_df = pd.read_csv(ground_truth_csv, sep='\t|,', engine='python')
    tracking_df = pd.read_csv(tracking_csv)

    # Get unique frame IDs
    frame_ids = sorted(set(predictions_df['sample_id'].unique()) &
                       set(ground_truth_df['numSample'].unique()) &
                       set(tracking_df['sample_id'].unique()))

    print(f"Evaluating {len(frame_ids)} frames...")

    # Evaluate each frame
    for frame_id in frame_ids:
        pred_frame = predictions_df[predictions_df['sample_id'] == frame_id]
        gt_frame = ground_truth_df[ground_truth_df['numSample'] == frame_id]
        track_frame = tracking_df[tracking_df['sample_id'] == frame_id]

        evaluator.evaluate_frame(pred_frame, gt_frame, track_frame, frame_id)

    # Generate and save report
    report = evaluator.generate_comprehensive_report()

    output_path = Path(output_dir) / 'tracking_evaluation_report.json'
    evaluator.save_report(str(output_path))

    return report