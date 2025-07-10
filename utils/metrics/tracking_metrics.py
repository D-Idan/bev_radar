"""
Comprehensive tracking and detection evaluation metrics.
Implements precision, DetA, distance-based metrics, and general statistics.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import json

from utils.metrics.camera_iou_metrics import CameraIoUCalculator
from utils.radar_camera_relation import is_radar_point_in_camera_view


@dataclass
class FrameMetrics:
    """Container for single frame metrics."""
    frame_id: int
    eval_type: str
    precision: float
    recall: float
    f1_score: float
    det_a: float
    mean_euclidean_distance: float
    std_euclidean_distance: float
    min_distance: float
    max_distance: float
    motp: float
    true_positives: int
    false_positives: int
    false_negatives: int
    total_associations: int


@dataclass
class AggregatedMetrics:
    """Container for aggregated metrics across all frames."""
    precision: float
    recall: float
    f1_score: float
    det_a: float
    mean_euclidean_distance: float
    std_euclidean_distance: float
    min_distance: float
    max_distance: float
    motp: float
    total_associations: int
    true_positives: int
    false_positives: int
    false_negatives: int
    frames_evaluated: int
    avg_detections_per_frame: float
    avg_ground_truth_per_frame: float


class TrackingDetectionEvaluator:
    """Comprehensive evaluation for both detection and tracking performance."""

    def __init__(self, distance_threshold: float = 5.0, iou_threshold: float = 0.3,
                 use_camera_fov_filter: bool = True):
        """Initialize evaluator with thresholds."""
        self.distance_threshold = distance_threshold
        self.iou_threshold = iou_threshold
        self.use_camera_fov_filter = use_camera_fov_filter
        self.camera_iou_calculator = CameraIoUCalculator()

        # Storage for evaluation data
        self.frame_results = []
        self.detection_frame_results = []
        self.tracking_frame_results = []
        self.camera_iou_results = []

    def evaluate_frame(self, predictions: pd.DataFrame, ground_truth: pd.DataFrame,
                       tracks: pd.DataFrame, frame_id: int) -> Dict[str, Any]:
        """Evaluate single frame performance for both detection and tracking."""
        # Extract data for evaluation
        pred_data = self._extract_detection_data(predictions)
        gt_data = self._extract_ground_truth_data(ground_truth)
        track_data = self._extract_tracking_data(tracks)

        # Evaluate detection and tracking performance
        detection_metrics = self._evaluate_associations(pred_data, gt_data, "detection", frame_id)
        tracking_metrics = self._evaluate_associations(track_data, gt_data, "tracking", frame_id)

        # Camera IoU evaluation
        camera_iou_result = self.camera_iou_calculator.evaluate_camera_iou_single_frame(
            predictions, ground_truth, tracks, frame_id, image_shape=(540, 960)
        )
        self.camera_iou_results.append(camera_iou_result)

        frame_result = {
            'frame_id': frame_id,
            'detection_results': asdict(detection_metrics),
            'tracking_results': asdict(tracking_metrics),
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
        """Extract detection data from dataframe."""
        if df.empty:
            return {'positions': np.zeros((0, 2)), 'confidences': np.zeros(0)}

        # Filter by camera FOV if enabled
        if self.use_camera_fov_filter and 'range_m' in df.columns and 'azimuth_deg' in df.columns:
            # Filter to keep only detections visible in camera
            visible_mask = df.apply(
                lambda row: is_radar_point_in_camera_view(row['range_m'], row['azimuth_deg']),
                axis=1
            )
            df = df[visible_mask].copy()

            if df.empty:  # All detections were filtered out
                return {'positions': np.zeros((0, 2)), 'confidences': np.zeros(0)}

        ranges = df['range_m'].values
        azimuths = np.deg2rad(df['azimuth_deg'].values)
        x = ranges * np.sin(azimuths)
        y = ranges * np.cos(azimuths)
        positions = np.column_stack([x, y])
        confidences = df['confidence'].values if 'confidence' in df.columns else np.ones(len(df))

        return {'positions': positions, 'confidences': confidences}

    def _extract_ground_truth_data(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract ground truth data from dataframe."""
        if df.empty:
            return {'positions': np.zeros((0, 2))}

        if 'radar_R_m' in df.columns and 'radar_A_deg' in df.columns:
            ranges = df['radar_R_m'].values
            azimuths = np.deg2rad(df['radar_A_deg'].values)
            x = ranges * np.sin(azimuths)
            y = ranges * np.cos(azimuths)
            positions = np.column_stack([x, y])
        else:
            positions = np.zeros((0, 2))

        return {'positions': positions}

    def _extract_tracking_data(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract tracking data from dataframe."""
        if df.empty:
            return {'positions': np.zeros((0, 2)), 'confidences': np.zeros(0), 'track_ids': np.zeros(0)}

        # Filter by camera FOV if enabled
        if self.use_camera_fov_filter and 'range_m' in df.columns and 'azimuth_deg' in df.columns:
            # Filter to keep only tracks visible in camera
            visible_mask = df.apply(
                lambda row: is_radar_point_in_camera_view(row['range_m'], row['azimuth_deg']),
                axis=1
            )
            df = df[visible_mask].copy()

            if df.empty:  # All tracks were filtered out
                return {'positions': np.zeros((0, 2)), 'confidences': np.zeros(0), 'track_ids': np.zeros(0)}

        ranges = df['range_m'].values
        azimuths = np.deg2rad(df['azimuth_deg'].values)
        x = ranges * np.sin(azimuths)
        y = ranges * np.cos(azimuths)
        positions = np.column_stack([x, y])
        confidences = df['confidence'].values if 'confidence' in df.columns else np.ones(len(df))
        track_ids = df['track_id'].values if 'track_id' in df.columns else np.arange(len(df))

        return {'positions': positions, 'confidences': confidences, 'track_ids': track_ids}

    def _evaluate_associations(self, pred_data: Dict, gt_data: Dict,
                               eval_type: str, frame_id: int) -> FrameMetrics:
        """Evaluate associations between predictions/tracks and ground truth."""
        pred_positions = pred_data['positions']
        gt_positions = gt_data['positions']

        if len(pred_positions) == 0 or len(gt_positions) == 0:
            return FrameMetrics(
                frame_id=frame_id, eval_type=eval_type, precision=0.0, recall=0.0,
                f1_score=0.0, det_a=0.0, mean_euclidean_distance=float('inf'),
                std_euclidean_distance=0.0, min_distance=float('inf'), max_distance=0.0,
                motp=float('inf'), true_positives=0, false_positives=len(pred_positions),
                false_negatives=len(gt_positions), total_associations=0
            )

        # Calculate distance matrix and optimal assignment
        distances = cdist(pred_positions, gt_positions)
        pred_indices, gt_indices = linear_sum_assignment(distances)

        # Determine valid matches
        valid_matches_mask = distances[pred_indices, gt_indices] <= self.distance_threshold
        valid_distances = distances[pred_indices, gt_indices][valid_matches_mask]

        # Calculate metrics
        true_positives = len(valid_distances)
        false_positives = len(pred_positions) - true_positives
        false_negatives = len(gt_positions) - true_positives

        precision = true_positives / len(pred_positions) if len(pred_positions) > 0 else 0.0
        recall = true_positives / len(gt_positions) if len(gt_positions) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        det_a = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0.0

        # Distance-based metrics
        if len(valid_distances) > 0:
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

        return FrameMetrics(
            frame_id=frame_id, eval_type=eval_type, precision=precision, recall=recall,
            f1_score=f1_score, det_a=det_a, mean_euclidean_distance=mean_euclidean_distance,
            std_euclidean_distance=std_euclidean_distance, min_distance=min_distance,
            max_distance=max_distance, motp=motp, true_positives=true_positives,
            false_positives=false_positives, false_negatives=false_negatives,
            total_associations=true_positives
        )

    def _aggregate_metrics(self, frame_results: List[FrameMetrics], eval_type: str) -> AggregatedMetrics:
        """Aggregate metrics across all frames."""
        if not frame_results:
            return AggregatedMetrics(
                precision=0.0, recall=0.0, f1_score=0.0, det_a=0.0,
                mean_euclidean_distance=float('inf'), std_euclidean_distance=0.0,
                min_distance=float('inf'), max_distance=0.0, motp=float('inf'),
                total_associations=0, true_positives=0, false_positives=0, false_negatives=0,
                frames_evaluated=0, avg_detections_per_frame=0.0, avg_ground_truth_per_frame=0.0
            )

        # Sum up counts
        total_tp = sum(result.true_positives for result in frame_results)
        total_fp = sum(result.false_positives for result in frame_results)
        total_fn = sum(result.false_negatives for result in frame_results)

        # Calculate overall metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        det_a = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

        # Distance metrics
        valid_distances = [r.mean_euclidean_distance for r in frame_results if r.mean_euclidean_distance != float('inf')]
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

        # Frame statistics
        frames_evaluated = len(frame_results)
        if eval_type == "detection":
            avg_detections_per_frame = np.mean([r['num_predictions'] for r in self.frame_results])
        else:
            avg_detections_per_frame = np.mean([r['num_tracks'] for r in self.frame_results])
        avg_ground_truth_per_frame = np.mean([r['num_ground_truth'] for r in self.frame_results])

        return AggregatedMetrics(
            precision=precision, recall=recall, f1_score=f1_score, det_a=det_a,
            mean_euclidean_distance=mean_euclidean_distance, std_euclidean_distance=std_euclidean_distance,
            min_distance=min_distance, max_distance=max_distance, motp=motp,
            total_associations=total_tp, true_positives=total_tp, false_positives=total_fp,
            false_negatives=total_fn, frames_evaluated=frames_evaluated,
            avg_detections_per_frame=avg_detections_per_frame, avg_ground_truth_per_frame=avg_ground_truth_per_frame
        )

    def _calculate_camera_iou_summary(self) -> Dict[str, Any]:
        """Calculate camera IoU summary statistics."""
        all_detection_camera_ious = []
        all_tracking_camera_ious = []

        for result in self.camera_iou_results:
            all_detection_camera_ious.extend(result['detection_vs_labels_ious'])
            if result['tracking_vs_labels_ious'] is not None:
                all_tracking_camera_ious.extend(result['tracking_vs_labels_ious'])

        return {
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

    def _calculate_improvements(self, tracking_metrics: AggregatedMetrics,
                                detection_metrics: AggregatedMetrics) -> Dict[str, Dict[str, float]]:
        """Calculate improvement percentages between tracking and detection."""
        def calc_improvement(tracking_val, detection_val):
            if detection_val == 0:
                return float('inf') if tracking_val > 0 else 0.0
            if detection_val == float('inf'):
                return -100.0 if tracking_val != float('inf') else 0.0
            return ((tracking_val - detection_val) / detection_val) * 100

        return {
            'precision': {
                'detection': detection_metrics.precision,
                'tracking': tracking_metrics.precision,
                'improvement_percent': calc_improvement(tracking_metrics.precision, detection_metrics.precision)
            },
            'recall': {
                'detection': detection_metrics.recall,
                'tracking': tracking_metrics.recall,
                'improvement_percent': calc_improvement(tracking_metrics.recall, detection_metrics.recall)
            },
            'f1_score': {
                'detection': detection_metrics.f1_score,
                'tracking': tracking_metrics.f1_score,
                'improvement_percent': calc_improvement(tracking_metrics.f1_score, detection_metrics.f1_score)
            },
            'det_a': {
                'detection': detection_metrics.det_a,
                'tracking': tracking_metrics.det_a,
                'improvement_percent': calc_improvement(tracking_metrics.det_a, detection_metrics.det_a)
            },
            'mean_euclidean_distance_m': {
                'detection': detection_metrics.mean_euclidean_distance,
                'tracking': tracking_metrics.mean_euclidean_distance,
                'improvement_percent': -calc_improvement(tracking_metrics.mean_euclidean_distance,
                                                         detection_metrics.mean_euclidean_distance)
            },
            'motp_m': {
                'detection': detection_metrics.motp,
                'tracking': tracking_metrics.motp,
                'improvement_percent': -calc_improvement(tracking_metrics.motp, detection_metrics.motp)
            }
        }

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate the single source of truth for all evaluation metrics."""
        if not self.frame_results:
            return {'error': 'No evaluation data available'}

        # Aggregate metrics
        detection_metrics = self._aggregate_metrics(self.detection_frame_results, "detection")
        tracking_metrics = self._aggregate_metrics(self.tracking_frame_results, "tracking")

        # Calculate improvements and camera IoU
        improvements = self._calculate_improvements(tracking_metrics, detection_metrics)
        camera_iou_summary = self._calculate_camera_iou_summary()

        return {
            'evaluation_summary': {
                'frames_evaluated': len(self.frame_results),
                'distance_threshold_m': self.distance_threshold,
                'iou_threshold': self.iou_threshold
            },
            'detection_performance': asdict(detection_metrics),
            'tracking_performance': asdict(tracking_metrics),
            'performance_comparison': improvements,
            'camera_iou_performance': camera_iou_summary,
            'frame_by_frame_results': self.frame_results
        }

    def save_json_report(self, output_path: str) -> None:
        """Save comprehensive JSON report split into summary and detailed frame files."""
        report = self.generate_comprehensive_report()

        # Convert numpy types for JSON serialization
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

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Split the report into summary and detailed files
        summary_report = {k: v for k, v in report.items() if k != 'frame_by_frame_results'}
        frame_details = report.get('frame_by_frame_results', [])

        # Save summary metrics (lightweight for aggregation)
        summary_path = output_path.parent / f"{output_path.stem}_summary{output_path.suffix}"
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2)

        # Save detailed frame-by-frame results (heavy data)
        details_path = output_path.parent / f"{output_path.stem}_frame_details{output_path.suffix}"
        with open(details_path, 'w') as f:
            json.dump({
                'evaluation_info': {
                    'frames_evaluated': len(frame_details),
                    'distance_threshold_m': self.distance_threshold,
                    'iou_threshold': self.iou_threshold
                },
                'frame_by_frame_results': frame_details
            }, f, indent=2)

    def save_text_report(self, output_path: str) -> None:
        """Save human-readable text report using the comprehensive report data."""
        report = self.generate_comprehensive_report()

        if 'error' in report:
            with open(output_path, 'w') as f:
                f.write(f"Error: {report['error']}\n")
            return

        with open(output_path, 'w') as f:
            self._write_report_content(f, report)

    def print_summary_report(self) -> None:
        """Print formatted summary using the comprehensive report data."""
        report = self.generate_comprehensive_report()

        if 'error' in report:
            print(f"Error: {report['error']}")
            return

        print("\n" + "=" * 80)
        print("           TRACKING AND DETECTION EVALUATION REPORT")
        print("=" * 80)

        self._print_report_content(report)
        print("\n" + "=" * 80)

    def _write_report_content(self, f, report: Dict[str, Any]) -> None:
        """Write formatted report content to file."""
        summary = report['evaluation_summary']
        det = report['detection_performance']
        track = report['tracking_performance']
        comp = report['performance_comparison']
        camera_iou = report.get('camera_iou_performance', {})

        # Header
        f.write("═" * 100 + "\n")
        f.write("                        TRACKING AND DETECTION EVALUATION REPORT\n")
        f.write("═" * 100 + "\n\n")

        # Executive Summary
        f.write("📋 EXECUTIVE SUMMARY\n")
        f.write("─" * 50 + "\n")
        f.write(f"Frames: {summary['frames_evaluated']:,} | Distance Threshold: {summary['distance_threshold_m']:.1f}m | "
                f"IoU Threshold: {summary['iou_threshold']:.2f} | Avg GT/Frame: {det['avg_ground_truth_per_frame']:.1f}\n\n")

        # Performance Summary Table
        f.write("🎯 PERFORMANCE SUMMARY\n")
        f.write("─" * 50 + "\n")
        f.write(f"{'Metric':<25} {'Detection':<12} {'Tracking':<12} {'Improvement':<12}\n")
        f.write("─" * 80 + "\n")

        metrics_to_show = ['precision', 'recall', 'f1_score', 'det_a']
        for metric in metrics_to_show:
            f.write(f"{metric.replace('_', ' ').title():<25} {det[metric]:<12.4f} {track[metric]:<12.4f} "
                   f"{comp[metric]['improvement_percent']:>+10.1f}%\n")

        # Distance metrics
        det_dist = "N/A" if det['mean_euclidean_distance'] == float('inf') else f"{det['mean_euclidean_distance']:.2f}"
        track_dist = "N/A" if track['mean_euclidean_distance'] == float('inf') else f"{track['mean_euclidean_distance']:.2f}"
        dist_improvement = "N/A" if comp['mean_euclidean_distance_m']['improvement_percent'] == float('inf') else f"{comp['mean_euclidean_distance_m']['improvement_percent']:+10.1f}%"
        f.write(f"{'Mean Distance (m)':<25} {det_dist:<12} {track_dist:<12} {dist_improvement:>11}\n")

        # Camera IoU
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

        # Performance Analysis
        f.write("📊 PERFORMANCE ANALYSIS\n")
        f.write("─" * 50 + "\n")
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
        f.write("\n")
        f.write("═" * 100 + "\n")

    def _print_report_content(self, report: Dict[str, Any]) -> None:
        """Print formatted report content to console."""
        summary = report['evaluation_summary']
        det = report['detection_performance']
        track = report['tracking_performance']
        comp = report['performance_comparison']

        print(f"\n📊 EVALUATION SUMMARY:")
        print(f"   • Frames Evaluated: {summary['frames_evaluated']}")
        print(f"   • Distance Threshold: {summary['distance_threshold_m']:.1f}m")
        print(f"   • IoU Threshold: {summary['iou_threshold']:.2f}")

        print(f"\n🎯 DETECTION PERFORMANCE:")
        print(f"   • Precision: {det['precision']:.3f} | Recall: {det['recall']:.3f} | F1: {det['f1_score']:.3f}")
        print(f"   • DetA: {det['det_a']:.3f} | Mean Distance: {det['mean_euclidean_distance']:.2f}m")
        print(f"   • TP/FP/FN: {det['true_positives']}/{det['false_positives']}/{det['false_negatives']}")

        print(f"\n🔄 TRACKING PERFORMANCE:")
        print(f"   • Precision: {track['precision']:.3f} | Recall: {track['recall']:.3f} | F1: {track['f1_score']:.3f}")
        print(f"   • DetA: {track['det_a']:.3f} | Mean Distance: {track['mean_euclidean_distance']:.2f}m")
        print(f"   • TP/FP/FN: {track['true_positives']}/{track['false_positives']}/{track['false_negatives']}")

        print(f"\n📈 IMPROVEMENTS (Tracking vs Detection):")
        print(f"   • Precision: {comp['precision']['improvement_percent']:+.1f}% | "
              f"Recall: {comp['recall']['improvement_percent']:+.1f}% | "
              f"F1: {comp['f1_score']['improvement_percent']:+.1f}%")


def evaluate_tracking_sequence(predictions_csv: str, ground_truth_csv: str,
                               tracking_csv: str, output_dir: str,
                               distance_threshold: float = 5.0,
                               iou_threshold: float = 0.3,
                               max_frames: Optional[int] = None,
                               skip_initial_frames: int = 3,
                               max_frame_gap_time: float = 5.0,
                               use_camera_fov_filter: bool = True) -> Tuple[Path, Dict[str, Any]]:
    """Evaluate complete tracking sequence and save both JSON and text reports.
    
    Args:
        predictions_csv: Path to predictions CSV file
        ground_truth_csv: Path to ground truth CSV file  
        tracking_csv: Path to tracking results CSV file
        output_dir: Directory to save evaluation results
        distance_threshold: Distance threshold for valid associations
        iou_threshold: IoU threshold for evaluation
        max_frames: Maximum number of frames to evaluate
        skip_initial_frames: Number of initial frames to skip from metrics
        max_frame_gap_time: Maximum time gap (seconds) before tracks are cleared
        use_camera_fov_filter: Filter detections/tracks to camera FOV during evaluation
    """
    evaluator = TrackingDetectionEvaluator(
        distance_threshold=distance_threshold,
        iou_threshold=iou_threshold,
        use_camera_fov_filter=use_camera_fov_filter,
    )

    # Load data
    predictions_df = pd.read_csv(predictions_csv)
    ground_truth_df = pd.read_csv(ground_truth_csv, sep='\t|,', engine='python')
    tracking_df = pd.read_csv(tracking_csv)

    # Get frames with ground truth
    gt_frames = set(ground_truth_df['numSample'].unique())
    frame_ids = sorted(gt_frames)

    # Skip initial frames
    if skip_initial_frames > 0 and len(frame_ids) > skip_initial_frames:
        skipped_frames = frame_ids[:skip_initial_frames]
        frame_ids = frame_ids[skip_initial_frames:]
        print(f"Skipping first {skip_initial_frames} frames from metrics: {skipped_frames}")

    if max_frames:
        frame_ids = frame_ids[:max_frames]

    # Identify frames to skip after time gaps
    frames_to_skip = set()
    if 'time_gap' in tracking_df.columns:
        # Find frames that follow large time gaps
        large_gap_frames = tracking_df[tracking_df['time_gap'] > max_frame_gap_time]['sample_id'].unique()
        
        # For each frame after a large gap, skip the next min_hits frames
        for gap_frame in large_gap_frames:
            # Find the position of this frame in our evaluation frame list
            try:
                gap_idx = frame_ids.index(gap_frame)
                # Skip the next min_hits frames after the gap
                for i in range(skip_initial_frames):
                    if gap_idx + i < len(frame_ids):
                        frames_to_skip.add(frame_ids[gap_idx + i])
            except ValueError:
                # Gap frame not in our evaluation list, skip
                continue
        
        if frames_to_skip:
            print(f"Skipping {len(frames_to_skip)} frames after large time gaps (>{max_frame_gap_time:.1f}s): {sorted(frames_to_skip)}")
    else:
        print(" the column 'time_gap' in tracking_df.columns is missing, check the tracking csv output")
    # Evaluate each frame (excluding skipped frames)
    evaluated_frames = []
    for frame_id in frame_ids:
        if frame_id not in frames_to_skip:
            pred_frame = predictions_df[predictions_df['sample_id'] == frame_id]
            gt_frame = ground_truth_df[ground_truth_df['numSample'] == frame_id]
            track_frame = tracking_df[tracking_df['sample_id'] == frame_id]
            evaluator.evaluate_frame(pred_frame, gt_frame, track_frame, frame_id)
            evaluated_frames.append(frame_id)

    print(f"Evaluated {len(evaluated_frames)} frames (skipped {len(frame_ids) - len(evaluated_frames)} frames total)")

    # Save reports
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save JSON report for aggregation
    evaluator.save_json_report(str(output_path / 'evaluation_metrics.json'))

    # Save text report for human reading
    evaluator.save_text_report(str(output_path / 'evaluation_summary.txt'))

    return output_path, evaluator.generate_comprehensive_report()


if __name__ == "__main__":
    # Example usage
    predictions_csv = "predictions.csv"
    ground_truth_csv = "ground_truth.csv"
    tracking_csv = "tracking_results.csv"
    output_dir = "evaluation_results"

    output_path, report = evaluate_tracking_sequence(
        predictions_csv=predictions_csv,
        ground_truth_csv=ground_truth_csv,
        tracking_csv=tracking_csv,
        output_dir=output_dir,
        distance_threshold=5.0,
        iou_threshold=0.3
    )

    print(f"Reports saved to: {output_path}")