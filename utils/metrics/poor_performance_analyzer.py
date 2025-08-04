"""
Poor Performance Frame Analyzer
Identifies and analyzes frames where tracking performs poorly compared to raw predictions.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class PoorPerformanceFrame:
    """Container for poor performance frame data."""
    frame_id: int
    metric_name: str
    raw_prediction_value: float
    tracking_value: float
    degradation: float  # Percentage degradation
    detection_count: int
    track_count: int
    ground_truth_count: int


class PoorPerformanceAnalyzer:
    """Analyzes frames with poor tracking performance compared to raw predictions."""

    def __init__(self, metric_name: str = 'det_a', top_k: int = 20):
        """
        Initialize analyzer.

        Args:
            metric_name: Metric to analyze ('det_a', 'precision', 'recall', 'f1_score', 'camera_iou_mean')
            top_k: Number of worst frames to identify
        """
        self.metric_name = metric_name
        self.top_k = top_k
        self.distance_metrics = ['ncle']
        self.primary_metrics = ['hota', 'mota', 'det_a', 'precision', 'tracking_iou', 'camera_iou_mean', 'ncle', 'tp',
                                'fp', 'precision_ratio']

    def analyze_poor_performance_frames(self, evaluation_details_path: Path, output_dir: Path) -> List[
        PoorPerformanceFrame]:
        with open(evaluation_details_path, 'r') as f:
            frame_data = json.load(f)

        frames_info = frame_data.get('frame_by_frame_results', [])
        if not frames_info:
            return []

        poor_frames = []
        for frame_info in frames_info:
            frame_id = int(frame_info['frame_id'])  # Convert to int

            # Handle camera IoU separately
            if self.metric_name == 'camera_iou_mean':
                camera_iou_data = frame_info.get('camera_iou_results', {})
                raw_list = camera_iou_data.get('detection_vs_labels_ious', [])
                raw_value = np.mean(raw_list) if raw_list else 0.0
                track_list = camera_iou_data.get('tracking_vs_labels_ious', [])
                tracking_value = np.mean(track_list) if track_list else 0.0
            else:
                # Get values from metrics dictionary with prefixed keys
                metrics = frame_info.get('metrics', {})
                if self.metric_name == 'precision_ratio':
                    # Calculate from TP/FP values
                    det_tp = metrics.get('detection_tp', 0)
                    det_fp = metrics.get('detection_fp', 0)
                    track_tp = metrics.get('tracking_tp', 0)
                    track_fp = metrics.get('tracking_fp', 0)
                    raw_value = det_tp / (det_tp + det_fp) if (det_tp + det_fp) > 0 else 0.0
                    tracking_value = track_tp / (track_tp + track_fp) if (track_tp + track_fp) > 0 else 0.0
                else:
                    raw_value = metrics.get(f'detection_{self.metric_name}', 0)
                    tracking_value = metrics.get(f'tracking_{self.metric_name}', 0)

            # Skip invalid frames
            if raw_value == 0 or raw_value == float('inf'):
                continue

            # Calculate degradation
            degradation = ((raw_value - tracking_value) / raw_value) * 100

            # Get counts with fallback values
            poor_frame = PoorPerformanceFrame(
                frame_id=frame_id,
                metric_name=self.metric_name,
                raw_prediction_value=raw_value,
                tracking_value=tracking_value,
                degradation=degradation,
                detection_count=frame_info.get('num_predictions', 0),
                track_count=frame_info.get('num_tracks', 0),
                ground_truth_count=frame_info.get('num_ground_truth_total', 0)  # Updated key
            )
            poor_frames.append(poor_frame)

        # Sort by degradation (worst first)
        poor_frames.sort(key=lambda x: x.degradation, reverse=True)

        # Take top K worst frames
        worst_frames = poor_frames[:self.top_k]

        # Save analysis results
        self._save_analysis_results(worst_frames, poor_frames, output_dir)

        return worst_frames

    def _save_analysis_results(self, worst_frames: List[PoorPerformanceFrame],
                               all_frames: List[PoorPerformanceFrame],
                               output_dir: Path):
        """Save analysis results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save worst frames list to text file
        txt_file = output_dir / f'worst_{self.metric_name}_frames.txt'
        with open(txt_file, 'w') as f:
            f.write(f"WORST PERFORMING FRAMES FOR {self.metric_name.upper()}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Analyzed {len(all_frames)} frames, showing top {len(worst_frames)} worst\n\n")

            f.write(f"{'Frame ID':<10} {'Raw Pred':<12} {'Tracking':<12} {'Degradation':<12} "
                    f"{'Dets':<8} {'Tracks':<8} {'GT':<8}\n")
            f.write("-" * 80 + "\n")

            for frame in worst_frames:
                raw_str = f"{frame.raw_prediction_value:.4f}"
                track_str = f"{frame.tracking_value:.4f}" if frame.tracking_value != float('inf') else "INF"
                f.write(f"{frame.frame_id:<10} {raw_str:<12} {track_str:<12} "
                        f"{frame.degradation:>10.1f}% {frame.detection_count:<8} "
                        f"{frame.track_count:<8} {frame.ground_truth_count:<8}\n")

        # Save frame IDs only for easy parsing
        frame_ids_file = output_dir / f'worst_{self.metric_name}_frame_ids.txt'
        with open(frame_ids_file, 'w') as f:
            for frame in worst_frames:
                f.write(f"{frame.frame_id}\n")

        # Save detailed JSON report
        json_file = output_dir / f'worst_{self.metric_name}_analysis.json'
        degradation_values = [f.degradation for f in all_frames]
        degradation_values = degradation_values if degradation_values else 0.0
        analysis_data = {
            'metric_analyzed': self.metric_name,
            'total_frames_analyzed': len(all_frames),
            'worst_frames_count': len(worst_frames),
            'worst_frames': [
                {
                    'frame_id': f.frame_id,
                    'raw_prediction_value': f.raw_prediction_value,
                    'tracking_value': f.tracking_value,
                    'degradation_percent': f.degradation,
                    'detection_count': f.detection_count,
                    'track_count': f.track_count,
                    'ground_truth_count': f.ground_truth_count
                }
                for f in worst_frames
            ],

            'degradation_statistics': {
                'mean': np.mean(degradation_values),
                'std': np.std(degradation_values),
                'median': np.median(degradation_values),
                'max': np.max(degradation_values),
                'min': np.min(degradation_values)
            }
        }

        with open(json_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)

        # Create visualization
        self._create_degradation_visualization(all_frames, worst_frames, output_dir)

    def _create_degradation_visualization(self, all_frames: List[PoorPerformanceFrame],
                                          worst_frames: List[PoorPerformanceFrame],
                                          output_dir: Path):
        """Create visualization of performance degradation."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.metric_name.upper()} Performance Degradation Analysis', fontsize=16)

        # 1. Histogram of degradation
        ax = axes[0, 0]
        degradations = [f.degradation for f in all_frames]
        if not np.isnan(degradations).all() and len(degradations) > 0:
            ax.hist(degradations, bins=30, alpha=0.7, color='blue', edgecolor='black')
        else:
            return
        ax.axvline(np.mean(degradations), color='red', linestyle='--', label=f'Mean: {np.mean(degradations):.1f}%')
        ax.set_xlabel('Degradation (%)')
        ax.set_ylabel('Number of Frames')
        ax.set_title('Distribution of Performance Degradation')
        ax.legend()

        # 2. Scatter plot: Raw vs Tracking performance
        ax = axes[0, 1]
        raw_values = [f.raw_prediction_value for f in all_frames if f.tracking_value != float('inf')]
        track_values = [f.tracking_value for f in all_frames if f.tracking_value != float('inf')]

        if raw_values and track_values:
            ax.scatter(raw_values, track_values, alpha=0.5, s=20)

            # Highlight worst frames
            worst_raw = [f.raw_prediction_value for f in worst_frames if f.tracking_value != float('inf')]
            worst_track = [f.tracking_value for f in worst_frames if f.tracking_value != float('inf')]
            ax.scatter(worst_raw, worst_track, color='red', s=50, label='Worst frames')

            # Add diagonal line
            min_val = min(min(raw_values), min(track_values))
            max_val = max(max(raw_values), max(track_values))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect tracking')

            ax.set_xlabel(f'Raw Prediction {self.metric_name}')
            ax.set_ylabel(f'Tracking {self.metric_name}')
            ax.set_title('Raw Prediction vs Tracking Performance')
            ax.legend()

        # 3. Top worst frames bar chart
        ax = axes[1, 0]
        frame_ids = [f"Frame {f.frame_id}" for f in worst_frames[:10]]
        degradations = [f.degradation for f in worst_frames[:10]]

        bars = ax.bar(range(len(frame_ids)), degradations, color='red', alpha=0.7)
        ax.set_xticks(range(len(frame_ids)))
        ax.set_xticklabels(frame_ids, rotation=45, ha='right')
        ax.set_ylabel('Degradation (%)')
        ax.set_title('Top 10 Worst Performing Frames')

        # Add value labels on bars
        for bar, val in zip(bars, degradations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

        # 4. Degradation vs Frame characteristics
        ax = axes[1, 1]
        det_counts = [f.detection_count for f in all_frames]
        degradations = [f.degradation for f in all_frames]

        scatter = ax.scatter(det_counts, degradations,
                             c=[f.ground_truth_count for f in all_frames],
                             cmap='viridis', alpha=0.6, s=30)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Ground Truth Count')

        ax.set_xlabel('Detection Count')
        ax.set_ylabel('Degradation (%)')
        ax.set_title('Degradation vs Detection Count')

        plt.tight_layout()
        plt.savefig(output_dir / f'{self.metric_name}_degradation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


def analyze_poor_performance_for_configuration(dataset_root: Path,
                                               config_name: str,
                                               metric_name: str = 'det_a',
                                               top_k: int = 20) -> Optional[List[PoorPerformanceFrame]]:
    """
    Analyze poor performance frames for a specific tracking configuration.

    Args:
        dataset_root: Root directory of the dataset
        config_name: Name of the tracking configuration
        metric_name: Metric to analyze
        top_k: Number of worst frames to identify

    Returns:
        List of PoorPerformanceFrame objects or None if analysis fails
    """
    # Construct paths
    eval_details_path = (dataset_root / 'plots' / 'tracking_output' /
                         config_name / 'logs' / 'evaluation_metrics_frame_details.json')

    if not eval_details_path.exists():
        print(f"Evaluation details not found: {eval_details_path}")
        return None

    # Create output directory
    output_dir = (dataset_root / 'plots' / 'tracking_output' /
                  config_name / 'analysis' / 'poor_performance')

    # Run analysis
    analyzer = PoorPerformanceAnalyzer(metric_name=metric_name, top_k=top_k)
    worst_frames = analyzer.analyze_poor_performance_frames(eval_details_path, output_dir)

    return worst_frames