# offline_tracking.py
import copy

import numpy as np
import pandas as pd
from typing import List, Optional

from tqdm import tqdm

from radar_tracking import TrackletManager, Detection, Track
from visualizations.track_viz import (
    visualize_counts_vs_tracks_per_frame,
    visualize_frame_radar_azimuth,
    visualize_tracklet_lifetime_histogram,
    visualize_avg_confidence_over_time,
    visualize_all_frames_3d_overview,
    visualize_tracking_temporal_evolution, visualize_timing_analysis, visualize_tracking_during_gaps
)
from visualizations.tracking_visualization import create_tracking_video
from visualizations.visualize_timing import plot_timing_analysis, plot_detailed_timing_analysis
from utils.tracking_metrics import TrackingEvaluator

def setup_tracking_system():
    """
    Initialize and return a TrackletManager and its config dict.
    """
    tracker_config = {
        'max_age': 3,
        'min_hits': 3,
        'iou_threshold': 5.0,
        'dt': 0.1,

        # Confidence-based parameters
        'min_confidence_init': 0.7,
        'min_confidence_assoc': 0.4,
        'confidence_weight': 0.3,
        'association_strategy': 'confidence_weighted',

        # Range culling parameters - configured for your radar
        'enable_range_culling': True,
        'max_range': 103.0,  # Your radar's max range
        'min_azimuth_deg': -90.0,  # Your radar's azimuth limits
        'max_azimuth_deg': 90.0,
        'range_buffer': 10.0,  # 10m buffer to avoid killing tracks just outside
        'azimuth_buffer_deg': 5.0  # 5° buffer for azimuth
    }
    manager = TrackletManager(tracker_config=tracker_config)
    return manager, tracker_config


def load_predictions(pred_csv_path: str) -> pd.DataFrame:
    """
    Load all_predictions.csv into a DataFrame and sort by sample_id/frame order.
    Expects columns: detection_id, confidence, x1, y1, …, range_m, azimuth_deg, sample_id
    """
    df = pd.read_csv(pred_csv_path, sep=r'\s+|,', engine='python')
    # Ensure it's sorted by sample_id (i.e., the frame order)
    df = df.sort_values(by='sample_id').reset_index(drop=True)
    return df


def load_labels(label_csv_path: str) -> pd.DataFrame:
    """
    Load labels.csv into a DataFrame. Expects columns including: numSample, radar_R_m, radar_A_deg, etc.
    We'll use numSample as sample_id, and radar_R_m, radar_A_deg as ground‐truth.
    """
    df = pd.read_csv(label_csv_path, sep=r'\s+|,', engine='python')
    return df


def build_detections_for_frame(preds_df: pd.DataFrame, frame_id: int,
                              timestamp_us: float) -> List[Detection]:
    """
    Given the full predictions DataFrame and a specific frame_id (sample_id),
    return a list of Detection objects for that frame.
    """
    # Filter all rows whose sample_id == frame_id
    sub = preds_df[preds_df['sample_id'] == frame_id]
    dets: list[Detection] = []

    # Convert timestamp from microseconds to seconds
    timestamp_s = timestamp_us / 1_000_000

    for _, row in sub.iterrows():
        det = Detection(
            range_m=float(row['range_m']),
            azimuth_rad=np.radians(float(row['azimuth_deg'])),
            confidence=float(row['confidence']),
            timestamp=timestamp_s,  # Use actual timestamp
            frame_id=frame_id
        )
        det._detection_id = int(row['detection_id'])
        dets.append(det)

    return dets


def build_ground_truth_for_frame(
        labels_df: pd.DataFrame,
        frame_id: int
) -> list[Detection]:
    """
    Convert ground‐truth (labels.csv) for that frame_id into Detection objects.
    Assumes columns: numSample (→ sample_id), radar_R_m, radar_A_deg.
    """
    sub = labels_df[labels_df['numSample'] == frame_id]
    gt_list: list[Detection] = []
    for _, row in sub.iterrows():
        r = float(row['radar_R_m'])
        az_deg = float(row['radar_A_deg'])
        gt = Detection(
            range_m=r,
            azimuth_rad=np.radians(az_deg),
            confidence=1.0,  # ground truth = perfect confidence
            timestamp=float(frame_id)
        )
        gt_list.append(gt)
    return gt_list


def setup_output_directories(output_dir: str) -> dict:
    """
    Create standardized output directory structure.

    Args:
        output_dir: Root output directory path

    Returns:
        Dictionary with paths to different output subdirectories
    """
    output_path = Path(output_dir)

    # Define subdirectory structure
    subdirs = {
        'root': output_path,
        'tracks': output_path / 'tracks',
        'visualizations': output_path / 'visualizations',
        'frame_images': output_path / 'visualizations' / 'frames',
        'summary_plots': output_path / 'visualizations' / 'summary',
        'logs': output_path / 'logs',
        'config': output_path / 'config'
    }

    # Create all directories
    for subdir_path in subdirs.values():
        subdir_path.mkdir(parents=True, exist_ok=True)

    return subdirs


def save_config_info(config: dict, tracker_config: dict, output_paths: dict):
    """Save configuration information to the config directory."""
    config_info = {
        'tracker_config': tracker_config,
        'processing_config': config,
        'output_structure': {k: str(v) for k, v in output_paths.items()}
    }

    config_file = output_paths['config'] / 'tracking_config.json'
    with open(config_file, 'w') as f:
        json.dump(config_info, f, indent=2)

    print(f"Configuration saved to: {config_file}")


def offline_tracking(
        preds_csv: str,
        labels_csv: str,
        output_dir: str = "tracking_output",
        tracker_config: Optional[dict] = None,
        create_video: bool = True,
        max_video_samples: Optional[int] = 100
):
    """
    Main offline‐tracking function with visualization and evaluation with timestamp support:
      1) Reads predictions + labels
      2) Iterates over each unique frame_id in ascending order
      3) Builds Detection objects, calls tracker.update(...)
      4) Writes out one row per active track each frame into tracking.csv
      5) Saves comprehensive visualizations (PNG files)

    Args:
        preds_csv: Path to predictions CSV file
        labels_csv: Path to labels CSV file
        output_dir: Root directory for all outputs
        tracker_config: Optional tracker configuration override
        create_video: Whether to create tracking visualization video
        max_video_samples: Maximum samples to include in video (for performance)
    """

    # Setup output directory structure
    output_paths = setup_output_directories(output_dir)
    print(f"Output directory structure created at: {output_paths['root']}")

    # 1) Load all_predictions.csv and labels.csv
    preds_df = load_predictions(preds_csv)
    labels_df = load_labels(labels_csv)

    # Get timestamps from labels
    timestamps_df = labels_df[['numSample', 'timestamp_us']].drop_duplicates()
    timestamps_dict = dict(zip(timestamps_df['numSample'],
                             timestamps_df['timestamp_us']))

    # 2) Unique frame IDs (sample_id) in sorted order
    all_frames = sorted(preds_df['sample_id'].unique().tolist())

    # 3) Initialize tracker
    if tracker_config is not None:
        # Override default config
        manager = TrackletManager(tracker_config=tracker_config)
        config = tracker_config
    else:
        manager, config = setup_tracking_system()

    # Save configuration
    save_config_info({'preds_csv': preds_csv, 'labels_csv': labels_csv},
                     config, output_paths)

    # For storing the CSV rows
    tracking_rows = []

    # For visualization data storage
    det_counts: List[int] = []
    track_counts: List[int] = []
    avg_confidence_per_frame: List[float] = []

    # Store all data for comprehensive visualizations
    all_detections: List[List[Detection]] = []
    all_ground_truth: List[List[Detection]] = []
    all_tracks: List[List[Track]] = []

    # Add timing analysis storage
    frame_times = []
    time_gaps = []

    # 4) Loop over frames
    prev_timestamp = None
    for frame_id in tqdm(all_frames,
                         total=len(all_frames),
                         desc="Processing Frames",
                         unit="frame",
                         colour="green",
                         dynamic_ncols=True):

        # Get timestamp for this frame
        timestamp_us = timestamps_dict.get(frame_id, frame_id * 1_000_000)
        timestamp_s = timestamp_us / 1_000_000

        frame_times.append((frame_id, timestamp_s))

        # Calculate time gap
        if prev_timestamp is not None:
            gap = timestamp_s - prev_timestamp
            time_gaps.append(gap)
            dt = gap
        else:
            # First frame - no gap to calculate
            dt = config.get('base_dt', 0.1)

        # a) Build detections for this frame
        detections = build_detections_for_frame(preds_df, frame_id, timestamp_us)
        det_counts.append(len(detections))

        # b) Build ground truth for this frame
        ground_truth = build_ground_truth_for_frame(labels_df, frame_id)

        # c) Update tracker
        active_tracks = manager.update(detections, ground_truth, current_time=timestamp_s)

        # Store data for comprehensive visualizations
        all_detections.append(detections)
        all_ground_truth.append(ground_truth)
        all_tracks.append(copy.deepcopy(active_tracks))

        # Count how many tracks are currently "confirmed" or "tentative"
        track_counts.append(len(active_tracks))

        # Compute average confidence among active tracks this frame
        if active_tracks:
            confidences = [t.last_detection.confidence for t in active_tracks if t.last_detection is not None]
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
        else:
            avg_conf = 0.0
        avg_confidence_per_frame.append(avg_conf)

        # d) Record one row per active track
        for track in active_tracks:
            # Last detection from this track
            last_det = track.last_detection
            if last_det is None:
                continue

            rid = getattr(last_det, '_detection_id', -1)
            r = float(last_det.range_m)
            az = float(np.degrees(last_det.azimuth_rad))
            conf = float(last_det.confidence)

            row = {
                'sample_id': int(frame_id),
                'frame_id': int(frame_id),
                'track_id': int(track.id),
                'detection_id': int(rid),
                'confidence': conf,
                'range_m': r,
                'azimuth_deg': az,
                'track_age': int(track.age),
                'hits': int(track.hits),
                'track_state': track.state.name if hasattr(track.state, 'name') else str(track.state),
                'timestamp': timestamp_s,
                'time_gap': gap if prev_timestamp else 0.0,
                # x1..y4 left as NaN placeholders; replace if you have pixel‐corner data
                'x1': np.nan, 'y1': np.nan,
                'x2': np.nan, 'y2': np.nan,
                'x3': np.nan, 'y3': np.nan,
                'x4': np.nan, 'y4': np.nan
            }
            tracking_rows.append(row)

        prev_timestamp = timestamp_s

        # e) Visualize this frame (individual frame visualization)
        visualize_frame_radar_azimuth(
            frame_id=frame_id,
            detections=detections,
            ground_truth=ground_truth,
            active_tracks=active_tracks,
            output_dir=str(output_paths['frame_images']),
            radar_config=config
        )

    # 5) Build DataFrame and write tracking.csv
    track_df = pd.DataFrame(tracking_rows)
    cols = [
        'sample_id', 'frame_id', 'timestamp', 'time_gap', 'track_id',
        'detection_id', 'confidence', 'range_m', 'azimuth_deg',
        'track_age', 'hits', 'track_state',
        'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'
    ]
    track_df = track_df[cols]

    # Save tracking results
    tracking_csv_path = output_paths['tracks'] / 'tracking.csv'
    track_df.to_csv(tracking_csv_path, index=False)

    # 6) Generate all visualizations using the visualization functions

    # a) Network Output per Frame vs. Confirmed Tracks per Frame
    visualize_counts_vs_tracks_per_frame(
        all_frames=all_frames,
        det_counts=det_counts,
        track_counts=track_counts,
        output_dir=str(output_paths['summary_plots'])
    )

    # b) Histogram of Tracklet Lifetimes
    visualize_tracklet_lifetime_histogram(
        manager=manager,
        output_dir=str(output_paths['summary_plots'])
    )

    # c) Average Confidence of Active Tracks Over Time
    visualize_avg_confidence_over_time(
        all_frames=all_frames,
        all_tracks=all_tracks,
        frame_times=frame_times,
        avg_confidence_per_frame=avg_confidence_per_frame,
        output_dir=str(output_paths['summary_plots'])
    )

    # d) Comprehensive overview showing all frames data
    visualize_all_frames_3d_overview(
        all_detections=all_detections,
        all_ground_truth=all_ground_truth,
        all_tracks=all_tracks,
        all_frames=all_frames,
        frame_times=frame_times,
        output_dir=str(output_paths['summary_plots'])
    )

    # e) Temporal evolution visualization
    visualize_tracking_temporal_evolution(
        all_detections=all_detections,
        all_ground_truth=all_ground_truth,
        all_tracks=all_tracks,
        all_frames=all_frames,
        frame_times=frame_times,
        output_dir=str(output_paths['summary_plots'])
    )

    # Visualize timing analysis
    visualize_timing_analysis(
        frame_times=frame_times,
        time_gaps=time_gaps,
        output_dir=str(output_paths['summary_plots'])
    )

    # Visualize tracking during gaps
    gap_threshold = np.percentile(time_gaps, 90) if time_gaps else 0.5  # Top 10% gaps
    visualize_tracking_during_gaps(
        all_tracks=all_tracks,
        frame_times=frame_times,
        gap_threshold=gap_threshold,
        output_dir=str(output_paths['summary_plots']),
        radar_config=config
    )

    # Visualize timing analysis
    plot_timing_analysis(labels_csv, output_path=Path(output_paths['summary_plots']) / Path("timing_analysis.png"))
    plot_detailed_timing_analysis(labels_csv,
                                  output_path=Path(output_paths['summary_plots']) / Path("detailed_timing_analysis.png"))

    # Save tracking summary
    summary_file = output_paths['logs'] / 'tracking_summary.txt'
    with open(summary_file, 'w') as f:
        # Redirect print output to file
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        manager.print_summary()

        # Add timing statistics
        if time_gaps:
            print("\n" + "=" * 50)
            print("TIMING STATISTICS")
            print("=" * 50)
            print(f"Total duration: {frame_times[-1][1] - frame_times[0][1]:.2f} seconds")
            print(f"Number of frames: {len(frame_times)}")
            print(f"Average time gap: {np.mean(time_gaps):.3f} seconds")
            print(f"Median time gap: {np.median(time_gaps):.3f} seconds")
            print(f"Max time gap: {np.max(time_gaps):.3f} seconds")
            print(f"Min time gap: {np.min(time_gaps):.3f} seconds")
            print(f"Gaps > 0.5s: {sum(1 for g in time_gaps if g > 0.5)}")
            print("=" * 50)

        sys.stdout = original_stdout

    # ===== ENHANCED EVALUATION METRICS =====
    print("\nRunning comprehensive tracking evaluation...")

    evaluator = TrackingEvaluator(distance_threshold=config.get('iou_threshold', 5.0))

    # Evaluate frame by frame
    for frame_id in all_frames:
        # Filter data for this frame
        frame_preds = preds_df[preds_df['sample_id'] == frame_id]
        frame_labels = labels_df[labels_df['numSample'] == frame_id]
        frame_tracks = track_df[track_df['sample_id'] == frame_id] if not track_df.empty else pd.DataFrame()

        if not frame_labels.empty:  # Only evaluate frames with ground truth
            evaluator.evaluate_frame(frame_preds, frame_labels, frame_tracks, frame_id)

    # Generate and save comprehensive evaluation report
    evaluation_report = evaluator.generate_comprehensive_report()
    eval_report_path = output_paths['logs'] / 'comprehensive_evaluation.json'
    evaluator.save_report(str(eval_report_path))

    # Save summary metrics to text file
    summary_path = output_paths['logs'] / 'evaluation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("COMPREHENSIVE TRACKING EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        if 'summary' in evaluation_report:
            summary = evaluation_report['summary']
            f.write(f"Frames Evaluated: {summary.get('frames_evaluated', 0)}\n")
            f.write(f"MOTA (Multiple Object Tracking Accuracy): {summary.get('mota', 0):.3f}\n")
            f.write(f"MOTP (Multiple Object Tracking Precision): {summary.get('motp', 0):.3f} meters\n\n")

        if 'detection_performance' in evaluation_report:
            det_perf = evaluation_report['detection_performance']
            f.write("DETECTION PERFORMANCE:\n")
            f.write(f"  Overall Precision: {det_perf.get('overall_precision', 0):.3f}\n")
            f.write(f"  Overall Recall: {det_perf.get('overall_recall', 0):.3f}\n")
            f.write(f"  Overall F1-Score: {det_perf.get('overall_f1_score', 0):.3f}\n\n")

        if 'tracking_performance' in evaluation_report:
            track_perf = evaluation_report['tracking_performance']
            f.write("TRACKING PERFORMANCE:\n")
            f.write(f"  Overall Precision: {track_perf.get('overall_precision', 0):.3f}\n")
            f.write(f"  Overall Recall: {track_perf.get('overall_recall', 0):.3f}\n")
            f.write(f"  Overall F1-Score: {track_perf.get('overall_f1_score', 0):.3f}\n\n")

        if 'distance_analysis' in evaluation_report:
            dist_analysis = evaluation_report['distance_analysis']
            f.write("DISTANCE ANALYSIS:\n")
            f.write(f"  Mean Distance Error: {dist_analysis.get('overall_mean_distance', 0):.2f} meters\n")
            f.write(f"  Distance Std Dev: {dist_analysis.get('overall_std_distance', 0):.2f} meters\n")
            f.write(f"  Frames with Valid Associations: {dist_analysis.get('frames_with_valid_associations', 0)}\n\n")

    # ===== ENHANCED VISUALIZATION AND VIDEO CREATION =====
    if create_video:
        print("\nCreating tracking visualization video...")

        # Setup data directory path
        data_dir = Path(labels_csv).parent

        try:
            video_path = create_tracking_video(
                data_dir=data_dir,
                output_dir=output_paths['visualizations'],
                labels_csv=labels_csv,
                predictions_csv=preds_csv,
                tracking_csv=str(tracking_csv_path),
                max_samples=max_video_samples
            )
            print(f"Tracking video created: {video_path}")

        except Exception as e:
            print(f"Video creation failed: {e}")
            print("Continuing without video...")

    # Print completion message
    print(f"\nOffline tracking completed. Files written to: {output_paths['root']}")
    print(f"  • Tracking results: {tracking_csv_path}")
    print(f"  • Visualizations: {output_paths['visualizations']}")
    print(f"  • Evaluation report: {eval_report_path}")
    print(f"  • Configuration: {output_paths['config']}")
    print(f"  • Logs: {output_paths['logs']}")
    if create_video:
        print(f"  • Tracking video: {output_paths['visualizations']}")
    print(f"  • Tracker configuration: {config}")


if __name__ == "__main__":
    from pathlib import Path
    import json

    path_config_default = Path('ADCProcessing/data_config.json')
    config = json.load(open(path_config_default))
    record = config['target_value']
    root_folder = Path(config['Data_Dir'], 'RadIal_Data', record)
    labels_csv = Path(root_folder, 'labels.csv')

    path_file_par = Path(__file__).parent

    # Example with custom tracker config for confidence-based tracking
    custom_tracker_config = {
        'max_age': 3,
        'min_hits': 3,
        'iou_threshold': 60.0,
        'base_dt': 0.2,  # 200ms base time step
        'max_dt_gap': 1.0,  # Trigger multi-step prediction for gaps > 1.0s

        # Confidence-based parameters
        'min_confidence_init': 0.5,
        'min_confidence_assoc': 0.2,
        'confidence_weight': 0.3,
        'association_strategy': 'confidence_weighted', #  "distance_only", "confidence_weighted", "confidence_gated", "hybrid_score"

        # Range culling parameters - configured for your radar
        'enable_range_culling': True,
        'max_range': 103.0,  # Your radar's max range
        'min_azimuth_deg': -90.0,  # Your radar's azimuth limits
        'max_azimuth_deg': 90.0,
        'range_buffer': 10.0,  # 10m buffer to avoid killing tracks just outside
        'azimuth_buffer_deg': 5.0  # 5° buffer for azimuth
    }

    args = {
        'preds_csv': str(path_file_par / Path('plots/predictions/all_predictions.csv')),
        'labels_csv': str(labels_csv),
        'output_dir': str(path_file_par / Path('plots/tracking_output')),
        'tracker_config': custom_tracker_config,
        'create_video': True,
        'max_video_samples': 50  # Limit video to first 50 samples for performance
    }

    offline_tracking(**args)