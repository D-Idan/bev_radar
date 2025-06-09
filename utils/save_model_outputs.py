# save_model_outputs.py
import pandas as pd
import numpy as np
from pathlib import Path
from utils.util import process_predictions_FFT
import json
import pickle
from typing import Dict, List
from radar_tracking import Detection


def extract_model_predictions(model_outputs, encoder, confidence_threshold=0.2):
    """Extract predictions from model outputs (original function, unchanged)"""
    pred_obj = model_outputs['Detection'].detach().cpu().numpy().copy()[0]
    pred_obj = encoder.decode(pred_obj, 0.05)
    pred_obj = np.asarray(pred_obj)
    predictions = []

    if len(pred_obj) > 0:
        processed_pred = process_predictions_FFT(pred_obj, confidence_threshold=confidence_threshold)
        for i, detection in enumerate(processed_pred):
            predictions.append({
                'detection_id': i,
                'confidence': detection[0],
                'x1': detection[1],
                'y1': detection[2],
                'x2': detection[3],
                'y2': detection[4],
                'x3': detection[5],
                'y3': detection[6],
                'x4': detection[7],
                'y4': detection[8],
                'range_m': detection[9],
                'azimuth_deg': detection[10]
            })
    return pd.DataFrame(predictions)


def convert_model_output_to_detections(model_outputs, encoder, frame_timestamp: float,
                                       confidence_threshold: float = 0.2) -> List[Detection]:
    """
    Convert model outputs to Detection objects for tracking.
    Uses the same extraction logic as extract_model_predictions.
    """
    detections = []

    # Extract using the same method as your existing code
    pred_obj = model_outputs['Detection'].detach().cpu().numpy().copy()[0]
    pred_obj = encoder.decode(pred_obj, 0.05)
    pred_obj = np.asarray(pred_obj)

    if len(pred_obj) > 0:
        processed_pred = process_predictions_FFT(pred_obj, confidence_threshold=confidence_threshold)
        for detection in processed_pred:
            # Create Detection object for tracking
            detection_obj = Detection(
                range_m=float(detection[9]),  # range_m
                azimuth_rad=np.radians(detection[10]),  # convert azimuth_deg to radians
                confidence=float(detection[0]),  # confidence
                timestamp=frame_timestamp,
                # Store additional bounding box info for later use
                bbox_corners={
                    'x1': detection[1], 'y1': detection[2],
                    'x2': detection[3], 'y2': detection[4],
                    'x3': detection[5], 'y3': detection[6],
                    'x4': detection[7], 'y4': detection[8]
                }
            )
            detections.append(detection_obj)

    return detections


def extract_tracking_predictions(tracks, frame_id):
    """Extract tracking results into a DataFrame similar to model predictions"""
    predictions = []

    for i, track in enumerate(tracks):
        # Get the latest state from the track
        # Extract state directly from track attributes
        x, y, _, _ = track.state  # [x, y, vx, vy]

        # Convert Cartesian to polar coordinates
        range_m = np.sqrt(x ** 2 + y ** 2)
        azimuth_rad = np.arctan2(x, y)  # Swap x and y for Y=1, X=0 to be 0 degrees (forward axis)
        azimuth_deg = np.degrees(azimuth_rad)

        # Get bounding box from last detection if available
        bbox_corners = track.last_detection.bbox_corners if track.last_detection else None

        prediction = {
            'track_id': track.id,  # Fixed attribute name
            'detection_id': track.id,  # Using track ID as placeholder
            'confidence': track.confidence,
            'range_m': range_m,
            'azimuth_deg': azimuth_deg,
            'track_age': track.age,
            'hits': track.hits,
            'track_state': 'active' if track.time_since_update == 0 else 'coasting'
        }

        # Add bounding box if available
        if bbox_corners:
            prediction.update(bbox_corners)
        else:
            for coord in ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']:
                prediction[coord] = np.nan

        predictions.append(prediction)

    return pd.DataFrame(predictions)


def save_predictions_to_csv(model_outputs, encoder, sample_id, output_path):
    """Save model predictions for a sample to CSV (original function, unchanged)"""
    df = extract_model_predictions(model_outputs, encoder)
    df['sample_id'] = sample_id
    # Reorder columns
    cols = ['sample_id', 'detection_id', 'confidence', 'range_m', 'azimuth_deg'] + \
           [f'{coord}{i}' for coord in ['x', 'y'] for i in range(1, 5)]
    df = df[cols]
    df.to_csv(output_path, index=False)
    return df


def save_tracking_predictions_to_csv(tracks, sample_id, frame_id, output_path):
    """Save tracking predictions for a sample to CSV"""
    df = extract_tracking_predictions(tracks, frame_id)
    df['sample_id'] = sample_id
    df['frame_id'] = frame_id

    # Reorder columns to match original format plus tracking info
    base_cols = ['sample_id', 'frame_id', 'track_id', 'detection_id', 'confidence',
                 'range_m', 'azimuth_deg']
    bbox_cols = [f'{coord}{i}' for coord in ['x', 'y'] for i in range(1, 5)]
    tracking_cols = ['track_age', 'hits', 'track_state']

    cols = base_cols + bbox_cols + tracking_cols
    # Only include columns that exist in the DataFrame
    available_cols = [col for col in cols if col in df.columns]
    df = df[available_cols] if available_cols else df

    df.to_csv(output_path, index=False)
    return df


def batch_save_predictions(model_outputs_dict, encoder, output_dir):
    """Save predictions for multiple samples (original function, unchanged)"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    for sample_id, outputs in model_outputs_dict.items():
        df = extract_model_predictions(outputs, encoder)
        df['sample_id'] = sample_id
        all_predictions.append(df)

    # Combine all predictions
    combined_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    combined_df.to_csv(output_dir / 'all_predictions.csv', index=False)
    return combined_df


def batch_save_tracking_results(tracking_results_dict, output_dir):
    """Save tracking results for multiple samples"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_tracking_predictions = []

    for frame_id, frame_data in tracking_results_dict.items():
        tracks = frame_data.get('tracks', [])
        sample_id = frame_data.get('frame_id', frame_id)  # Use stored sample_id if available

        df = extract_tracking_predictions(tracks, frame_id)
        df['sample_id'] = sample_id
        df['frame_id'] = frame_id
        all_tracking_predictions.append(df)

    # Combine all tracking predictions
    if all_tracking_predictions:
        combined_tracking_df = pd.concat(all_tracking_predictions, ignore_index=True)
        combined_tracking_df.to_csv(output_dir / 'all_tracking_predictions.csv', index=False)
    else:
        combined_tracking_df = pd.DataFrame()

    return combined_tracking_df


def save_comprehensive_results(results_comparison: Dict, encoder, output_dir: str = "plots/tracking_results/"):
    """
    Enhanced comprehensive results saving using existing CSV structure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Save original predictions using existing function
    print("Saving original model predictions...")
    original_predictions_df = batch_save_predictions(
        results_comparison['without_tracking']['model_outputs'],
        encoder,
        output_path / "original_predictions"
    )

    # 2. Save tracking predictions
    print("Saving tracking predictions...")
    tracking_predictions_df = batch_save_tracking_results(
        results_comparison['with_tracking']['tracks_per_frame'],
        output_path / "tracking_predictions"
    )

    # 3. Create frame-by-frame comparison DataFrame
    print("Creating frame-by-frame comparison...")
    comparison_data = []

    for frame_id in results_comparison['without_tracking']['detections_per_frame'].keys():
        without_data = results_comparison['without_tracking']['detections_per_frame'][frame_id]
        with_data = results_comparison['with_tracking']['tracks_per_frame'].get(frame_id, {})

        comparison_data.append({
            'frame_id': frame_id,
            'sample_id': without_data.get('frame_id', frame_id),
            'original_detections_count': without_data['num_detections'],
            'tracking_tracks_count': with_data.get('num_tracks', 0),
            'ground_truth_count': len(without_data.get('ground_truth', [])),
            'original_processing_time': results_comparison['without_tracking']['processing_times'][
                frame_id] if frame_id < len(results_comparison['without_tracking']['processing_times']) else np.nan,
            'tracking_processing_time': results_comparison['with_tracking']['processing_times'][
                frame_id] if frame_id < len(results_comparison['with_tracking']['processing_times']) else np.nan
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_path / 'frame_comparison.csv', index=False)

    # 4. Save tracking statistics
    tracking_stats = results_comparison['with_tracking']['tracklet_statistics']
    stats_df = pd.DataFrame([tracking_stats])
    stats_df.to_csv(output_path / 'tracking_statistics.csv', index=False)

    # 5. Save metadata and summary
    summary = {
        'metadata': results_comparison['metadata'],
        'summary_statistics': {
            'total_frames': len(results_comparison['without_tracking']['detections_per_frame']),
            'total_original_detections': results_comparison['without_tracking']['total_detections'],
            'total_tracks_created': results_comparison['with_tracking']['total_tracks_created'],
            'avg_original_processing_time': np.mean(results_comparison['without_tracking']['processing_times']),
            'avg_tracking_processing_time': np.mean(results_comparison['with_tracking']['processing_times']),
            'tracking_overhead_ratio': np.mean(results_comparison['with_tracking']['processing_times']) / np.mean(
                results_comparison['without_tracking']['processing_times']) if np.mean(
                results_comparison['without_tracking']['processing_times']) > 0 else np.nan
        }
    }

    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # 6. Save complete results as pickle for detailed analysis
    with open(output_path / 'complete_results.pkl', 'wb') as f:
        pickle.dump(results_comparison, f)

    print(f"\nComprehensive results saved to: {output_path}")
    print("Files created:")
    print(f"  - original_predictions/all_predictions.csv ({len(original_predictions_df)} detections)")
    print(f"  - tracking_predictions/all_tracking_predictions.csv ({len(tracking_predictions_df)} tracks)")
    print(f"  - frame_comparison.csv ({len(comparison_df)} frames)")
    print(f"  - tracking_statistics.csv")
    print(f"  - summary.json")
    print(f"  - complete_results.pkl (for detailed analysis)")

    return {
        'original_predictions': original_predictions_df,
        'tracking_predictions': tracking_predictions_df,
        'frame_comparison': comparison_df,
        'tracking_statistics': stats_df
    }


def analyze_tracking_benefits(results_comparison: Dict, output_dir: str = "plots/tracking_results/"):
    """
    Analyze and report the benefits of using tracking
    """
    output_path = Path(output_dir)

    # Calculate key metrics
    total_frames = len(results_comparison['without_tracking']['detections_per_frame'])
    total_detections = results_comparison['without_tracking']['total_detections']
    total_tracks = results_comparison['with_tracking']['total_tracks_created']

    avg_detections_per_frame = total_detections / total_frames if total_frames > 0 else 0
    tracking_stats = results_comparison['with_tracking']['tracklet_statistics']

    # Processing time analysis
    avg_original_time = np.mean(results_comparison['without_tracking']['processing_times'])
    avg_tracking_time = np.mean(results_comparison['with_tracking']['processing_times'])
    overhead_ratio = avg_tracking_time / avg_original_time if avg_original_time > 0 else np.inf

    # Create analysis report
    analysis = {
        'detection_analysis': {
            'total_frames_processed': total_frames,
            'total_detections_without_tracking': total_detections,
            'avg_detections_per_frame': avg_detections_per_frame,
            'detection_density': total_detections / total_frames if total_frames > 0 else 0
        },
        'tracking_analysis': {
            'total_tracks_created': total_tracks,
            'track_creation_ratio': total_tracks / total_detections if total_detections > 0 else 0,
            'avg_track_lifetime': tracking_stats.get('average_lifetime', 0),
            'active_tracks': tracking_stats.get('active_tracklets', 0),
            'overall_match_ratio': tracking_stats.get('overall_match_ratio', 0),
            'avg_match_distance': tracking_stats.get('overall_average_distance', 0)
        },
        'performance_analysis': {
            'avg_detection_processing_time_ms': avg_original_time * 1000,
            'avg_tracking_processing_time_ms': avg_tracking_time * 1000,
            'tracking_overhead_ratio': overhead_ratio,
            'tracking_overhead_ms': (avg_tracking_time - avg_original_time) * 1000
        }
    }

    # Save analysis
    with open(output_path / 'tracking_benefits_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 80)
    print("TRACKING BENEFITS ANALYSIS")
    print("=" * 80)
    print(f"Frames Processed: {total_frames}")
    print(f"Total Detections: {total_detections} (avg: {avg_detections_per_frame:.1f}/frame)")
    print(f"Total Tracks Created: {total_tracks}")
    print(f"Track Creation Efficiency: {analysis['tracking_analysis']['track_creation_ratio']:.2f}")
    print(f"Average Track Lifetime: {analysis['tracking_analysis']['avg_track_lifetime']:.1f} frames")

    if tracking_stats.get('overall_match_ratio'):
        print(f"Overall Match Ratio: {analysis['tracking_analysis']['overall_match_ratio']:.3f}")
        print(f"Average Match Distance: {analysis['tracking_analysis']['avg_match_distance']:.2f}m")

    print(f"\nPerformance:")
    print(f"Detection Processing: {analysis['performance_analysis']['avg_detection_processing_time_ms']:.2f}ms/frame")
    print(f"Tracking Processing: {analysis['performance_analysis']['avg_tracking_processing_time_ms']:.2f}ms/frame")
    print(
        f"Tracking Overhead: {analysis['performance_analysis']['tracking_overhead_ms']:.2f}ms/frame ({overhead_ratio:.2f}x)")
    print("=" * 80)

    return analysis