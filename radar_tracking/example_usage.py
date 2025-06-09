"""
Example usage of the radar tracking system.
Demonstrates how to use all components together.
"""
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from data_structures import Detection, Track
from tracklet_manager import TrackletManager
from metrics import RadarMetrics
from coordinate_transforms import polar_to_cartesian


def generate_synthetic_radar_data(num_frames: int = 50,
                                  num_objects: int = 3) -> tuple:
    """
    Generate synthetic radar data for testing.

    Args:
        num_frames: Number of frames to generate
        num_objects: Number of objects to track

    Returns:
        Tuple of (detections_list, ground_truth_list)
    """
    # Initialize object trajectories
    objects = []
    for i in range(num_objects):
        # Random starting position and velocity
        start_range = np.random.uniform(50, 200)  # 50-200 meters
        start_azimuth = np.random.uniform(-np.pi / 3, np.pi / 3)  # ±60 degrees
        velocity = np.random.uniform(5, 15)  # 5-15 m/s
        direction = np.random.uniform(0, 2 * np.pi)

        objects.append({
            'start_pos': polar_to_cartesian(start_range, start_azimuth),
            'velocity': (velocity * np.cos(direction), velocity * np.sin(direction)),
            'confidence': np.random.uniform(0.7, 0.95)
        })

    detections_frames = []
    ground_truth_frames = []

    for frame in range(num_frames):
        frame_detections = []
        frame_ground_truth = []

        for obj_id, obj in enumerate(objects):
            # Calculate true position
            true_x = obj['start_pos'][0] + obj['velocity'][0] * frame
            true_y = obj['start_pos'][1] + obj['velocity'][1] * frame

            # Add noise to detections
            noise_x = np.random.normal(0, 1.0)  # 1m standard deviation
            noise_y = np.random.normal(0, 1.0)

            det_x = true_x + noise_x
            det_y = true_y + noise_y

            # Convert to polar coordinates
            from coordinate_transforms import cartesian_to_polar
            det_range, det_azimuth = cartesian_to_polar(det_x, det_y)
            true_range, true_azimuth = cartesian_to_polar(true_x, true_y)

            # Create detection (with noise)
            detection = Detection(
                range_m=det_range,
                azimuth_rad=det_azimuth,
                confidence=obj['confidence'] + np.random.normal(0, 0.05),
                timestamp=frame
            )

            # Create ground truth (without noise)
            ground_truth = Detection(
                range_m=true_range,
                azimuth_rad=true_azimuth,
                confidence=1.0,
                timestamp=frame
            )

            # Sometimes miss detections (simulate real-world conditions)
            if np.random.random() > 0.1:  # 90% detection rate
                frame_detections.append(detection)

            frame_ground_truth.append(ground_truth)

        # Add false alarms
        num_false_alarms = np.random.poisson(0.5)  # Average 0.5 false alarms per frame
        for _ in range(num_false_alarms):
            false_range = np.random.uniform(30, 300)
            false_azimuth = np.random.uniform(-np.pi / 2, np.pi / 2)
            false_detection = Detection(
                range_m=false_range,
                azimuth_rad=false_azimuth,
                confidence=np.random.uniform(0.3, 0.7),
                timestamp=frame
            )
            frame_detections.append(false_detection)

        detections_frames.append(frame_detections)
        ground_truth_frames.append(frame_ground_truth)

    return detections_frames, ground_truth_frames


def run_tracking_example():
    """Run complete tracking example with synthetic data."""
    print("Generating synthetic radar data...")
    detections_frames, ground_truth_frames = generate_synthetic_radar_data(
        num_frames=100, num_objects=4
    )

    # Configure tracker
    tracker_config = {
        'max_age': 5,
        'min_hits': 3,
        'iou_threshold': 8.0,  # 8 meter association threshold
        'dt': 0.1  # 100ms between frames
    }

    # Initialize tracklet manager
    manager = TrackletManager(tracker_config=tracker_config)

    print("Running tracking...")
    all_tracks = []

    # Process all frames
    for frame_id, (detections, ground_truth) in enumerate(
            zip(detections_frames, ground_truth_frames)
    ):
        tracks = manager.update(detections, ground_truth)
        all_tracks.append(tracks)

        # Print progress every 20 frames
        if frame_id % 20 == 0:
            print(f"Processed frame {frame_id}, active tracks: {len(tracks)}")

    # Print final results
    manager.print_summary()

    return manager, all_tracks


def plot_tracking_results(manager: TrackletManager,
                          detections_frames: List[List[Detection]],
                          max_frames: int = 50):
    """
    Plot tracking results for visualization.

    Args:
        manager: TrackletManager with results
        detections_frames: Original detection data
        max_frames: Maximum number of frames to plot
    """
    plt.figure(figsize=(12, 8))

    # Plot detections
    for frame_id in range(min(len(detections_frames), max_frames)):
        detections = detections_frames[frame_id]
        for det in detections:
            x, y = det.cartesian_pos
            plt.scatter(x, y, c='lightblue', alpha=0.3, s=20)

    # Plot tracks
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx = 0

    # Get all tracklets (active and historical)
    all_tracklets = {**manager.active_tracklets, **manager.historical_tracklets}

    for track_id, tracklet in all_tracklets.items():
        if tracklet.total_detections > 5:  # Only plot substantial tracks
            color = colors[color_idx % len(colors)]
            plt.scatter([], [], c=color, label=f'Track {track_id}', s=50)
            color_idx += 1

    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.title('Radar Tracking Results')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def evaluate_tracking_performance(manager: TrackletManager):
    """Evaluate and print detailed tracking performance metrics."""
    if not manager.frame_results:
        print("No evaluation data available")
        return

    results = manager.frame_results

    # Calculate metrics
    match_ratios = [r.match_ratio for r in results]
    distances = [r.average_distance for r in results if r.average_distance != float('inf')]
    total_matches = sum(r.num_matches for r in results)
    total_gt = sum(r.num_ground_truth for r in results)
    total_predictions = sum(r.num_predictions for r in results)

    print("\n" + "=" * 60)
    print("DETAILED TRACKING PERFORMANCE EVALUATION")
    print("=" * 60)
    print(f"Total Frames Processed: {len(results)}")
    print(f"Total Ground Truth Objects: {total_gt}")
    print(f"Total Predictions: {total_predictions}")
    print(f"Total Successful Matches: {total_matches}")
    print(f"\nOverall Match Ratio: {total_matches / total_gt:.3f}")
    print(f"Average Match Ratio per Frame: {np.mean(match_ratios):.3f}")
    print(f"Match Ratio Std Dev: {np.std(match_ratios):.3f}")

    if distances:
        print(f"\nAverage Match Distance: {np.mean(distances):.2f} meters")
        print(f"Distance Std Dev: {np.std(distances):.2f} meters")
        print(f"Min Match Distance: {np.min(distances):.2f} meters")
        print(f"Max Match Distance: {np.max(distances):.2f} meters")

    # False positive and negative analysis
    false_positives = total_predictions - total_matches
    false_negatives = total_gt - total_matches

    if total_predictions > 0:
        precision = total_matches / total_predictions
        print(f"\nPrecision: {precision:.3f}")

    if total_gt > 0:
        recall = total_matches / total_gt
        print(f"Recall: {recall:.3f}")

    print(f"\nFalse Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print("=" * 60)


if __name__ == "__main__":
    # Run the complete example
    print("Starting Radar Tracking System Example")
    print("=" * 50)

    # Run tracking
    manager, tracks = run_tracking_example()

    # Evaluate performance
    evaluate_tracking_performance(manager)

    # Uncomment to show plot (requires matplotlib)
    detections_frames, _ = generate_synthetic_radar_data(num_frames=50, num_objects=4)
    plot_tracking_results(manager, detections_frames)

    print("\nExample completed successfully!")