import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from radar_tracking import Detection, Track


def prepare_output_directories(output_dir: str):
    """
    Create output directory structure for visualizations.

    Args:
        output_dir: Base output directory
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def visualize_frame_radar_azimuth(
        frame_id: int,
        detections: List[Detection],
        ground_truth: List[Detection],
        active_tracks: List[Track],
        output_dir: str,
        show_coverage_bounds: bool = True
):
    """
    Plot one frame in (azimuth_deg, range_m) space with radar coverage overlay.
    """
    prepare_output_directories(output_dir)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Show radar coverage bounds
    if show_coverage_bounds:
        # Draw coverage area
        ax.axhline(y=103, color='gray', linestyle='--', alpha=0.5, label='Max Range')
        ax.axvline(x=-90, color='gray', linestyle='--', alpha=0.5, label='Azimuth Limits')
        ax.axvline(x=90, color='gray', linestyle='--', alpha=0.5)

        # Shade out-of-coverage areas
        ax.fill_between([-90, 90], 103, 120, color='red', alpha=0.1, label='Out of Coverage')
        ax.fill([-100, -90, -90, -100], [0, 0, 120, 120], color='red', alpha=0.1)
        ax.fill([90, 100, 100, 90], [0, 0, 120, 120], color='red', alpha=0.1)

    # Plot network output (blue circles)
    if detections:
        az_det = [np.degrees(d.azimuth_rad) for d in detections]
        rng_det = [d.range_m for d in detections]
        conf_det = [d.confidence for d in detections]

        # Color by confidence
        scatter = ax.scatter(az_det, rng_det, c=conf_det, s=20,
                             cmap='Blues', alpha=0.8, vmin=0, vmax=1,
                             label='Network Output')
        plt.colorbar(scatter, ax=ax, label='Confidence')

    # Plot ground truth (green X)
    if ground_truth:
        az_gt = [np.degrees(d.azimuth_rad) for d in ground_truth]
        rng_gt = [d.range_m for d in ground_truth]
        ax.scatter(az_gt, rng_gt, c='green', marker='x', s=60, label='Ground Truth')

    # Plot tracks using Kalman state (red triangles)
    for track in active_tracks:
        range_m, azimuth_rad = track.kalman_polar_position
        az_tr = np.degrees(azimuth_rad)
        rng_tr = range_m

        ax.scatter(az_tr, rng_tr, marker='^', s=20, facecolors='none', edgecolors='red',
                   linewidths=0.8, label='Tracker Estimate' if track == active_tracks[0] else "")
        ax.text(az_tr + 0.2, rng_tr + 0.2, f"T{track.id}", color='red', fontsize=8)

    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Range (m)")
    ax.set_title(f"Frame {frame_id:06d} - Radar Coverage Visualization")
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-100, 100)  # Show slightly beyond coverage for context
    ax.set_ylim(0, 120)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(output_dir) / f"frame_{frame_id:06d}.jpg"
    plt.savefig(out_path, dpi=150)
    plt.close()


def visualize_counts_vs_tracks_per_frame(
        all_frames: List[int],
        det_counts: List[int],
        track_counts: List[int],
        output_dir: str
):
    """Plot Network Output per Frame vs. Confirmed Tracks per Frame."""
    prepare_output_directories(output_dir)

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(all_frames, det_counts, marker='o', linestyle='-', label='Network Output/frame')
    ax1.plot(all_frames, track_counts, marker='s', linestyle='--', label='Active tracks/frame')
    ax1.set_xlabel("Frame ID (sample_id)")
    ax1.set_ylabel("Count")
    ax1.set_title("Network Output vs. Active Tracks per Frame")
    ax1.legend(loc='upper right')
    plt.tight_layout()

    save_path = Path(output_dir) / "counts_vs_tracks_per_frame.png"
    fig1.savefig(save_path)
    plt.close(fig1)


def visualize_tracklet_lifetime_histogram(
        manager,
        output_dir: str
):
    """Generate histogram of tracklet lifetimes."""
    prepare_output_directories(output_dir)

    # Get all stats from active and historical tracklets
    all_stats = {**manager.active_tracklets, **manager.historical_tracklets}
    lifetimes = np.array([sts.lifetime_frames for sts in all_stats.values()]) if all_stats else np.array([])

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    if len(lifetimes) > 0:
        ax2.hist(lifetimes, bins=20, edgecolor='black')
    ax2.set_xlabel("Tracklet Lifetime (frames)")
    ax2.set_ylabel("Number of Tracklets")
    ax2.set_title("Histogram of Tracklet Lifetimes")
    plt.tight_layout()

    save_path = Path(output_dir) / "tracklet_lifetime_histogram.png"
    fig2.savefig(save_path)
    plt.close(fig2)


def visualize_avg_confidence_over_time(
        all_frames: List[int],
        avg_confidence_per_frame: List[float],
        all_tracks: List[List[Track]],
        frame_times: List[Tuple[int, float]],
        output_dir: str,
        window_size: int = 3
):
    """Plot confidence over time for each track with rolling window averaging."""
    prepare_output_directories(output_dir)

    frame_to_time = dict(frame_times)

    # Collect confidence data per track
    track_confidence_data = {}

    for frame_idx, (frame_id, tracks) in enumerate(zip(all_frames, all_tracks)):
        timestamp = frame_to_time.get(frame_id, frame_id)

        for track in tracks:
            if track.id not in track_confidence_data:
                track_confidence_data[track.id] = {
                    'times': [],
                    'confidences': [],
                    'raw_confidences': []
                }

            if track.last_detection and track.last_detection.confidence > 0:
                track_confidence_data[track.id]['times'].append(timestamp)
                track_confidence_data[track.id]['raw_confidences'].append(
                    track.last_detection.confidence
                )

    # Apply rolling window averaging
    for track_id, data in track_confidence_data.items():
        raw_conf = data['raw_confidences']
        smoothed_conf = []

        for i in range(len(raw_conf)):
            start_idx = max(0, i - window_size + 1)
            window = raw_conf[start_idx:i + 1]
            smoothed_conf.append(np.mean(window))

        data['confidences'] = smoothed_conf

    # Sort tracks by lifetime for better visualization
    sorted_tracks = sorted(track_confidence_data.items(),
                           key=lambda x: len(x[1]['times']), reverse=True)

    # Select top tracks for individual subplots
    max_individual_tracks = 6
    individual_tracks = sorted_tracks[:max_individual_tracks]

    # Create figure with subplots
    if len(individual_tracks) > 0:
        fig_height = 4 + 2.5 * len(individual_tracks)
        fig, axes = plt.subplots(len(individual_tracks) + 1, 1,
                                 figsize=(14, fig_height),
                                 gridspec_kw={'height_ratios': [3] + [2] * len(individual_tracks)})

        if len(individual_tracks) == 1:
            axes = [axes[0], axes[1]]

        main_ax = axes[0]
        track_axes = axes[1:]
    else:
        fig, main_ax = plt.subplots(1, 1, figsize=(14, 6))
        track_axes = []

    # Main plot: Overall average and all tracks overview
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, (track_id, data) in enumerate(sorted_tracks[:10]):  # Show top 10 in overview
        if len(data['times']) < 3:  # Skip very short tracks
            continue

        color = colors[i % len(colors)]
        main_ax.plot(data['times'], data['confidences'],
                     color=color, linewidth=1.5, alpha=0.7,
                     label=f'Track {track_id}')

    # Add overall average
    timestamps = [frame_to_time.get(f, f) for f in all_frames]
    main_ax.plot(timestamps, avg_confidence_per_frame,
                 'k-', linewidth=3, alpha=0.8,
                 label='Overall Average')

    main_ax.set_ylabel('Detection Confidence', fontsize=12)
    main_ax.set_title(f'Track Confidence Overview (Rolling Window Size: {window_size})',
                      fontsize=14)
    main_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    main_ax.grid(True, alpha=0.3)
    main_ax.set_ylim(0, 1.05)

    # Individual track subplots with independent y-axis scaling
    for i, (track_id, data) in enumerate(individual_tracks):
        if i >= len(track_axes):
            break

        ax = track_axes[i]
        color = colors[i % len(colors)]

        # Plot raw confidence as scatter with transparency
        ax.scatter(data['times'], data['raw_confidences'],
                   color=color, s=15, alpha=0.4, label='Raw')

        # Plot smoothed confidence
        ax.plot(data['times'], data['confidences'],
                color=color, linewidth=2.5, alpha=0.9,
                label='Smoothed', marker='o', markersize=3)

        # Calculate confidence statistics for this track
        conf_range = max(data['confidences']) - min(data['confidences'])
        conf_mean = np.mean(data['confidences'])

        # Set y-axis range to highlight variations
        if conf_range > 0.1:  # Significant variation
            y_margin = conf_range * 0.1
            ax.set_ylim(min(data['confidences']) - y_margin,
                        max(data['confidences']) + y_margin)
        else:  # Small variation - use fixed range around mean
            ax.set_ylim(max(0, conf_mean - 0.1), min(1, conf_mean + 0.1))

        # Styling
        ax.set_ylabel(f'Confidence\n(Track {track_id})', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

        # Add statistics annotation
        stats_text = f'μ={conf_mean:.3f}, σ={np.std(data["confidences"]):.3f}'
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=8, verticalalignment='top')

    # Set x-label on bottom subplot
    if track_axes.any():
        track_axes[-1].set_xlabel('Time (seconds)', fontsize=12)
    else:
        main_ax.set_xlabel('Time (seconds)', fontsize=12)

    plt.tight_layout()
    save_path = Path(output_dir) / "enhanced_confidence_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_all_frames_3d_overview(
        all_detections: List[List[Detection]],
        all_ground_truth: List[List[Detection]],
        all_tracks: List[List[Track]],
        all_frames: List[int],
        frame_times: List[Tuple[int, float]],  # Added parameter
        output_dir: str
):
    """Create 3D visualization with real time axis showing tracker predictions during gaps."""
    prepare_output_directories(output_dir)

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Create frame_id to timestamp mapping
    frame_to_time = dict(frame_times)

    # Collect all data points
    gt_times, gt_az, gt_rng = [], [], []
    det_times, det_az, det_rng = [], [], []

    # Create a color map for tracks
    unique_track_ids = set()
    for frame_tracks in all_tracks:
        for track in frame_tracks:
            unique_track_ids.add(track.id)
    num_tracks = len(unique_track_ids)
    color_map = cm.get_cmap('tab10', min(num_tracks + 1, 10))
    track_colors = {track_id: mcolors.to_hex(color_map(i % 10))
                    for i, track_id in enumerate(unique_track_ids)}

    # Store track data with actual Kalman predictions
    track_data = {}

    for frame_idx, frame_id in enumerate(all_frames):
        timestamp = frame_to_time.get(frame_id, frame_id)

        # Ground truth
        for gt in all_ground_truth[frame_idx]:
            gt_times.append(timestamp)
            gt_az.append(np.degrees(gt.azimuth_rad))
            gt_rng.append(gt.range_m)

        # Detections
        for det in all_detections[frame_idx]:
            det_times.append(timestamp)
            det_az.append(np.degrees(det.azimuth_rad))
            det_rng.append(det.range_m)

        # Tracks with enhanced prediction handling
        for track in all_tracks[frame_idx]:
            range_m, azimuth_rad = track.kalman_polar_position
            azimuth_deg = np.degrees(azimuth_rad)

            if track.id not in track_data:
                track_data[track.id] = {
                    'measurements': [],  # Actual measurements
                    'predictions': [],  # Kalman predictions during gaps
                    'has_detection': []
                }

            track_data[track.id]['measurements'].append((timestamp, azimuth_deg, range_m))
            track_data[track.id]['has_detection'].append(track.last_detection is not None)

    # Generate Kalman predictions for gaps
    for track_id, data in track_data.items():
        measurements = data['measurements']
        has_detections = data['has_detection']

        # Find gaps and generate actual Kalman predictions
        for i in range(len(measurements) - 1):
            curr_time, curr_az, curr_rng = measurements[i]
            next_time, next_az, next_rng = measurements[i + 1]

            time_gap = next_time - curr_time
            if time_gap > 0.2:  # Significant gap
                # Create temporary track for prediction
                from radar_tracking.data_structures import Track
                from radar_tracking.coordinate_transforms import polar_to_cartesian

                # Convert current position back to Cartesian for state
                x_curr, y_curr = polar_to_cartesian(curr_rng, np.radians(curr_az))

                # Estimate velocity from previous measurement if available
                if i > 0:
                    prev_time, prev_az, prev_rng = measurements[i - 1]
                    prev_x, prev_y = polar_to_cartesian(prev_rng, np.radians(prev_az))
                    dt_prev = curr_time - prev_time
                    vx = (x_curr - prev_x) / dt_prev if dt_prev > 0 else 0
                    vy = (y_curr - prev_y) / dt_prev if dt_prev > 0 else 0
                else:
                    vx, vy = 0, 0

                # Create state vector and covariance
                state = np.array([x_curr, y_curr, vx, vy])
                covariance = np.eye(4) * 10.0  # Initial uncertainty

                # Generate actual Kalman predictions
                temp_track = type('TempTrack', (), {
                    'state': state,
                    'covariance': covariance
                })()

                predictions = get_actual_kalman_predictions(
                    temp_track, curr_time, next_time, dt=0.05
                )

                data['predictions'].extend(predictions)

    # Plot ground truth and detections
    if gt_times:
        ax.scatter(gt_times, gt_az, gt_rng, c='green', marker='x', s=40,
                   alpha=0.7, label='Ground Truth')
    if det_times:
        ax.scatter(det_times, det_az, det_rng, c='blue', s=15,
                   alpha=0.5, label='Detections')

    # Plot tracks with clear distinction between measurements and predictions
    for track_id, data in track_data.items():
        if not data['measurements']:
            continue

        color = track_colors[track_id]

        # Plot measurements (solid line with markers)
        times, azimuths, ranges = zip(*data['measurements'])
        ax.plot(times, azimuths, ranges, color=color, linewidth=2.5, alpha=0.9,
                label=f'Track {track_id} (Measurements)', marker='o', markersize=4)

        # Plot predictions (dashed line, different markers)
        if data['predictions']:
            pred_times, pred_ranges, pred_azimuths, _ = zip(*data['predictions'])
            pred_azimuths_deg = [np.degrees(az) for az in pred_azimuths]
            ax.plot(pred_times, pred_azimuths_deg, pred_ranges,
                    color=color, linewidth=1.5, alpha=0.6, linestyle='--',
                    marker='^', markersize=2,
                    label=f'Track {track_id} (Kalman Predictions)')

        # Add track ID at start
        if times:
            ax.text(times[0], azimuths[0], ranges[0], f"T{track_id}",
                    color=color, fontsize=8, fontweight='bold')

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Azimuth (degrees)', fontsize=12)
    ax.set_zlabel('Range (meters)', fontsize=12)
    ax.set_title('3D Radar Tracking: Measurements vs Kalman Predictions\n'
                 'Solid lines: actual measurements, Dashed lines: Kalman filter predictions',
                 fontsize=14)

    # Enhanced legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10)

    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    save_path = Path(output_dir) / "3d_tracking_enhanced.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def visualize_tracking_temporal_evolution(
        all_detections: List[List[Detection]],
        all_ground_truth: List[List[Detection]],
        all_tracks: List[List[Track]],
        all_frames: List[int],
        frame_times: List[Tuple[int, float]],  # Added parameter
        num_tracks: Optional[int] = 3,
        output_dir: str = "tracking_temporal_evolution.png",
):
    """Create temporal visualization with real time axis showing predictions during gaps."""
    prepare_output_directories(output_dir)

    frame_to_time = dict(frame_times)

    # Find longest-lived tracks
    track_lifespans = {}
    for frame_idx, tracks in enumerate(all_tracks):
        timestamp = frame_to_time.get(all_frames[frame_idx], all_frames[frame_idx])
        for track in tracks:
            if track.id not in track_lifespans:
                track_lifespans[track.id] = []
            track_lifespans[track.id].append((timestamp, track, all_frames[frame_idx]))

    # Select top num_tracks longest tracks
    longest_tracks = sorted(track_lifespans.items(), key=lambda x: len(x[1]), reverse=True)[:num_tracks]

    fig, axes = plt.subplots(num_tracks, 1, figsize=(16, 4 * num_tracks))
    if num_tracks == 1:
        axes = [axes]

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for i, (track_id, track_history) in enumerate(longest_tracks):
        if i >= len(axes):
            break

        ax = axes[i]

        # Extract data
        timestamps = [t for t, _, _ in track_history]
        ranges = [track.kalman_polar_position[0] for _, track, _ in track_history]
        has_detection = [track.last_detection is not None for _, track, _ in track_history]

        # Generate actual Kalman predictions for gaps
        enhanced_times = []
        enhanced_ranges = []
        prediction_mask = []

        for j in range(len(timestamps) - 1):
            enhanced_times.append(timestamps[j])
            enhanced_ranges.append(ranges[j])
            prediction_mask.append(False)  # Measurement

            # Check for significant time gap
            time_gap = timestamps[j + 1] - timestamps[j]
            if time_gap > 0.15:  # Generate predictions for gaps > 150ms
                # Get current track state
                _, current_track, _ = track_history[j]

                # Generate actual Kalman predictions
                predictions = get_actual_kalman_predictions(
                    current_track, timestamps[j], timestamps[j + 1], dt=0.05
                )

                for pred_time, pred_range, _, is_pred in predictions:
                    if pred_time < timestamps[j + 1]:  # Don't overlap with next measurement
                        enhanced_times.append(pred_time)
                        enhanced_ranges.append(pred_range)
                        prediction_mask.append(True)  # Prediction

        # Add last point
        enhanced_times.append(timestamps[-1])
        enhanced_ranges.append(ranges[-1])
        prediction_mask.append(False)  # Measurement

        # Plot measurements vs predictions with clear distinction
        meas_times = [t for t, is_pred in zip(enhanced_times, prediction_mask) if not is_pred]
        meas_ranges = [r for r, is_pred in zip(enhanced_ranges, prediction_mask) if not is_pred]
        pred_times = [t for t, is_pred in zip(enhanced_times, prediction_mask) if is_pred]
        pred_ranges = [r for r, is_pred in zip(enhanced_ranges, prediction_mask) if is_pred]

        # Plot measurements (solid line with circles)
        ax.plot(meas_times, meas_ranges, color=colors[i % len(colors)],
                linewidth=3, alpha=0.9, label=f'Track {track_id} (Measurements)',
                marker='o', markersize=5, markerfacecolor='white', markeredgewidth=2)

        # Plot Kalman predictions (dashed line with triangles)
        if pred_times:
            ax.plot(pred_times, pred_ranges, color=colors[i % len(colors)],
                    linewidth=2, alpha=0.6, linestyle='--',
                    label=f'Track {track_id} (Kalman Predictions)',
                    marker='^', markersize=3)

        # Mark significant time gaps
        for j in range(len(timestamps) - 1):
            gap = timestamps[j + 1] - timestamps[j]
            if gap > 0.5:  # Mark gaps > 500ms
                ax.axvspan(timestamps[j], timestamps[j + 1], alpha=0.15,
                           color='red', label='Time Gap >0.5s' if j == 0 else "")
                # Add gap annotation
                mid_time = (timestamps[j] + timestamps[j + 1]) / 2
                ax.text(mid_time, ax.get_ylim()[1] * 0.95, f'{gap:.1f}s gap',
                        ha='center', va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

        # Styling and annotations
        ax.set_ylabel('Range (m)', fontsize=12)
        ax.set_title(f'Track {track_id}: Measurements vs Kalman Predictions During Gaps',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Set appropriate axis limits
        if enhanced_times:
            ax.set_xlim(min(enhanced_times) - 0.5, max(enhanced_times) + 0.5)

        # Add track statistics
        num_measurements = len(meas_times)
        num_predictions = len(pred_times)
        track_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0

        stats_text = (f'Measurements: {num_measurements}\n'
                      f'Predictions: {num_predictions}\n'
                      f'Duration: {track_duration:.1f}s')

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                fontsize=9, verticalalignment='top')

    # Set x-label on bottom subplot
    if axes:
        axes[-1].set_xlabel('Time (seconds)', fontsize=12)

    plt.suptitle('Enhanced Temporal Evolution: Actual Kalman Filter Predictions\n'
                 'Solid lines: measurements, Dashed lines: Kalman filter predictions',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle

    save_path = Path(output_dir) / "enhanced_tracking_temporal_evolution.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def visualize_tracking_temporal_evolution(
        all_detections: List[List[Detection]],
        all_ground_truth: List[List[Detection]],
        all_tracks: List[List[Track]],
        all_frames: List[int],
        frame_times: List[Tuple[int, float]],  # Added parameter
        num_tracks: Optional[int] = 3,
        output_dir: str = "tracking_temporal_evolution.png",
):
    """Create temporal visualization with real time axis showing predictions during gaps."""
    prepare_output_directories(output_dir)

    # Create frame_id to timestamp mapping
    frame_to_time = dict(frame_times)

    # Find longest-lived tracks
    track_lifespans = {}
    for frame_idx, tracks in enumerate(all_tracks):
        timestamp = frame_to_time.get(all_frames[frame_idx], all_frames[frame_idx])
        for track in tracks:
            if track.id not in track_lifespans:
                track_lifespans[track.id] = []
            track_lifespans[track.id].append((timestamp, track, all_frames[frame_idx]))

    # Select top num_tracks longest tracks
    longest_tracks = sorted(track_lifespans.items(), key=lambda x: len(x[1]), reverse=True)[:num_tracks]

    fig, axes = plt.subplots(num_tracks, 1, figsize=(14, 3.5 * num_tracks))
    if num_tracks == 1:
        axes = [axes]

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for i, (track_id, track_history) in enumerate(longest_tracks):
        if i >= len(axes):
            break

        ax = axes[i]

        # Extract data
        timestamps = [t for t, _, _ in track_history]
        ranges = [track.kalman_polar_position[0] for _, track, _ in track_history]
        has_detection = [track.last_detection is not None for _, track, _ in track_history]

        # Create dense time series for predictions
        dense_times = []
        dense_ranges = []
        prediction_mask = []

        for j in range(len(timestamps) - 1):
            dense_times.append(timestamps[j])
            dense_ranges.append(ranges[j])
            prediction_mask.append(True)  # Actual measurement

            # Fill gaps with predictions
            time_gap = timestamps[j + 1] - timestamps[j]
            if time_gap > 0.15:  # Show predictions for gaps > 150ms
                num_pred = int(time_gap / 0.05)  # Prediction every 50ms
                for k in range(1, num_pred):
                    t = timestamps[j] + k * 0.05
                    # Linear interpolation for smooth visualization
                    alpha = k / num_pred
                    r = ranges[j] + alpha * (ranges[j + 1] - ranges[j])
                    dense_times.append(t)
                    dense_ranges.append(r)
                    prediction_mask.append(False)  # Prediction

        # Add last point
        dense_times.append(timestamps[-1])
        dense_ranges.append(ranges[-1])
        prediction_mask.append(True)

        # Plot Kalman filtered trajectory
        ax.plot(dense_times, dense_ranges, color=colors[i % len(colors)],
                linewidth=2.5, label=f'Track {track_id} (Kalman Filtered)', zorder=2)

        # Highlight predictions with different style
        pred_times = [t for t, is_meas in zip(dense_times, prediction_mask) if not is_meas]
        pred_ranges = [r for r, is_meas in zip(dense_ranges, prediction_mask) if not is_meas]
        if pred_times:
            ax.plot(pred_times, pred_ranges, 'o', color=colors[i % len(colors)],
                    markersize=3, alpha=0.3, label='Predictions during gaps')

        # Plot raw detections
        detection_times = []
        detection_ranges = []
        for timestamp, track, _ in track_history:
            if track.last_detection:
                detection_times.append(timestamp)
                detection_ranges.append(track.last_detection.range_m)

        if detection_times:
            ax.scatter(detection_times, detection_ranges,
                       color=colors[i % len(colors)], alpha=0.6, s=40,
                       marker='o', edgecolors='black', linewidth=0.5,
                       label='Raw Detections', zorder=3)

        # Mark time gaps
        for j in range(len(timestamps) - 1):
            gap = timestamps[j + 1] - timestamps[j]
            if gap > 0.5:  # Mark gaps > 500ms
                ax.axvspan(timestamps[j], timestamps[j + 1], alpha=0.1,
                           color='red', label='Time gap' if j == 0 else "")
                # Add text annotation
                mid_time = (timestamps[j] + timestamps[j + 1]) / 2
                ax.text(mid_time, ax.get_ylim()[1] * 0.95, f'{gap:.1f}s gap',
                        ha='center', va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        ax.set_ylabel('Range (m)', fontsize=11)
        ax.set_title(f'Track {track_id}: Continuous Tracking Through Time Gaps', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Set x-axis to show actual time
        ax.set_xlim(min(dense_times) - 0.5, max(dense_times) + 0.5)

    if len(axes) > 0:
        axes[-1].set_xlabel('Time (seconds)', fontsize=11)

    plt.suptitle('Temporal Evolution: Real-Time Tracking with Predictions\n'
                 'Tracker maintains position estimates during measurement gaps',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = Path(output_dir) / "tracking_temporal_evolution.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def visualize_timing_analysis(frame_times: List[Tuple[int, float]],
                              time_gaps: List[float],
                              output_dir: str):
    """Visualize frame timing and gaps."""
    prepare_output_directories(output_dir)

    frames, timestamps = zip(*frame_times)

    # Recalculate time gaps correctly (excluding first measurement)
    corrected_gaps = []
    for i in range(1, len(timestamps)):
        gap = timestamps[i] - timestamps[i - 1]
        corrected_gaps.append(gap)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # 1. Timestamps vs Frame ID
    ax1.plot(frames, timestamps, 'b-', linewidth=1.5, marker='o', markersize=3)
    ax1.set_xlabel('Frame ID')
    ax1.set_ylabel('Timestamp (seconds)')
    ax1.set_title('Frame Timestamps - Corrected Timeline')
    ax1.grid(True, alpha=0.3)

    # Highlight large gaps (but not the first "gap")
    for i in range(1, len(timestamps)):
        gap = timestamps[i] - timestamps[i - 1]
        if gap > 0.5:  # Gaps > 0.5s
            ax1.axvspan(frames[i - 1], frames[i], alpha=0.2, color='red')
            # Add gap duration annotation
            mid_frame = (frames[i - 1] + frames[i]) / 2
            ax1.text(mid_frame, timestamps[i], f'{gap:.2f}s',
                     ha='center', va='bottom', fontsize=8,
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # 2. Corrected time gaps histogram
    if corrected_gaps:
        ax2.hist(corrected_gaps, bins=50, edgecolor='black', alpha=0.7, color='green')
        ax2.set_xlabel('Time Gap (seconds)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Time Gaps Between Consecutive Frames\n(Excluding Initial Measurement)')
        ax2.axvline(np.mean(corrected_gaps), color='red', linestyle='--',
                    label=f'Mean: {np.mean(corrected_gaps):.3f}s')
        ax2.axvline(np.median(corrected_gaps), color='orange', linestyle='--',
                    label=f'Median: {np.median(corrected_gaps):.3f}s')
        ax2.legend()

    # 3. Corrected time gaps over frame sequence
    if corrected_gaps:
        ax3.plot(frames[1:], corrected_gaps, 'g-', linewidth=1.5, marker='o', markersize=2)
        ax3.set_xlabel('Frame ID')
        ax3.set_ylabel('Time Gap to Previous Frame (s)')
        ax3.set_title('Time Gaps Throughout Sequence (Corrected)')
        ax3.grid(True, alpha=0.3)

        # Mark large gaps
        large_gap_threshold = np.percentile(corrected_gaps, 95)
        large_gaps = [(frames[i + 1], gap) for i, gap in enumerate(corrected_gaps)
                      if gap > large_gap_threshold]
        if large_gaps:
            gap_frames, gap_values = zip(*large_gaps)
            ax3.scatter(gap_frames, gap_values, color='red', s=50, zorder=5,
                        label=f'Large gaps (>{large_gap_threshold:.2f}s)')
            ax3.legend()

        # Add statistics box
        stats_text = (f'Total Frames: {len(frames)}\n'
                      f'Time Span: {timestamps[-1] - timestamps[0]:.2f}s\n'
                      f'Avg Gap: {np.mean(corrected_gaps):.3f}s\n'
                      f'Max Gap: {np.max(corrected_gaps):.3f}s\n'
                      f'Gaps >0.5s: {sum(1 for g in corrected_gaps if g > 0.5)}')

        ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                 fontsize=9, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'corrected_timing_analysis.png', dpi=150)
    plt.close()

    print(f"Timing Analysis Summary:")
    print(f"  - Total measurements: {len(timestamps)}")
    print(f"  - Time gaps calculated: {len(corrected_gaps)}")
    if corrected_gaps:
        print(f"  - Average gap: {np.mean(corrected_gaps):.3f}s")
        print(f"  - Large gaps (>0.5s): {sum(1 for g in corrected_gaps if g > 0.5)}")

def visualize_tracking_during_gaps(all_tracks: List[List[Track]],
                                   frame_times: List[Tuple[int, float]],
                                   gap_threshold: float,
                                   output_dir: str):
    """Visualize how tracking maintains estimates during time gaps."""
    prepare_output_directories(output_dir)

    # Find frames with large preceding gaps
    gap_frames = []
    for i in range(1, len(frame_times)):
        time_gap = frame_times[i][1] - frame_times[i - 1][1]
        if time_gap > gap_threshold:
            gap_frames.append({
                'before_idx': i - 1,
                'after_idx': i,
                'before': frame_times[i - 1],
                'after': frame_times[i],
                'gap': time_gap
            })

    if not gap_frames:
        print(f"No gaps larger than {gap_threshold}s found")
        return

    # Visualize tracking through largest gap
    largest_gap = max(gap_frames, key=lambda x: x['gap'])

    # Create larger figure for better clarity
    fig = plt.figure(figsize=(16, 8))

    # Use GridSpec for better layout control
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, width_ratios=[3, 1], height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[:, 0])  # Left: main tracking plot
    ax2 = fig.add_subplot(gs[0, 1])  # Top right: uncertainty
    ax3 = fig.add_subplot(gs[1, 1])  # Bottom right: legend/info

    # Get gap information
    before_idx = largest_gap['before_idx']
    after_idx = largest_gap['after_idx']
    before_frame_id = largest_gap['before'][0]
    after_frame_id = largest_gap['after'][0]
    before_time = largest_gap['before'][1]
    after_time = largest_gap['after'][1]
    gap_duration = largest_gap['gap']

    # Get tracks
    before_tracks = all_tracks[before_idx] if before_idx < len(all_tracks) else []
    after_tracks = all_tracks[after_idx] if after_idx < len(all_tracks) else []

    # Main tracking visualization
    track_data = []

    for track in before_tracks:
        if track is None or not hasattr(track, 'state'):
            continue

        x, y = track.position
        vx, vy = track.velocity

        # Find matching track after gap
        after_track = None
        for at in after_tracks:
            if at is not None and at.id == track.id:
                after_track = at
                break

        track_data.append({
            'id': track.id,
            'before_pos': (x, y),
            'velocity': (vx, vy),
            'after_pos': after_track.position if after_track else None,
            'after_track': after_track
        })

    # Plot with enhanced clarity
    if track_data:
        # Create time points for smooth prediction curves
        t_pred = np.linspace(0, gap_duration, 100)

        for i, td in enumerate(track_data):
            color = plt.cm.Set1(i % 9)

            x0, y0 = td['before_pos']
            vx, vy = td['velocity']

            # Starting position
            ax1.scatter(x0, y0, color=color, s=200, marker='o',
                        edgecolors='black', linewidth=2, zorder=5,
                        label=f"Track {td['id']} - Start")

            # Predicted trajectory
            x_pred = x0 + vx * t_pred
            y_pred = y0 + vy * t_pred
            ax1.plot(x_pred, y_pred, '--', color=color, linewidth=3,
                     alpha=0.7, label=f"Track {td['id']} - Prediction")

            # Add arrows to show direction
            for t in [gap_duration * 0.25, gap_duration * 0.5, gap_duration * 0.75]:
                x_arr = x0 + vx * t
                y_arr = y0 + vy * t
                ax1.annotate('', xy=(x_arr + vx * 0.5, y_arr + vy * 0.5),
                             xytext=(x_arr, y_arr),
                             arrowprops=dict(arrowstyle='->', color=color,
                                             lw=2, alpha=0.5))

            # End position and correction
            if td['after_pos']:
                x1, y1 = td['after_pos']
                ax1.scatter(x1, y1, color=color, s=200, marker='s',
                            edgecolors='black', linewidth=2, zorder=5,
                            label=f"Track {td['id']} - End")

                # Correction arrow
                x_pred_end = x0 + vx * gap_duration
                y_pred_end = y0 + vy * gap_duration
                ax1.annotate('', xy=(x1, y1), xytext=(x_pred_end, y_pred_end),
                             arrowprops=dict(arrowstyle='->', color='red',
                                             lw=3, alpha=0.8))

                # Error distance
                error = np.sqrt((x1 - x_pred_end) ** 2 + (y1 - y_pred_end) ** 2)
                mid_x = (x_pred_end + x1) / 2
                mid_y = (y_pred_end + y1) / 2
                ax1.text(mid_x, mid_y, f'{error:.1f}m',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                         fontsize=10, ha='center')
            else:
                # Lost track
                x_end = x0 + vx * gap_duration
                y_end = y0 + vy * gap_duration
                ax1.scatter(x_end, y_end, color=color, s=200, marker='x',
                            linewidth=3, label=f"Track {td['id']} - Lost")

    else:
        ax1.text(0.5, 0.5, 'No active tracks found',
                 transform=ax1.transAxes, ha='center', va='center',
                 fontsize=16, bbox=dict(boxstyle='round', facecolor='yellow'))

    # Configure main plot
    ax1.set_xlabel('X Position (m)', fontsize=14)
    ax1.set_ylabel('Y Position (m)', fontsize=14)
    ax1.set_title(f'Track Prediction During {gap_duration:.2f}s Time Gap\n'
                  f'Time: {before_time:.2f}s → {after_time:.2f}s '
                  f'(Frames {before_frame_id} → {after_frame_id})',
                  fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axis('equal')

    # Add scale reference
    if track_data:
        xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()
        scale_length = 10  # 10 meters
        scale_x = xlim[0] + (xlim[1] - xlim[0]) * 0.1
        scale_y = ylim[0] + (ylim[1] - ylim[0]) * 0.1
        ax1.plot([scale_x, scale_x + scale_length], [scale_y, scale_y],
                 'k-', linewidth=3)
        ax1.text(scale_x + scale_length / 2, scale_y - 1, '10m',
                 ha='center', va='top', fontsize=10)

    # Uncertainty plot
    if before_tracks:
        initial_variance = 2.0
        if hasattr(before_tracks[0], 'covariance'):
            initial_variance = (before_tracks[0].covariance[0, 0] +
                                before_tracks[0].covariance[1, 1]) / 2

        time_points = np.linspace(0, gap_duration, 50)
        process_noise = 10.0
        variance_growth = initial_variance + process_noise * time_points

        ax2.plot(time_points, np.sqrt(variance_growth), 'r-', linewidth=3)
        ax2.fill_between(time_points, 0, np.sqrt(variance_growth),
                         alpha=0.2, color='red')
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Position σ (m)', fontsize=12)
        ax2.set_title('Uncertainty Growth', fontsize=14)
        ax2.grid(True, alpha=0.3)

    # Information panel
    ax3.axis('off')
    info_text = f"Gap Analysis:\n"
    info_text += f"• Gap Duration: {gap_duration:.2f} seconds\n"
    info_text += f"• Tracks Before Gap: {len(before_tracks)}\n"
    info_text += f"• Tracks After Gap: {len(after_tracks)}\n"
    info_text += f"• Tracks Maintained: {len([td for td in track_data if td['after_pos']])}\n"

    if track_data:
        avg_error = np.mean(
            [np.sqrt((td['after_pos'][0] - (td['before_pos'][0] + td['velocity'][0] * gap_duration)) ** 2 +
                     (td['after_pos'][1] - (td['before_pos'][1] + td['velocity'][1] * gap_duration)) ** 2)
             for td in track_data if td['after_pos']])
        info_text += f"• Avg Prediction Error: {avg_error:.2f} meters"

    ax3.text(0.1, 0.8, info_text, transform=ax3.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'tracking_through_gaps.png', dpi=150)
    plt.close()

    # Create summary of all gaps (enhanced version)
    if len(gap_frames) > 1:
        fig2, (ax4, ax5) = plt.subplots(2, 1, figsize=(12, 8),
                                        gridspec_kw={'height_ratios': [2, 1]})

        gaps = [g['gap'] for g in gap_frames]
        frame_ids = [g['before'][0] for g in gap_frames]
        times = [g['before'][1] for g in gap_frames]

        # Bar chart of gaps
        bars = ax4.bar(range(len(gaps)), gaps, color='blue', alpha=0.7, edgecolor='black')

        # Color code by severity
        for i, (bar, gap) in enumerate(zip(bars, gaps)):
            if gap > 2.0:
                bar.set_color('red')
                bar.set_alpha(0.8)
            elif gap > 1.0:
                bar.set_color('orange')
                bar.set_alpha(0.7)

        ax4.set_xlabel('Gap Index', fontsize=12)
        ax4.set_ylabel('Gap Duration (seconds)', fontsize=12)
        ax4.set_title(f'All Time Gaps > {gap_threshold}s in Tracking Sequence', fontsize=14)

        # Add annotations
        for i, (gap, fid, t) in enumerate(zip(gaps, frame_ids, times)):
            ax4.text(i, gap + 0.05, f'Frame {fid}\n@ {t:.1f}s',
                     ha='center', va='bottom', fontsize=9)

        ax4.grid(True, alpha=0.3, axis='y')

        # Timeline view
        ax5.scatter(times, [1] * len(times), s=100, c=gaps, cmap='hot_r',
                    edgecolors='black', linewidth=1)
        ax5.set_xlabel('Time (seconds)', fontsize=12)
        ax5.set_ylabel('')
        ax5.set_title('Gap Locations in Timeline', fontsize=12)
        ax5.set_ylim(0.5, 1.5)
        ax5.set_yticks([])
        ax5.grid(True, alpha=0.3, axis='x')

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='hot_r',
                                   norm=plt.Normalize(vmin=min(gaps), vmax=max(gaps)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax5, orientation='horizontal', pad=0.1)
        cbar.set_label('Gap Duration (s)', fontsize=11)

        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'all_time_gaps.png', dpi=150)
        plt.close()


def get_actual_kalman_predictions(track, start_time, end_time, dt=0.05):
    """
    Generate actual Kalman filter predictions between two timestamps.

    Args:
        track: Track object with current state
        start_time: Start timestamp
        end_time: End timestamp
        dt: Prediction time step

    Returns:
        List of (timestamp, range, azimuth, is_prediction) tuples
    """
    from radar_tracking.kalman_filter import RadarKalmanFilter
    from radar_tracking.coordinate_transforms import cartesian_to_polar

    predictions = []
    kf = RadarKalmanFilter()

    # Start from track's current state
    current_state = track.state.copy()
    current_covariance = track.covariance.copy()

    # Generate predictions at regular intervals
    current_time = start_time
    while current_time < end_time:
        # Predict next state
        current_state, current_covariance = kf.predict(
            current_state, current_covariance, dt
        )
        current_time += dt

        # Convert to polar coordinates
        x, y = current_state[0], current_state[1]
        range_m, azimuth_rad = cartesian_to_polar(x, y)

        predictions.append((current_time, range_m, azimuth_rad, True))  # True = prediction

    return predictions