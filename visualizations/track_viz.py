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
        show_coverage_bounds: bool = True,
        show_confidence_ellipses: bool = True,
        radar_config: Optional[dict] = None,
):
    """
    Plot one frame in (azimuth_deg, range_m) space with radar coverage overlay and confidence ellipses.
    """
    prepare_output_directories(output_dir)

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Extract radar parameters from config (with fallback defaults)
    if radar_config is None:
        radar_config = {}

    max_range = radar_config.get('max_range', 103.0)
    min_azimuth = radar_config.get('min_azimuth_deg', -90.0)
    max_azimuth = radar_config.get('max_azimuth_deg', 90.0)
    range_buffer = radar_config.get('range_buffer', 10.0)
    azimuth_buffer = radar_config.get('azimuth_buffer_deg', 5.0)

    # Show radar coverage bounds
    if show_coverage_bounds:
        # Draw coverage area using config values
        ax.axhline(y=max_range, color='gray', linestyle='--', alpha=0.5, label='Max Range')
        ax.axvline(x=min_azimuth, color='gray', linestyle='--', alpha=0.5, label='Azimuth Limits')
        ax.axvline(x=max_azimuth, color='gray', linestyle='--', alpha=0.5)

        # Calculate display bounds with buffer
        display_max_range = max_range + range_buffer
        display_min_azimuth = min_azimuth - azimuth_buffer
        display_max_azimuth = max_azimuth + azimuth_buffer

        # Shade out-of-coverage areas
        ax.fill_between([min_azimuth, max_azimuth], max_range, display_max_range,
                        color='red', alpha=0.1, label='Out of Coverage')
        ax.fill([display_min_azimuth, min_azimuth, min_azimuth, display_min_azimuth],
                [0, 0, display_max_range, display_max_range], color='red', alpha=0.1)
        ax.fill([max_azimuth, display_max_azimuth, display_max_azimuth, max_azimuth],
                [0, 0, display_max_range, display_max_range], color='red', alpha=0.1)

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

    # Plot tracks with confidence ellipses
    for i, track in enumerate(active_tracks):
        range_m, azimuth_rad = track.kalman_polar_position
        az_tr = np.degrees(azimuth_rad)
        rng_tr = range_m

        # Plot track position (update state)
        ax.scatter(az_tr, rng_tr, marker='^', s=20, facecolors='none', edgecolors='red',
                   linewidths=0.8, label='Track Position' if i == 0 else "")
        ax.text(az_tr + 0.2, rng_tr + 0.2, f"T{track.id}", color='red', fontsize=8)

        # Plot prediction state if available
        if hasattr(track, 'predicted_state') and track.predicted_state is not None:
            from radar_tracking.coordinate_transforms import cartesian_to_polar
            pred_range, pred_azimuth = cartesian_to_polar(
                track.predicted_state[0], track.predicted_state[1]
            )
            pred_az_deg = np.degrees(pred_azimuth)
            ax.scatter(pred_az_deg, pred_range, marker='o', s=15,
                       facecolors='orange', edgecolors='darkorange', alpha=0.7,
                       label='Prediction' if i == 0 else "")

        # Draw confidence ellipses if enabled and covariance data is available
        if show_confidence_ellipses and hasattr(track, 'covariance'):
            # Update state ellipse (solid)
            update_ellipse = create_confidence_ellipse_polar(
                track.position, track.covariance[:2, :2], color='red', alpha=0.2
            )
            if update_ellipse:
                ax.add_patch(update_ellipse)

            # Prediction state ellipse (dashed) if available
            if hasattr(track, 'predicted_covariance') and track.predicted_covariance is not None:
                pred_pos = (track.predicted_state[0], track.predicted_state[1])
                pred_ellipse = create_confidence_ellipse_polar(
                    pred_pos, track.predicted_covariance[:2, :2],
                    color='orange', alpha=0.15, linestyle='--'
                )
                if pred_ellipse:
                    ax.add_patch(pred_ellipse)

    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Range (m)")
    ax.set_title(f"Frame {frame_id:06d} - Enhanced Radar Tracking Visualization")
    ax.legend(loc='upper right', fontsize=8)

    # Use config-based limits
    display_min_azimuth = radar_config.get('min_azimuth_deg', -90.0) - radar_config.get('azimuth_buffer_deg', 5.0) - 5
    display_max_azimuth = radar_config.get('max_azimuth_deg', 90.0) + radar_config.get('azimuth_buffer_deg', 5.0) + 5
    display_max_range = radar_config.get('max_range', 103.0) + radar_config.get('range_buffer', 10.0) + 10

    ax.set_xlim(display_min_azimuth, display_max_azimuth)
    ax.set_ylim(0, display_max_range)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(output_dir) / f"frame_{frame_id:06d}.jpg"
    plt.savefig(out_path, dpi=150)
    plt.close()


def create_confidence_ellipse_polar(position: Tuple[float, float],
                                    covariance: np.ndarray,
                                    color: str = 'red',
                                    alpha: float = 0.3,
                                    linestyle: str = '-') -> Optional['Ellipse']:
    """Create confidence ellipse in polar coordinates for radar display."""
    from matplotlib.patches import Ellipse
    from radar_tracking.coordinate_transforms import cartesian_to_polar

    try:
        # Convert position to polar
        range_m, azimuth_rad = cartesian_to_polar(position[0], position[1])
        azimuth_deg = np.degrees(azimuth_rad)

        # Eigenvalues and eigenvectors of covariance
        eigenvals, eigenvecs = np.linalg.eigh(covariance)

        # Convert eigenvalues to standard deviations (95% confidence)
        chi2_val = 5.991  # 95% confidence
        width = 2 * np.sqrt(chi2_val * eigenvals[0])
        height = 2 * np.sqrt(chi2_val * eigenvals[1])

        # Angle of ellipse
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))

        # Create ellipse in Cartesian space, then transform display coordinates
        # For radar display, we approximate the ellipse in polar coordinates
        ellipse = Ellipse((azimuth_deg, range_m),
                          width=np.degrees(width / range_m), height=height,
                          angle=angle, alpha=alpha, facecolor=color,
                          edgecolor=color, linestyle=linestyle)
        return ellipse
    except:
        return None


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
        frame_times: List[Tuple[int, float]],
        output_dir: str
):
    """Create 3D visualization using actual stored prediction and update states."""
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

    # Store track data with actual stored states
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

        # Tracks with stored prediction/update states
        for track in all_tracks[frame_idx]:
            if track.id not in track_data:
                track_data[track.id] = {
                    'update_states': [],  # After update (measurements)
                    'prediction_states': [],  # After prediction (no measurement)
                    'timestamps': []
                }

            # Use actual Kalman state (after update)
            range_m, azimuth_rad = track.kalman_polar_position
            azimuth_deg = np.degrees(azimuth_rad)

            track_data[track.id]['update_states'].append((timestamp, azimuth_deg, range_m))
            track_data[track.id]['timestamps'].append(timestamp)

            # Add prediction state if available and different from update
            if (hasattr(track, 'predicted_state') and track.predicted_state is not None and
                    hasattr(track, 'state_history') and track.state_history):

                # Get latest prediction from state history
                for entry in reversed(track.state_history):
                    if entry['step_type'] == 'prediction':
                        from radar_tracking.coordinate_transforms import cartesian_to_polar
                        pred_range, pred_azimuth = cartesian_to_polar(
                            entry['state'][0], entry['state'][1]
                        )
                        pred_azimuth_deg = np.degrees(pred_azimuth)
                        track_data[track.id]['prediction_states'].append(
                            (timestamp, pred_azimuth_deg, pred_range)
                        )
                        break

    # Plot ground truth and detections
    if gt_times:
        ax.scatter(gt_times, gt_az, gt_rng, c='green', marker='x', s=40,
                   alpha=0.7, label='Ground Truth')
    if det_times:
        ax.scatter(det_times, det_az, det_rng, c='blue', s=15,
                   alpha=0.5, label='Detections')

    # Plot tracks with stored states
    for track_id, data in track_data.items():
        if not data['update_states']:
            continue

        color = track_colors[track_id]

        # Plot update states (solid line with circles)
        times, azimuths, ranges = zip(*data['update_states'])
        ax.plot(times, azimuths, ranges, color=color, linewidth=2.5, alpha=0.9,
                label=f'Track {track_id} (Updates)', marker='o', markersize=4)

        # Plot prediction states (dashed line with triangles)
        if data['prediction_states']:
            pred_times, pred_azimuths, pred_ranges = zip(*data['prediction_states'])
            ax.plot(pred_times, pred_azimuths, pred_ranges,
                    color=color, linewidth=1.5, alpha=0.6, linestyle='--',
                    marker='^', markersize=3,
                    label=f'Track {track_id} (Predictions)')

        # Add track ID at start
        if times:
            ax.text(times[0], azimuths[0], ranges[0], f"T{track_id}",
                    color=color, fontsize=8, fontweight='bold')

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Azimuth (degrees)', fontsize=12)
    ax.set_zlabel('Range (meters)', fontsize=12)
    ax.set_title('3D Radar Tracking: Stored Prediction vs Update States\n'
                 'Solid: update states, Dashed: prediction states',
                 fontsize=14)

    # Enhanced legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10)

    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    save_path = Path(output_dir) / "3d_tracking_stored_states.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def visualize_tracking_temporal_evolution(
        all_detections: List[List[Detection]],
        all_ground_truth: List[List[Detection]],
        all_tracks: List[List[Track]],
        all_frames: List[int],
        frame_times: List[Tuple[int, float]],
        num_tracks: Optional[int] = 3,
        output_dir: str = "tracking_temporal_evolution.png",
):
    """Create temporal visualization using stored prediction and update states."""
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

        # Extract stored state data
        timestamps = []
        update_ranges = []
        prediction_ranges = []
        update_times = []
        prediction_times = []
        detection_ranges = []
        detection_times = []

        for timestamp, track, frame_id in track_history:
            # Update state (current position after measurement)
            range_m, _ = track.kalman_polar_position
            timestamps.append(timestamp)
            update_ranges.append(range_m)
            update_times.append(timestamp)

            # Detection data if available
            if track.last_detection:
                detection_ranges.append(track.last_detection.range_m)
                detection_times.append(timestamp)

            # Prediction state from stored history if available
            if (hasattr(track, 'state_history') and track.state_history):
                for entry in reversed(track.state_history):
                    if entry['step_type'] == 'prediction' and entry.get('timestamp'):
                        from radar_tracking.coordinate_transforms import cartesian_to_polar
                        pred_range, _ = cartesian_to_polar(
                            entry['state'][0], entry['state'][1]
                        )
                        prediction_ranges.append(pred_range)
                        prediction_times.append(entry['timestamp'])
                        break

        # Plot update states (solid line with circles)
        if update_times and update_ranges:
            ax.plot(update_times, update_ranges, color=colors[i % len(colors)],
                    linewidth=3, alpha=0.9, label=f'Track {track_id} (Kalman Updates)',
                    marker='o', markersize=5, markerfacecolor='white',
                    markeredgewidth=2, zorder=3)

        # Plot prediction states (dashed line with triangles)
        if prediction_times and prediction_ranges:
            ax.plot(prediction_times, prediction_ranges, color=colors[i % len(colors)],
                    linewidth=2, alpha=0.6, linestyle='--',
                    label=f'Track {track_id} (Kalman Predictions)',
                    marker='^', markersize=4, zorder=2)

        # Plot raw detections (scatter)
        if detection_times and detection_ranges:
            ax.scatter(detection_times, detection_ranges,
                       color=colors[i % len(colors)], alpha=0.7, s=30,
                       marker='s', edgecolors='black', linewidth=0.5,
                       label=f'Track {track_id} (Raw Detections)', zorder=4)

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
        ax.set_title(f'Track {track_id}: Stored Kalman States Analysis',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Set appropriate axis limits
        if timestamps:
            ax.set_xlim(min(timestamps) - 0.5, max(timestamps) + 0.5)

        # Add uncertainty analysis if available
        if hasattr(track_history[0][1], 'state_history'):
            uncertainties = []
            uncertainty_times = []
            for timestamp, track, _ in track_history:
                if hasattr(track, 'covariance'):
                    pos_uncertainty = np.sqrt(track.covariance[0, 0] + track.covariance[1, 1])
                    uncertainties.append(pos_uncertainty)
                    uncertainty_times.append(timestamp)

            if uncertainties:
                # Add uncertainty as shaded area
                ax2 = ax.twinx()
                ax2.plot(uncertainty_times, uncertainties, 'gray', alpha=0.5,
                         linewidth=1, label='Position Uncertainty')
                ax2.set_ylabel('Uncertainty (m)', fontsize=10, color='gray')
                ax2.tick_params(axis='y', labelcolor='gray')

        # Add track statistics
        num_updates = len(update_times)
        num_predictions = len(prediction_times)
        track_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0

        stats_text = (f'Updates: {num_updates}\n'
                      f'Predictions: {num_predictions}\n'
                      f'Duration: {track_duration:.1f}s')

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                fontsize=9, verticalalignment='top')

    # Set x-label on bottom subplot
    if axes.any():
        axes[-1].set_xlabel('Time (seconds)', fontsize=12)

    plt.suptitle('Temporal Evolution: Stored Kalman Filter States\n'
                 'Actual prediction and update states from tracking system',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    save_path = Path(output_dir) / "temporal_evolution_stored_states.png"
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
                                   output_dir: str,
                                   radar_config: Optional[dict] = None):
    """Visualize tracking during gaps using stored prediction states."""
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

    fig = plt.figure(figsize=(16, 10))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, width_ratios=[3, 1], height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[:, 0])  # Left: main tracking plot
    ax2 = fig.add_subplot(gs[0, 1])  # Top right: uncertainty
    ax3 = fig.add_subplot(gs[1, 1])  # Bottom right: info

    # Get gap information
    before_idx = largest_gap['before_idx']
    after_idx = largest_gap['after_idx']
    gap_duration = largest_gap['gap']
    before_time = largest_gap['before'][1]
    after_time = largest_gap['after'][1]

    # Get tracks
    before_tracks = all_tracks[before_idx] if before_idx < len(all_tracks) else []
    after_tracks = all_tracks[after_idx] if after_idx < len(all_tracks) else []

    # Analyze tracks using stored states
    track_analysis = []

    for track in before_tracks:
        if track is None or not hasattr(track, 'state'):
            continue

        # Get stored prediction and update states
        prediction_state = None
        update_state = track.state.copy()
        uncertainty_before = None
        uncertainty_after = None

        # Extract prediction state if available
        if (hasattr(track, 'predicted_state') and track.predicted_state is not None):
            prediction_state = track.predicted_state.copy()

        # Get uncertainty from covariance
        if hasattr(track, 'covariance'):
            uncertainty_before = np.sqrt(track.covariance[0, 0] + track.covariance[1, 1])
        if (hasattr(track, 'predicted_covariance') and track.predicted_covariance is not None):
            pred_uncertainty = np.sqrt(track.predicted_covariance[0, 0] + track.predicted_covariance[1, 1])

        # Find matching track after gap
        after_track = None
        for at in after_tracks:
            if at is not None and at.id == track.id:
                after_track = at
                if hasattr(at, 'covariance'):
                    uncertainty_after = np.sqrt(at.covariance[0, 0] + at.covariance[1, 1])
                break

        track_analysis.append({
            'id': track.id,
            'before_pos': track.position,
            'velocity': track.velocity,
            'prediction_pos': (prediction_state[0], prediction_state[1]) if prediction_state is not None else None,
            'after_pos': after_track.position if after_track else None,
            'uncertainty_before': uncertainty_before,
            'uncertainty_after': uncertainty_after,
            'has_prediction': prediction_state is not None,
            'survived_gap': after_track is not None
        })

    # Plot tracking analysis
    colors = plt.cm.Set1(np.arange(len(track_analysis)))

    for i, analysis in enumerate(track_analysis):
        color = colors[i % len(colors)]

        # Plot before position
        x0, y0 = analysis['before_pos']
        ax1.scatter(x0, y0, color=color, s=150, marker='o',
                    edgecolors='black', linewidth=2, zorder=5,
                    label=f"Track {analysis['id']} - Before")

        # Plot prediction if available
        if analysis['prediction_pos']:
            px, py = analysis['prediction_pos']
            ax1.scatter(px, py, color=color, s=100, marker='d',
                        edgecolors='black', linewidth=1, alpha=0.7, zorder=4,
                        label=f"Track {analysis['id']} - Prediction")

            # Arrow from update to prediction
            ax1.annotate('', xy=(px, py), xytext=(x0, y0),
                         arrowprops=dict(arrowstyle='->', color=color,
                                         lw=2, alpha=0.6, linestyle='--'))

        # Plot after position if track survived
        if analysis['after_pos']:
            x1, y1 = analysis['after_pos']
            ax1.scatter(x1, y1, color=color, s=150, marker='s',
                        edgecolors='black', linewidth=2, zorder=5,
                        label=f"Track {analysis['id']} - After")

            # Calculate and show tracking accuracy
            if analysis['prediction_pos']:
                px, py = analysis['prediction_pos']
                error = np.sqrt((x1 - px) ** 2 + (y1 - py) ** 2)
                # Correction arrow from prediction to actual
                ax1.annotate('', xy=(x1, y1), xytext=(px, py),
                             arrowprops=dict(arrowstyle='->', color='red',
                                             lw=2, alpha=0.8))
                # Error annotation
                mid_x, mid_y = (px + x1) / 2, (py + y1) / 2
                ax1.text(mid_x, mid_y, f'{error:.1f}m error',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                         fontsize=9, ha='center')
            else:
                # Direct connection if no prediction stored
                ax1.plot([x0, x1], [y0, y1], color=color, linewidth=2,
                         alpha=0.7, linestyle=':')

        # Plot uncertainty ellipses if available
        if analysis['uncertainty_before']:
            from matplotlib.patches import Circle
            circle = Circle((x0, y0), analysis['uncertainty_before'],
                            color=color, alpha=0.2, fill=True)
            ax1.add_patch(circle)

    # Configure main plot
    ax1.set_xlabel('X Position (m)', fontsize=14)
    ax1.set_ylabel('Y Position (m)', fontsize=14)
    ax1.set_title(f'Stored State Analysis During {gap_duration:.2f}s Gap\n'
                  f'Using Actual Kalman Filter States',
                  fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Uncertainty evolution plot
    if any(t['uncertainty_before'] for t in track_analysis):
        track_ids = [t['id'] for t in track_analysis if t['uncertainty_before']]
        before_uncertainties = [t['uncertainty_before'] for t in track_analysis if t['uncertainty_before']]
        after_uncertainties = [t['uncertainty_after'] if t['uncertainty_after'] else t['uncertainty_before'] * 2
                               for t in track_analysis if t['uncertainty_before']]

        x_pos = np.arange(len(track_ids))
        ax2.bar(x_pos - 0.2, before_uncertainties, 0.4, label='Before Gap', alpha=0.7)
        ax2.bar(x_pos + 0.2, after_uncertainties, 0.4, label='After Gap', alpha=0.7)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'T{tid}' for tid in track_ids])
        ax2.set_ylabel('Position σ (m)', fontsize=12)
        ax2.set_title('Uncertainty Growth', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Information panel
    ax3.axis('off')

    # Calculate statistics
    num_tracks_before = len(track_analysis)
    num_survived = sum(1 for t in track_analysis if t['survived_gap'])
    num_with_predictions = sum(1 for t in track_analysis if t['has_prediction'])

    if track_analysis and any(t['after_pos'] and t['prediction_pos'] for t in track_analysis):
        errors = [np.sqrt((t['after_pos'][0] - t['prediction_pos'][0]) ** 2 +
                          (t['after_pos'][1] - t['prediction_pos'][1]) ** 2)
                  for t in track_analysis
                  if t['after_pos'] and t['prediction_pos']]
        avg_error = np.mean(errors) if errors else 0
    else:
        avg_error = 0

    info_text = f"Stored State Analysis:\n"
    info_text += f"• Gap Duration: {gap_duration:.2f}s\n"
    info_text += f"• Tracks Before: {num_tracks_before}\n"
    info_text += f"• Tracks Survived: {num_survived}\n"
    info_text += f"• With Stored Predictions: {num_with_predictions}\n"
    info_text += f"• Avg Prediction Error: {avg_error:.2f}m\n"
    info_text += f"• Survival Rate: {100 * num_survived / max(num_tracks_before, 1):.1f}%"

    ax3.text(0.1, 0.8, info_text, transform=ax3.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'stored_states_gap_analysis.png', dpi=150)
    plt.close()

    print(f"Gap analysis completed using stored Kalman states")
    print(f"  - Analyzed {num_tracks_before} tracks through {gap_duration:.2f}s gap")
    print(f"  - {num_with_predictions} tracks had stored prediction states")
    print(f"  - {num_survived} tracks survived the gap")