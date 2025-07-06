import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from radar_tracking import Detection, Track
from copy import deepcopy


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
        show_association_distances: bool = True,
        radar_config: Optional[dict] = None,
):
    """
    Plot one frame in (azimuth_deg, range_m) space with radar coverage overlay and confidence ellipses.
    Now includes a zoomed-in subplot showing only the data region.
    """
    prepare_output_directories(output_dir)

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)

    # Extract radar parameters from config (with fallback defaults)
    if radar_config is None:
        radar_config = {}

    max_range = radar_config.get('max_range', 103.0)
    min_azimuth = radar_config.get('min_azimuth_deg', -90.0)
    max_azimuth = radar_config.get('max_azimuth_deg', 90.0)
    range_buffer = radar_config.get('range_buffer', 10.0)
    azimuth_buffer = radar_config.get('azimuth_buffer_deg', 5.0)

    # Chi-square thresholds for visualization
    chi2_95 = 5.991
    chi2_99 = 9.210
    chi2_99_9 = 13.816

    # Collect data points for zoom calculation
    all_azimuths = []
    all_ranges = []

    if detections:
        all_azimuths.extend([np.degrees(d.azimuth_rad) for d in detections])
        all_ranges.extend([d.range_m for d in detections])

    if ground_truth:
        all_azimuths.extend([np.degrees(d.azimuth_rad) for d in ground_truth])
        all_ranges.extend([d.range_m for d in ground_truth])

    for track in active_tracks:
        range_m, azimuth_rad = track.kalman_polar_position
        all_azimuths.append(np.degrees(azimuth_rad))
        all_ranges.append(range_m)

        # Add prediction state if available
        if hasattr(track, 'predicted_state') and track.predicted_state is not None:
            from radar_tracking.coordinate_transforms import cartesian_to_polar
            pred_range, pred_azimuth = cartesian_to_polar(
                track.predicted_state[0], track.predicted_state[1]
            )
            all_azimuths.append(np.degrees(pred_azimuth))
            all_ranges.append(pred_range)

    # Calculate zoom bounds with minimal borders
    zoom_border_az = 2.0  # degrees
    zoom_border_range = 5.0  # meters

    if all_azimuths and all_ranges:
        zoom_min_az = min(all_azimuths) - zoom_border_az
        zoom_max_az = max(all_azimuths) + zoom_border_az
        zoom_min_range = max(0, min(all_ranges) - zoom_border_range)
        zoom_max_range = max(all_ranges) + zoom_border_range
    else:
        # Fallback if no data
        zoom_min_az, zoom_max_az = -10, 10
        zoom_min_range, zoom_max_range = 0, 50

    # Function to plot data on both axes
    def plot_data_on_axis(ax, is_zoomed=False):
        # Show radar coverage bounds (only on full view)
        if show_coverage_bounds and not is_zoomed:
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
        scatter = None
        if detections:
            az_det = [np.degrees(d.azimuth_rad) for d in detections]
            rng_det = [d.range_m for d in detections]
            conf_det = [d.confidence for d in detections]

            # Create 5 discrete blue bins
            conf_bins = np.digitize(conf_det, bins=[0, 0.4, 0.6, 0.8, 0.9, 1.0]) - 1
            blue_colors = plt.cm.get_cmap('Blues')(np.linspace(0.3, 1.0, 5))  # 5 blue shades
            colors = [blue_colors[bin_idx] for bin_idx in conf_bins]

            scatter = ax.scatter(az_det, rng_det, c=colors, s=50, alpha=0.8,
                                 label='Network Output')

        # Plot ground truth (green X)
        if ground_truth:
            az_gt = [np.degrees(d.azimuth_rad) for d in ground_truth]
            rng_gt = [d.range_m for d in ground_truth]
            ax.scatter(az_gt, rng_gt, c='green', marker='x', s=100, label='Ground Truth')

        # Track whether we've added ellipse labels
        has_update_ellipse = False
        has_pred_ellipse = False
        has_chi2_95 = False
        has_chi2_99 = False
        has_chi2_99_9 = False

        # Plot tracks with multiple threshold ellipses
        for i, track in enumerate(active_tracks):
            range_m, azimuth_rad = track.kalman_polar_position
            az_tr = np.degrees(azimuth_rad)
            rng_tr = range_m

            # Plot track position (update state)
            ax.scatter(az_tr, rng_tr, marker='^', s=50, facecolors='none', edgecolors='red',
                       linewidths=0.8, label='Track Position' if i == 0 else "")

            # Add track ID and distance annotation
            track_text = f"T{track.id}"

            # Show distance information based on association status
            if show_association_distances:
                if hasattr(track, 'has_association_this_frame') and track.has_association_this_frame:
                    # Track was associated - show association distance
                    if hasattr(track, 'last_association_distance') and not np.isnan(track.last_association_distance):
                        distance = track.last_association_distance
                        strategy = getattr(track, 'last_association_strategy', 'unknown')

                        distance_text = f"\nd={distance:.2f}"
                        if strategy == 'mahalanobis_distance':
                            distance_text += " (χ²)"
                        elif 'confidence' in strategy:
                            distance_text += " (conf)"
                        track_text += distance_text
                else:
                    # Track was NOT associated - show nearest detection distance
                    if hasattr(track, 'nearest_detection_distance') and track.nearest_detection_distance is not None:
                        distance = track.nearest_detection_distance
                        strategy = getattr(track, 'last_association_strategy', 'unknown')

                        distance_text = f"\nd={distance:.2f}"
                        if strategy == 'mahalanobis_distance':
                            distance_text += " (χ²✗)"  # X mark to indicate failed gating
                        else:
                            distance_text += " (✗)"
                        track_text += distance_text

            # Position text with smart offset
            if is_zoomed:
                text_offset_az = 0.1
                text_offset_rng = 0.5
                fontsize = 7
            else:
                text_offset_az = 0.5
                text_offset_rng = 2.0
                fontsize = 8

            text_az = az_tr + text_offset_az
            text_rng = rng_tr + text_offset_rng

            # Color-code text based on association status
            if hasattr(track, 'has_association_this_frame') and track.has_association_this_frame:
                text_color = 'darkred'  # Associated track
                edge_color = 'red'
                bg_color = 'white'
            else:
                text_color = 'darkorange'  # Unassociated track
                edge_color = 'orange'
                bg_color = 'lightyellow'

            ax.text(text_az, text_rng, track_text, color=text_color,
                    fontsize=fontsize, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor=bg_color,
                              alpha=0.9,
                              edgecolor=edge_color,
                              linewidth=1),
                    ha='left', va='bottom')

            # Add a thin line connecting text to track point
            if show_association_distances and hasattr(track, 'last_association_distance'):
                if not np.isnan(track.last_association_distance):
                    ax.plot([az_tr, text_az], [rng_tr, text_rng],
                            color='red', linewidth=0.5, alpha=0.6, linestyle=':')

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

            # Draw multiple threshold ellipses if enabled
            if show_confidence_ellipses:
                # Determine ellipse center and covariance to use
                ellipse_center = None
                ellipse_covariance = None

                # Prefer prediction state and covariance if available
                if (hasattr(track, 'predicted_state') and track.predicted_state is not None and
                        hasattr(track, 'predicted_covariance') and track.predicted_covariance is not None):
                    ellipse_center = track.predicted_state[:2]  # [x, y]
                    ellipse_covariance = track.predicted_covariance[:2, :2]  # 2x2 position covariance
                elif hasattr(track, 'covariance') and track.covariance is not None:
                    # Fallback to current position and covariance
                    ellipse_center = track.position
                    ellipse_covariance = track.covariance[:2, :2]

                if ellipse_center is not None and ellipse_covariance is not None:
                    # Chi-square 95% ellipse (solid red)
                    ellipse_95 = create_chi2_confidence_ellipse_polar(
                        ellipse_center, ellipse_covariance, chi2_95,
                        color='red', alpha=0.15, linestyle='-', linewidth=1
                    )
                    if ellipse_95:
                        ax.add_patch(ellipse_95)
                        if not has_chi2_95:
                            ax.scatter([], [], c='red', alpha=0.15, s=80, marker='o',
                                       label=f'χ² 95% ({chi2_95:.3f})')
                            has_chi2_95 = True

                    # Chi-square 99% ellipse (dashed orange)
                    ellipse_99 = create_chi2_confidence_ellipse_polar(
                        ellipse_center, ellipse_covariance, chi2_99,
                        color='orange', alpha=0.12, linestyle='--', linewidth=1.5
                    )
                    if ellipse_99:
                        ax.add_patch(ellipse_99)
                        if not has_chi2_99:
                            ax.scatter([], [], c='orange', alpha=0.12, s=80, marker='s',
                                       label=f'χ² 99% ({chi2_99:.3f})')
                            has_chi2_99 = True

                    # Chi-square 99.9% ellipse (dotted purple)
                    ellipse_99_9 = create_chi2_confidence_ellipse_polar(
                        ellipse_center, ellipse_covariance, chi2_99_9,
                        color='purple', alpha=0.1, linestyle=':', linewidth=2
                    )
                    if ellipse_99_9:
                        ax.add_patch(ellipse_99_9)
                        if not has_chi2_99_9:
                            ax.scatter([], [], c='purple', alpha=0.1, s=80, marker='^',
                                       label=f'χ² 99.9% ({chi2_99_9:.3f})')
                            has_chi2_99_9 = True

        return scatter

    # Plot on both axes
    scatter1 = plot_data_on_axis(ax1, is_zoomed=False)
    scatter2 = plot_data_on_axis(ax2, is_zoomed=True)

    # Set titles and labels
    ax1.set_xlabel("Azimuth (deg)")
    ax1.set_ylabel("Range (m)")
    ax1.set_title(f"Frame {frame_id:06d} - Full Coverage View")

    ax2.set_xlabel("Azimuth (deg)")
    ax2.set_ylabel("Range (m)")
    ax2.set_title(f"Frame {frame_id:06d} - Zoomed Data View")

    # Set axis limits
    display_min_azimuth = min_azimuth - azimuth_buffer
    display_max_azimuth = max_azimuth + azimuth_buffer
    display_max_range = max_range + range_buffer

    ax1.set_xlim(display_max_azimuth, display_min_azimuth)
    ax1.set_ylim(0, display_max_range)

    ax2.set_xlim(zoom_max_az, zoom_min_az)
    ax2.set_ylim(zoom_min_range, zoom_max_range)

    # Add grids
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # Add colorbar for confidence if we have detections
    if detections:
        from matplotlib.colors import ListedColormap, BoundaryNorm
        # Define the actual bin boundaries
        bin_boundaries = [0, 0.4, 0.6, 0.8, 0.9, 1.0]
        blue_colors = plt.cm.get_cmap('Blues')(np.linspace(0.3, 1.0, 5))
        blue_cmap = ListedColormap(blue_colors)
        # Use BoundaryNorm to map the unequal bins correctly
        norm = BoundaryNorm(bin_boundaries, blue_cmap.N)

        sm = plt.cm.ScalarMappable(cmap=blue_cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=[ax1, ax2], label='Confidence',
                            pad=0.02, boundaries=bin_boundaries ,ticks=bin_boundaries)
        cbar.set_ticklabels([str(i) for i in bin_boundaries])

    # Add legend to the first subplot
    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, loc='upper right', fontsize=7, ncol=2)

    # Draw rectangle on full view showing zoom area
    if all_azimuths and all_ranges:
        from matplotlib.patches import Rectangle
        zoom_rect = Rectangle((zoom_min_az, zoom_min_range),
                              zoom_max_az - zoom_min_az,
                              zoom_max_range - zoom_min_range,
                              linewidth=2, edgecolor='black', facecolor='none',
                              linestyle=':', alpha=0.7)
        ax1.add_patch(zoom_rect)
        ax1.text(zoom_min_az, zoom_max_range + 2, 'Zoom Area',
                 fontsize=8, ha='left', style='italic')

    plt.suptitle(f"Mahalanobis Distance Tracking - Frame {frame_id:06d}",
                 fontsize=14, fontweight='bold')

    out_path = Path(output_dir) / f"frame_{frame_id:06d}.jpg"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_chi2_confidence_ellipse_polar(position: Tuple[float, float],
                                         covariance: np.ndarray,
                                         chi2_threshold: float,
                                         color: str = 'red',
                                         alpha: float = 0.3,
                                         linestyle: str = '-',
                                         linewidth: float = 1) -> Optional['Ellipse']:
    """Create chi-square confidence ellipse in polar coordinates for radar display."""
    from matplotlib.patches import Ellipse
    from radar_tracking.coordinate_transforms import cartesian_to_polar

    try:
        # Convert position to polar
        range_m, azimuth_rad = cartesian_to_polar(position[0], position[1])
        azimuth_deg = np.degrees(azimuth_rad)

        # Eigenvalues and eigenvectors of covariance
        eigenvals, eigenvecs = np.linalg.eigh(covariance)

        # Convert eigenvalues to ellipse dimensions using chi-square threshold
        width = 2 * np.sqrt(chi2_threshold * eigenvals[0])
        height = 2 * np.sqrt(chi2_threshold * eigenvals[1])

        # Angle of ellipse
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))

        # Create ellipse in polar coordinates (approximate)
        ellipse = Ellipse((azimuth_deg, range_m),
                          width=np.degrees(width / max(range_m, 1)), height=height,
                          angle=angle, alpha=alpha, facecolor=color,
                          edgecolor=color, linestyle=linestyle, linewidth=linewidth)
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
    if np.any(track_axes):
        track_axes[-1].set_xlabel('Time (seconds)', fontsize=12)
    else:
        main_ax.set_xlabel('Time (seconds)', fontsize=12)

    plt.tight_layout()
    save_path = Path(output_dir) / "detection_per_track_confidence.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def extract_all_predictions_for_track(track_history):
    """Extract all unique prediction entries from a track's lifetime."""
    all_predictions = {}  # timestamp -> (azimuth, range) to avoid duplicates

    for timestamp, track, frame_id in track_history:
        if hasattr(track, 'state_history') and track.state_history:
            for entry in track.state_history:
                if entry['step_type'] == 'prediction' and entry.get('timestamp') is not None:
                    pred_timestamp = entry['timestamp']
                    if pred_timestamp not in all_predictions:
                        from radar_tracking.coordinate_transforms import cartesian_to_polar
                        pred_range, pred_azimuth = cartesian_to_polar(
                            entry['state'][0], entry['state'][1]
                        )
                        pred_azimuth_deg = np.degrees(pred_azimuth)
                        all_predictions[pred_timestamp] = (pred_azimuth_deg, pred_range)

    return [(timestamp, az, rng) for timestamp, (az, rng) in sorted(all_predictions.items())]

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

    # Create frame_id to timestamp mapping
    frame_to_time = dict(frame_times)

    # Get time range for segmentation
    timestamps = [frame_to_time.get(frame_id, frame_id) for frame_id in all_frames]
    min_time = min(timestamps)
    max_time = max(timestamps)
    total_duration = max_time - min_time

    # Create 3D subfolder
    subfolder_3d = Path(output_dir) / "3d_segments"
    subfolder_3d.mkdir(parents=True, exist_ok=True)

    # Collect all data points first
    all_gt_times, all_gt_az, all_gt_rng = [], [], []
    all_det_times, all_det_az, all_det_rng = [], [], []

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
    all_track_data = {}
    track_lifespans = {}

    for frame_idx, frame_id in enumerate(all_frames):
        timestamp = frame_to_time.get(frame_id, frame_id)

        # Ground truth
        for gt in all_ground_truth[frame_idx]:
            all_gt_times.append(timestamp)
            all_gt_az.append(np.degrees(gt.azimuth_rad))
            all_gt_rng.append(gt.range_m)

        # Detections
        for det in all_detections[frame_idx]:
            all_det_times.append(timestamp)
            all_det_az.append(np.degrees(det.azimuth_rad))
            all_det_rng.append(det.range_m)

        # Collect track lifespans
        for track in all_tracks[frame_idx]:
            if track.id not in track_lifespans:
                track_lifespans[track.id] = []
            track_lifespans[track.id].append((timestamp, track, frame_id))

    # Second pass: extract predictions and updates
    for track_id, track_history in track_lifespans.items():
        all_track_data[track_id] = {
            'update_states': [],
            'prediction_states': [],
            'timestamps': []
        }

        # Extract all predictions for this track
        prediction_data = extract_all_predictions_for_track(track_history)
        all_track_data[track_id]['prediction_states'] = prediction_data

        # Extract update states
        for timestamp, track, frame_id in track_history:
            range_m, azimuth_rad = track.kalman_polar_position
            azimuth_deg = np.degrees(azimuth_rad)
            all_track_data[track_id]['update_states'].append((timestamp, azimuth_deg, range_m))
            all_track_data[track_id]['timestamps'].append(timestamp)

    # Function to create a 3D plot for a time segment
    def create_3d_plot(gt_times, gt_az, gt_rng, det_times, det_az, det_rng, track_data,
                       time_start, time_end, filename, title_suffix=""):
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Plot ground truth and detections
        if gt_times:
            ax.scatter(gt_times, gt_az, gt_rng, c='green', marker='x', s=40,
                       alpha=0.7, label='Ground Truth')
        if det_times:
            ax.scatter(det_times, det_az, det_rng, c='blue', s=15,
                       alpha=0.5, label='Detections')

        # Plot tracks with stored states - simplified legend
        legend_entries = ['Ground Truth', 'Detections']
        has_updates = False
        has_predictions = False

        for track_id, data in track_data.items():
            if not data['update_states']:
                continue

            color = track_colors[track_id]

            # Plot update states (solid line with circles)
            times, azimuths, ranges = zip(*data['update_states'])
            # Only add to legend for first track
            update_label = 'Track Updates' if not has_updates else ""
            ax.plot(times, azimuths, ranges, color=color, linewidth=2.5, alpha=0.9,
                    label=update_label, marker='o', markersize=4)
            has_updates = True

            # Plot prediction states (dashed line with triangles)
            if data['prediction_states']:
                pred_times, pred_azimuths, pred_ranges = zip(*data['prediction_states'])
                # Only add to legend for first track
                pred_label = 'Track Predictions' if not has_predictions else ""
                ax.plot(pred_times, pred_azimuths, pred_ranges,
                        color=color, linewidth=1.5, alpha=0.6, linestyle='--',
                        marker='^', markersize=3, label=pred_label)
                has_predictions = True

            # Add track ID at start
            if times:
                ax.text(times[0], azimuths[0], ranges[0], f"T{track_id}",
                        color=color, fontsize=8, fontweight='bold')

        # Add legend entries for what we actually have
        if has_updates:
            legend_entries.append('Track Updates')
        if has_predictions:
            legend_entries.append('Track Predictions')

        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Azimuth (degrees)', fontsize=12)
        ax.set_zlabel('Range (meters)', fontsize=12)
        ax.set_title(f'3D Radar Tracking: Stored Prediction vs Update States{title_suffix}\n'
                     f'Time: {time_start:.1f}s - {time_end:.1f}s\n'
                     'Solid: update states, Dashed: prediction states\n'
                     'Each color represents a unique track ID',
                     fontsize=14)

        # Clean legend with only the entries we want
        handles, labels = ax.get_legend_handles_labels()
        filtered_handles = []
        filtered_labels = []

        for handle, label in zip(handles, labels):
            if label in legend_entries:
                filtered_handles.append(handle)
                filtered_labels.append(label)

        if filtered_handles:
            ax.legend(filtered_handles, filtered_labels, loc='best', fontsize=10)

        ax.view_init(elev=20, azim=45)

        plt.tight_layout()
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close()

    # Create 20-second segments
    segment_duration = 20.0
    num_segments = int(np.ceil(total_duration / segment_duration))

    for segment in range(num_segments):
        segment_start = min_time + segment * segment_duration
        segment_end = min(min_time + (segment + 1) * segment_duration, max_time)

        # Filter data for this time segment
        segment_gt_times = [t for t, az, rng in zip(all_gt_times, all_gt_az, all_gt_rng)
                            if segment_start <= t <= segment_end]
        segment_gt_az = [az for t, az, rng in zip(all_gt_times, all_gt_az, all_gt_rng)
                         if segment_start <= t <= segment_end]
        segment_gt_rng = [rng for t, az, rng in zip(all_gt_times, all_gt_az, all_gt_rng)
                          if segment_start <= t <= segment_end]

        segment_det_times = [t for t, az, rng in zip(all_det_times, all_det_az, all_det_rng)
                             if segment_start <= t <= segment_end]
        segment_det_az = [az for t, az, rng in zip(all_det_times, all_det_az, all_det_rng)
                          if segment_start <= t <= segment_end]
        segment_det_rng = [rng for t, az, rng in zip(all_det_times, all_det_az, all_det_rng)
                           if segment_start <= t <= segment_end]

        # Filter track data for this time segment
        segment_track_data = {}
        for track_id, data in all_track_data.items():
            segment_track_data[track_id] = {
                'update_states': [(t, az, rng) for t, az, rng in data['update_states']
                                  if segment_start <= t <= segment_end],
                'prediction_states': [(t, az, rng) for t, az, rng in data['prediction_states']
                                      if segment_start <= t <= segment_end],
                'timestamps': [t for t in data['timestamps']
                               if segment_start <= t <= segment_end]
            }

        # Create plot for this segment
        filename = subfolder_3d / f"3d_segment_{segment:02d}_{segment_start:.1f}s-{segment_end:.1f}s.png"
        create_3d_plot(segment_gt_times, segment_gt_az, segment_gt_rng,
                       segment_det_times, segment_det_az, segment_det_rng,
                       segment_track_data, segment_start, segment_end, filename,
                       f" (Segment {segment + 1}/{num_segments})")

    # Create full scenario plot and save to summary folder
    summary_folder = Path(output_dir).parent / "summary"
    summary_folder.mkdir(parents=True, exist_ok=True)

    full_scenario_filename = summary_folder / "3d_tracking_full_scenario.png"
    create_3d_plot(all_gt_times, all_gt_az, all_gt_rng,
                   all_det_times, all_det_az, all_det_rng,
                   all_track_data, min_time, max_time, full_scenario_filename,
                   f" (Full Scenario - {total_duration:.1f}s)")

    print(f"Created {num_segments} 3D segments of {segment_duration}s each")
    print(f"Segments saved to: {subfolder_3d}")
    print(f"Full scenario saved to: {full_scenario_filename}")


def visualize_tracking_temporal_evolution(
        all_tracks: List[List[Track]],
        all_frames: List[int],
        frame_times: List[Tuple[int, float]],
        num_tracks: Optional[int] = 2,
        radar_config: Optional[dict] = None,
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

        # Use the helper function to extract predictions
        prediction_data = extract_all_predictions_for_track(track_history)

        # Convert to range-only data for temporal plot
        prediction_times = [item[0] for item in prediction_data]
        prediction_ranges = [item[2] for item in prediction_data]  # item[2] is range

        # Extract update states and detections
        update_data = []
        detection_data = []
        timestamps = []

        for timestamp, track, frame_id in track_history:
            # Update state (current position after measurement)
            range_m, _ = track.kalman_polar_position
            timestamps.append(timestamp)
            update_data.append((timestamp, range_m))

            # Detection data if available
            if track.last_detection:
                detection_data.append((timestamp, track.last_detection.range_m))

        # Sort all data by timestamp
        update_data.sort(key=lambda x: x[0])
        detection_data.sort(key=lambda x: x[0])
        timestamps = np.sort(timestamps)

        # Extract sorted times and ranges
        update_times = [item[0] for item in update_data]
        update_ranges = [item[1] for item in update_data]
        detection_times = [item[0] for item in detection_data]
        detection_ranges = [item[1] for item in detection_data]

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
        if timestamps.any():
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

    plt.suptitle('Temporal Evolution:\n'
                 'Stored Kalman Filter States\n'
                 'Actual prediction and update states from tracking system',
                 fontsize=14, fontweight='bold', y=0.93)

    plt.tight_layout()
    plt.subplots_adjust(top=0.78)

    save_path = Path(output_dir) / "temporal_evolution_stored_states.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def visualize_timing_analysis(frame_times: List[Tuple[int, float]],
                              time_gaps: List[float],
                              output_dir: str):
    """Visualize frame timing and gaps - simplified to show only timeline."""
    prepare_output_directories(output_dir)

    frames, timestamps = zip(*frame_times)

    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()

    # Plot timestamps vs Frame ID
    ax1.plot(frames, timestamps, 'b-', linewidth=1.5, marker='o', markersize=3)
    ax1.set_xlabel('Frame ID')
    ax1.set_ylabel('Timestamp (seconds)')
    ax1.set_title('Frame Timestamps - Corrected Timeline')
    ax1.grid(True, alpha=0.3)

    # Highlight large gaps
    for i in range(1, len(timestamps)):
        gap = timestamps[i] - timestamps[i - 1]
        if gap > 0.5:  # Gaps > 0.5s
            ax1.axvspan(frames[i - 1], frames[i], alpha=0.2, color='red')
            # Add gap duration annotation
            mid_frame = (frames[i - 1] + frames[i]) / 2
            ax1.text(mid_frame, timestamps[i], f'{gap:.2f}s',
                     ha='center', va='bottom', fontsize=8,
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'frame_timestamps_timeline.png', dpi=150)
    plt.close()

