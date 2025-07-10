from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from matplotlib import pyplot as plt, gridspec
from radar_tracking import Track, Detection, cartesian_to_polar, euclidean_distance
from visualizations.track_viz import prepare_output_directories
from visualizations.visualization_config import VisualizationConfig as VizConfig


def extract_kalman_data_for_track(track_history: List[Tuple], all_frames: List[int],
                                  all_ground_truth: List[List[Detection]],
                                  association_distance_threshold: float) -> Dict:
    """
    Extract prediction, update, and ground truth data for a specific track.

    Args:
        track_history: List of (timestamp, track, frame_id) tuples
        all_frames: List of frame IDs
        all_ground_truth: List of ground truth detections for each frame
        association_distance_threshold: Max distance for ground truth association

    Returns:
        Dictionary containing extracted data arrays
    """
    prediction_data = []
    update_data = []
    ground_truth_data = []

    for timestamp, track, frame_id in track_history:
        # Extract predictions and updates from state history
        if hasattr(track, 'state_history') and track.state_history:
            for entry in track.state_history:
                if entry['step_type'] == 'prediction':
                    # Prior state (after prediction, before update)
                    state = entry['state']
                    covariance = entry['covariance']
                    pos_uncertainty = entry.get('position_uncertainty', 0)

                    prediction_data.append({
                        'timestamp': entry.get('timestamp', timestamp),
                        'x': float(state[0]),
                        'y': float(state[1]),
                        'vx': float(state[2]),
                        'vy': float(state[3]),
                        'covariance': covariance,
                        'pos_uncertainty': pos_uncertainty,
                        'x_uncertainty': np.sqrt(covariance[0, 0]),
                        'y_uncertainty': np.sqrt(covariance[1, 1])
                    })

                elif entry['step_type'] == 'update':
                    # Posterior state (after update with measurement)
                    state = entry['state']
                    covariance = entry['covariance']
                    pos_uncertainty = entry.get('position_uncertainty', 0)

                    update_data.append({
                        'timestamp': entry.get('timestamp', timestamp),
                        'x': float(state[0]),
                        'y': float(state[1]),
                        'vx': float(state[2]),
                        'vy': float(state[3]),
                        'covariance': covariance,
                        'pos_uncertainty': pos_uncertainty,
                        'x_uncertainty': np.sqrt(covariance[0, 0]),
                        'y_uncertainty': np.sqrt(covariance[1, 1])
                    })

        # Find associated ground truth for this frame
        frame_idx = all_frames.index(frame_id) if frame_id in all_frames else None
        if frame_idx is not None and frame_idx < len(all_ground_truth):
            track_x, track_y = track.position

            closest_gt = None
            min_distance = float('inf')

            for gt in all_ground_truth[frame_idx]:
                gt_x, gt_y = gt.cartesian_pos
                distance = euclidean_distance((track_x, track_y), (gt_x, gt_y))

                if distance <= association_distance_threshold and distance < min_distance:
                    min_distance = distance
                    closest_gt = gt

            if closest_gt is not None:
                gt_x, gt_y = closest_gt.cartesian_pos
                ground_truth_data.append({
                    'timestamp': timestamp,
                    'x': float(gt_x),
                    'y': float(gt_y)
                })

    # Sort all data by timestamp
    prediction_data.sort(key=lambda x: x['timestamp'])
    update_data.sort(key=lambda x: x['timestamp'])
    ground_truth_data.sort(key=lambda x: x['timestamp'])

    return {
        'predictions': prediction_data,
        'updates': update_data,
        'ground_truth': ground_truth_data
    }


def create_kalman_analysis_plot(data: Dict, track_id: int, coordinate: str,
                                output_dir: Path) -> None:
    """
    Create Kalman filter analysis plot for a specific coordinate (x or y).

    Args:
        data: Dictionary containing prediction, update, and ground truth data
        track_id: Track ID for title
        coordinate: 'x' or 'y' coordinate
        output_dir: Output directory path
    """
    # Create figure with layout similar to GitHub code
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Main trajectory plot (top span)
    ax_main = fig.add_subplot(gs[0, :])

    # Extract coordinate-specific data
    coord_label = coordinate.upper()
    coord_unit = "meters"

    # Plot ground truth (True State)
    if data['ground_truth']:
        gt_times = [d['timestamp'] for d in data['ground_truth']]
        gt_coords = [d[coordinate] for d in data['ground_truth']]
        ax_main.plot(gt_times, gt_coords,
                     color=VizConfig.GROUND_TRUTH['color'],
                     linewidth=2,
                     label='True State',
                     marker=VizConfig.GROUND_TRUTH['marker'],
                     markersize=6,
                     alpha=VizConfig.GROUND_TRUTH['alpha'])

    # Plot prior estimates (Predictions)
    if data['predictions']:
        pred_times = [d['timestamp'] for d in data['predictions']]
        pred_coords = [d[coordinate] for d in data['predictions']]
        ax_main.plot(pred_times, pred_coords,
                     color=VizConfig.PREDICTIONS['color'],
                     linestyle=VizConfig.PREDICTIONS['linestyle'],
                     linewidth=VizConfig.PREDICTIONS['linewidth'],
                     label='Prior Estimate',
                     marker=VizConfig.PREDICTIONS['marker'],
                     markersize=4,
                     alpha=VizConfig.PREDICTIONS['alpha'])

    # Plot posterior estimates (Updates/Tracks)
    if data['updates']:
        update_times = [d['timestamp'] for d in data['updates']]
        update_coords = [d[coordinate] for d in data['updates']]
        ax_main.plot(update_times, update_coords,
                     color=VizConfig.TRACKS['color'],
                     linestyle='-.',
                     linewidth=2,
                     label='Posterior Estimate',
                     marker=VizConfig.TRACKS['marker'],
                     markersize=4,
                     alpha=VizConfig.TRACKS['alpha'])

    ax_main.set_title(f'{coord_label} Tracking using Kalman Filter - Track ID {track_id}')
    ax_main.set_xlabel('Time (seconds)')
    ax_main.set_ylabel(f'{coord_label} Position ({coord_unit})')
    ax_main.legend(loc='upper right', fontsize='small')
    ax_main.grid(True, alpha=0.3)

    # Prior error plot (bottom left)
    ax_prior = fig.add_subplot(gs[1, 0])
    if data['predictions'] and data['ground_truth']:
        gt_times_array = np.array([d['timestamp'] for d in data['ground_truth']])
        gt_coords_array = np.array([d[coordinate] for d in data['ground_truth']])

        prior_errors = []
        prior_uncertainties = []
        error_times = []

        for pred in data['predictions']:
            # Find closest ground truth in time
            time_diffs = np.abs(gt_times_array - pred['timestamp'])
            closest_idx = np.argmin(time_diffs)

            if time_diffs[closest_idx] < 0.5:  # Within 0.5 seconds
                error = pred[coordinate] - gt_coords_array[closest_idx]
                uncertainty = pred[f'{coordinate}_uncertainty']

                prior_errors.append(error)
                prior_uncertainties.append(uncertainty)
                error_times.append(pred['timestamp'])

        if prior_errors:
            # Plot prior error
            ax_prior.plot(error_times, prior_errors,
                          color=VizConfig.PREDICTIONS['color'],
                          linewidth=1.5,
                          label='Prior Error')

            # Plot uncertainty bounds
            ax_prior.plot(error_times, prior_uncertainties,
                          color='red',
                          linestyle='--',
                          linewidth=1,
                          label='±σ')
            ax_prior.plot(error_times, [-u for u in prior_uncertainties],
                          color='red',
                          linestyle='--',
                          linewidth=1)

            # Fill uncertainty region
            ax_prior.fill_between(error_times, prior_uncertainties,
                                  [-u for u in prior_uncertainties],
                                  alpha=0.2, color='red')

    ax_prior.set_title(f'Prior Estimation Error ({coord_label})')
    ax_prior.set_xlabel('Time (seconds)')
    ax_prior.set_ylabel('Error (meters)')
    ax_prior.legend(fontsize='small')
    ax_prior.grid(True, alpha=0.3)

    # Posterior error plot (bottom right)
    ax_posterior = fig.add_subplot(gs[1, 1])
    if data['updates'] and data['ground_truth']:
        gt_times_array = np.array([d['timestamp'] for d in data['ground_truth']])
        gt_coords_array = np.array([d[coordinate] for d in data['ground_truth']])

        posterior_errors = []
        posterior_uncertainties = []
        error_times = []

        for update in data['updates']:
            # Find closest ground truth in time
            time_diffs = np.abs(gt_times_array - update['timestamp'])
            closest_idx = np.argmin(time_diffs)

            if time_diffs[closest_idx] < 0.5:  # Within 0.5 seconds
                error = update[coordinate] - gt_coords_array[closest_idx]
                uncertainty = update[f'{coordinate}_uncertainty']

                posterior_errors.append(error)
                posterior_uncertainties.append(uncertainty)
                error_times.append(update['timestamp'])

        if posterior_errors:
            # Plot posterior error
            ax_posterior.plot(error_times, posterior_errors,
                              color=VizConfig.TRACKS['color'],
                              linewidth=1.5,
                              label='Posterior Error')

            # Plot uncertainty bounds
            ax_posterior.plot(error_times, posterior_uncertainties,
                              color='red',
                              linestyle='--',
                              linewidth=1,
                              label='±σ')
            ax_posterior.plot(error_times, [-u for u in posterior_uncertainties],
                              color='red',
                              linestyle='--',
                              linewidth=1)

            # Fill uncertainty region
            ax_posterior.fill_between(error_times, posterior_uncertainties,
                                      [-u for u in posterior_uncertainties],
                                      alpha=0.2, color='red')

    ax_posterior.set_title(f'Posterior Estimation Error ({coord_label})')
    ax_posterior.set_xlabel('Time (seconds)')
    ax_posterior.set_ylabel('Error (meters)')
    ax_posterior.legend(fontsize='small')
    ax_posterior.grid(True, alpha=0.3)

    # Synchronize y-axis limits for error plots
    if ax_prior.get_ylim() and ax_posterior.get_ylim():
        all_errors = []
        if 'prior_errors' in locals():
            all_errors.extend(prior_errors)
        if 'posterior_errors' in locals():
            all_errors.extend(posterior_errors)
        if 'prior_uncertainties' in locals():
            all_errors.extend(prior_uncertainties + [-u for u in prior_uncertainties])
        if 'posterior_uncertainties' in locals():
            all_errors.extend(posterior_uncertainties + [-u for u in posterior_uncertainties])

        if all_errors:
            y_max = max(abs(e) for e in all_errors) * 1.1
            ax_prior.set_ylim(-y_max, y_max)
            ax_posterior.set_ylim(-y_max, y_max)

    # Add statistics text box
    stats_text = f'{coord_label} Coordinate Statistics:\n'
    stats_text += f'Track ID: {track_id}\n'
    stats_text += f'Predictions: {len(data["predictions"])}\n'
    stats_text += f'Updates: {len(data["updates"])}\n'
    stats_text += f'GT Associations: {len(data["ground_truth"])}'

    # Add performance metrics if available
    if data['predictions'] and data['ground_truth'] and 'prior_errors' in locals():
        prior_rmse = np.sqrt(np.mean(np.array(prior_errors) ** 2))
        stats_text += f'\nPrior RMSE: {prior_rmse:.3f}m'

    if data['updates'] and data['ground_truth'] and 'posterior_errors' in locals():
        posterior_rmse = np.sqrt(np.mean(np.array(posterior_errors) ** 2))
        stats_text += f'\nPosterior RMSE: {posterior_rmse:.3f}m'

    ax_main.text(0.02, 0.98, stats_text, transform=ax_main.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                 fontsize=9, verticalalignment='top')

    # Save the plot
    save_path = output_dir / f"kalman_filter_analysis_{coordinate}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Kalman filter {coord_label} analysis plot saved to: {save_path}")


def visualize_kalman_filter_analysis_longest_track(
        all_tracks: List[List[Track]],
        all_ground_truth: List[List[Detection]],
        all_frames: List[int],
        frame_times: List[Tuple[int, float]],
        output_dir: str,
        association_distance_threshold: float = 2.0
):
    """
    Create comprehensive Kalman filter analysis plots for the longest-lived track.
    Generates separate plots for X and Y coordinates showing prior/posterior estimates and errors.

    Args:
        all_tracks: List of tracks for each frame
        all_ground_truth: List of ground truth detections for each frame
        all_frames: List of frame IDs
        frame_times: List of (frame_id, timestamp) tuples
        output_dir: Output directory for saving the plots
        association_distance_threshold: Max distance for ground truth association (meters)
    """
    print("\n" + "=" * 60)
    print("KALMAN FILTER ANALYSIS")
    print("=" * 60)

    prepare_output_directories(output_dir)
    output_path = Path(output_dir)

    frame_to_time = dict(frame_times)

    # Find all track lifespans
    track_lifespans = {}
    for frame_idx, tracks in enumerate(all_tracks):
        timestamp = frame_to_time.get(all_frames[frame_idx], all_frames[frame_idx])
        for track in tracks:
            if track.id not in track_lifespans:
                track_lifespans[track.id] = []
            track_lifespans[track.id].append((timestamp, track, all_frames[frame_idx]))

    if not track_lifespans:
        print("No tracks found for Kalman filter analysis")
        return

    # Select longest track
    longest_track_id, longest_history = max(track_lifespans.items(), key=lambda x: len(x[1]))
    print(f"Analyzing longest track ID {longest_track_id} ({len(longest_history)} frames)")

    # Extract Kalman filter data
    kalman_data = extract_kalman_data_for_track(
        longest_history, all_frames, all_ground_truth, association_distance_threshold
    )

    # Validate data availability
    if not kalman_data['predictions'] and not kalman_data['updates']:
        print("No state history data found for Kalman filter analysis")
        return

    # Print data summary
    print(f"  Prior states (predictions): {len(kalman_data['predictions'])}")
    print(f"  Posterior states (updates): {len(kalman_data['updates'])}")
    print(f"  Ground truth associations: {len(kalman_data['ground_truth'])}")

    # Create plots for both X and Y coordinates
    for coordinate in ['x', 'y']:
        print(f"  Generating {coordinate.upper()} coordinate analysis...")
        create_kalman_analysis_plot(kalman_data, longest_track_id, coordinate, output_path)

    print(f"Kalman filter analysis completed for track {longest_track_id}")
    print("=" * 60)