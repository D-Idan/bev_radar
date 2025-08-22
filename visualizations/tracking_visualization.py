"""
Enhanced tracking visualization with improved styling and coordinate systems.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import Optional, Tuple
from utils.util import worldToImage


class TrackingVisualizationTool:
    """Enhanced visualization tool for radar tracking results."""

    def __init__(self, camera_params=None, radar_params=None):
        # Default camera parameters
        self.camera_matrix = np.array([
            [1845.41929, 0.0, 855.802458],
            [0.0, 1788.69210, 607.342667],
            [0.0, 0.0, 1.0]
        ]) if camera_params is None else camera_params['matrix']

        self.image_width, self.image_height = 1920, 1080

        # Radar parameters
        self.radar_params = radar_params or {
            'max_range': 103.0,
            'min_azimuth': -90.0,
            'max_azimuth': 90.0
        }

        # Enhanced styling with improved symbols and sizing
        # Base sizes for radar maps: X=4x base, circles=2x base, triangles=1x base
        base_size = 50
        self.style_config = {
            'labels': {
                'color': 'green',
                'marker': 'x',
                'size_base': base_size * 4,  # 4x base size
                'alpha': 0.9,
                'label': 'Labels',
                'linewidth': 3
            },
            'detections': {
                'color': 'blue',
                'marker': 'o',
                'size_base': base_size * 2,  # 2x base size
                'size_max': base_size * 3,
                'alpha_base': 0.4,
                'alpha_max': 0.8,
                'label': 'Network Output'
            },
            'tracks': {
                'color': 'red',
                'marker': '^',
                'size_base': base_size,  # 1x base size
                'alpha': 0.9,
                'label': 'Tracker State',
                'facecolors': 'none',
                'edgecolors': 'red',
                'linewidth': 1
            },
            'coverage_bounds': {'color': 'gray', 'linestyle': '--', 'alpha': 0.5},
            'out_of_coverage': {'color': 'red', 'alpha': 0.1}
        }

        # Unified colormap for all radar displays
        self.radar_colormap = 'viridis'

    def load_data(self, labels_csv: str, predictions_csv: str, tracking_csv: Optional[str] = None):
        """Load all data sources."""
        new__labels_path = Path(labels_csv).with_name('car_labels.csv')
        labels_df = pd.read_csv(new__labels_path)
        predictions_df = pd.read_csv(predictions_csv)
        tracking_df = pd.read_csv(tracking_csv) if tracking_csv is not None else None
        return labels_df, predictions_df, tracking_df

    def create_simplified_visualization(self, sample_id: int, labels_df: pd.DataFrame,
                                        predictions_df: pd.DataFrame, tracking_df: Optional[pd.DataFrame],
                                        image_path: Path, ra_path: Path) -> plt.Figure:
        """Create simplified 2-panel visualization: camera image and range-azimuth map."""

        # Load data files
        image = cv2.imread(str(image_path))
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ra_map = np.load(ra_path)

        # Filter data for this sample
        sample_predictions = predictions_df[predictions_df['sample_id'] == sample_id]
        sample_tracks = (tracking_df[tracking_df['sample_id'] == sample_id]
                         if tracking_df is not None else None)

        # Get time information
        timestamp, time_gap = self._get_time_info(sample_id, tracking_df, labels_df)

        # Create figure with 1x2 layout for side-by-side plots
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))

        # Enhanced title with time information
        title = f'Radar Tracking Visualization - Sample {sample_id}'
        if timestamp is not None and time_gap is not None:
            title += f' | Time: {timestamp:.3f}s | Δt: {time_gap:.3f}s'

        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Panel 1: Camera image (left side)
        if image is not None:
            sample_labels = labels_df[labels_df['numSample'] == sample_id]
            image_viz = self._annotate_camera_image_simplified(image.copy(), sample_labels, sample_predictions,
                                                               sample_tracks)
            axs[0].imshow(image_viz)
        axs[0].set_title('Camera View', fontweight='bold', fontsize=14)
        axs[0].axis('off')

        # Add legend to camera view
        from matplotlib.patches import Rectangle
        camera_legend = [
            Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='green', linewidth=2, alpha=0.7,
                      label='BOTSort'),
            Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='blue', linewidth=2, alpha=0.7,
                      label='Network Detection'),
            Rectangle((0, 0), 1, 1, facecolor='red', edgecolor='red', linewidth=1, alpha=0.3,
                      label='Corrector')
        ]
        axs[0].legend(handles=camera_legend, loc='upper right', fontsize=12)

        # Panel 2: Range-Azimuth map with limited azimuth range (right side)
        self._create_ra_visualization_limited(axs[1], ra_map, sample_predictions, sample_tracks)
        axs[1].set_title('Range-Azimuth Map (-60° to +60°)', fontweight='bold', fontsize=14)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])

        return fig

    def _annotate_camera_image_simplified(self, image: np.ndarray, labels_df: pd.DataFrame,
                                          predictions_df: pd.DataFrame,
                                          tracks_df: Optional[pd.DataFrame]) -> np.ndarray:
        """Annotate camera image with labels, predictions, and tracks."""

        def draw_simple_bbox(img, bbox, color, thickness=2, fill_color=None, fill_alpha=0.3):
            """Draw bounding box without text labels."""
            x1, y1, x2, y2 = map(int, bbox)

            # Draw filled rectangle if fill_color is provided
            if fill_color is not None:
                # Create overlay for alpha blending
                overlay = img.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)
                cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0, img)

            # Draw border
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            return img

        def add_track_flag(img, bbox, track_id):
            """Add track ID flag similar to BEV visualization."""
            x1, y1, x2, y2 = map(int, bbox)
            flag_x, flag_y = x2 + 5, y1 - 5

            flag_text = f"T{track_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(flag_text, font, font_scale, thickness)

            padding = 1
            bg_x1 = max(0, flag_x - padding)
            bg_y1 = max(0, flag_y - text_height - baseline - padding)
            bg_x2 = min(img.shape[1], flag_x + text_width + padding)
            bg_y2 = min(img.shape[0], flag_y + baseline + padding)

            # Draw white background and red border
            cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
            cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 0, 0), 1)

            # Draw text
            text_y = flag_y - baseline
            cv2.putText(img, flag_text, (flag_x, text_y), font, font_scale, (255, 0, 0), thickness)

            return img

        # Ground truth labels (green) - draw first so they appear behind
        for _, row in labels_df.iterrows():
            # Skip rows with NaN pixel coordinates
            if pd.isna(row['x1_pix']) or pd.isna(row['y1_pix']) or pd.isna(row['x2_pix']) or pd.isna(row['y2_pix']):
                continue
            x1, y1, x2, y2 = int(row['x1_pix']), int(row['y1_pix']), int(row['x2_pix']), int(row['y2_pix'])
            label_id = int(row['ID'])

            # Draw green bounding box
            bbox = (x1, y1, x2, y2)
            image = draw_simple_bbox(image, bbox, (0, 255, 0), 2)  # Green color

            # Add label ID text
            cv2.putText(image, f"BS:{label_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Network predictions (blue) - now with larger boxes (same size as old labels)
        for _, row in predictions_df.iterrows():
            range_m, azimuth_deg = row['range_m'], row['azimuth_deg']
            x = np.sin(np.deg2rad(azimuth_deg)) * range_m
            y = np.cos(np.deg2rad(azimuth_deg)) * range_m

            u1, v1 = worldToImage(-x - 0.9, y, 0)
            u2, v2 = worldToImage(-x + 0.9, y, 1.6)
            u1, v1 = int(u1 / 2), int(v1 / 2)
            u2, v2 = int(u2 / 2), int(v2 / 2)

            bbox = (u1, v1, u2, v2)
            image = draw_simple_bbox(image, bbox, (0, 0, 255), 3)  # Blue, thicker line like old labels

        # Tracker predictions (red) - now with medium boxes (same size as old network output)
        if tracks_df is not None and not tracks_df.empty:
            for _, row in tracks_df.iterrows():
                range_m, azimuth_deg = row['range_m'], row['azimuth_deg']
                track_id = row['track_id']

                x = np.sin(np.deg2rad(azimuth_deg)) * range_m
                y = np.cos(np.deg2rad(azimuth_deg)) * range_m

                u1, v1 = worldToImage(-x - 0.9, y, 0)
                u2, v2 = worldToImage(-x + 0.9, y, 1.6)
                u1, v1 = int(u1 / 2), int(v1 / 2)
                u2, v2 = int(u2 / 2), int(v2 / 2)

                bbox = (u1, v1, u2, v2)
                image = draw_simple_bbox(image, bbox, (255, 0, 0), 1, fill_color=(255, 0, 0),
                                         fill_alpha=0.3)  # Red, medium thickness with fill

                # # Add track flag
                # image = add_track_flag(image, bbox, track_id)

        return image


    def _create_ra_visualization_limited(self, ax: plt.Axes, ra_map: np.ndarray,
                                         predictions_df: pd.DataFrame, tracks_df: Optional[pd.DataFrame]):
        """Create Range-Azimuth visualization limited to -60° to +60° azimuth range."""

        # Crop the RA map to show only -60° to +60° range
        height, width = ra_map.shape

        # Calculate pixel bounds for -60° to +60° range
        # Original range is -90° to +90° (180° total)
        # We want -60° to +60° (120° total), centered
        azimuth_start = -60  # degrees
        azimuth_end = 60  # degrees

        # Convert to pixel coordinates
        x_start = int(((azimuth_start + 90) / 180) * width)  # -60° maps to pixel position
        x_end = int(((azimuth_end + 90) / 180) * width)  # +60° maps to pixel position

        # Crop the image
        ra_cropped = ra_map[:, x_start:x_end]

        # Display cropped map
        ax.imshow(ra_cropped, aspect='auto', origin='lower', alpha=0.8, cmap=self.radar_colormap)
        ax.invert_xaxis()

        # Add coverage bounds for limited range
        self._add_coverage_bounds_ra_limited(ax, ra_cropped.shape)

        # Plot detections (blue circles) - no labels
        for _, row in predictions_df.iterrows():
            azimuth_deg = row['azimuth_deg']
            # Only plot if within -60° to +60° range
            if -60 <= azimuth_deg <= 60:
                x, y = self._convert_ra_coords_limited(row['range_m'], azimuth_deg, ra_cropped.shape)
                confidence = row['confidence']

                style = self.style_config['detections']
                size = style['size_base'] + (style['size_max'] - style['size_base']) * confidence
                alpha = style['alpha_base'] + (style['alpha_max'] - style['alpha_base']) * confidence

                ax.scatter(x, y, c=style['color'], marker=style['marker'], s=size,
                           alpha=alpha, edgecolors='darkblue', linewidth=1, zorder=10)

        # Plot tracks (red triangles) with flags
        if tracks_df is not None and not tracks_df.empty:
            for _, row in tracks_df.iterrows():
                azimuth_deg = row['azimuth_deg']
                # Only plot if within -60° to +60° range
                if -60 <= azimuth_deg <= 60:
                    x, y = self._convert_ra_coords_limited(row['range_m'], azimuth_deg, ra_cropped.shape)
                    track_id = row['track_id']
                    style = self.style_config['tracks']

                    # Plot triangle marker
                    ax.scatter(x, y, marker=style['marker'], s=style['size_base'],
                               alpha=style['alpha'], facecolors=style['facecolors'],
                               edgecolors=style['edgecolors'], linewidth=style['linewidth'] * 1.5,
                               zorder=15)

                    # Add track flag
                    ax.annotate(f"T{track_id}", (x, y), xytext=(5, 5),
                                textcoords='offset points', fontsize=9, color='red',
                                fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                          edgecolor='red', alpha=0.8),
                                zorder=16)

        ax.scatter([], [], c='blue', marker='o', s=100, alpha=0.7,
                   edgecolors='darkblue', label='Network Detection')
        ax.scatter([], [], marker='^', s=100, alpha=0.9, facecolors='none',
                   edgecolors='red', linewidth=2, label='Corrector')

        self._setup_ra_axes_limited(ax, ra_cropped.shape)

        # Simple automatic legend
        ax.legend(loc='upper right', fontsize=12)

    def _convert_ra_coords_limited(self, range_m: float, azimuth_deg: float, ra_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Convert range-azimuth to image coordinates for limited azimuth range (-60° to +60°)."""
        height, width = ra_shape
        y = np.clip(int((range_m / self.radar_params['max_range']) * height), 0, height - 1)
        # Map -60° to +60° to full width
        x = np.clip(int(((azimuth_deg + 60) / 120) * width), 0, width - 1)
        return x, y


    def _add_coverage_bounds_ra_limited(self, ax: plt.Axes, ra_shape: Tuple[int, int]):
        """Add radar coverage bounds to limited RA plot."""
        height, width = ra_shape

        # Max range line
        max_range_y = height - 1
        style = self.style_config['coverage_bounds']
        ax.axhline(y=max_range_y, color=style['color'], linestyle=style['linestyle'],
                   alpha=style['alpha'], linewidth=2)

        # Azimuth bounds (-60° and +60°)
        ax.axvline(x=0, color=style['color'], linestyle=style['linestyle'],
                   alpha=style['alpha'], linewidth=2)
        ax.axvline(x=width - 1, color=style['color'], linestyle=style['linestyle'],
                   alpha=style['alpha'], linewidth=2)


    def _setup_ra_axes_limited(self, ax: plt.Axes, shape: Tuple[int, int]):
        """Setup Range-Azimuth axes for limited range (-60° to +60°)."""
        height, width = shape

        # Azimuth axis for -60° to +60°
        azimuth_ticks = [-60, -30, 0, 30, 60]
        x_ticks = [((a + 60) / 120) * width for a in azimuth_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{a}°" for a in azimuth_ticks])
        ax.set_xlabel("Azimuth Angle (degrees)", fontweight='bold', fontsize=12)

        # Range axis (same as before)
        range_ticks = [0, 20, 40, 60, 80, 100]
        y_ticks = [(r / self.radar_params['max_range']) * height for r in range_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{r}m" for r in range_ticks])
        ax.set_ylabel("Range (m)", fontweight='bold', fontsize=12)

        # Move y-axis (range) to the right side
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    def _get_time_info(self, sample_id: int, tracking_df: Optional[pd.DataFrame], labels_df: pd.DataFrame) -> Tuple[
        Optional[float], Optional[float]]:
        """Get timestamp and time gap for the current sample, with labels as fallback."""

        # First try to get from tracking data (preferred source)
        if tracking_df is not None and not tracking_df.empty:
            sample_tracks = tracking_df[tracking_df['sample_id'] == sample_id]
            if not sample_tracks.empty:
                # Get the first row's timestamp and time_gap (should be same for all tracks in this sample)
                timestamp = sample_tracks.iloc[0]['timestamp']
                time_gap = sample_tracks.iloc[0]['time_gap']
                return timestamp, time_gap

        # Fallback to labels data
        sample_labels = labels_df[labels_df['numSample'] == sample_id]
        if sample_labels.empty:
            return None, None

        # Check if timestamp_us column exists in new format
        if 'timestamp_us' in sample_labels.columns:
            timestamp_us = sample_labels.iloc[0]['timestamp_us']
            timestamp = timestamp_us / 1_000_000.0  # Convert to seconds
        else:
            # New format might not have timestamp, return None
            return None, None

        # Calculate time gap from previous sample
        time_gap = None
        prev_sample_id = sample_id - 1
        prev_labels = labels_df[labels_df['numSample'] == prev_sample_id]
        if not prev_labels.empty:
            prev_timestamp_us = prev_labels.iloc[0]['timestamp_us']
            prev_timestamp = prev_timestamp_us / 1_000_000.0
            time_gap = timestamp - prev_timestamp

        return timestamp, time_gap


def create_tracking_video(data_dir: Path, output_dir: Path, labels_csv: str,
                         predictions_csv: str, tracking_csv: Optional[str] = None,
                         max_samples: Optional[int] = None) -> str:
    """Create video from enhanced tracking visualizations."""

    viz_tool = TrackingVisualizationTool()
    labels_df, predictions_df, tracking_df = viz_tool.load_data(labels_csv, predictions_csv, tracking_csv)

    # Create temporary directory for frames
    frames_dir = output_dir / 'video_frames'
    frames_dir.mkdir(exist_ok=True)

    # Get sample IDs
    sample_ids = sorted(labels_df['numSample'].unique())
    if max_samples:
        sample_ids = sample_ids[:max_samples]

    created_frames = []

    for i, sample_id in enumerate(sample_ids):
        # File paths
        image_path = data_dir / 'camera' / f"image_{sample_id:06d}.jpg"
        rd_path = data_dir / 'radar_RD' / f"rd_{sample_id:06d}.npy"
        ra_path = data_dir / 'radar_RA' / f"ra_{sample_id:06d}.npy"

        if not all(p.exists() for p in [image_path, rd_path, ra_path]):
            continue

        try:
            # Create enhanced visualization
            fig = viz_tool.create_simplified_visualization(
                sample_id, labels_df, predictions_df, tracking_df,
                image_path, ra_path
            )

            # Save frame with higher DPI for better quality
            frame_path = frames_dir / f"frame_{i:06d}.png"
            fig.savefig(frame_path, dpi=90, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            created_frames.append(frame_path)
        except Exception as e:
            print(f"Error processing frame {i} (sample {sample_id}): {str(e)}")
            import traceback
            traceback.print_exc()
            continue

        if i % 10 == 0:
            print(f"Created enhanced frame {i+1}/{len(sample_ids)}")

    # Create video using ffmpeg (if available)
    video_path = output_dir / 'enhanced_tracking_visualization.mp4'
    try:
        import subprocess
        cmd = [
            "ffmpeg", "-y", "-framerate", "5",
            "-i", str(frames_dir / "frame_%06d.png"),
            "-vf", "scale=2160:-2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-b:v", "600k", "-preset", "fast",
            str(video_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Enhanced video created: {video_path}")

        # Clean up frames
        for frame in created_frames:
            frame.unlink()
        frames_dir.rmdir()

    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"FFmpeg not available. Enhanced frame images saved in: {frames_dir}")
        video_path = frames_dir

    return str(video_path)