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
        labels_df = pd.read_csv(labels_csv, sep='\t|,', engine='python')
        predictions_df = pd.read_csv(predictions_csv)
        tracking_df = pd.read_csv(tracking_csv) if tracking_csv is not None else None
        return labels_df, predictions_df, tracking_df

    def create_comprehensive_visualization(self, sample_id: int, labels_df: pd.DataFrame,
                                         predictions_df: pd.DataFrame, tracking_df: Optional[pd.DataFrame],
                                         image_path: Path, rd_path: Path, ra_path: Path) -> plt.Figure:
        """Create comprehensive 4-panel visualization with enhanced styling."""

        # Load data files
        image = cv2.imread(str(image_path))
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rd_map = np.load(rd_path)
        ra_map = np.load(ra_path)

        # Filter data for this sample
        sample_labels = labels_df[labels_df['numSample'] == sample_id]
        sample_predictions = predictions_df[predictions_df['sample_id'] == sample_id]
        sample_tracks = (tracking_df[tracking_df['sample_id'] == sample_id]
                        if tracking_df is not None else None)

        # Create figure with improved layout - add more space at bottom for legend
        fig, axs = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(f'Radar Tracking Visualization - Sample {sample_id}', fontsize=18, fontweight='bold')

        # Panel 1: Camera image with simplified bounding boxes
        if image is not None:
            image_viz = self._annotate_camera_image(image.copy(), sample_labels,
                                                   sample_predictions, sample_tracks)
            axs[0, 0].imshow(image_viz)
        axs[0, 0].set_title('Camera View', fontweight='bold', fontsize=14)
        axs[0, 0].axis('off')
        # Add legend to camera view
        from matplotlib.patches import Rectangle
        camera_legend = [
            Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.7, label='Labels'),
            Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.7, label='Network Output'),
            Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.7, label='Tracker State')
        ]
        axs[0, 0].legend(handles=camera_legend, loc='upper right', fontsize=10)

        # Panel 2: Fixed BEV with proper coordinate system
        self._create_bev_visualization(axs[0, 1], ra_map, sample_labels,
                                     sample_predictions, sample_tracks)
        axs[0, 1].set_title('Bird\'s Eye View (Cartesian Coordinates)', fontweight='bold', fontsize=14)

        # Panel 3: Range-Doppler map with unified colormap
        self._create_rd_visualization(axs[1, 0], rd_map)
        axs[1, 0].set_title('Range-Doppler Map', fontweight='bold', fontsize=14)

        # Panel 4: Range-Azimuth map with unified colormap
        self._create_ra_visualization(axs[1, 1], ra_map, sample_labels,
                                    sample_predictions, sample_tracks)
        axs[1, 1].set_title('Range-Azimuth Map', fontweight='bold', fontsize=14)

        # Add unified legend with improved styling
        self._add_unified_legend(fig)

        # Adjust layout to prevent legend overlap
        plt.tight_layout(rect=[0, 0.05, 1, 0.96], h_pad=3.0)  # h_pad increases vertical padding

        return fig

    def _annotate_camera_image(self, image: np.ndarray, labels_df: pd.DataFrame,
                              predictions_df: pd.DataFrame, tracks_df: Optional[pd.DataFrame]) -> np.ndarray:
        """Annotate camera image with simplified bounding boxes (no text labels)."""

        def draw_simple_bbox(img, bbox, color, thickness=2):
            """Draw bounding box without text labels."""
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            return img

        def add_track_flag(img, bbox, track_id):
            """Add track ID flag similar to BEV visualization."""
            x1, y1, x2, y2 = map(int, bbox)
            # Position flag at top-left corner of bounding box
            flag_x, flag_y = x2 + 5, y1 - 5  # 5 pixels right of box, 5 pixels above

            # Create flag text
            flag_text = f"T{track_id}"

            # Get text size for background rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(flag_text, font, font_scale, thickness)

            # Draw background rectangle (white with red border)
            padding = 1
            bg_x1 = flag_x - padding
            bg_y1 = flag_y - text_height - baseline - padding
            bg_x2 = flag_x + text_width + padding
            bg_y2 = flag_y + baseline + padding

            # Ensure flag stays within image bounds
            bg_x1 = max(0, bg_x1)
            bg_y1 = max(0, bg_y1)
            bg_x2 = min(img.shape[1], bg_x2)
            bg_y2 = min(img.shape[0], bg_y2)

            # Draw white background
            cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
            # Draw red border
            cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 0, 0), 1)

            # Draw text
            text_y = flag_y - baseline
            cv2.putText(img, flag_text, (flag_x, text_y), font, font_scale, (255, 0, 0), thickness)

            return img

        # Labels (green)
        for _, row in labels_df.iterrows():
            scale_w, scale_h = self._get_scale_factor(image)
            bbox = (row['x1_pix'] * scale_w, row['y1_pix'] * scale_h,
                   row['x2_pix'] * scale_w, row['y2_pix'] * scale_h)
            image = draw_simple_bbox(image, bbox, (0, 255, 0), 5)  # Green in RGB

        # Network predictions (blue)
        for _, row in predictions_df.iterrows():
            range_m, azimuth_deg = row['range_m'], row['azimuth_deg']
            x = np.sin(np.deg2rad(azimuth_deg)) * range_m
            y = np.cos(np.deg2rad(azimuth_deg)) * range_m

            u1, v1 = worldToImage(-x - 0.9, y, 0)
            u2, v2 = worldToImage(-x + 0.9, y, 1.6)
            u1, v1 = int(u1 / 2), int(v1 / 2)
            u2, v2 = int(u2 / 2), int(v2 / 2)

            bbox = (u1, v1, u2, v2)
            image = draw_simple_bbox(image, bbox, (0, 0, 255), 3)  # Blue in RGB

        # Tracker predictions (red)
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
                image = draw_simple_bbox(image, bbox, (255, 0, 0), 1)  # Red in RGB

                # Add track flag
                image = add_track_flag(image, bbox, track_id)

        return image

    def _create_bev_visualization(self, ax: plt.Axes, ra_map: np.ndarray,
                                  labels_df: pd.DataFrame, predictions_df: pd.DataFrame,
                                  tracks_df: Optional[pd.DataFrame]):
        """Create BEV visualization using the working polar transform method."""

        # Create properly oriented BEV background map
        bev_img = self._convert_ra_to_bev_fixed(ra_map)
        max_range = self.radar_params['max_range']

        # Display with correct coordinate system - flip x-axis to have positive on right
        ax.imshow(bev_img, extent=[max_range, -max_range, 0, max_range],
                  origin='lower', alpha=0.7, cmap=self.radar_colormap)

        # Add radar coverage visualization
        self._add_coverage_bounds_bev(ax)

        # Plot annotations with proper layering
        # Layer 1: Labels (green X - bottom layer but largest)
        for _, row in labels_df.iterrows():
            x, y = self._polar_to_cartesian(row['radar_R_m'], row['radar_A_deg'])
            style = self.style_config['labels']
            ax.scatter(x, y, c=style['color'], marker=style['marker'], s=style['size_base'],
                       alpha=style['alpha'], linewidth=style['linewidth'], zorder=5)

        # Layer 2: Detections (blue circles - middle layer)
        for _, row in predictions_df.iterrows():
            x, y = self._polar_to_cartesian(row['range_m'], row['azimuth_deg'])
            confidence = row['confidence']

            # Scale size based on confidence
            style = self.style_config['detections']
            size = style['size_base'] + (style['size_max'] - style['size_base']) * confidence
            alpha = style['alpha_base'] + (style['alpha_max'] - style['alpha_base']) * confidence

            ax.scatter(x, y, c=style['color'], marker=style['marker'], s=size,
                       alpha=alpha, edgecolors='darkblue', linewidth=1, zorder=10)

        # Layer 3: Tracks (red triangles - top layer)
        if tracks_df is not None and not tracks_df.empty:
            for _, row in tracks_df.iterrows():
                x, y = self._polar_to_cartesian(row['range_m'], row['azimuth_deg'])
                style = self.style_config['tracks']
                ax.scatter(x, y, marker=style['marker'], s=style['size_base'],
                           alpha=style['alpha'], facecolors=style['facecolors'],
                           edgecolors=style['edgecolors'], linewidth=style['linewidth']*1.5,
                           zorder=15)  # Highest zorder
                # Add track ID annotation
                ax.annotate(f"T{row['track_id']}", (x, y), xytext=(5, 5),
                            textcoords='offset points', fontsize=9, color='red',
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                      edgecolor='red', alpha=0.8),
                            zorder=16)

        self._setup_bev_axes_fixed(ax)

        # flip the x - axis
        ax.invert_xaxis()

    def _create_rd_visualization(self, ax: plt.Axes, rd_map: np.ndarray):
        """Create Range-Doppler visualization with unified colormap."""
        ax.imshow(rd_map, aspect='auto', origin='lower', cmap=self.radar_colormap)
        ax.invert_xaxis()
        self._setup_rd_axes(ax, rd_map.shape)

    def _create_ra_visualization(self, ax: plt.Axes, ra_map: np.ndarray,
                               labels_df: pd.DataFrame, predictions_df: pd.DataFrame,
                               tracks_df: Optional[pd.DataFrame]):
        """Create Range-Azimuth visualization with unified colormap and enhanced symbols."""

        # Display background map with unified colormap
        ax.imshow(ra_map, aspect='auto', origin='lower', alpha=0.8, cmap=self.radar_colormap)
        ax.invert_xaxis()

        # Add coverage bounds
        self._add_coverage_bounds_ra(ax, ra_map.shape)

        # Plot in proper order with correct zorder
        # Layer 1: Labels (green X - bottom but largest)
        for _, row in labels_df.iterrows():
            x, y = self._convert_ra_coords(row['radar_R_m'], row['radar_A_deg'], ra_map.shape)
            style = self.style_config['labels']
            ax.scatter(x, y, c=style['color'], marker=style['marker'], s=style['size_base'],
                      alpha=style['alpha'], linewidth=style['linewidth'], zorder=5)

        # Layer 2: Detections (blue circles)
        for _, row in predictions_df.iterrows():
            x, y = self._convert_ra_coords(row['range_m'], row['azimuth_deg'], ra_map.shape)
            confidence = row['confidence']

            # Scale size and alpha based on confidence
            style = self.style_config['detections']
            size = style['size_base'] + (style['size_max'] - style['size_base']) * confidence
            alpha = style['alpha_base'] + (style['alpha_max'] - style['alpha_base']) * confidence

            ax.scatter(x, y, c=style['color'], marker=style['marker'], s=size,
                      alpha=alpha, edgecolors='darkblue', linewidth=1, zorder=10)

        # Layer 3: Tracks (red hollow triangles - top layer)
        if tracks_df is not None and not tracks_df.empty:
            for _, row in tracks_df.iterrows():
                x, y = self._convert_ra_coords(row['range_m'], row['azimuth_deg'], ra_map.shape)
                style = self.style_config['tracks']
                ax.scatter(x, y, marker=style['marker'], s=style['size_base'],
                          alpha=style['alpha'], facecolors=style['facecolors'],
                          edgecolors=style['edgecolors'], linewidth=style['linewidth']*1.5,
                          zorder=15)  # Highest zorder

        self._setup_ra_axes(ax, ra_map.shape)

    def _convert_ra_to_bev_fixed(self, ra_map: np.ndarray) -> np.ndarray:
        """Convert Range-Azimuth map to BEV using polarTransform (working method)."""
        from polarTransform import convertToCartesianImage

        # 1. Normalize & make proper format
        if ra_map.dtype != np.uint8:
            ra_norm = ((ra_map - ra_map.min()) /
                       (ra_map.max() - ra_map.min()) * 255).astype(np.uint8)
        else:
            ra_norm = ra_map.copy()

        # 2. Polar→Cartesian (BEV)
        ra_for_polar = ra_norm.T
        num_range_bins, num_az_bins = ra_norm.shape
        RA_cartesian, _ = convertToCartesianImage(
            ra_for_polar,
            useMultiThreading=True,
            initialAngle=-np.pi / 2,
            finalAngle=+np.pi / 2,
            order=1,
            hasColor=False,
            finalRadius=num_range_bins
        )

        # Fix orientation: flip vertically to correct upside down
        bev = RA_cartesian
        # bev = cv2.flip(RA_cartesian, flipCode=1)  # Flip vertically
        bev = cv2.rotate(bev, cv2.ROTATE_90_CLOCKWISE)
        bev = cv2.resize(bev, dsize=(400, 512))

        return bev

    def _polar_to_cartesian(self, range_m: float, azimuth_deg: float) -> Tuple[float, float]:
        """Convert polar coordinates to Cartesian with proper radar convention."""
        azimuth_rad = np.deg2rad(azimuth_deg)
        # Standard radar convention: 0° = North (forward), positive clockwise
        x = range_m * np.sin(azimuth_rad)  # Lateral (positive = right)
        y = range_m * np.cos(azimuth_rad)  # Forward (positive = forward)
        return x, y

    def _convert_ra_coords(self, range_m: float, azimuth_deg: float, ra_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Convert range-azimuth to image coordinates."""
        height, width = ra_shape
        y = np.clip(int((range_m / self.radar_params['max_range']) * height), 0, height - 1)
        x = np.clip(int(((azimuth_deg + 90) / 180) * width), 0, width - 1)
        return x, y

    def _add_coverage_bounds_bev(self, ax: plt.Axes):
        """Add radar coverage bounds to BEV plot."""
        max_range = self.radar_params['max_range']

        # Draw range circle
        circle = plt.Circle((0, 0), max_range, fill=False,
                           color=self.style_config['coverage_bounds']['color'],
                           linestyle=self.style_config['coverage_bounds']['linestyle'],
                           alpha=self.style_config['coverage_bounds']['alpha'], linewidth=2)
        ax.add_patch(circle)

        # Draw azimuth sector boundaries
        min_az = self.radar_params['min_azimuth']
        max_az = self.radar_params['max_azimuth']

        # Left boundary
        x1 = max_range * np.sin(np.deg2rad(min_az))
        y1 = max_range * np.cos(np.deg2rad(min_az))
        ax.plot([0, x1], [0, y1], color=self.style_config['coverage_bounds']['color'],
                linestyle=self.style_config['coverage_bounds']['linestyle'],
                alpha=self.style_config['coverage_bounds']['alpha'], linewidth=2)

        # Right boundary
        x2 = max_range * np.sin(np.deg2rad(max_az))
        y2 = max_range * np.cos(np.deg2rad(max_az))
        ax.plot([0, x2], [0, y2], color=self.style_config['coverage_bounds']['color'],
                linestyle=self.style_config['coverage_bounds']['linestyle'],
                alpha=self.style_config['coverage_bounds']['alpha'], linewidth=2)

    def _add_coverage_bounds_ra(self, ax: plt.Axes, ra_shape: Tuple[int, int]):
        """Add radar coverage bounds to RA plot."""
        height, width = ra_shape

        # Max range line
        max_range_y = height - 1  # Top of the image
        style = self.style_config['coverage_bounds']
        ax.axhline(y=max_range_y, color=style['color'], linestyle=style['linestyle'],
                  alpha=style['alpha'], linewidth=2)

        # Azimuth bounds
        min_az_x = 0  # Left edge
        max_az_x = width - 1  # Right edge
        ax.axvline(x=min_az_x, color=style['color'], linestyle=style['linestyle'],
                  alpha=style['alpha'], linewidth=2)
        ax.axvline(x=max_az_x, color=style['color'], linestyle=style['linestyle'],
                  alpha=style['alpha'], linewidth=2)

    def _setup_bev_axes_fixed(self, ax: plt.Axes):
        """Setup BEV axes with fixed coordinate system."""
        max_range = self.radar_params['max_range']
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(0, max_range)
        ax.set_xlabel('Lateral Distance (m)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Forward Distance (m)', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Add more detailed tick marks
        x_ticks = np.arange(-100, 101, 20)
        y_ticks = np.arange(0, max_range+1, 20)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

    def _setup_rd_axes(self, ax: plt.Axes, shape: Tuple[int, int]):
        """Setup Range-Doppler axes."""
        height, width = shape

        # Doppler axis
        doppler_ticks = np.linspace(-width // 2 * 0.1, width // 2 * 0.1, 5)
        x_ticks = doppler_ticks / 0.1 + width / 2
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{t:.1f}" for t in doppler_ticks])
        ax.set_xlabel("Doppler Velocity (m/s)", fontweight='bold', fontsize=12)

        # Range axis
        range_ticks = [0, 20, 40, 60, 80, 100]
        y_ticks = [(r / self.radar_params['max_range']) * height for r in range_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{r}m" for r in range_ticks])
        ax.set_ylabel("Range (m)", fontweight='bold', fontsize=12)

    def _setup_ra_axes(self, ax: plt.Axes, shape: Tuple[int, int]):
        """Setup Range-Azimuth axes."""
        height, width = shape

        # Azimuth axis
        azimuth_ticks = [-90, -45, 0, 45, 90]
        x_ticks = [((a + 90) / 180) * width for a in azimuth_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{a}°" for a in azimuth_ticks])
        ax.set_xlabel("Azimuth Angle (degrees)", fontweight='bold', fontsize=12)

        # Range axis
        range_ticks = [0, 20, 40, 60, 80, 100]
        y_ticks = [(r / self.radar_params['max_range']) * height for r in range_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{r}m" for r in range_ticks])
        ax.set_ylabel("Range (m)", fontweight='bold', fontsize=12)

    def _add_unified_legend(self, fig: plt.Figure):
        """Add unified legend with enhanced styling."""
        legend_elements = []

        # Labels (green X with no line)
        legend_elements.append(plt.Line2D([0], [0], marker='x', color='w',
                                          markerfacecolor='green', markersize=12,
                                          markeredgecolor='green', markeredgewidth=3,
                                          linestyle='none', label='Labels'))

        # Network Output (blue circles)
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor='blue', markersize=10,
                                         alpha=0.7, markeredgecolor='darkblue',
                                         linestyle='none', label='Network Output'))

        # Tracker State (red hollow triangles)
        legend_elements.append(plt.Line2D([0], [0], marker='^', color='w',
                                         markerfacecolor='none', markersize=12,
                                         markeredgecolor='red', markeredgewidth=2,
                                         linestyle='none', label='Tracker State'))

        # Position legend below the plots with no overlap
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.01),
                  ncol=len(legend_elements), fontsize=14, frameon=True, fancybox=True,
                  shadow=True, borderpad=1)

    def _get_scale_factor(self, image: np.ndarray) -> Tuple[float, float]:
        """Calculate scale factors for image."""
        height, width = image.shape[:2]
        return width / self.image_width, height / self.image_height


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

        # Create enhanced visualization
        fig = viz_tool.create_comprehensive_visualization(
            sample_id, labels_df, predictions_df, tracking_df,
            image_path, rd_path, ra_path
        )

        # Save frame with higher DPI for better quality
        frame_path = frames_dir / f"frame_{i:06d}.png"
        fig.savefig(frame_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        created_frames.append(frame_path)

        if i % 10 == 0:
            print(f"Created enhanced frame {i+1}/{len(sample_ids)}")

    # Create video using ffmpeg (if available)
    video_path = output_dir / 'enhanced_tracking_visualization.mp4'
    try:
        import subprocess
        cmd = [
            'ffmpeg', '-y', '-framerate', '5',
            '-i', str(frames_dir / 'frame_%06d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-crf', '18',  # Higher quality
            str(video_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Enhanced video created: {video_path}")

        # Clean up frames
        for frame in created_frames:
            frame.unlink()
        frames_dir.rmdir()

    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"FFmpeg not available. Enhanced frame images saved in: {frames_dir}")
        video_path = frames_dir

    return str(video_path)