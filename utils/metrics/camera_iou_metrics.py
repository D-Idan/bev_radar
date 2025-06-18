"""
Camera Image IoU Calculation for Radar Tracking Evaluation.
Converts world coordinates to camera pixel coordinates and calculates IoU.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from utils.util import worldToImage
import cv2


class CameraIoUCalculator:
    """Calculate IoU between bounding boxes in camera image space."""

    def __init__(self, image_width: int = 960, image_height: int = 540):
        """
        Initialize camera IoU calculator.

        Args:
            image_width: Camera image width in pixels
            image_height: Camera image height in pixels
        """
        self.image_width = image_width
        self.image_height = image_height

        self.labels_image_width, self.labels_image_height = 1920, 1080

    def _get_scale_factor(self, image_shape: Tuple[int, int]) -> Tuple[float, float]:
        """
        Get scale factor to match visualization scaling.

        Args:
            image_shape: (height, width) of the actual image

        Returns:
            (scale_w, scale_h): Scale factors for width and height
        """
        height, width = image_shape[:2]
        scale_w = width / self.labels_image_width
        scale_h = height / self.labels_image_height
        return scale_w, scale_h

    def range_azimuth_to_camera_bbox_consistent(self, range_m: float, azimuth_deg: float,
                                              image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Convert range/azimuth to camera bounding box using SAME method as visualization.

        Args:
            range_m: Range in meters
            azimuth_deg: Azimuth angle in degrees
            image_shape: (height, width) of the actual image

        Returns:
            (x_min, y_min, x_max, y_max): Camera pixel coordinates
        """
        # Use EXACT same calculation as visualization
        x = np.sin(np.deg2rad(azimuth_deg)) * range_m
        y = np.cos(np.deg2rad(azimuth_deg)) * range_m

        # Use same dimensions as visualization: 0.9m half-width, heights 0 to 1.6m
        u1, v1 = worldToImage(-x - 0.9, y, 0)
        u2, v2 = worldToImage(-x + 0.9, y, 1.6)

        # Apply same scaling as visualization
        u1, v1 = int(u1 / 2), int(v1 / 2)
        u2, v2 = int(u2 / 2), int(v2 / 2)

        # Ensure proper bounding box format (min, max)
        x_min = min(u1, u2)
        x_max = max(u1, u2)
        y_min = min(v1, v2)
        y_max = max(v1, v2)

        # Clamp to image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image_shape[1] - 1, x_max)
        y_max = min(image_shape[0] - 1, y_max)

        return x_min, y_min, x_max, y_max

    def get_label_bbox_consistent(self, label_row: pd.Series,
                                image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Get label bounding box using SAME scaling as visualization.

        Args:
            label_row: Label data row
            image_shape: (height, width) of the actual image

        Returns:
            (x_min, y_min, x_max, y_max): Camera pixel coordinates
        """
        # Use same scaling method as visualization
        scale_w, scale_h = self._get_scale_factor(image_shape)

        x_min = int(label_row['x1_pix'] * scale_w)
        y_min = int(label_row['y1_pix'] * scale_h)
        x_max = int(label_row['x2_pix'] * scale_w)
        y_max = int(label_row['y2_pix'] * scale_h)

        # Clamp to image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image_shape[1] - 1, x_max)
        y_max = min(image_shape[0] - 1, y_max)

        return x_min, y_min, x_max, y_max

    def calculate_bbox_iou(self, bbox1: Tuple[int, int, int, int],
                           bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate IoU between two rectangular bounding boxes.

        Args:
            bbox1: (x_min, y_min, x_max, y_max) for first box
            bbox2: (x_min, y_min, x_max, y_max) for second box

        Returns:
            IoU value between 0 and 1
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        # Check if there's intersection
        if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
            return 0.0

        # Calculate areas
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)

        # Avoid division by zero
        if area1 <= 0 or area2 <= 0:
            return 0.0

        # Calculate IoU
        union_area = area1 + area2 - inter_area
        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def evaluate_camera_iou_single_frame(self, predictions_df: pd.DataFrame,
                                         labels_df: pd.DataFrame,
                                         tracking_df: Optional[pd.DataFrame] = None,
                                         sample_id: int = None,
                                         image_shape: Tuple[int, int] = (540, 960)) -> Dict[str, any]:
        """
        Evaluate camera IoU for a single frame using consistent scaling.

        Args:
            predictions_df: Network predictions for the frame
            labels_df: Ground truth labels for the frame
            tracking_df: Tracking results for the frame (optional)
            sample_id: Sample ID for filtering (optional)
            image_shape: (height, width) of the actual image used in visualization

        Returns:
            Dictionary with IoU metrics
        """
        # Filter data for specific sample if provided
        if sample_id is not None:
            predictions_df = predictions_df[predictions_df['sample_id'] == sample_id]
            labels_df = labels_df[labels_df['numSample'] == sample_id]
            if tracking_df is not None:
                tracking_df = tracking_df[tracking_df['sample_id'] == sample_id]

        results = {
            'detection_vs_labels_ious': [],
            'tracking_vs_labels_ious': [] if tracking_df is not None else None,
            'sample_id': sample_id,
            'debug_info': {
                'num_predictions': len(predictions_df),
                'num_labels': len(labels_df),
                'num_tracks': len(tracking_df) if tracking_df is not None else 0
            }
        }

        # Convert labels to camera bounding boxes using consistent scaling
        label_bboxes = []
        for _, label_row in labels_df.iterrows():
            bbox = self.get_label_bbox_consistent(label_row, image_shape)
            label_bboxes.append(bbox)

        # Evaluate detection predictions vs labels
        for _, pred_row in predictions_df.iterrows():
            pred_bbox = self.range_azimuth_to_camera_bbox_consistent(
                pred_row['range_m'], pred_row['azimuth_deg'], image_shape
            )

            # Calculate IoU with all labels and take maximum
            max_iou = 0.0
            for label_bbox in label_bboxes:
                iou = self.calculate_bbox_iou(pred_bbox, label_bbox)
                max_iou = max(max_iou, iou)

            results['detection_vs_labels_ious'].append(max_iou)

        # Evaluate tracking predictions vs labels (if available)
        if tracking_df is not None and not tracking_df.empty:
            for _, track_row in tracking_df.iterrows():
                track_bbox = self.range_azimuth_to_camera_bbox_consistent(
                    track_row['range_m'], track_row['azimuth_deg'], image_shape
                )

                # Calculate IoU with all labels and take maximum
                max_iou = 0.0
                for label_bbox in label_bboxes:
                    iou = self.calculate_bbox_iou(track_bbox, label_bbox)
                    max_iou = max(max_iou, iou)

                results['tracking_vs_labels_ious'].append(max_iou)

        return results

    def evaluate_camera_iou_sequence(self, predictions_csv: str, labels_csv: str,
                                     tracking_csv: Optional[str] = None,
                                     max_frames: Optional[int] = None,
                                     image_shape: Tuple[int, int] = (540, 960)) -> Dict[str, any]:
        """
        Evaluate camera IoU for entire sequence using consistent scaling.

        Args:
            predictions_csv: Path to predictions CSV file
            labels_csv: Path to labels CSV file
            tracking_csv: Path to tracking CSV file (optional)
            max_frames: Maximum number of frames to evaluate
            image_shape: (height, width) of the actual image used in visualization

        Returns:
            Dictionary with sequence-level IoU metrics
        """
        # Load data
        predictions_df = pd.read_csv(predictions_csv)
        labels_df = pd.read_csv(labels_csv, sep='\t|,', engine='python')
        tracking_df = pd.read_csv(tracking_csv) if tracking_csv else None

        # Get sample IDs from ground truth
        sample_ids = sorted(labels_df['numSample'].unique())
        if max_frames:
            sample_ids = sample_ids[:max_frames]

        all_detection_ious = []
        all_tracking_ious = []
        frame_results = []

        print(f"Evaluating camera IoU for {len(sample_ids)} frames...")

        for i, sample_id in enumerate(sample_ids):
            frame_result = self.evaluate_camera_iou_single_frame(
                predictions_df, labels_df, tracking_df, sample_id, image_shape
            )

            frame_results.append(frame_result)
            all_detection_ious.extend(frame_result['detection_vs_labels_ious'])

            if frame_result['tracking_vs_labels_ious'] is not None:
                all_tracking_ious.extend(frame_result['tracking_vs_labels_ious'])

            # Print progress and debug info
            if i % 10 == 0 or i < 5:
                debug = frame_result['debug_info']
                det_ious = frame_result['detection_vs_labels_ious']
                mean_iou = np.mean(det_ious) if det_ious else 0.0
                print(f"Frame {sample_id}: {debug['num_predictions']} preds, {debug['num_labels']} labels, mean IoU: {mean_iou:.3f}")

        # Calculate summary statistics
        summary_results = {
            'detection_camera_iou': {
                'mean': np.mean(all_detection_ious) if all_detection_ious else 0.0,
                'std': np.std(all_detection_ious) if all_detection_ious else 0.0,
                'median': np.median(all_detection_ious) if all_detection_ious else 0.0,
                'max': np.max(all_detection_ious) if all_detection_ious else 0.0,
                'min': np.min(all_detection_ious) if all_detection_ious else 0.0,
                'count': len(all_detection_ious)
            },
            'tracking_camera_iou': {
                'mean': np.mean(all_tracking_ious) if all_tracking_ious else 0.0,
                'std': np.std(all_tracking_ious) if all_tracking_ious else 0.0,
                'median': np.median(all_tracking_ious) if all_tracking_ious else 0.0,
                'max': np.max(all_tracking_ious) if all_tracking_ious else 0.0,
                'min': np.min(all_tracking_ious) if all_tracking_ious else 0.0,
                'count': len(all_tracking_ious)
            },
            'frames_evaluated': len(sample_ids),
            'frame_by_frame_results': frame_results
        }

        print(f"Final camera IoU - Detection: {summary_results['detection_camera_iou']['mean']:.3f}, "
              f"Tracking: {summary_results['tracking_camera_iou']['mean']:.3f}")

        return summary_results


# Update the convenience function to include image_shape parameter
def calculate_camera_iou_metrics(predictions_csv: str, labels_csv: str,
                                 tracking_csv: Optional[str] = None,
                                 max_frames: Optional[int] = None,
                                 image_shape: Tuple[int, int] = (540, 960)) -> Dict[str, any]:
    """
    Convenience function to calculate camera IoU metrics with consistent scaling.

    Args:
        predictions_csv: Path to predictions CSV file
        labels_csv: Path to labels CSV file
        tracking_csv: Path to tracking CSV file (optional)
        max_frames: Maximum number of frames to evaluate
        image_shape: (height, width) of the actual image used in visualization

    Returns:
        Dictionary with IoU metrics
    """
    calculator = CameraIoUCalculator()
    return calculator.evaluate_camera_iou_sequence(
        predictions_csv, labels_csv, tracking_csv, max_frames, image_shape
    )


# Rest of the convenience functions remain the same...
def print_camera_iou_summary(iou_results: Dict[str, any]) -> None:
    """Print formatted summary of camera IoU results."""
    print("\n" + "=" * 60)
    print("           CAMERA IMAGE IOU EVALUATION")
    print("=" * 60)

    print(f"\n📊 EVALUATION SUMMARY:")
    print(f"   • Frames Evaluated: {iou_results['frames_evaluated']}")

    det_iou = iou_results['detection_camera_iou']
    print(f"\n🎯 DETECTION CAMERA IOU:")
    print(f"   • Mean IoU:            {det_iou['mean']:.3f}")
    print(f"   • Std IoU:             {det_iou['std']:.3f}")
    print(f"   • Median IoU:          {det_iou['median']:.3f}")
    print(f"   • Max IoU:             {det_iou['max']:.3f}")
    print(f"   • Min IoU:             {det_iou['min']:.3f}")
    print(f"   • Total Detections:    {det_iou['count']}")

    track_iou = iou_results['tracking_camera_iou']
    if track_iou['count'] > 0:
        print(f"\n🔄 TRACKING CAMERA IOU:")
        print(f"   • Mean IoU:            {track_iou['mean']:.3f}")
        print(f"   • Std IoU:             {track_iou['std']:.3f}")
        print(f"   • Median IoU:          {track_iou['median']:.3f}")
        print(f"   • Max IoU:             {track_iou['max']:.3f}")
        print(f"   • Min IoU:             {track_iou['min']:.3f}")
        print(f"   • Total Tracks:        {track_iou['count']}")

        if det_iou['mean'] > 0:
            improvement = ((track_iou['mean'] - det_iou['mean']) / det_iou['mean']) * 100
            print(f"\n📈 TRACKING IMPROVEMENT:")
            print(f"   • Mean IoU Improvement: {improvement:+.1f}%")

    print("\n" + "=" * 60)


def debug_detailed_iou():
    """Detailed debug function to understand the coordinate transformation"""
    calculator = CameraIoUCalculator()

    # Test with sample data
    sample_range = 10.0
    sample_azimuth = 0.0
    image_shape = (540, 960)

    print("=== DEBUGGING COORDINATE TRANSFORMATION ===")

    # Step 1: Check world coordinates
    x = np.sin(np.deg2rad(sample_azimuth)) * sample_range
    y = np.cos(np.deg2rad(sample_azimuth)) * sample_range
    print(f"World coordinates: x={x}, y={y}")

    # Step 2: Check worldToImage function
    from utils.util import worldToImage  # Make sure this import works

    u1, v1 = worldToImage(-x - 0.9, y, 0)
    u2, v2 = worldToImage(-x + 0.9, y, 1.6)
    print(f"Before scaling: corner1=({u1}, {v1}), corner2=({u2}, {v2})")

    # Step 3: Check scaling
    u1_scaled, v1_scaled = int(u1 / 2), int(v1 / 2)
    u2_scaled, v2_scaled = int(u2 / 2), int(v2 / 2)
    print(f"After scaling: corner1=({u1_scaled}, {v1_scaled}), corner2=({u2_scaled}, {v2_scaled})")

    # Step 4: Generate prediction bbox
    pred_bbox = calculator.range_azimuth_to_camera_bbox_consistent(
        sample_range, sample_azimuth, image_shape
    )
    print(f"Final prediction bbox: {pred_bbox}")

    # Step 5: Test with different label bboxes
    test_cases = [
        (100, 100, 200, 200),  # Original test
        (350, 200, 450, 300),  # Closer to prediction
        (400, 248, 500, 350),  # Overlapping with prediction
        (450, 300, 550, 400),  # Partially overlapping
    ]

    print("\n=== TESTING DIFFERENT LABEL BBOXES ===")
    for i, label_bbox in enumerate(test_cases):
        iou = calculator.calculate_bbox_iou(pred_bbox, label_bbox)
        print(f"Test {i + 1}: Label bbox {label_bbox} -> IoU: {iou}")

        # Manual intersection check
        x1_min, y1_min, x1_max, y1_max = pred_bbox
        x2_min, y2_min, x2_max, y2_max = label_bbox

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        has_intersection = inter_x_min < inter_x_max and inter_y_min < inter_y_max
        print(f"  Intersection area: ({inter_x_min}, {inter_y_min}) to ({inter_x_max}, {inter_y_max})")
        print(f"  Has intersection: {has_intersection}")

        if has_intersection:
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = area1 + area2 - inter_area
            manual_iou = inter_area / union_area
            print(f"  Manual IoU calculation: {manual_iou}")
        print()


def debug_real_data_alignment(predictions_csv, labels_csv, max_frames=5):
    """Debug the alignment between real predictions and labels"""
    import pandas as pd

    predictions_df = pd.read_csv(predictions_csv)
    labels_df = pd.read_csv(labels_csv, sep='\t|,', engine='python')

    # Find frames with both predictions and labels
    pred_frames = set(predictions_df['sample_id'].unique())
    label_frames = set(labels_df['numSample'].unique())
    common_frames = sorted(list(pred_frames.intersection(label_frames)))[:max_frames]

    print(f"Found {len(common_frames)} common frames to analyze")

    calculator = CameraIoUCalculator()
    image_shape = (540, 960)

    for frame_id in common_frames:
        print(f"\n{'=' * 50}")
        print(f"FRAME {frame_id}")
        print(f"{'=' * 50}")

        pred_frame = predictions_df[predictions_df['sample_id'] == frame_id]
        label_frame = labels_df[labels_df['numSample'] == frame_id]

        print(f"Predictions: {len(pred_frame)}, Labels: {len(label_frame)}")

        # Analyze all predictions in this frame
        pred_bboxes = []
        for _, pred_row in pred_frame.iterrows():
            bbox = calculator.range_azimuth_to_camera_bbox_consistent(
                pred_row['range_m'], pred_row['azimuth_deg'], image_shape
            )
            pred_bboxes.append(bbox)
            print(f"  Pred: range={pred_row['range_m']:.1f}m, azimuth={pred_row['azimuth_deg']:.1f}° -> bbox={bbox}")

        # Analyze all labels in this frame
        label_bboxes = []
        for _, label_row in label_frame.iterrows():
            # Check what pixel coordinate columns exist
            pix_cols = [col for col in label_row.keys() if 'pix' in col.lower()]
            print(f"  Label pixel columns available: {pix_cols}")

            if all(col in label_row.keys() for col in ['x1_pix', 'y1_pix', 'x2_pix', 'y2_pix']):
                raw_bbox = (label_row['x1_pix'], label_row['y1_pix'], label_row['x2_pix'], label_row['y2_pix'])
                bbox = calculator.get_label_bbox_consistent(label_row, image_shape)
                label_bboxes.append(bbox)
                print(f"  Label: raw_pix={raw_bbox} -> scaled_bbox={bbox}")

                # Check if label has radar coordinates too
                if 'radar_R_m' in label_row.keys() and 'radar_A_deg' in label_row.keys():
                    print(
                        f"    Label radar: range={label_row['radar_R_m']:.1f}m, azimuth={label_row['radar_A_deg']:.1f}°")
            else:
                print(f"  Label: Missing required pixel coordinates")

        # Calculate distances between prediction and label centers
        if pred_bboxes and label_bboxes:
            print(f"\n  Distance Analysis:")
            for i, pred_bbox in enumerate(pred_bboxes):
                pred_center = ((pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2)
                for j, label_bbox in enumerate(label_bboxes):
                    label_center = ((label_bbox[0] + label_bbox[2]) / 2, (label_bbox[1] + label_bbox[3]) / 2)
                    distance = np.sqrt(
                        (pred_center[0] - label_center[0]) ** 2 + (pred_center[1] - label_center[1]) ** 2)
                    iou = calculator.calculate_bbox_iou(pred_bbox, label_bbox)
                    print(
                        f"    Pred{i} center={pred_center} <-> Label{j} center={label_center}: distance={distance:.1f}px, IoU={iou:.3f}")

        print(f"\n  Image bounds check:")
        print(f"    Image shape: {image_shape} (height x width)")
        all_bboxes = pred_bboxes + label_bboxes
        for i, bbox in enumerate(all_bboxes):
            bbox_type = "Pred" if i < len(pred_bboxes) else "Label"
            in_bounds = (0 <= bbox[0] < image_shape[1] and 0 <= bbox[1] < image_shape[0] and
                         0 <= bbox[2] < image_shape[1] and 0 <= bbox[3] < image_shape[0])
            print(f"    {bbox_type} bbox {bbox}: in_bounds={in_bounds}")



if __name__ == "__main__":
    # Run debug function if this script is executed directly
    # debug_detailed_iou()

    # Run this with your actual files
    debug_real_data_alignment("/Users/daniel/Idan/University/Masters/Thesis/2025/repos/bev_radar/plots/predictions/all_predictions.csv",
                              "/Volumes/ELEMENTS/datasets/radial/labels_CVPR.csv")