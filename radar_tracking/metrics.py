"""
Evaluation metrics for radar object detection and tracking.
Uses Hungarian algorithm for optimal assignment between predictions and ground truth.
"""
import numpy as np
from typing import List, Optional
from scipy.optimize import linear_sum_assignment
from radar_tracking.data_structures import Detection, TrackingResult
from radar_tracking.coordinate_transforms import euclidean_distance


class RadarMetrics:
    """
    Evaluation metrics for radar detection and tracking performance.
    """

    def __init__(self, max_distance_threshold: float = 10.0):
        """
        Initialize metrics calculator.

        Args:
            max_distance_threshold: Maximum distance for valid matches (meters)
        """
        self.max_distance_threshold = max_distance_threshold

    def compute_detection_metrics(self,
                                  predictions: List[Detection],
                                  ground_truth: List[Detection],
                                  use_confidence: bool = True) -> TrackingResult:
        """
        Compute detection metrics using Hungarian algorithm.

        Args:
            predictions: List of predicted detections
            ground_truth: List of ground truth detections
            use_confidence: Whether to weight by detection confidence

        Returns:
            TrackingResult with evaluation metrics
        """
        if not predictions or not ground_truth:
            return TrackingResult(
                total_distance=float('inf'),
                num_matches=0,
                num_predictions=len(predictions),
                num_ground_truth=len(ground_truth),
                match_ratio=0.0,
                average_distance=float('inf')
            )

        # Create cost matrix (distances between all prediction-GT pairs)
        cost_matrix = self._create_cost_matrix(predictions, ground_truth, use_confidence)

        # Apply Hungarian algorithm
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

        # Calculate metrics
        total_distance = 0.0
        num_matches = 0

        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            distance = cost_matrix[pred_idx, gt_idx]

            # Only count as match if within threshold
            if distance <= self.max_distance_threshold:
                total_distance += distance
                num_matches += 1

        # Calculate derived metrics
        match_ratio = num_matches / len(ground_truth) if ground_truth else 0.0
        average_distance = total_distance / num_matches if num_matches > 0 else float('inf')

        return TrackingResult(
            total_distance=total_distance,
            num_matches=num_matches,
            num_predictions=len(predictions),
            num_ground_truth=len(ground_truth),
            match_ratio=match_ratio,
            average_distance=average_distance
        )

    def _create_cost_matrix(self,
                            predictions: List[Detection],
                            ground_truth: List[Detection],
                            use_confidence: bool = True) -> np.ndarray:
        """
        Create cost matrix for Hungarian algorithm.

        Args:
            predictions: List of predicted detections
            ground_truth: List of ground truth detections
            use_confidence: Whether to weight by detection confidence

        Returns:
            Cost matrix where cost[i,j] is the cost of assigning prediction i to GT j
        """
        n_pred = len(predictions)
        n_gt = len(ground_truth)
        cost_matrix = np.zeros((n_pred, n_gt))

        for i, pred in enumerate(predictions):
            for j, gt in enumerate(ground_truth):
                # Calculate Euclidean distance in Cartesian coordinates
                distance = euclidean_distance(pred.cartesian_pos, gt.cartesian_pos)

                # Apply confidence weighting if requested
                if use_confidence:
                    # Lower confidence increases cost (inverse weighting)
                    confidence_weight = 1.0 / max(pred.confidence, 0.1)  # Avoid division by zero
                    cost_matrix[i, j] = distance * confidence_weight
                else:
                    cost_matrix[i, j] = distance

        return cost_matrix

    def print_metrics(self, result: TrackingResult, frame_id: Optional[int] = None):
        """
        Print evaluation metrics in a readable format.

        Args:
            result: TrackingResult to print
            frame_id: Optional frame identifier
        """
        header = f"Frame {frame_id} Metrics:" if frame_id is not None else "Metrics:"
        print(f"\n{header}")
        print(f"  Total Distance: {result.total_distance:.2f} meters")
        print(f"  Matches: {result.num_matches}/{result.num_ground_truth}")
        print(f"  Match Ratio: {result.match_ratio:.2f}")
        print(f"  Average Distance: {result.average_distance:.2f} meters")
        print(f"  Predictions: {result.num_predictions}")
        print(f"  Ground Truth: {result.num_ground_truth}")