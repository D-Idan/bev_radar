# tracklet_manager.py

"""
Tracklet management for radar tracking system.
Handles track lifecycle, merging, and long-term tracking statistics.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
from radar_tracking.data_structures import Track, Detection, TrackingResult
from radar_tracking.tracker import RadarTracker
from radar_tracking.metrics import RadarMetrics


@dataclass
class TrackletStatistics:
    """Statistics for a tracklet over its lifetime."""
    track_id: int
    total_detections: int = 0
    average_confidence: float = 0.0
    max_confidence: float = 0.0
    lifetime_frames: int = 0
    total_distance_traveled: float = 0.0
    last_position: Optional[tuple] = None
    birth_frame: int = 0
    death_frame: Optional[int] = None

    def update(self, track: Track, frame_id: int):
        """Update statistics with new track information."""
        self.total_detections += 1
        self.lifetime_frames = frame_id - self.birth_frame + 1

        if track.last_detection:
            conf = track.last_detection.confidence
            self.average_confidence = (
                    (self.average_confidence * (self.total_detections - 1) + conf) /
                    self.total_detections
            )
            self.max_confidence = max(self.max_confidence, conf)

        # Calculate distance traveled
        current_pos = track.position
        if self.last_position is not None:
            distance = np.sqrt(
                (current_pos[0] - self.last_position[0]) ** 2 +
                (current_pos[1] - self.last_position[1]) ** 2
            )
            self.total_distance_traveled += distance

        self.last_position = current_pos


class TrackletManager:
    """
    Manages tracklets and provides high-level tracking functionality.
    Implements track lifecycle management similar to DeepSORT/SORT.
    """

    def __init__(self,
                 tracker_config: Dict = None,
                 track_buffer_size: int = 1000):
        """
        Initialize tracklet manager.

        Args:
            tracker_config: Configuration for underlying tracker
            track_buffer_size: Maximum number of historical tracks to keep
        """
        # Initialize tracker with provided or default config
        config = tracker_config or {}
        self.tracker = RadarTracker(**config)

        # Track management
        self.active_tracklets: Dict[int, TrackletStatistics] = {}
        self.historical_tracklets: Dict[int, TrackletStatistics] = {}
        self.track_buffer_size = track_buffer_size

        # Metrics
        self.metrics = RadarMetrics()
        self.frame_results: List[TrackingResult] = []

        # State
        self.current_frame = 0

    def update(self,
               detections: List[Detection],
               ground_truth: Optional[List[Detection]] = None,
               dt: Optional[float] = None,
               current_time: Optional[float] = None) -> List[Track]:
        """
        Update tracking with new detections.

        Args:
            detections: Current frame detections
            ground_truth: Optional ground truth for evaluation
            dt: Optional time step for this frame (if None, uses tracker default)
            current_time: Optional current timestamp in seconds

        Returns:
            List of active tracks
        """
        self.current_frame += 1

        # Update tracker with timestamp support
        if current_time is not None:
            # Pass current time to tracker
            tracks = self.tracker.update(detections, dt, current_time)
        else:
            # Fallback to dt-based update
            tracks = self.tracker.update(detections, dt)

        # Update tracklet statistics
        self._update_tracklet_statistics(tracks)

        # Evaluate against ground truth if provided
        if ground_truth is not None:
            # Convert tracks to detections for evaluation
            track_detections = self._tracks_to_detections(tracks)
            result = self.metrics.compute_detection_metrics(
                track_detections, ground_truth
            )
            self.frame_results.append(result)

        return tracks

    def _update_tracklet_statistics(self, tracks: List[Track]):
        """Update statistics for all active tracklets."""
        current_track_ids = {track.id for track in tracks}

        # Update existing tracklets
        for track in tracks:
            if track.id not in self.active_tracklets:
                # New tracklet
                self.active_tracklets[track.id] = TrackletStatistics(
                    track_id=track.id,
                    birth_frame=self.current_frame
                )

            self.active_tracklets[track.id].update(track, self.current_frame)

        # Move inactive tracklets to historical
        inactive_ids = set(self.active_tracklets.keys()) - current_track_ids
        for track_id in inactive_ids:
            tracklet = self.active_tracklets.pop(track_id)
            tracklet.death_frame = self.current_frame - 1
            self.historical_tracklets[track_id] = tracklet

            # Manage buffer size
            if len(self.historical_tracklets) > self.track_buffer_size:
                # Remove oldest tracklet
                oldest_id = min(self.historical_tracklets.keys())
                del self.historical_tracklets[oldest_id]

    def _tracks_to_detections(self, tracks: List[Track]) -> List[Detection]:
        """Convert tracks to detections for evaluation."""
        detections = []
        for track in tracks:
            # Create detection from track state
            pos = track.position
            # Convert back to polar coordinates for consistency
            from radar_tracking.coordinate_transforms import cartesian_to_polar
            range_m, azimuth_rad = cartesian_to_polar(pos[0], pos[1])

            detection = Detection(
                range_m=range_m,
                azimuth_rad=azimuth_rad,
                confidence=track.confidence,
                timestamp=self.current_frame,
                cartesian_pos=pos
            )
            detections.append(detection)

        return detections

    def get_tracking_summary(self) -> Dict:
        """Get comprehensive tracking summary."""
        all_tracklets = {**self.active_tracklets, **self.historical_tracklets}

        if not all_tracklets:
            return {"error": "No tracklets available"}

        # Calculate summary statistics
        total_tracklets = len(all_tracklets)
        active_tracklets = len(self.active_tracklets)

        lifetimes = [t.lifetime_frames for t in all_tracklets.values()]
        confidences = [t.average_confidence for t in all_tracklets.values()]
        distances = [t.total_distance_traveled for t in all_tracklets.values()]

        summary = {
            "total_tracklets": total_tracklets,
            "active_tracklets": active_tracklets,
            "completed_tracklets": len(self.historical_tracklets),
            "current_frame": self.current_frame,
            "average_lifetime": np.mean(lifetimes) if lifetimes else 0,
            "max_lifetime": max(lifetimes) if lifetimes else 0,
            "min_lifetime": min(lifetimes) if lifetimes else 0,
            "average_confidence": np.mean(confidences) if confidences else 0,
            "average_distance_traveled": np.mean(distances) if distances else 0,
            "total_distance_all_tracks": sum(distances)
        }

        # Add evaluation metrics if available
        if self.frame_results:
            avg_match_ratio = np.mean([r.match_ratio for r in self.frame_results])
            avg_distance = np.mean([r.average_distance for r in self.frame_results
                                    if r.average_distance != float('inf')])
            total_matches = sum([r.num_matches for r in self.frame_results])
            total_gt = sum([r.num_ground_truth for r in self.frame_results])

            summary.update({
                "evaluation_frames": len(self.frame_results),
                "overall_match_ratio": avg_match_ratio,
                "overall_average_distance": avg_distance if not np.isnan(avg_distance) else 0,
                "total_matches": total_matches,
                "total_ground_truth": total_gt
            })

        return summary

    def get_tracklet_details(self, track_id: int) -> Optional[TrackletStatistics]:
        """Get detailed statistics for a specific tracklet."""
        if track_id in self.active_tracklets:
            return self.active_tracklets[track_id]
        elif track_id in self.historical_tracklets:
            return self.historical_tracklets[track_id]
        return None

    def export_results(self) -> Dict:
        """Export all tracking results for analysis."""
        return {
            "frame_results": self.frame_results,
            "active_tracklets": self.active_tracklets,
            "historical_tracklets": self.historical_tracklets,
            "summary": self.get_tracking_summary()
        }

    def reset(self):
        """Reset all tracking state."""
        self.tracker.reset()
        self.active_tracklets.clear()
        self.historical_tracklets.clear()
        self.frame_results.clear()
        self.current_frame = 0

    def print_summary(self):
        """Print tracking summary in readable format."""
        summary = self.get_tracking_summary()

        print("\n" + "=" * 50)
        print("RADAR TRACKING SUMMARY")
        print("=" * 50)
        print(f"Total Tracklets: {summary['total_tracklets']}")
        print(f"Active Tracklets: {summary['active_tracklets']}")
        print(f"Completed Tracklets: {summary['completed_tracklets']}")
        print(f"Current Frame: {summary['current_frame']}")
        print(f"Average Tracklet Lifetime: {summary['average_lifetime']:.1f} frames")
        print(f"Max Tracklet Lifetime: {summary['max_lifetime']} frames")
        print(f"Average Confidence: {summary['average_confidence']:.3f}")
        print(f"Total Distance Traveled: {summary['total_distance_all_tracks']:.1f} meters")

        if 'evaluation_frames' in summary:
            print(f"\nEVALUATION METRICS:")
            print(f"Frames Evaluated: {summary['evaluation_frames']}")
            print(f"Overall Match Ratio: {summary['overall_match_ratio']:.3f}")
            print(f"Average Match Distance: {summary['overall_average_distance']:.2f} meters")
            print(f"Total Matches: {summary['total_matches']}")
            print(f"Total Ground Truth: {summary['total_ground_truth']}")

        print("=" * 50)