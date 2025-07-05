"""
Aggregate Poor Performance Analysis
Identifies consistently problematic frames across multiple configurations.
"""

import json
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


class AggregatePoorPerformanceAnalyzer:
    """Analyzes poor performance patterns across multiple configurations."""

    def analyze_dataset_poor_performance(self, dataset_root: Path,
                                         metric_name: str = 'det_a') -> Dict:
        """
        Analyze poor performance across all configurations for a dataset.

        Returns dictionary with frame IDs that appear as worst in multiple configs.
        """
        tracking_output_dir = dataset_root / 'plots' / 'tracking_output'

        # Collect worst frames from each configuration
        config_worst_frames = {}
        frame_appearance_count = defaultdict(int)
        frame_degradations = defaultdict(list)

        for config_dir in tracking_output_dir.iterdir():
            if not config_dir.is_dir() or config_dir.name == 'raw_predictions':
                continue

            analysis_file = (config_dir / 'analysis' / 'poor_performance' /
                             f'worst_{metric_name}_analysis.json')

            if analysis_file.exists():
                with open(analysis_file, 'r') as f:
                    data = json.load(f)

                worst_frames = data['worst_frames']
                config_worst_frames[config_dir.name] = worst_frames

                for frame in worst_frames:
                    frame_id = frame['frame_id']
                    frame_appearance_count[frame_id] += 1
                    frame_degradations[frame_id].append({
                        'config': config_dir.name,
                        'degradation': frame['degradation_percent']
                    })

        # Find frames that appear in multiple configurations
        problematic_frames = {
            frame_id: {
                'appearance_count': count,
                'configs': frame_degradations[frame_id]
            }
            for frame_id, count in frame_appearance_count.items()
            if count > 1
        }

        # Save aggregate analysis
        output_file = (dataset_root / 'plots' / 'tracking_output' /
                       f'aggregate_poor_{metric_name}_frames.txt')

        with open(output_file, 'w') as f:
            f.write(f"FRAMES WITH POOR {metric_name.upper()} ACROSS MULTIPLE CONFIGURATIONS\n")
            f.write("=" * 80 + "\n\n")

            sorted_frames = sorted(problematic_frames.items(),
                                   key=lambda x: x[1]['appearance_count'],
                                   reverse=True)

            for frame_id, info in sorted_frames:
                f.write(f"Frame {frame_id}: Appears in {info['appearance_count']} configurations\n")
                for config_info in info['configs']:
                    f.write(f"  - {config_info['config']}: {config_info['degradation']:.1f}% degradation\n")
                f.write("\n")

        print(f"Aggregate poor performance analysis saved to: {output_file}")

        return problematic_frames