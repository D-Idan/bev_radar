"""
Aggregate Analysis across Multiple Datasets
Generates comprehensive reports across all processed datasets.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional


class AggregateAnalysisGenerator:
    """Generates aggregate analysis reports across multiple datasets."""

    def __init__(self, key_metrics: Optional[List[str]] = None):
        """Initialize aggregate analysis generator."""
        self.key_metrics = key_metrics or [
            'hota', 'mota', 'precision', 'det_a'
        ]
        self.distance_metrics = []

        # Add IoU metrics
        self.iou_metrics = ['camera_iou_mean']

    def generate_aggregate_analysis(self, output_base: Path) -> Path:
        """
        Generate aggregate analysis across all processed datasets.

        Args:
            output_base: Base output directory containing all dataset results

        Returns:
            Path to the generated aggregate analysis directory
        """
        print("\nGenerating aggregate metrics across all datasets...")

        # Collect metrics from all datasets
        all_datasets_metrics = self._collect_all_datasets_metrics(output_base)

        if not all_datasets_metrics:
            print("No dataset metrics found for aggregation")
            return None

        # Create aggregate report
        aggregate_output_dir = output_base / 'aggregate_analysis'
        aggregate_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate and save analysis
        self._save_aggregate_analysis_report(all_datasets_metrics, aggregate_output_dir)

        print(f"Aggregate analysis saved to: {aggregate_output_dir}")
        return aggregate_output_dir

    def _collect_all_datasets_metrics(self, output_base: Path) -> Dict[str, Dict]:
        """Collect metrics from all dataset output directories."""
        # Store output_base for use in frame counting
        self.output_base = output_base

        all_datasets_metrics = {}

        for dataset_dir in output_base.iterdir():
            if not dataset_dir.is_dir() or not dataset_dir.name.startswith('RECORD'):
                continue

            comparison_file = (dataset_dir / 'plots' / 'configuration_comparison' /
                               'configuration_comparison.json')

            if comparison_file.exists():
                with open(comparison_file, 'r') as f:
                    all_datasets_metrics[dataset_dir.name] = json.load(f)

        return all_datasets_metrics

    def _save_aggregate_analysis_report(self, all_datasets_metrics: Dict[str, Dict],
                                        output_dir: Path):
        """Save comprehensive aggregate analysis across all datasets."""
        # Collect all configuration names across datasets
        all_configs = set()
        for dataset_metrics in all_datasets_metrics.values():
            all_configs.update(dataset_metrics.get('configurations_evaluated', []))

        all_configs = sorted(all_configs)

        # Build aggregate data structure
        aggregate_data = self._build_aggregate_data_structure(all_datasets_metrics, all_configs)

        # Calculate best configurations for each metric
        self._calculate_best_configurations(aggregate_data)

        # Generate per-dataset summaries
        self._generate_dataset_summaries(aggregate_data, all_datasets_metrics)

        # Save reports
        self._save_json_report(aggregate_data, output_dir)
        self._save_human_readable_report(aggregate_data, output_dir)

    def _build_aggregate_data_structure(self, all_datasets_metrics: Dict[str, Dict],
                                        all_configs: List[str]) -> Dict[str, Any]:
        """Build the main aggregate data structure."""
        aggregate_data = {
            'datasets_analyzed': list(all_datasets_metrics.keys()),
            'configurations_compared': all_configs,
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'aggregate_metrics': {},
            'dataset_summaries': {},
            'best_configurations': {},
            'configuration_success_rate': {}
        }

        # Process standard metrics
        all_metrics = self.key_metrics + self.iou_metrics

        for metric in all_metrics:
            aggregate_data['aggregate_metrics'][metric] = {}

            for config in all_configs:
                values = []
                for dataset_name, dataset_metrics in all_datasets_metrics.items():
                    # All metrics are now stored in the same way in metrics_comparison
                    metric_data = (dataset_metrics.get('metrics_comparison', {})
                                   .get(metric, {}).get(config, {}))

                    value = metric_data.get('value')
                    if value is not None and value != float('inf'):
                        values.append(value)

                if values:
                    aggregate_data['aggregate_metrics'][metric][config] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'values': values
                    }

        return aggregate_data

    def _calculate_best_configurations(self, aggregate_data: Dict[str, Any]):
        """Calculate best configuration for each metric across all datasets."""
        all_metrics = self.key_metrics + self.iou_metrics

        for metric in all_metrics:
            metric_data = aggregate_data['aggregate_metrics'].get(metric, {})
            if metric_data:
                if metric in self.distance_metrics:
                    best_config = min(metric_data.items(), key=lambda x: x[1]['mean'])
                else:
                    best_config = max(metric_data.items(), key=lambda x: x[1]['mean'])

                aggregate_data['best_configurations'][metric] = {
                    'config_name': best_config[0],
                    'mean_value': best_config[1]['mean'],
                    'std_value': best_config[1]['std'],
                    'count': best_config[1]['count']
                }

    def _generate_dataset_summaries(self, aggregate_data: Dict[str, Any],
                                    all_datasets_metrics: Dict[str, Dict]):
        """Generate summaries for each individual dataset."""
        for dataset_name, dataset_metrics in all_datasets_metrics.items():
            overall_ranking = dataset_metrics.get('overall_ranking', [])
            best_config = overall_ranking[0][0] if overall_ranking else None

            # Get frame count for this dataset
            frame_count = self._get_dataset_frame_count(dataset_name)

            aggregate_data['dataset_summaries'][dataset_name] = {
                'best_overall_config': best_config,
                'configurations_tested': len(dataset_metrics.get('configurations_evaluated', [])),
                'overall_ranking': overall_ranking[:3],  # Top 3
                'frame_count': frame_count  # Add frame count
            }

        # Calculate configuration success frequency
        config_wins = {}
        for dataset_summary in aggregate_data['dataset_summaries'].values():
            best_config = dataset_summary.get('best_overall_config')
            if best_config:
                config_wins[best_config] = config_wins.get(best_config, 0) + 1

        total_datasets = len(aggregate_data['dataset_summaries'])
        for config in aggregate_data['configurations_compared']:
            wins = config_wins.get(config, 0)
            aggregate_data['configuration_success_rate'][config] = {
                'wins': wins,
                'total_datasets': total_datasets,
                'success_rate': (wins / total_datasets) * 100 if total_datasets > 0 else 0.0
            }

    def _get_dataset_frame_count(self, dataset_name: str) -> int:
        """Get the number of frames for a dataset by counting unique frames in predictions CSV."""
        try:
            # Construct path to dataset
            dataset_path = None

            # Find the dataset in the output base directory
            for item in Path(self.output_base).iterdir():
                if item.is_dir() and item.name == dataset_name:
                    dataset_path = item
                    break

            if not dataset_path:
                return 0

            # Try predictions CSV first
            predictions_file = dataset_path / 'plots' / 'predictions' / 'all_predictions.csv'
            if predictions_file.exists():
                import pandas as pd
                df = pd.read_csv(predictions_file)
                if 'sample_id' in df.columns:
                    return len(df['sample_id'].unique())
                elif 'frame_id' in df.columns:
                    return len(df['frame_id'].unique())

            # Fallback: try any tracking CSV
            tracking_base = dataset_path / 'plots' / 'tracking_output'
            if tracking_base.exists():
                for config_dir in tracking_base.iterdir():
                    if config_dir.is_dir():
                        tracking_file = config_dir / 'tracks' / 'tracking.csv'
                        if tracking_file.exists():
                            import pandas as pd
                            df = pd.read_csv(tracking_file)
                            if 'sample_id' in df.columns:
                                return len(df['sample_id'].unique())
                            elif 'frame_id' in df.columns:
                                return len(df['frame_id'].unique())
                            break

            return 0

        except Exception as e:
            print(f"Warning: Could not get frame count for {dataset_name}: {e}")
            return 0

    def _save_json_report(self, aggregate_data: Dict[str, Any], output_dir: Path):
        """Save JSON aggregate report."""
        json_file = output_dir / 'aggregate_configuration_analysis.json'
        with open(json_file, 'w') as f:
            json.dump(aggregate_data, f, indent=2, default=str)

    def _save_human_readable_report(self, aggregate_data: Dict[str, Any], output_dir: Path):
        """Save human-readable aggregate analysis report."""
        txt_file = output_dir / 'aggregate_analysis_summary.txt'

        with open(txt_file, 'w') as f:
            self._write_header(f, aggregate_data)
            self._write_best_configurations_summary(f, aggregate_data)
            self._write_dataset_best_configs(f, aggregate_data)
            self._write_configuration_success_frequency(f, aggregate_data)
            self._write_detailed_metrics_analysis(f, aggregate_data)

    def _write_header(self, f, aggregate_data: Dict[str, Any]):
        """Write report header."""
        f.write("═" * 120 + "\n")
        f.write("                           AGGREGATE TRACKING CONFIGURATION ANALYSIS\n")
        f.write("═" * 120 + "\n\n")

        f.write(f"Datasets Analyzed: {len(aggregate_data['datasets_analyzed'])}\n")
        f.write(f"Configurations Compared: {len(aggregate_data['configurations_compared'])}\n")
        f.write(f"Analysis Date: {aggregate_data['analysis_timestamp']}\n\n")

    def _write_best_configurations_summary(self, f, aggregate_data: Dict[str, Any]):
        """Write best configurations summary."""
        f.write("🏆 BEST CONFIGURATIONS BY METRIC (ACROSS ALL DATASETS)\n")
        f.write("─" * 80 + "\n")

        for metric, best_data in aggregate_data['best_configurations'].items():
            config_name = best_data['config_name']
            mean_val = best_data['mean_value']
            std_val = best_data['std_value']
            count = best_data['count']

            if metric in self.distance_metrics:
                f.write(f"{metric.replace('_', ' ').title():<25}: {config_name:<25} "
                        f"({mean_val:.4f} ± {std_val:.4f}m, n={count})\n")
            else:
                f.write(f"{metric.replace('_', ' ').title():<25}: {config_name:<25} "
                        f"({mean_val:.4f} ± {std_val:.4f}, n={count})\n")

        f.write("\n")

    def _write_dataset_best_configs(self, f, aggregate_data: Dict[str, Any]):
        """Write per-dataset best configurations with frame counts."""
        f.write("📊 BEST CONFIGURATION PER DATASET\n")
        f.write("─" * 80 + "\n")

        for dataset_name, summary in aggregate_data['dataset_summaries'].items():
            best_config = summary.get('best_overall_config', 'N/A')
            frame_count = summary.get('frame_count', 0)

            # Format with frame count
            if frame_count > 0:
                dataset_display = f"{dataset_name} ({frame_count} frames)"
            else:
                dataset_display = f"{dataset_name} (frames: unknown)"

            f.write(f"{dataset_display:<50}: {best_config}\n")

        f.write("\n")

    def _write_configuration_success_frequency(self, f, aggregate_data: Dict[str, Any]):
        """Write configuration success frequency analysis."""
        f.write("📈 CONFIGURATION SUCCESS FREQUENCY (Best Overall per Dataset)\n")
        f.write("─" * 80 + "\n")
        f.write("Note: Shows how often each configuration was ranked as the best overall for a dataset\n\n")

        # Sort by success rate
        sorted_configs = sorted(
            aggregate_data['configuration_success_rate'].items(),
            key=lambda x: x[1]['success_rate'],
            reverse=True
        )

        for config, data in sorted_configs:
            wins = data['wins']
            total = data['total_datasets']
            rate = data['success_rate']
            f.write(f"{config:<30}: {wins:2d}/{total} datasets ({rate:.1f}%)\n")

        f.write("\n")

    def _write_detailed_metrics_analysis(self, f, aggregate_data: Dict[str, Any]):
        """Write detailed metrics analysis."""
        f.write("📋 DETAILED METRICS ANALYSIS\n")
        f.write("─" * 80 + "\n")

        all_metrics = self.key_metrics + self.iou_metrics

        for metric in all_metrics:
            f.write(f"\n{metric.replace('_', ' ').title()}:\n")
            metric_data = aggregate_data['aggregate_metrics'].get(metric, {})

            if not metric_data:
                f.write("  No data available\n")
                continue

            # Sort configurations by mean value
            if metric in self.distance_metrics:
                sorted_configs = sorted(metric_data.items(), key=lambda x: x[1]['mean'])
            else:
                sorted_configs = sorted(metric_data.items(), key=lambda x: x[1]['mean'], reverse=True)

            for i, (config_name, data) in enumerate(sorted_configs, 1):
                mean_val = data['mean']
                std_val = data['std']
                count = data['count']

                if metric in self.distance_metrics:
                    f.write(f"  {i:2d}. {config_name:<25} {mean_val:.4f} ± {std_val:.4f}m (n={count})\n")
                else:
                    f.write(f"  {i:2d}. {config_name:<25} {mean_val:.4f} ± {std_val:.4f} (n={count})\n")

        f.write("\n" + "═" * 120 + "\n")