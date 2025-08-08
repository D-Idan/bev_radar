"""
Configuration Comparison Metrics
Handles comparison between different tracking configurations.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ConfigurationComparison:
    """Results of comparing multiple configurations."""
    dataset_name: str
    configurations: List[str]
    baseline_config: str
    metrics_comparison: Dict[str, Dict[str, Any]]
    ranking: Dict[str, List[str]]
    overall_ranking: List[tuple]
    best_config: str
    improvement_summary: Dict[str, float]


class ConfigurationComparisonAnalyzer:
    """Analyzes and compares multiple tracking configurations against raw predictions baseline."""

    def __init__(self, key_metrics: Optional[List[str]] = None):
        """Initialize analyzer."""
        self.existing_comparison_data = None
        self.key_metrics = key_metrics or [
            'hota', 'mota', 'accuracy', 'det_a', 'precision',
            'avg_precision', 'avg_recall', 'avg_f1_score', 'ncle'
        ]
        self.distance_metrics = ['accuracy', 'motp', 'ncle']
        self.iou_metrics = ['camera_iou_mean']
        self.count_metrics = ['tp', 'fp']

    def load_existing_comparison(self, dataset_root: Path):
        """Load existing configuration comparison if it exists."""
        comparison_file = dataset_root / 'plots' / 'configuration_comparison' / 'configuration_comparison.json'
        if comparison_file.exists():
            with open(comparison_file, 'r') as f:
                self.existing_comparison_data = json.load(f)
                return True
        return False

    def compare_configurations(self, dataset_name: str, config_results: Dict[str, bool],
                               dataset_root: Path, baseline_config: str = 'network_output') -> ConfigurationComparison:

        """
        Compare tracking configurations against raw predictions baseline.

        Args:
            dataset_name: Name of the dataset
            config_results: Dictionary of configuration success status
            dataset_root: Root directory for the dataset
            baseline_config: Should be 'raw_predictions' for true baseline

        Returns:
            ConfigurationComparison object with analysis results
        """
        # Load existing comparison data if available
        self.load_existing_comparison(dataset_root)

        # Collect metrics from successful tracking configurations
        config_metrics = self._collect_configuration_metrics(config_results, dataset_root)

        # Merge with existing configurations if available
        if self.existing_comparison_data:
            existing_metrics = self.existing_comparison_data.get('metrics_comparison', {})
            # Add configurations from existing data that are not in current run
            for metric, metric_configs in existing_metrics.items():
                for config_name, config_data in metric_configs.items():
                    # Skip 'raw_predictions' if found in existing data
                    if config_name == 'raw_predictions':
                        continue
                    if config_name not in config_metrics:
                        config_metrics[config_name] = self._reconstruct_config_metrics_from_comparison(config_name,
                                                                                                       existing_metrics)

        # Add network output as baseline (not raw_predictions)
        network_output_metrics = self._collect_network_output_baseline(dataset_root)
        if network_output_metrics:
            config_metrics['network_output'] = network_output_metrics

        if not config_metrics:
            raise ValueError("No successful configurations found for comparison")

        # Use network output as baseline
        baseline_config = 'network_output'

        # Perform comparison analysis
        metrics_comparison = self._analyze_metrics_comparison(config_metrics, baseline_config)
        ranking = self._rank_configurations(config_metrics, baseline_config)
        overall_ranking = self._calculate_overall_ranking(ranking)
        improvement_summary = self._calculate_improvement_summary(metrics_comparison, baseline_config)

        return ConfigurationComparison(
            dataset_name=dataset_name,
            configurations=list(config_metrics.keys()),
            baseline_config=baseline_config,
            metrics_comparison=metrics_comparison,
            ranking=ranking,
            overall_ranking=overall_ranking,
            best_config=overall_ranking[0][0] if overall_ranking else None,
            improvement_summary=improvement_summary
        )

    def _extract_metric_value(self, metrics: Dict, config_name: str, metric: str, is_baseline: bool = False) -> float:
        """Extract metric value from metrics dictionary with consistent logic."""
        # Handle NCLE specifically from camera_iou_performance
        if metric == 'ncle':
            camera_iou = metrics.get('camera_iou_performance', {})
            if is_baseline or config_name == 'network_output':
                return camera_iou.get('detection_ncle', {}).get('mean', 0.0)
            else:
                return camera_iou.get('tracking_ncle', {}).get('mean', 0.0)

        if is_baseline or config_name == 'network_output':
            performance_data = metrics.get('detection_performance', {})
            source = 'detection'
        else:
            performance_data = metrics.get('tracking_performance', {})
            source = 'tracking'

        # Handle average metrics
        if metric in ['avg_precision', 'avg_recall', 'avg_f1_score']:
            avg_metrics = metrics.get('average_metrics', {})
            if avg_metrics and source in avg_metrics:
                metric_key = metric.replace('avg_', '')
                return avg_metrics[source].get(metric_key, 0.0)

        return performance_data.get(metric, 0.0)

    def _collect_network_output_baseline(self, dataset_root: Path) -> Optional[Dict]:
        """Collect metrics from network output (detection performance)."""
        # Look for evaluation metrics from any tracking configuration
        # (they all evaluate the same raw predictions)
        for config_dir in (dataset_root / 'plots' / 'tracking_output').iterdir():
            if config_dir.is_dir():
                metrics_file = config_dir / 'logs' / 'evaluation_metrics_summary.json'
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        full_metrics = json.load(f)

                    # Extract detection performance as baseline
                    detection_perf = full_metrics.get('detection_performance', {})
                    camera_iou = full_metrics.get('camera_iou_performance', {})

                    # Create baseline metrics structure
                    baseline_metrics = {
                        'evaluation_summary': full_metrics.get('evaluation_summary', {}),
                        'detection_performance': detection_perf,  # This becomes our baseline
                        'tracking_performance': detection_perf,  # Same as detection for baseline
                        'camera_iou_performance': camera_iou
                    }
                    return baseline_metrics

        return None

    def _collect_configuration_metrics(self, config_results: Dict[str, bool],
                                       dataset_root: Path) -> Dict[str, Dict]:
        """Collect metrics from all successful configurations."""
        config_metrics = {}

        for config_name, success in config_results.items():
            if not success:
                continue

            metrics_file = (dataset_root / 'plots' / 'tracking_output' /
                            config_name / 'logs' / 'evaluation_metrics_summary.json')

            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    config_metrics[config_name] = json.load(f)

        return config_metrics

    def _analyze_metrics_comparison(self, config_metrics: Dict[str, Dict],
                                    baseline_config: str) -> Dict[str, Dict]:
        """Analyze metrics comparison between tracking configs and raw predictions baseline."""
        comparison = {}

        # Get baseline metrics (raw predictions = detection performance)
        baseline_metrics = config_metrics.get(baseline_config, {}).get('detection_performance', {})
        baseline_iou = config_metrics.get(baseline_config, {}).get('camera_iou_performance', {})

        # Process standard metrics
        for metric in self.key_metrics:
            comparison[metric] = {}

            for config_name, metrics in config_metrics.items():
                metric_value = self._extract_metric_value(metrics, config_name, metric,
                                                          is_baseline=(config_name == baseline_config))

                if metric_value == float('inf'):
                    metric_value = None

                comparison[metric][config_name] = {
                    'value': metric_value,
                    'improvement_vs_baseline': None
                }

                # Calculate improvement vs baseline (skip for baseline itself)
                if config_name != baseline_config and baseline_metrics and metric in baseline_metrics and metric_value is not None:
                    baseline_value = baseline_metrics[metric]
                    if baseline_value != 0 and baseline_value != float('inf'):
                        if metric in self.distance_metrics:  # Lower is better
                            improvement = ((baseline_value - metric_value) / baseline_value) * 100
                        else:  # Higher is better
                            improvement = ((metric_value - baseline_value) / baseline_value) * 100
                        comparison[metric][config_name]['improvement_vs_baseline'] = improvement

        # Process single IoU metric
        comparison['camera_iou_mean'] = {}

        for config_name, metrics in config_metrics.items():
            camera_iou = metrics.get('camera_iou_performance', {})

            if config_name == baseline_config:
                # For raw predictions: use detection IoU
                metric_value = camera_iou.get('detection_camera_iou', {}).get('mean', 0.0)
            else:
                # For tracking configs: use tracking IoU
                metric_value = camera_iou.get('tracking_camera_iou', {}).get('mean', 0.0)

            comparison['camera_iou_mean'][config_name] = {
                'value': metric_value,
                'improvement_vs_baseline': None
            }

            # Calculate improvement vs baseline
            if config_name != baseline_config:
                baseline_value = baseline_iou.get('detection_camera_iou', {}).get('mean', 0.0)

                if baseline_value > 0 and metric_value > 0:
                    improvement = ((metric_value - baseline_value) / baseline_value) * 100
                    comparison['camera_iou_mean'][config_name]['improvement_vs_baseline'] = improvement

            # Process NCLE metric
            comparison['ncle'] = {}

            for config_name, metrics in config_metrics.items():
                camera_iou = metrics.get('camera_iou_performance', {})

                if config_name == baseline_config:
                    # For network output: use detection NCLE
                    metric_value = camera_iou.get('detection_ncle', {}).get('mean', 0.0)
                else:
                    # For tracking configs: use tracking NCLE
                    metric_value = camera_iou.get('tracking_ncle', {}).get('mean', 0.0)

                comparison['ncle'][config_name] = {
                    'value': metric_value,
                    'improvement_vs_baseline': None
                }

                # Calculate improvement vs baseline (lower is better for NCLE)
                if config_name != baseline_config:
                    baseline_value = config_metrics.get(baseline_config, {}).get('camera_iou_performance', {}).get(
                        'detection_ncle', {}).get('mean', 0.0)

                    if baseline_value > 0 and metric_value > 0:
                        improvement = ((baseline_value - metric_value) / baseline_value) * 100
                        comparison['ncle'][config_name]['improvement_vs_baseline'] = improvement

        return comparison

    def _rank_configurations(self, config_metrics: Dict[str, Dict],
                             baseline_config: str="network_output") -> Dict[str, List[str]]:
        """Rank configurations for each metric (including raw predictions baseline)."""
        ranking = {}

        # Process standard metrics
        for metric in self.key_metrics:
            metric_values = []

            for config_name, metrics in config_metrics.items():
                metric_value = self._extract_metric_value(metrics, config_name, metric,
                                                          is_baseline=(config_name == baseline_config))


                if metric_value != float('inf') and metric_value is not None:
                    metric_values.append((config_name, metric_value))

            if metric_values:
                # Handle metrics where lower is better
                if metric in self.distance_metrics:  # Lower is better (includes 'accuracy')
                    ranked = sorted(metric_values, key=lambda x: x[1])
                else:  # Higher is better (including precision_ratio)
                    ranked = sorted(metric_values, key=lambda x: x[1], reverse=True)

                ranking[metric] = [config_name for config_name, _ in ranked]

        # Process single IoU metric
        metric_values = []

        for config_name, metrics in config_metrics.items():
            camera_iou = metrics.get('camera_iou_performance', {})

            if config_name == 'network_output':
                metric_value = camera_iou.get('detection_camera_iou', {}).get('mean', 0.0)
            else:
                metric_value = camera_iou.get('tracking_camera_iou', {}).get('mean', 0.0)

            if metric_value > 0:
                metric_values.append((config_name, metric_value))

        if metric_values:
            # Higher IoU is better
            ranked = sorted(metric_values, key=lambda x: x[1], reverse=True)
            ranking['camera_iou_mean'] = [config_name for config_name, _ in ranked]

        return ranking

    def _calculate_overall_ranking(self, ranking: Dict[str, List[str]]) -> List[tuple]:
        """Calculate overall ranking across all metrics."""
        # Get all configurations
        all_configs = set()
        for config_list in ranking.values():
            all_configs.update(config_list)

        # Calculate overall scores
        overall_scores = {}
        all_metrics = self.key_metrics + self.iou_metrics

        for config_name in all_configs:
            score = 0
            valid_metrics = 0

            for metric in all_metrics:
                metric_ranking = ranking.get(metric, [])
                if config_name in metric_ranking:
                    rank_position = metric_ranking.index(config_name)
                    # Convert rank to score (best rank = highest score)
                    score += (len(metric_ranking) - rank_position)
                    valid_metrics += 1

            if valid_metrics > 0:
                overall_scores[config_name] = score / valid_metrics

        return sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)

    def _calculate_improvement_summary(self, metrics_comparison: Dict[str, Dict],
                                       baseline_config: str) -> Dict[str, float]:
        """Calculate summary of improvements for each configuration."""
        improvement_summary = {}

        for config_name in metrics_comparison.get(list(metrics_comparison.keys())[0], {}).keys():
            if config_name == baseline_config:
                continue

            improvements = []
            for metric in self.key_metrics:
                metric_data = metrics_comparison.get(metric, {}).get(config_name, {})
                improvement = metric_data.get('improvement_vs_baseline')
                if improvement is not None:
                    improvements.append(improvement)

            if improvements:
                improvement_summary[config_name] = np.mean(improvements)

        return improvement_summary

    def _reconstruct_config_metrics_from_comparison(self, config_name: str, existing_metrics: Dict) -> Dict:
        """Reconstruct config metrics structure from existing comparison data."""
        reconstructed = {
            'tracking_performance': {},
            'detection_performance': {},
            'camera_iou_performance': {}
        }

        # Extract values from existing comparison
        for metric, configs in existing_metrics.items():
            if config_name in configs:
                value = configs[config_name].get('value')
                if value is not None:
                    if metric == 'camera_iou_mean':
                        if config_name == 'network_output':
                            reconstructed['camera_iou_performance']['detection_camera_iou'] = {'mean': value}
                        else:
                            reconstructed['camera_iou_performance']['tracking_camera_iou'] = {'mean': value}
                    else:
                        if config_name == 'network_output':
                            reconstructed['detection_performance'][metric] = value
                        else:
                            reconstructed['tracking_performance'][metric] = value

                    # Handle average metrics
                    if metric in ['avg_precision', 'avg_recall', 'avg_f1_score']:
                        if config_name == 'network_output':
                            reconstructed['detection_performance'][metric] = value
                        else:
                            reconstructed['tracking_performance'][metric] = value
        return reconstructed

    def save_comparison_report(self, comparison: ConfigurationComparison, output_dir: Path):
        """Save configuration comparison report."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        comparison_data = {
            'dataset': comparison.dataset_name,
            'configurations_evaluated': comparison.configurations,
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'baseline_config': comparison.baseline_config,
            'metrics_comparison': comparison.metrics_comparison,
            'ranking': comparison.ranking,
            'overall_ranking': comparison.overall_ranking,
            'best_config': comparison.best_config,
            'improvement_summary': comparison.improvement_summary
        }

        json_file = output_dir / 'configuration_comparison.json'
        with open(json_file, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)

        # Save human-readable report
        self._save_human_readable_report(comparison, output_dir)

    def _save_human_readable_report(self, comparison: ConfigurationComparison, output_dir: Path):
        """Save human-readable comparison report."""
        txt_file = output_dir / 'configuration_comparison_summary.txt'

        with open(txt_file, 'w') as f:
            f.write("═" * 100 + "\n")
            f.write("                    TRACKING CONFIGURATION COMPARISON REPORT\n")
            f.write("═" * 100 + "\n\n")

            f.write(f"Dataset: {comparison.dataset_name}\n")
            f.write(f"Configurations Evaluated: {len(comparison.configurations)}\n")
            f.write(f"Baseline Configuration: {comparison.baseline_config}\n")
            f.write(f"Best Overall Configuration: {comparison.best_config}\n\n")

            # Overall ranking
            f.write("🏆 OVERALL PERFORMANCE RANKING\n")
            f.write("─" * 50 + "\n")
            for i, (config_name, score) in enumerate(comparison.overall_ranking, 1):
                improvement = comparison.improvement_summary.get(config_name, 0.0)
                f.write(f"{i:2d}. {config_name:<25} (Score: {score:.2f}, Avg Improvement: {improvement:+.1f}%)\n")
            f.write("\n")

            # Detailed metrics comparison
            f.write("📊 DETAILED METRICS COMPARISON\n")
            f.write("─" * 50 + "\n")

            all_metrics = self.key_metrics + self.iou_metrics

            for metric in all_metrics:
                # Format metric name for display
                if metric.startswith('avg_'):
                    display_name = metric.replace('avg_', 'Average ').replace('_', ' ').title()
                elif metric == 'accuracy':
                    display_name = "Accuracy (RA) [m]"
                else:
                    display_name = metric.replace('_', ' ').title()
                f.write(f"\n{display_name}:\n")
                f.write("  " + "─" * 80 + "\n")

                metric_data = comparison.metrics_comparison.get(metric, {})
                ranking = comparison.ranking.get(metric, [])

                for i, config_name in enumerate(ranking, 1):
                    data = metric_data.get(config_name, {})
                    value = data.get('value')
                    improvement = data.get('improvement_vs_baseline')

                    if value is None:
                        continue

                    improvement_str = ""
                    if improvement is not None:
                        improvement_str = f" ({improvement:+.1f}% vs baseline)"

                    if metric == 'accuracy':
                        f.write(f"  {i:2d}. {config_name:<25} {value:.4f}{improvement_str}\n")
                    elif metric in ['ncle', 'motp'] or 'iou' in metric:
                        f.write(f"  {i:2d}. {config_name:<25} {value:.4f}{improvement_str}\n")
                    else:
                        f.write(f"  {i:2d}. {config_name:<25} {value:.4f}{improvement_str}\n")

            f.write("\n" + "═" * 100 + "\n")