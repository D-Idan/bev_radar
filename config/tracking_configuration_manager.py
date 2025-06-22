"""
Tracking Configuration Manager with YAML support
Generates and manages multiple tracking configurations for comparison studies.
"""

import copy
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path


class TrackingConfigurationManager:
    """Manages multiple tracking configurations from YAML file."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager."""
        if config_file is None:
            config_file = Path(__file__).parent / "tracking_configurations.yaml"

        self.config_file = Path(config_file)
        self.config_data = self._load_configuration_file()

    def _load_configuration_file(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")

        with open(self.config_file, 'r') as f:
            return yaml.safe_load(f)

    def get_available_configurations(self) -> List[str]:
        """Get list of all available configuration names."""
        return list(self.config_data.get('configurations', {}).keys())

    def get_default_configurations(self) -> List[str]:
        """Get list of default configurations to run."""
        return self.config_data.get('default_configurations', self.get_available_configurations())

    def get_configuration_description(self, config_name: str) -> str:
        """Get description of a specific configuration."""
        config = self.config_data.get('configurations', {}).get(config_name, {})
        return config.get('description', f"Configuration: {config_name}")

    def get_baseline_config(self) -> str:
        """Get the name of the baseline configuration."""
        return self.config_data.get('baseline_config', 'baseline')

    def generate_configurations(self, base_config: Dict[str, Any],
                                output_base: Path, target_value: str,
                                specific_configs: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Generate tracking configurations based on YAML definitions.

        Args:
            base_config: Base tracking configuration
            output_base: Base output directory path
            target_value: Dataset name
            specific_configs: List of specific configurations to generate (None = default configs)

        Returns:
            Dictionary mapping configuration names to complete configurations
        """
        # Determine which configurations to generate
        if specific_configs is None:
            configs_to_generate = self.get_default_configurations()
        else:
            # Validate requested configurations
            available = self.get_available_configurations()
            invalid_configs = [c for c in specific_configs if c not in available]
            if invalid_configs:
                raise ValueError(f"Invalid configurations requested: {invalid_configs}. "
                                 f"Available: {available}")
            configs_to_generate = specific_configs

        configurations = {}
        dataset_root = output_base / target_value

        for config_name in configs_to_generate:
            config_def = self.config_data['configurations'][config_name]
            config = copy.deepcopy(base_config)

            # Apply overrides from YAML
            overrides = config_def.get('overrides', {})
            for key, value in overrides.items():
                config['tracker_config'][key] = value

            # Update output directory to include config name
            config['output_dir'] = str(dataset_root / 'plots' / 'tracking_output' / config_name)

            # Store metadata separately and add it to a wrapper
            config_with_metadata = {
                **config,
                'config_metadata': {
                    'name': config_name,
                    'description': config_def.get('description', ''),
                    'overrides_applied': overrides
                }
            }

            configurations[config_name] = config_with_metadata

        return configurations

    def print_available_configurations(self):
        """Print all available configurations with descriptions."""
        print("\nAvailable Tracking Configurations:")
        print("=" * 50)

        configs = self.config_data.get('configurations', {})
        for name, config in configs.items():
            description = config.get('description', 'No description')
            overrides = config.get('overrides', {})

            print(f"\n📋 {name}")
            print(f"   Description: {description}")
            print(f"   Overrides: {len(overrides)} parameters")
            for key, value in overrides.items():
                print(f"     - {key}: {value}")

        default_configs = self.get_default_configurations()
        print(f"\n🎯 Default configurations: {', '.join(default_configs)}")
        print(f"🏠 Baseline configuration: {self.get_baseline_config()}")