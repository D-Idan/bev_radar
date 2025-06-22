# BEV Radar: SORT-Oriented Deep Learning Radar Tracking with Confidence Integration

A research framework investigating the application of SORT (Simple Online Realtime Tracking) algorithms to deep neural network outputs on radar datasets, with novel confidence score integration in tracking and Kalman filtering.

## 🎯 Research Overview

This thesis work addresses a fundamental question in radar-based object detection: **Does post-processing deep learning detection outputs with tracking algorithms improve overall performance?** The research specifically focuses on integrating detection confidence scores into SORT-based tracking algorithms and Kalman filter measurement noise adaptation.

### Key Research Contributions

- **SORT-Oriented Tracking**: Implementation of SORT-like algorithms specifically designed for radar deep learning outputs
- **Confidence Score Integration**: Novel methods for incorporating neural network confidence scores into:
  - Track association strategies (5 different approaches)
  - Kalman filter measurement noise (R matrix) weighting
- **Variable Time Handling**: Robust temporal processing for real-world radar data with irregular frame rates
- **Comprehensive Evaluation**: Systematic comparison of tracking configurations against raw detection baselines

### Research Questions

1. How does SORT-based tracking improve upon raw neural network detection performance?
2. What is the optimal way to integrate confidence scores into tracking association?
3. How does confidence-based R matrix weighting affect Kalman filter performance?
4. Which association strategy performs best under different scenarios?

## 🏗️ System Architecture

### Core Pipeline
```
Raw Radar Data → T_FFTRadNet (ViT) → Confidence-Based SORT Tracking → Performance Evaluation
```

### Major Components

1. **T_FFTRadNet**: Vision Transformer with FFT preprocessing for radar object detection
2. **Confidence-Enhanced SORT**: Multiple association strategies with confidence integration
3. **Adaptive Kalman Filtering**: R matrix weighting based on detection confidence
4. **Comprehensive Evaluation**: Multi-metric comparison framework

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision numpy pandas tqdm opencv-python imageio matplotlib seaborn pyyaml
cd ADCProcessing/DBReader && pip install .
```

### Basic Usage

The main entry point is `main_pipeline.py` which orchestrates the complete pipeline:

#### Run Complete Pipeline (Default Configurations)
```bash
python main_pipeline.py --target RECORD@2020-11-22_12.45.05
```

#### List Available Configurations
```bash
python main_pipeline.py --list-tracking-configs
```

#### Run Specific Configurations
```bash
python main_pipeline.py --target RECORD@2020-11-22_12.45.05 \
    --tracking-configs baseline adaptive_assoc_linear adaptive_both_linear
```

#### Process All Datasets
```bash
python main_pipeline.py
```

## ⚙️ Configuration System

The pipeline uses YAML-based configuration with two main files:

### `config/pipeline_config.yaml`
- Tracking parameters (association strategies, confidence thresholds)
- Processing options (video creation, frame limits)
- Path configurations
- Output settings

### `config/radar_model_config.yaml`
- Radar system parameters (coverage, geometry)
- Neural network model architecture
- Data processing statistics

### Key Configuration Parameters

#### Association Strategies
```yaml
association:
  strategy: "mahalanobis_distance"  # Options:
    # - "distance_only": Pure Euclidean distance
    # - "confidence_weighted": Distance weighted by confidence  
    # - "confidence_gated": Confidence threshold filtering
    # - "hybrid_score": Combined distance + confidence
    # - "mahalanobis_distance": Statistical distance with gating
```

#### Confidence-Based R Matrix Weighting
```yaml
adaptive_r:
  use_in_association: false  # Apply to Mahalanobis distance calculation
  use_in_update: false       # Apply to Kalman filter update
  weighting_strategy: "linear"  # "linear", "squared", "stepped"
  config:
    r_min_factor: 0.3   # High confidence: 30% of base R
    r_max_factor: 15.0  # Low confidence: 15x base R
```

## 📊 Output Structure & Research Metrics

All outputs are organized in the `data/` directory:

```
data/
├── RECORD@2020-11-22_12.45.05/           # Dataset outputs
│   ├── plots/
│   │   ├── predictions/                   # Neural network outputs
│   │   ├── tracking_output/               # Per-configuration results
│   │   │   ├── baseline/                  # Raw detection performance
│   │   │   ├── adaptive_assoc_linear/     # Confidence-weighted association
│   │   │   └── adaptive_both_linear/      # Full confidence integration
│   │   └── configuration_comparison/      # Cross-configuration analysis
└── aggregate_analysis/                    # Multi-dataset comparison
```

### Research Metrics

The system evaluates multiple performance dimensions:

#### Detection Metrics
- **Precision, Recall, F1-Score**: Standard detection performance
- **DetA**: Detection accuracy metric

#### Tracking Metrics  
- **MOTA/MOTP**: Multi-Object Tracking Accuracy/Precision
- **Track Purity/Completeness**: Track quality measures
- **Mean Euclidean Distance**: Spatial accuracy

#### Confidence Integration Metrics
- **Association Success Rate**: How well confidence improves associations
- **R Matrix Effectiveness**: Impact of adaptive measurement noise

### Key Research Outputs

1. **Configuration Comparison Reports**: Systematic comparison of tracking strategies against raw detection baseline
2. **Aggregate Analysis**: Cross-dataset performance trends and optimal configurations
3. **Confidence Correlation Analysis**: Relationship between confidence scores and tracking performance

## 🔬 Research Methodology

### Tracking Configurations Evaluated

1. **Baseline (`raw_predictions`)**: Neural network outputs without tracking
2. **Standard SORT (`baseline`)**: Distance-only association 
3. **Confidence Association (`adaptive_assoc_*`)**: Confidence-weighted association strategies
4. **Full Integration (`adaptive_both_*`)**: Confidence in both association and R matrix

### Association Strategy Comparison

| Strategy | Description | Research Focus |
|----------|-------------|----------------|
| `distance_only` | Pure Euclidean distance | Baseline SORT performance |
| `confidence_weighted` | Distance × confidence factor | Simple confidence integration |
| `confidence_gated` | Confidence threshold + distance | Selective confidence filtering |
| `hybrid_score` | Combined distance-confidence score | Balanced integration approach |
| `mahalanobis_distance` | Statistical distance with gating | Principled uncertainty handling |

### R Matrix Weighting Strategies

- **Linear**: `R = base_R × (max_factor - (max_factor - min_factor) × confidence)`
- **Squared**: `R = base_R × (max_factor - (max_factor - min_factor) × confidence²)`
- **Stepped**: Discrete confidence thresholds with specific R factors

## 📈 Research Insights

### Expected Research Outcomes

1. **Tracking Improvement**: Quantified improvement of SORT tracking over raw detections
2. **Confidence Integration Benefits**: Measurable gains from confidence-based association
3. **R Matrix Adaptation**: Effectiveness of confidence-weighted measurement noise
4. **Optimal Strategy Identification**: Best performing configuration across different scenarios

### Evaluation Framework

The system generates comprehensive research reports including:
- **Performance Rankings**: Configurations ranked by multiple metrics
- **Statistical Significance**: Error bars and confidence intervals
- **Cross-Dataset Validation**: Generalization across different radar sequences
- **Ablation Studies**: Individual component contribution analysis

## 🛠️ Advanced Research Features

### Variable Time Interval Handling
Unlike synthetic datasets, RadIal has irregular frame rates. The system:
- Uses actual timestamp differences for Kalman prediction
- Handles large temporal gaps with multi-step prediction
- Automatically culls tracks during extended interruptions

### Confidence Score Utilization
The research explores multiple ways to use neural network confidence:
- **Association Gating**: Only consider high-confidence detections
- **Distance Weighting**: Scale association costs by confidence
- **Measurement Noise**: Adapt Kalman R matrix based on confidence
- **Track Initialization**: Require high confidence for new tracks

### Multi-Dataset Analysis
The framework supports:
- **Cross-Dataset Validation**: Test configurations on multiple radar sequences
- **Aggregate Performance**: Statistical analysis across all datasets
- **Configuration Robustness**: Identify strategies that work consistently

## 📚 Research Documentation

### Key Files for Research Understanding

- **`utils/metrics/configuration_comparison.py`**: Cross-configuration evaluation framework
- **`utils/reports/aggregate_analysis.py`**: Multi-dataset analysis generation
- **`radar_tracking/`**: Core SORT implementation with confidence integration
- **`config/tracking_configuration_manager.py`**: Automated configuration generation

### Research Paper Documentation

Algorithm details and research findings are documented in:
- `readme_offline_tracker.md`: Detailed algorithmic descriptions
- `utils/metrics/metrics.md`: Evaluation methodology
- Generated reports in `data/aggregate_analysis/`: Research results

## 🎯 Research Impact

This work contributes to:
1. **Radar Object Detection**: Novel confidence integration approaches
2. **Multi-Object Tracking**: SORT algorithm enhancements for radar data
3. **Deep Learning + Tracking**: Principled fusion of neural networks and tracking
4. **Automotive Applications**: Real-world radar processing improvements

## 🤝 Usage for Research

### For Thesis Writing
```bash
# Generate comprehensive results for multiple configurations
python main_pipeline.py --target RECORD@2020-11-22_12.45.05 \
    --tracking-configs baseline adaptive_assoc_linear adaptive_assoc_squared \
                      adaptive_both_linear adaptive_both_squared

# View generated research reports
ls data/RECORD@2020-11-22_12.45.05/plots/configuration_comparison/
ls data/aggregate_analysis/
```

### For Algorithm Development
```bash
# Test single configuration for development
python main_pipeline.py --target RECORD@2020-11-22_12.45.05 \
    --tracking-configs baseline

# List all available configurations
python main_pipeline.py --list-tracking-configs
```

---

*This framework provides a comprehensive foundation for investigating confidence-based SORT tracking in radar applications, with systematic evaluation capabilities for academic research.*