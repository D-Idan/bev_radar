# Radar Tracking Visualization Package

## Overview
A modular visualization system for radar tracking data, supporting camera views, bird's eye view (BEV), range-doppler maps, and temporal analysis.

## Directory Structure

```
visualizations/
├── core/                 # Core functionality and shared utilities
│   ├── base_visualizer.py    # Abstract base class for all visualizers
│   ├── config.py             # Configuration (camera/radar params, styling)
│   └── utils.py              # Data loading, coordinate transforms
│
├── components/           # Individual visualization components
│   ├── camera_view.py        # Camera image with bounding boxes
│   ├── bev_view.py           # Bird's eye view visualization
│   ├── radar_maps.py         # Range-Doppler and Range-Azimuth maps
│   └── annotations.py        # Annotation helpers (legends, labels)
│
├── analysis/            # Analysis modules
│   ├── temporal_analysis.py  # Time-series analysis, evolution plots
│   ├── tracking_analysis.py  # Performance metrics, confidence analysis
│   └── gap_analysis.py       # Time gap detection and visualization
│
├── plotters/            # Output generation
│   ├── frame_plotter.py      # Single frame visualizations
│   ├── sequence_plotter.py   # Multi-frame plots (3D, temporal)
│   └── video_generator.py    # Video creation (FFmpeg/OpenCV)
│
└── tools/               # High-level interfaces
    ├── comprehensive_tool.py # Main tool combining all components
    └── timing_tool.py        # Standalone timing analysis
```

## Quick Start

```python
from visualizations import ComprehensiveVisualizationTool

# Create single frame visualization
tool = ComprehensiveVisualizationTool()
fig = tool.create_frame_visualization(
    sample_id=100,
    labels_df=labels,
    predictions_df=predictions,
    tracking_df=tracks,
    data_paths={'image': ..., 'rd': ..., 'ra': ...}
)

# Generate video
video_path = tool.create_tracking_video(
    data_dir=Path("data"),
    output_dir=Path("output"),
    labels_csv="labels.csv",
    predictions_csv="predictions.csv"
)
```

## Adding New Features

### 1. New Visualization Component
Create in `components/`:
```python
# components/new_view.py
from ..core.base_visualizer import BaseVisualizer

class NewView(BaseVisualizer):
    def create(self, ax, data, **kwargs):
        # Your visualization logic
        pass
```

### 2. New Analysis Type
Add to `analysis/`:
```python
# analysis/new_analysis.py
class NewAnalyzer:
    def analyze(self, data):
        # Analysis logic
        return results
    
    def plot_results(self, results, ax):
        # Plotting logic
        pass
```

### 3. Custom Styling
Modify `core/config.py`:
```python
self.style_config['new_element'] = {
    'color': 'purple',
    'marker': 's',
    'size_base': 100,
    'alpha': 0.8
}
```

## Debugging Tips

### Common Issues

1. **Missing Data Files**
   - Check paths in `DataLoader.load_*` methods
   - Verify file existence before processing
   - Look for error messages in console output

2. **Coordinate System Issues**
   - `CoordinateTransform` in `core/utils.py` handles all conversions
   - BEV: X=lateral (right+), Y=forward (forward+)
   - Camera: Check `world_to_image` implementation

3. **Visualization Artifacts**
   - Check `zorder` values for layer ordering
   - Verify alpha values for transparency
   - Review axis limits and scaling

### Debug Mode
Enable verbose logging:
```python
# In any component
if self.config.debug:
    print(f"Processing {len(data)} points...")
```

### Data Flow
1. **Input**: CSV files + numpy arrays (radar maps) + images
2. **Loading**: `DataLoader` → filtered DataFrames
3. **Processing**: Components process data independently
4. **Rendering**: Matplotlib axes → Figure → File/Video

## Key Classes

- **VisualizationConfig**: Central configuration hub
- **BaseVisualizer**: Interface for all visualization components
- **DataLoader**: Handles all file I/O operations
- **CoordinateTransform**: Manages coordinate system conversions
- **ComprehensiveVisualizationTool**: Main entry point

## Performance Considerations

- Video generation: Use FFmpeg when available (faster than OpenCV)
- Large datasets: Process in batches, use generators
- Memory: Close matplotlib figures after saving (`plt.close()`)

## Dependencies
- numpy, pandas, matplotlib
- opencv-python (cv2)
- polarTransform (for BEV conversion)
- ffmpeg (optional, for video generation)