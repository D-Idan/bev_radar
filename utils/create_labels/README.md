## Usage Instructions:

1. **Install dependencies:**
   ```bash
   pip install ultralytics opencv-python pandas numpy tqdm
   ```

2. **Update configuration:**
   - Edit `config.py` and set `INPUT_FOLDER` to your image directory

3. **Run the main script:**
   ```python
   python main.py
   ```

4. **Choose tracking mode:**
   - Set `use_tracking = True` in `main.py` for consistent IDs across frames
   - Set `use_tracking = False` for unique IDs per detection

## Features:

- **Automatic car detection** using YOLOv8
- **Object tracking** for consistent IDs across frames
- **Sorted processing** by filename
- **CSV output** with all required columns
- **Visualization** with bounding boxes and IDs
- **Modular design** for easy integration and enhancement
- **Configurable** detection parameters
- **Multiple vehicle types** (cars, buses, trucks)

## Notes:

- The tracker assigns consistent IDs to the same car across multiple frames
- If you don't need tracking, use the non-tracking mode for faster processing
- You can adjust the model size in `config.py` for speed vs accuracy trade-off
- The visualization folder will contain images with drawn bounding boxes

This solution provides a robust, production-ready system for car detection and tracking that you can easily extend and integrate into other projects.