"""Example of using the car detection system from another script."""

from main import process_images_with_tracking, process_images_without_tracking
import pandas as pd


# Example 1: Process with tracking
def analyze_traffic_flow(image_folder: str):
    """Analyze traffic flow with consistent car tracking."""
    results_df = process_images_with_tracking(
        input_folder=image_folder,
        output_csv="traffic_analysis.csv",
        save_viz=True
    )

    # Analyze results
    print("\nTraffic Analysis:")
    print(f"Total vehicles detected: {results_df['ID'].nunique()}")

    # Count vehicles per frame
    vehicles_per_frame = results_df.groupby('numSample')['ID'].nunique()
    print(f"Average vehicles per frame: {vehicles_per_frame.mean():.2f}")

    return results_df


# Example 2: Custom detection parameters
def detect_large_vehicles(image_folder: str):
    """Detect only buses and trucks."""
    from config import MODEL_NAME

    results_df = process_images_without_tracking(
        input_folder=image_folder,
        output_csv="large_vehicles.csv",
        model_name=MODEL_NAME,
        car_classes=[5, 7],  # bus and truck only
        save_viz=True
    )

    return results_df


# Example 3: Post-process results
def filter_by_size(csv_path: str, min_area: int = 1000):
    """Filter detections by minimum bounding box area."""
    df = pd.read_csv(csv_path)

    # Calculate area
    df['area'] = (df['x2_pix'] - df['x1_pix']) * (df['y2_pix'] - df['y1_pix'])

    # Filter
    large_objects = df[df['area'] >= min_area]

    # Save filtered results
    large_objects.to_csv("large_objects.csv", index=False)

    return large_objects


if __name__ == "__main__":
    # Run analysis
    image_folder = "path/to/images"

    # Analyze traffic
    traffic_df = analyze_traffic_flow(image_folder)

    # Filter results
    filtered_df = filter_by_size("traffic_analysis.csv", min_area=2000)
    print(f"\nLarge vehicles: {len(filtered_df)}")