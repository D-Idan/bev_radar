import pandas as pd
import numpy as np


def calculate_time_differences(labels_csv_path):
    """
    Calculate time differences between consecutive samples.

    Args:
        labels_csv_path: Path to the labels.csv file with timestamp_us column

    Returns:
        DataFrame with additional columns for time differences
    """
    # Read the labels
    df = pd.read_csv(labels_csv_path)

    # Ensure data is sorted by timestamp
    df = df.sort_values('timestamp_us')

    # Calculate time difference in microseconds
    df['time_diff_us'] = df['timestamp_us'].diff()

    # Convert to seconds
    df['time_diff_seconds'] = df['time_diff_us'] / 1_000_000

    # Calculate time since start (in seconds)
    df['time_since_start_seconds'] = (df['timestamp_us'] - df['timestamp_us'].iloc[0]) / 1_000_000

    return df


def get_time_between_samples(labels_csv_path, sample1_num, sample2_num):
    """
    Get time difference in seconds between two specific samples.

    Args:
        labels_csv_path: Path to the labels.csv file
        sample1_num: numSample value of first sample
        sample2_num: numSample value of second sample

    Returns:
        Time difference in seconds (positive if sample2 is later than sample1)
    """
    df = pd.read_csv(labels_csv_path)

    # Get timestamps for both samples
    ts1 = df[df['numSample'] == sample1_num]['timestamp_us'].iloc[0]
    ts2 = df[df['numSample'] == sample2_num]['timestamp_us'].iloc[0]

    # Calculate difference in seconds
    time_diff_seconds = (ts2 - ts1) / 1_000_000

    return time_diff_seconds


def get_sample_rate_statistics(labels_csv_path):
    """
    Calculate statistics about the sampling rate.

    Args:
        labels_csv_path: Path to the labels.csv file

    Returns:
        Dictionary with statistics about timing
    """
    df = calculate_time_differences(labels_csv_path)

    # Remove NaN from first row
    time_diffs = df['time_diff_seconds'].dropna()

    stats = {
        'mean_interval_seconds': time_diffs.mean(),
        'std_interval_seconds': time_diffs.std(),
        'min_interval_seconds': time_diffs.min(),
        'max_interval_seconds': time_diffs.max(),
        'mean_fps': 1.0 / time_diffs.mean() if time_diffs.mean() > 0 else 0,
        'total_duration_seconds': df['time_since_start_seconds'].iloc[-1]
    }

    return stats


# Example usage:
if __name__ == "__main__":
    # Example path to your labels.csv file
    labels_path = "/path/to/RadIal_Data/RECORD@2020-11-22_12.45.05/labels.csv"

    # Calculate time differences for all samples
    df_with_times = calculate_time_differences(labels_path)
    print("First few rows with time information:")
    print(df_with_times[['numSample', 'timestamp_us', 'time_diff_seconds', 'time_since_start_seconds']].head())

    # Get time between specific samples
    time_diff = get_time_between_samples(labels_path, sample1_num=100, sample2_num=200)
    print(f"\nTime between samples 100 and 200: {time_diff:.3f} seconds")

    # Get sampling statistics
    stats = get_sample_rate_statistics(labels_path)
    print("\nSampling statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")