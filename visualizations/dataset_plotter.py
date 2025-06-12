import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
from collections import Counter


def analyze_dataset_frames(labels_csv_path, min_frames=10, output_path='dataset_frames_plot.png'):
    """
    Plot dataset values that have more than a chosen number of frames.

    Args:
        labels_csv_path: Path to the labels.csv file
        min_frames: Minimum number of frames required for a dataset to be included
        output_path: Path to save the output figure
    """
    # Read the CSV file
    df = pd.read_csv(labels_csv_path)

    # Count frames per dataset (using 'index' column as frame identifier)
    dataset_frame_counts = df.groupby('dataset')['index'].nunique().reset_index()
    dataset_frame_counts.columns = ['dataset', 'frame_count']

    # Filter datasets with more than min_frames
    filtered_datasets = dataset_frame_counts[dataset_frame_counts['frame_count'] > min_frames]

    if filtered_datasets.empty:
        print(f"No datasets found with more than {min_frames} frames")
        return

    # Sort by frame count for better visualization
    filtered_datasets = filtered_datasets.sort_values('frame_count', ascending=False)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Create bar plot
    bars = plt.bar(range(len(filtered_datasets)), filtered_datasets['frame_count'],
                   color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)

    # Customize the plot
    plt.xlabel('Dataset', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Frames', fontsize=12, fontweight='bold')
    plt.title(f'Datasets with More Than {min_frames} Frames\n({len(filtered_datasets)} datasets shown)',
              fontsize=14, fontweight='bold')

    # Set x-axis labels
    plt.xticks(range(len(filtered_datasets)), filtered_datasets['dataset'],
               rotation=45, ha='right')

    # Add value labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, filtered_datasets['frame_count'])):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                 f'{count}', ha='center', va='bottom', fontweight='bold')

    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Display statistics
    print(f"\nDataset Statistics:")
    print(f"Total datasets: {len(dataset_frame_counts)}")
    print(f"Datasets with >{min_frames} frames: {len(filtered_datasets)}")
    print(f"Max frames: {filtered_datasets['frame_count'].max()}")
    print(f"Min frames (filtered): {filtered_datasets['frame_count'].min()}")
    print(f"Average frames (filtered): {filtered_datasets['frame_count'].mean():.1f}")

    # Show top datasets
    print(f"\nTop 10 datasets by frame count:")
    for idx, row in filtered_datasets.head(10).iterrows():
        print(f"  {row['dataset']}: {row['frame_count']} frames")

    # Show the plot
    plt.show()

    return filtered_datasets


def get_dataset_sample_distribution(labels_csv_path):
    """
    Analyze the distribution of samples per frame for each dataset.

    Args:
        labels_csv_path: Path to the labels.csv file
    """
    df = pd.read_csv(labels_csv_path)

    # Count samples per frame per dataset
    samples_per_frame = df.groupby(['dataset', 'index']).size().reset_index(name='sample_count')

    # Get statistics per dataset
    dataset_stats = samples_per_frame.groupby('dataset')['sample_count'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)
    dataset_stats.columns = ['total_frames', 'avg_samples_per_frame', 'std_samples_per_frame',
                             'min_samples_per_frame', 'max_samples_per_frame']

    return dataset_stats


def main():
    parser = argparse.ArgumentParser(description='Plot datasets with frame counts above threshold')
    parser.add_argument('labels_csv', help='Path to the labels.csv file')
    parser.add_argument('--min_frames', type=int, default=10,
                        help='Minimum number of frames required (default: 10)')
    parser.add_argument('--output', default='dataset_frames_plot.png',
                        help='Output path for the plot (default: dataset_frames_plot.png)')
    parser.add_argument('--show_sample_stats', action='store_true',
                        help='Show additional statistics about samples per frame')

    args = parser.parse_args()

    # Create the main plot
    filtered_datasets = analyze_dataset_frames(args.labels_csv, args.min_frames, args.output)

    # Show sample distribution statistics if requested
    if args.show_sample_stats:
        print("\n" + "=" * 50)
        print("SAMPLE DISTRIBUTION STATISTICS")
        print("=" * 50)
        sample_stats = get_dataset_sample_distribution(args.labels_csv)
        print(sample_stats)


if __name__ == "__main__":
    main()

# Example usage:
# python dataset_plotter.py /path/to/labels.csv --min_frames 20 --output my_plot.png --show_sample_stats

# ## Key Features:
#
# 1. **Frame Count Analysis**: Counts the number of unique frames (using the `index` column) for each dataset
# 2. **Filtering**: Only includes datasets that have more than a specified minimum number of frames
# 3. **Visualization**: Creates a bar chart showing datasets and their frame counts
# 4. **Statistics**: Provides detailed statistics about the filtered datasets
# 5. **Sample Distribution**: Optional analysis of how many samples exist per frame
#
# ## How to Use:
#
# ```bash
# # Basic usage
# python dataset_plotter.py /path/to/labels.csv --min_frames 15 --output my_dataset_plot.png
#
# # With additional sample statistics
# python dataset_plotter.py /path/to/labels.csv --min_frames 10 --show_sample_stats
# ```
#
# ## Command Line Arguments:
#
# - `labels_csv`: Path to your labels.csv file (required)
# - `--min_frames`: Minimum number of frames required (default: 10)
# - `--output`: Output path for the plot (default: 'dataset_frames_plot.png')
# - `--show_sample_stats`: Show additional statistics about samples per frame
#
# ## Output Features:
#
# 1. **Bar Chart**: Shows datasets with frame counts above the threshold
# 2. **Value Labels**: Each bar shows the exact frame count
# 3. **Statistics Summary**: Displays total datasets, filtered count, max/min/average frames
# 4. **Top 10 List**: Shows the datasets with highest frame counts
# 5. **High-Quality Save**: Saves plot at 300 DPI with proper formatting
#
# The script handles the relationship between samples and frames correctly - since each frame can have multiple samples (multiple objects detected), it counts unique frame indices per dataset rather than just counting rows.