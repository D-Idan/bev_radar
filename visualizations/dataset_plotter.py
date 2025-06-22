import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
from collections import Counter


def analyze_dataset_frames(labels_csv_path, min_frames=10, output_path='dataset_frames_plot.png'):
    """
    Create comprehensive dataset analysis with multiple plots.

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
    filtered_datasets_by_count = filtered_datasets.sort_values('frame_count', ascending=False)

    # Sort by name for alphabetical plot
    filtered_datasets_by_name = filtered_datasets.sort_values('dataset', ascending=True)

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(28, 18))

    # Create a more flexible grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.25,
                          left=0.06, right=0.98, top=0.88, bottom=0.05)

    # Plot 1: Datasets sorted by frame count (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(range(len(filtered_datasets_by_count)), filtered_datasets_by_count['frame_count'],
                    color='steelblue', alpha=0.8, edgecolor='navy', linewidth=0.5)

    ax1.set_xlabel('Dataset Index (Sorted by Frame Count)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Frames', fontsize=14, fontweight='bold')
    ax1.set_title(f'Datasets by Frame Count (>{min_frames} frames) - {len(filtered_datasets_by_count)} datasets',
                  fontsize=16, fontweight='bold', pad=20)

    # Use simple index numbers for x-axis
    ax1.set_xticks(range(0, len(filtered_datasets_by_count), max(1, len(filtered_datasets_by_count) // 15)))
    ax1.set_xticklabels(
        [str(i + 1) for i in range(0, len(filtered_datasets_by_count), max(1, len(filtered_datasets_by_count) // 15))],
        fontsize=12)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on select bars
    for i, (bar, count) in enumerate(zip(bars1, filtered_datasets_by_count['frame_count'])):
        if i < 5 or i % max(1, len(filtered_datasets_by_count) // 8) == 0:  # Show top 5 and every nth
            ax1.text(bar.get_x() + bar.get_width() / 2.,
                     bar.get_height() + max(filtered_datasets_by_count['frame_count']) * 0.02,
                     f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 2: Datasets sorted alphabetically (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(range(len(filtered_datasets_by_name)), filtered_datasets_by_name['frame_count'],
                    color='forestgreen', alpha=0.8, edgecolor='darkgreen', linewidth=0.5)

    ax2.set_xlabel('Dataset Index (Alphabetical Order)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Frames', fontsize=14, fontweight='bold')
    ax2.set_title(f'Datasets Alphabetically (>{min_frames} frames)', fontsize=16, fontweight='bold', pad=20)

    ax2.set_xticks(range(0, len(filtered_datasets_by_name), max(1, len(filtered_datasets_by_name) // 15)))
    ax2.set_xticklabels(
        [str(i + 1) for i in range(0, len(filtered_datasets_by_name), max(1, len(filtered_datasets_by_name) // 15))],
        fontsize=12)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on select bars
    for i, (bar, count) in enumerate(zip(bars2, filtered_datasets_by_name['frame_count'])):
        if i < 5 or i % max(1, len(filtered_datasets_by_name) // 8) == 0:
            ax2.text(bar.get_x() + bar.get_width() / 2.,
                     bar.get_height() + max(filtered_datasets_by_name['frame_count']) * 0.02,
                     f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 3: Distribution histogram (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    n_bins = min(20, max(5, len(filtered_datasets) // 3))
    ax3.hist(filtered_datasets['frame_count'], bins=n_bins,
             color='orange', alpha=0.7, edgecolor='darkorange', linewidth=0.8)
    ax3.set_xlabel('Number of Frames', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Number of Datasets', fontsize=14, fontweight='bold')
    ax3.set_title('Distribution of Frame Counts', fontsize=16, fontweight='bold', pad=20)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # Add statistics lines
    mean_frames = filtered_datasets['frame_count'].mean()
    median_frames = filtered_datasets['frame_count'].median()
    ax3.axvline(mean_frames, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Mean: {mean_frames:.1f}')
    ax3.axvline(median_frames, color='purple', linestyle='--', alpha=0.8, linewidth=2,
                label=f'Median: {median_frames:.1f}')
    ax3.legend(fontsize=12)

    # Plot 4: Summary statistics (middle-right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Create summary statistics
    total_datasets = len(dataset_frame_counts)
    filtered_count = len(filtered_datasets)
    max_frames = filtered_datasets['frame_count'].max()
    min_frames_filtered = filtered_datasets['frame_count'].min()
    avg_frames = filtered_datasets['frame_count'].mean()
    std_frames = filtered_datasets['frame_count'].std()

    stats_text = f"""SUMMARY STATISTICS
{'=' * 40}

Total datasets in file: {total_datasets}
Datasets with >{min_frames} frames: {filtered_count}
Percentage filtered: {(filtered_count / total_datasets) * 100:.1f}%

FRAME COUNT STATISTICS (FILTERED)
{'=' * 40}
Maximum frames: {max_frames}
Minimum frames: {min_frames_filtered}
Average frames: {avg_frames:.1f}
Standard deviation: {std_frames:.1f}

TOP 10 DATASETS BY FRAME COUNT
{'=' * 40}"""

    # Add top 10 datasets
    top_10 = filtered_datasets_by_count.head(10)
    for i, (idx, row) in enumerate(top_10.iterrows(), 1):
        # Truncate dataset name if too long for display
        display_name = row['dataset'][:35] + "..." if len(row['dataset']) > 35 else row['dataset']
        stats_text += f"\n{i:2d}. {display_name:<38} ({row['frame_count']:3d} frames)"

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.9))

    ax4.set_title('Key Statistics & Top Datasets', fontsize=16, fontweight='bold')

    # Plot 5: Complete dataset list (bottom - spans both columns)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    # Create complete alphabetical list in multiple columns
    datasets_alpha = filtered_datasets_by_name['dataset'].tolist()
    frame_counts = filtered_datasets_by_name['frame_count'].tolist()

    # Calculate number of columns based on dataset count
    n_datasets = len(datasets_alpha)
    n_cols = min(4, max(2, n_datasets // 15))  # 2-4 columns
    n_rows = (n_datasets + n_cols - 1) // n_cols  # Ceiling division

    list_text = f"COMPLETE DATASET LIST - ALPHABETICAL ORDER (for downloading)\n"
    list_text += f"{'=' * 100}\n"
    list_text += f"Total: {n_datasets} datasets with >{min_frames} frames\n\n"

    # Create multi-column layout
    for row in range(n_rows):
        line = ""
        for col in range(n_cols):
            idx = row + col * n_rows
            if idx < n_datasets:
                dataset_name = datasets_alpha[idx]
                frame_count = frame_counts[idx]
                # Format each entry: "001. dataset_name (123 frames)"
                entry = f"{idx + 1:3d}. {dataset_name:<45} ({frame_count:3d} frames)"
                if col < n_cols - 1:  # Not the last column
                    entry = entry[:70] + " | "  # Truncate and add separator
                line += entry
        list_text += line + "\n"

    ax5.text(0.02, 0.98, list_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.9))

    ax5.set_title('All Filtered Datasets - Ready for Download', fontsize=16, fontweight='bold')

    # Add overall title with proper spacing
    fig.suptitle(f'Comprehensive Dataset Analysis - Frame Counts > {min_frames}',
                 fontsize=22, fontweight='bold', y=0.95)

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Multi-plot analysis saved to: {output_path}")

    # Display complete dataset list in console for easy copying
    print(f"\n{'=' * 80}")
    print("COMPLETE DATASET LIST - ALPHABETICAL ORDER (for downloading)")
    print('=' * 80)
    for i, (dataset, frames) in enumerate(zip(datasets_alpha, frame_counts), 1):
        print(f"{i:3d}. {dataset:<60} ({frames:3d} frames)")

    print(f"\n{'=' * 80}")
    print(f"SUMMARY: {n_datasets} datasets with >{min_frames} frames")
    print(f"Average frames per dataset: {avg_frames:.1f}")
    print(f"Total frames across all filtered datasets: {sum(frame_counts)}")
    print('=' * 80)

    # Show the plot
    plt.show()

    return filtered_datasets_by_count


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
    parser = argparse.ArgumentParser(description='Create comprehensive dataset analysis with multiple plots')
    parser.add_argument('labels_csv', help='Path to the labels.csv file')
    parser.add_argument('--min_frames', type=int, default=10,
                        help='Minimum number of frames required (default: 10)')
    parser.add_argument('--output', default='dataset_comprehensive_analysis.png',
                        help='Output path for the plot (default: dataset_comprehensive_analysis.png)')
    parser.add_argument('--show_sample_stats', action='store_true',
                        help='Show additional statistics about samples per frame')

    args = parser.parse_args()

    # Create the comprehensive analysis
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