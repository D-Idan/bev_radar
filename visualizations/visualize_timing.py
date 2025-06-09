import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_timing_analysis(labels_csv_path, output_path=None, remove_outliers_for_hist=True):
    """
    Create visualizations of timing information from labels.csv

    Args:
        labels_csv_path: Path to labels.csv with timestamp information
        output_path: Optional path to save the plot
        remove_outliers_for_hist: If True, removes outliers for histogram visualization
    """
    # Read and prepare data
    df = pd.read_csv(labels_csv_path)
    df = df.sort_values('timestamp_us')

    # Calculate time differences in seconds
    df['time_diff_s'] = df['timestamp_us'].diff() / 1_000_000  # Convert to seconds
    df['time_since_start_s'] = (df['timestamp_us'] - df['timestamp_us'].iloc[0]) / 1_000_000

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Time intervals between consecutive samples
    axes[0].plot(df['numSample'].iloc[1:], df['time_diff_s'].iloc[1:], 'b-', linewidth=0.5)
    axes[0].set_xlabel('Sample Number', fontsize=12)
    axes[0].set_ylabel('Time Interval (seconds)', fontsize=12)
    axes[0].set_title('Time Intervals Between Consecutive Samples', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Add mean line and statistics
    mean_interval = df['time_diff_s'].mean()
    median_interval = df['time_diff_s'].median()
    axes[0].axhline(y=mean_interval, color='r', linestyle='--',
                    label=f'Mean: {mean_interval:.3f} s', linewidth=2)
    axes[0].axhline(y=median_interval, color='g', linestyle=':',
                    label=f'Median: {median_interval:.3f} s', linewidth=2)

    # Identify and annotate outliers
    outlier_threshold = df['time_diff_s'].quantile(0.99)
    outliers = df[df['time_diff_s'] > outlier_threshold]
    if len(outliers) > 0:
        for idx, row in outliers.iterrows():
            axes[0].annotate(f'{row["time_diff_s"]:.1f}s',
                             xy=(row['numSample'], row['time_diff_s']),
                             xytext=(row['numSample'], row['time_diff_s'] * 0.8),
                             arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                             fontsize=10, color='red')

    axes[0].legend(fontsize=10)
    axes[0].set_ylim(bottom=-0.1)  # Start from near 0

    # Plot 2: Histogram of time intervals
    # For better visualization, we'll create two histograms: one for normal values and one including outliers
    time_diffs_clean = df['time_diff_s'].dropna()

    if remove_outliers_for_hist:
        # Calculate percentiles for better visualization
        p5 = time_diffs_clean.quantile(0.05)
        p95 = time_diffs_clean.quantile(0.95)

        # Filter data for main histogram
        time_diffs_filtered = time_diffs_clean[(time_diffs_clean >= p5) & (time_diffs_clean <= p95)]

        # Create main histogram
        n, bins, patches = axes[1].hist(time_diffs_filtered, bins=50,
                                        edgecolor='black', alpha=0.7, color='skyblue')

        # Add text about outliers
        n_outliers = len(time_diffs_clean) - len(time_diffs_filtered)
        if n_outliers > 0:
            axes[1].text(0.98, 0.95, f'{n_outliers} outliers removed\n(outside 5th-95th percentile)',
                         transform=axes[1].transAxes,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                         horizontalalignment='right', verticalalignment='top')
    else:
        # Show all data
        axes[1].hist(time_diffs_clean, bins=100, edgecolor='black', alpha=0.7, color='skyblue')

    axes[1].set_xlabel('Time Interval (seconds)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Time Intervals (seconds)', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add vertical lines for mean and median
    axes[1].axvline(x=mean_interval, color='r', linestyle='--',
                    label=f'Mean: {mean_interval:.3f} s', linewidth=2)
    axes[1].axvline(x=median_interval, color='g', linestyle=':',
                    label=f'Median: {median_interval:.3f} s', linewidth=2)
    axes[1].legend(fontsize=10)

    # Plot 3: Cumulative time vs sample number
    axes[2].plot(df['numSample'], df['time_since_start_s'], 'g-', linewidth=2, label='Actual')
    axes[2].set_xlabel('Sample Number', fontsize=12)
    axes[2].set_ylabel('Time Since Start (seconds)', fontsize=12)
    axes[2].set_title('Cumulative Time Progress', fontsize=14)
    axes[2].grid(True, alpha=0.3)

    # Add ideal linear progression based on median interval
    expected_interval = median_interval  # Use median as it's more robust to outliers
    ideal_time = np.arange(len(df)) * expected_interval
    axes[2].plot(df['numSample'], ideal_time, 'r--', alpha=0.7,
                 label=f'Expected (at {expected_interval:.3f}s intervals)', linewidth=1.5)

    # Calculate and display drift
    final_drift = df['time_since_start_s'].iloc[-1] - ideal_time[-1]
    axes[2].text(0.02, 0.95, f'Total drift: {final_drift:.2f} seconds',
                 transform=axes[2].transAxes,
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                 fontsize=10)

    axes[2].legend(fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return fig


def plot_detailed_timing_analysis(labels_csv_path, output_path=None):
    """
    Create a more detailed timing analysis with separate plots for outliers
    """
    # Read and prepare data
    df = pd.read_csv(labels_csv_path)
    df = df.sort_values('timestamp_us')

    # Calculate time differences in seconds
    df['time_diff_s'] = df['timestamp_us'].diff() / 1_000_000
    df['time_since_start_s'] = (df['timestamp_us'] - df['timestamp_us'].iloc[0]) / 1_000_000

    # Identify outliers using IQR method
    Q1 = df['time_diff_s'].quantile(0.25)
    Q3 = df['time_diff_s'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold_low = Q1 - 1.5 * IQR
    outlier_threshold_high = Q3 + 1.5 * IQR

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()

    # Plot 1: Full time series
    axes[0].plot(df['numSample'].iloc[1:], df['time_diff_s'].iloc[1:], 'b-', linewidth=0.5)
    axes[0].set_xlabel('Sample Number')
    axes[0].set_ylabel('Time Interval (seconds)')
    axes[0].set_title('All Time Intervals')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Zoomed view without outliers
    mask_normal = (df['time_diff_s'] >= outlier_threshold_low) & (df['time_diff_s'] <= outlier_threshold_high)
    axes[1].plot(df.loc[mask_normal, 'numSample'], df.loc[mask_normal, 'time_diff_s'], 'g.', markersize=2)
    axes[1].set_xlabel('Sample Number')
    axes[1].set_ylabel('Time Interval (seconds)')
    axes[1].set_title('Time Intervals (Outliers Removed)')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Histogram of normal values
    normal_values = df.loc[mask_normal, 'time_diff_s'].dropna()
    axes[2].hist(normal_values, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[2].set_xlabel('Time Interval (seconds)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title(f'Distribution of Normal Intervals\n({len(normal_values)} samples)')
    axes[2].grid(True, alpha=0.3, axis='y')

    # Add statistics
    stats_text = f'Mean: {normal_values.mean():.4f}s\n'
    stats_text += f'Std: {normal_values.std():.4f}s\n'
    stats_text += f'Min: {normal_values.min():.4f}s\n'
    stats_text += f'Max: {normal_values.max():.4f}s'
    axes[2].text(0.7, 0.95, stats_text, transform=axes[2].transAxes,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 verticalalignment='top', fontsize=10)

    # Plot 4: Summary statistics
    axes[3].axis('off')

    # Calculate comprehensive statistics
    total_samples = len(df) - 1  # -1 because diff() creates NaN for first row
    n_outliers = (~mask_normal).sum() - 1  # -1 for the NaN
    outlier_samples = df.loc[~mask_normal & df['time_diff_s'].notna(), ['numSample', 'time_diff_s']]

    summary_text = f"TIMING ANALYSIS SUMMARY\n"
    summary_text += f"{'=' * 40}\n\n"
    summary_text += f"Total samples: {len(df)}\n"
    summary_text += f"Total duration: {df['time_since_start_s'].iloc[-1]:.2f} seconds\n"
    summary_text += f"Expected duration (at median rate): {len(df) * df['time_diff_s'].median():.2f} seconds\n\n"
    summary_text += f"INTERVAL STATISTICS:\n"
    summary_text += f"Mean interval: {df['time_diff_s'].mean():.4f} seconds\n"
    summary_text += f"Median interval: {df['time_diff_s'].median():.4f} seconds\n"
    summary_text += f"Std deviation: {df['time_diff_s'].std():.4f} seconds\n\n"
    summary_text += f"OUTLIERS:\n"
    summary_text += f"Number of outliers: {n_outliers} ({n_outliers / total_samples * 100:.1f}%)\n"
    summary_text += f"Outlier threshold: < {outlier_threshold_low:.4f}s or > {outlier_threshold_high:.4f}s\n\n"

    if len(outlier_samples) > 0 and len(outlier_samples) <= 10:
        summary_text += f"Outlier details:\n"
        for idx, row in outlier_samples.iterrows():
            summary_text += f"  Sample {int(row['numSample'])}: {row['time_diff_s']:.3f}s\n"
    elif len(outlier_samples) > 10:
        summary_text += f"Top 5 longest intervals:\n"
        top_outliers = outlier_samples.nlargest(5, 'time_diff_s')
        for idx, row in top_outliers.iterrows():
            summary_text += f"  Sample {int(row['numSample'])}: {row['time_diff_s']:.3f}s\n"

    axes[3].text(0.1, 0.9, summary_text, transform=axes[3].transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return fig


# Example usage
if __name__ == "__main__":
    # labels_path = "/path/to/RadIal_Data/RECORD@2020-11-22_12.45.05/labels.csv"
    # plot_timing_analysis(labels_path, output_path="timing_analysis.png")

    from pathlib import Path
    import json

    path_repo = Path('/Users/daniel/Idan/University/Masters/Thesis/2024/radar_sort/utils')
    if not path_repo.exists():
        path_repo = Path('/utils')

    path_config_default = path_repo / Path('T_FFTRadNet/RadIal/ADCProcessing/data_config.json')
    config = json.load(open(path_config_default))
    record = config['target_value']
    root_folder = Path(config['Data_Dir'], 'RadIal_Data', record)
    labels_path = Path(root_folder, 'labels.csv')


    plot_timing_analysis(labels_path, output_path="timing_analysis.png")
    plot_detailed_timing_analysis(labels_path, output_path="detailed_timing_analysis.png")