import numpy as np
import pandas as pd
from radar_tracking.data_structures import EgoMotion


def load_odometry_data(labels_df: pd.DataFrame) -> dict:
    """
    Load odometry data from labels DataFrame.

    Args:
        labels_df: Labels DataFrame containing odometry columns

    Returns:
        Dictionary mapping sample_id (numSample) to EgoMotion objects
    """
    print("Loading odometry data from labels...")

    # Check if odometry columns exist in labels
    required_columns = ['ego_steering_wheel_deg', 'ego_yaw_rate_deg_s', 'ego_speed_kph']
    missing_columns = [col for col in required_columns if col not in labels_df.columns]

    assert not missing_columns, f"Missing required odometry columns: {missing_columns}"

    # Extract unique samples with their odometry data
    # Group by numSample and take first row (all rows for same sample should have same odometry)
    unique_samples = labels_df.groupby('numSample').first().reset_index()

    odometry_dict = {}
    successful_loads = 0

    for _, row in unique_samples.iterrows():
        sample_id = int(row['numSample'])
        timestamp_s = row['timestamp_us'] / 1_000_000

        try:
            # Check if odometry data is available (not NaN)
            steering_deg = row['ego_steering_wheel_deg']
            yaw_rate_deg_s = row['ego_yaw_rate_deg_s']
            speed_kph = row['ego_speed_kph']

            # Skip if any odometry value is NaN
            if pd.isna(steering_deg) or pd.isna(yaw_rate_deg_s) or pd.isna(speed_kph):
                continue

            # Create odometry tuple in expected format
            odometry_tuple = (float(steering_deg), float(yaw_rate_deg_s), float(speed_kph))

            # Create EgoMotion object using existing from_odometry method
            ego_motion = EgoMotion.from_odometry(odometry_tuple, timestamp_s)
            odometry_dict[sample_id] = ego_motion
            successful_loads += 1

        except Exception as e:
            print(f"Warning: Failed to parse odometry for sample {sample_id}: {e}")
            continue

    print(f"Loaded odometry data for {successful_loads}/{len(unique_samples)} frames")
    return odometry_dict


def get_custom_odometry(db, can_decoder, timestamp):
    """
    Custom odometry extraction function that works with the available CAN IDs.
    Returns: (steering_deg, yaw_rate_deg_s, speed_kph)
    """
    results = []

    try:
        # Steering: ID=485
        IDX = np.where(db.can_frames['ID'] == 485)[0]
        if len(IDX) > 0:
            timediff = np.abs(db.can_frames['timestamp'][IDX] - timestamp)
            id_steer = IDX[np.argmin(timediff)]
            message = can_decoder.decode([{
                'timestamp': db.can_frames['timestamp'][id_steer],
                'ID': int(db.can_frames['ID'][id_steer]),
                'DATA': db.can_frames['data'][id_steer]
            }])
            steering = message[0]['signals']['Steering_Wheel_Angle_deg']
            results.append(steering)
        else:
            results.append(np.nan)

        # Yaw Rate: ID=489
        IDX = np.where(db.can_frames['ID'] == 489)[0]
        if len(IDX) > 0:
            timediff = np.abs(db.can_frames['timestamp'][IDX] - timestamp)
            id_yaw = IDX[np.argmin(timediff)]
            message = can_decoder.decode([{
                'timestamp': db.can_frames['timestamp'][id_yaw],
                'ID': int(db.can_frames['ID'][id_yaw]),
                'DATA': db.can_frames['data'][id_yaw]
            }])
            yaw_rate = message[0]['signals']['YawRate_deg']
            results.append(yaw_rate)
        else:
            results.append(np.nan)

        # Speed: ID=1001
        IDX = np.where(db.can_frames['ID'] == 1001)[0]
        if len(IDX) > 0:
            timediff = np.abs(db.can_frames['timestamp'][IDX] - timestamp)
            id_speed = IDX[np.argmin(timediff)]
            message = can_decoder.decode([{
                'timestamp': db.can_frames['timestamp'][id_speed],
                'ID': int(db.can_frames['ID'][id_speed]),
                'DATA': db.can_frames['data'][id_speed]
            }])
            speed = message[0]['signals']['Speed_kph']
            results.append(speed)
        else:
            results.append(np.nan)

        return tuple(results)

    except Exception as e:
        print(f"Error extracting odometry: {e}")
        return (np.nan, np.nan, np.nan)


def get_optimized_odometry(db, can_decoder, timestamp, time_window=200000):
    """Optimized odometry with smaller time window"""
    results = []

    try:
        # Use smaller time window (50ms instead of searching all data)
        for can_id, signal_name in [(485, 'Steering_Wheel_Angle_deg'),
                                    (489, 'YawRate_deg'),
                                    (1001, 'Speed_kph')]:

            # Find messages within time window
            id_mask = db.can_frames['ID'] == can_id
            time_mask = np.abs(db.can_frames['timestamp'] - timestamp) <= time_window
            valid_indices = np.where(id_mask & time_mask)[0]

            if len(valid_indices) > 0:
                # Get closest message
                time_diffs = np.abs(db.can_frames['timestamp'][valid_indices] - timestamp)
                best_idx = valid_indices[np.argmin(time_diffs)]

                message = can_decoder.decode([{
                    'timestamp': db.can_frames['timestamp'][best_idx],
                    'ID': int(can_id),
                    'DATA': db.can_frames['data'][best_idx]
                }])
                value = message[0]['signals'][signal_name]
                results.append(value)
            else:
                results.append(np.nan)

        return tuple(results)

    except Exception:
        return (np.nan, np.nan, np.nan)