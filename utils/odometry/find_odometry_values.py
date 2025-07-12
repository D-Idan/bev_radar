#!/usr/bin/env python3
"""
Find odometry values by testing all CAN IDs and looking for expected values
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append('/Users/daniel/Idan/University/Masters/Thesis/2025/repos/bev_radar/ADCProcessing')

from DBReader.DBReader import SyncReader
from DBReader.DBReader.SensorsReaders import CANDecoder


def find_odometry_values(record_path, dbc_path):
    """Find CAN IDs that produce values matching the expected odometry values."""

    # Expected values from the example
    expected_steering = 2.1875  # degrees
    expected_yaw = 0.75  # deg/sec
    expected_speed = 93.6875  # kph

    tolerance = 0.1  # Allow small differences

    print(f"Looking for odometry values:")
    print(f"  Steering: ~{expected_steering}°")
    print(f"  Yaw rate: ~{expected_yaw}°/s")
    print(f"  Speed: ~{expected_speed} kph")
    print("-" * 50)

    db = SyncReader(record_path, tolerance=200000, silent=True)
    can_decoder = CANDecoder(str(dbc_path))

    # Get sample data like in example
    data = db.GetSensorData(120)
    target_timestamp = data['camera']['timestamp']

    # Get all unique CAN IDs
    unique_ids = np.unique(db.can_frames['ID'])
    print(f"Testing {len(unique_ids)} CAN IDs...")

    matches = {'steering': [], 'yaw': [], 'speed': []}

    for can_id in unique_ids:
        try:
            # Find messages near target timestamp
            id_indices = np.where(db.can_frames['ID'] == can_id)[0]
            if len(id_indices) == 0:
                continue

            time_diffs = np.abs(db.can_frames['timestamp'][id_indices] - target_timestamp)
            closest_idx = id_indices[np.argmin(time_diffs)]

            # Try to decode
            message_data = {
                'timestamp': db.can_frames['timestamp'][closest_idx],
                'ID': int(can_id),
                'DATA': db.can_frames['data'][closest_idx]
            }

            decoded = can_decoder.decode([message_data])

            if decoded and len(decoded) > 0 and 'signals' in decoded[0]:
                signals = decoded[0]['signals']

                # Check each signal for matches
                for signal_name, value in signals.items():
                    # Check steering
                    if abs(value - expected_steering) < tolerance:
                        matches['steering'].append((can_id, signal_name, value))
                        print(f"🎯 STEERING MATCH: ID {can_id}, {signal_name} = {value}")

                    # Check yaw rate
                    if abs(value - expected_yaw) < tolerance:
                        matches['yaw'].append((can_id, signal_name, value))
                        print(f"🎯 YAW RATE MATCH: ID {can_id}, {signal_name} = {value}")

                    # Check speed
                    if abs(value - expected_speed) < tolerance:
                        matches['speed'].append((can_id, signal_name, value))
                        print(f"🎯 SPEED MATCH: ID {can_id}, {signal_name} = {value}")

        except Exception:
            continue

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)

    if matches['steering']:
        can_id, signal_name, value = matches['steering'][0]
        print(f"Steering: ID {can_id}, signal '{signal_name}' = {value}")

    if matches['yaw']:
        can_id, signal_name, value = matches['yaw'][0]
        print(f"Yaw Rate: ID {can_id}, signal '{signal_name}' = {value}")

    if matches['speed']:
        can_id, signal_name, value = matches['speed'][0]
        print(f"Speed: ID {can_id}, signal '{signal_name}' = {value}")

    # Generate custom GetMostRecentOdometry function
    print("\n" + "=" * 60)
    print("CUSTOM FUNCTION CODE:")
    print("=" * 60)

    if all(matches.values()):
        steering_id, steering_signal, _ = matches['steering'][0]
        yaw_id, yaw_signal, _ = matches['yaw'][0]
        speed_id, speed_signal, _ = matches['speed'][0]

        print(f"""
def GetMostRecentOdometry(self, decoder, time):
    results = []

    # Steering: ID={steering_id}
    IDX = np.where(self.can_frames['ID']=={steering_id})[0]
    if len(IDX) > 0:
        timediff = np.abs(self.can_frames['timestamp'][IDX] - time)
        id_steer = IDX[np.argmin(timediff)]
        message = decoder.decode([{{'timestamp':self.can_frames['timestamp'][id_steer],'ID':self.can_frames['ID'][id_steer],'DATA':self.can_frames['data'][id_steer]}}])
        steering = message[0]['signals']['{steering_signal}']
        results.append(steering)

    # Yaw Rate: ID={yaw_id}  
    IDX = np.where(self.can_frames['ID']=={yaw_id})[0]
    if len(IDX) > 0:
        timediff = np.abs(self.can_frames['timestamp'][IDX] - time)
        id_yaw = IDX[np.argmin(timediff)]
        message = decoder.decode([{{'timestamp':self.can_frames['timestamp'][id_yaw],'ID':self.can_frames['ID'][id_yaw],'DATA':self.can_frames['data'][id_yaw]}}])
        yaw_rate = message[0]['signals']['{yaw_signal}']
        results.append(yaw_rate)

    # Speed: ID={speed_id}
    IDX = np.where(self.can_frames['ID']=={speed_id})[0]
    if len(IDX) > 0:
        timediff = np.abs(self.can_frames['timestamp'][IDX] - time)
        id_speed = IDX[np.argmin(timediff)]
        message = decoder.decode([{{'timestamp':self.can_frames['timestamp'][id_speed],'ID':self.can_frames['ID'][id_speed],'DATA':self.can_frames['data'][id_speed]}}])
        speed = message[0]['signals']['{speed_signal}']
        results.append(speed)

    return results
        """)


def main():
    record_path = "/Volumes/ELEMENTS/datasets/radial/RECORD@2020-11-22_12.28.47"
    dbc_path = "/Users/daniel/Idan/University/Masters/Thesis/2025/repos/bev_radar/ADCProcessing/DBReader/examples/can_database.dbc"

    if not Path(record_path).exists():
        print(f"Error: Record path not found: {record_path}")
        return

    if not Path(dbc_path).exists():
        print(f"Error: DBC file not found: {dbc_path}")
        return

    find_odometry_values(record_path, dbc_path)


if __name__ == "__main__":
    main()