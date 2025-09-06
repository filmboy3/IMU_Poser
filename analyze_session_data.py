#!/usr/bin/env python3
"""
Analyze captured session data to debug rep counting accuracy
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

def load_session_data(session_path):
    """Load IMU and metadata from session directory."""
    
    # Load metadata
    metadata_path = os.path.join(session_path, 'session_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load IMU data
    imu_path = os.path.join(session_path, 'imu_data.json')
    with open(imu_path, 'r') as f:
        imu_data = json.load(f)
    
    return metadata, imu_data

def analyze_acceleration_patterns(imu_data):
    """Analyze acceleration patterns to understand rep detection"""
    
    # Extract acceleration data
    timestamps = [sample['timestamp'] for sample in imu_data]
    acc_x = [sample['acceleration']['x'] for sample in imu_data]
    acc_y = [sample['acceleration']['y'] for sample in imu_data]
    acc_z = [sample['acceleration']['z'] for sample in imu_data]
    
    # Convert to numpy arrays
    timestamps = np.array(timestamps)
    acc_x = np.array(acc_x)
    acc_y = np.array(acc_y)
    acc_z = np.array(acc_z)
    
    # Normalize timestamps to start from 0
    timestamps = timestamps - timestamps[0]
    
    print(f"Acceleration Analysis:")
    print(f"- Y-axis (vertical) range: {acc_y.min():.3f} to {acc_y.max():.3f}")
    print(f"- Y-axis std deviation: {acc_y.std():.3f}")
    print(f"- X-axis range: {acc_x.min():.3f} to {acc_x.max():.3f}")
    print(f"- Z-axis range: {acc_z.min():.3f} to {acc_z.max():.3f}")
    print()
    
    # Simulate current rep counting algorithm
    simulate_rep_counting(timestamps, acc_y)
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: All acceleration components
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, acc_x, label='X (lateral)', alpha=0.7)
    plt.plot(timestamps, acc_y, label='Y (vertical)', alpha=0.7, linewidth=2)
    plt.plot(timestamps, acc_z, label='Z (forward/back)', alpha=0.7)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Peak threshold')
    plt.axhline(y=-0.5, color='blue', linestyle='--', alpha=0.5, label='Valley threshold')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration (g)')
    plt.title('Raw Acceleration Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Smoothed Y-axis with rep detection simulation
    plt.subplot(3, 1, 2)
    smoothed_y = moving_average(acc_y, window=10)
    plt.plot(timestamps, acc_y, alpha=0.3, label='Raw Y')
    plt.plot(timestamps[9:], smoothed_y, label='Smoothed Y (10-sample avg)', linewidth=2)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Peak threshold')
    plt.axhline(y=-0.5, color='blue', linestyle='--', alpha=0.5, label='Valley threshold')
    plt.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Neutral threshold')
    plt.axhline(y=-0.2, color='green', linestyle='--', alpha=0.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration (g)')
    plt.title('Y-Axis Acceleration with Current Algorithm Thresholds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Magnitude and potential better detection
    plt.subplot(3, 1, 3)
    magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    plt.plot(timestamps, magnitude, label='Acceleration Magnitude', alpha=0.7)
    plt.plot(timestamps, acc_y, label='Y-axis', alpha=0.7)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration (g)')
    plt.title('Acceleration Magnitude vs Y-axis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/jonathanschwartz/Documents/Social_Content/QuantumLeapValidator/session_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return timestamps, acc_x, acc_y, acc_z

def moving_average(data, window):
    """Calculate moving average"""
    return np.convolve(data, np.ones(window)/window, mode='valid')

def simulate_rep_counting(timestamps, acc_y):
    """Simulate the current rep counting algorithm"""
    
    # Current algorithm parameters
    peak_threshold = 0.5
    valley_threshold = -0.5
    neutral_threshold = 0.2
    buffer_size = 10
    
    # State machine
    state = 'neutral'  # neutral, descending, ascending
    rep_count = 0
    buffer = []
    
    detected_reps = []
    state_changes = []
    
    for i, acc in enumerate(acc_y):
        buffer.append(acc)
        if len(buffer) > buffer_size:
            buffer.pop(0)
        
        smoothed_acc = sum(buffer) / len(buffer)
        
        if state == 'neutral':
            if smoothed_acc < valley_threshold:
                state = 'descending'
                state_changes.append((timestamps[i], 'descending', smoothed_acc))
        elif state == 'descending':
            if smoothed_acc > peak_threshold:
                state = 'ascending'
                state_changes.append((timestamps[i], 'ascending', smoothed_acc))
        elif state == 'ascending':
            if abs(smoothed_acc) < neutral_threshold:
                rep_count += 1
                state = 'neutral'
                detected_reps.append((timestamps[i], rep_count, smoothed_acc))
                state_changes.append((timestamps[i], 'rep_completed', smoothed_acc))
    
    print(f"Rep Counting Simulation:")
    print(f"- Algorithm detected: {rep_count} reps")
    print(f"- State changes: {len(state_changes)}")
    print()
    
    print("Detected reps:")
    for timestamp, rep_num, acc_val in detected_reps:
        print(f"  Rep {rep_num}: {timestamp:.2f}s (acc: {acc_val:.3f})")
    print()
    
    print("All state changes:")
    for timestamp, new_state, acc_val in state_changes:
        print(f"  {timestamp:.2f}s: {new_state} (acc: {acc_val:.3f})")
    print()
    
    # Analyze why detection might be poor
    analyze_detection_issues(acc_y, peak_threshold, valley_threshold)

def analyze_detection_issues(acc_y, peak_threshold, valley_threshold):
    """Analyze potential issues with current detection algorithm"""
    
    acc_y = np.array(acc_y)
    
    # Check threshold crossings
    peaks = acc_y > peak_threshold
    valleys = acc_y < valley_threshold
    
    peak_count = np.sum(np.diff(peaks.astype(int)) == 1)  # Rising edges
    valley_count = np.sum(np.diff(valleys.astype(int)) == 1)  # Rising edges
    
    print(f"Threshold Analysis:")
    print(f"- Raw peak threshold crossings: {peak_count}")
    print(f"- Raw valley threshold crossings: {valley_count}")
    print(f"- Y-axis values > peak threshold: {np.sum(peaks)}")
    print(f"- Y-axis values < valley threshold: {np.sum(valleys)}")
    print(f"- Y-axis mean: {acc_y.mean():.3f}")
    print(f"- Y-axis median: {np.median(acc_y):.3f}")
    print()
    
    # Suggest better thresholds
    y_std = acc_y.std()
    y_mean = acc_y.mean()
    
    suggested_peak = y_mean + 1.5 * y_std
    suggested_valley = y_mean - 1.5 * y_std
    
    print(f"Suggested Improvements:")
    print(f"- Current thresholds: peak={peak_threshold}, valley={valley_threshold}")
    print(f"- Suggested thresholds: peak={suggested_peak:.3f}, valley={suggested_valley:.3f}")
    print(f"- Consider using acceleration magnitude instead of just Y-axis")
    print(f"- Consider adaptive thresholds based on recent data")
    print()

if __name__ == "__main__":
    session_dir = "/Users/jonathanschwartz/Downloads/session_20250903_143530_B7576CC0"
    
    # Load and analyze data
    metadata, imu_data = load_session_data(session_dir)
    timestamps, acc_x, acc_y, acc_z = analyze_acceleration_patterns(imu_data)
    
    print("Analysis complete. Check session_analysis.png for visualizations.")
