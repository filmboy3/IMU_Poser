#!/usr/bin/env python3
"""
Debug script for analyzing failed session where no reps were detected
despite user performing ~5 reps.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_session_data(session_path):
    """Load IMU data from session directory"""
    with open(f"{session_path}/imu_data.json", 'r') as f:
        imu_data = json.load(f)
    
    with open(f"{session_path}/session_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return imu_data, metadata

def analyze_motion_patterns(imu_data):
    """Analyze motion patterns to understand why reps weren't detected"""
    
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
    
    print(f"Session Duration: {timestamps[-1]:.2f} seconds")
    print(f"Total Samples: {len(timestamps)}")
    print(f"Sample Rate: {len(timestamps) / timestamps[-1]:.1f} Hz")
    
    # Analyze Y-axis (vertical) motion for squats
    print(f"\nY-Axis Analysis:")
    print(f"Mean: {np.mean(acc_y):.3f}g")
    print(f"Std: {np.std(acc_y):.3f}g")
    print(f"Min: {np.min(acc_y):.3f}g")
    print(f"Max: {np.max(acc_y):.3f}g")
    print(f"Range: {np.max(acc_y) - np.min(acc_y):.3f}g")
    
    # Simulate SmartExerciseDetector motion classification
    motion_types = []
    buffer_size = 100  # ~1 second at 100Hz
    
    for i in range(len(acc_y)):
        if i < buffer_size:
            motion_types.append("insufficient_data")
            continue
            
        # Get recent motion window
        window_y = acc_y[i-buffer_size:i]
        window_x = acc_x[i-buffer_size:i]
        window_z = acc_z[i-buffer_size:i]
        
        # Calculate motion metrics
        activity = np.sqrt(np.mean(window_y**2))
        jerkiness = np.std(np.diff(window_y))
        y_dominance = np.std(window_y) / (np.std(window_x) + np.std(window_z) + 1e-6)
        
        # Classify motion (using thresholds from SmartExerciseDetector)
        if activity < 0.02:
            motion_type = "stable"
        elif jerkiness > 0.15:
            motion_type = "rustling"
        elif activity > 0.05 and np.std(window_y) < 0.03:
            motion_type = "handling"
        elif activity > 0.08 and y_dominance > 1.5:
            motion_type = "exercise"
        else:
            motion_type = "unknown"
            
        motion_types.append(motion_type)
    
    # Count motion types
    motion_counts = {}
    for mt in motion_types:
        motion_counts[mt] = motion_counts.get(mt, 0) + 1
    
    print(f"\nMotion Type Distribution:")
    for mt, count in motion_counts.items():
        percentage = (count / len(motion_types)) * 100
        print(f"  {mt}: {count} samples ({percentage:.1f}%)")
    
    # Simulate rep counting algorithm
    print(f"\nSimulating Rep Counter:")
    
    # Apply smoothing (8-sample buffer as in RepCounter)
    smoothed_y = []
    buffer = []
    
    for y_val in acc_y:
        buffer.append(y_val)
        if len(buffer) > 8:
            buffer.pop(0)
        smoothed_y.append(sum(buffer) / len(buffer))
    
    smoothed_y = np.array(smoothed_y)
    
    # Rep detection thresholds (from RepCounter)
    peak_threshold = 0.12
    valley_threshold = -0.10
    neutral_threshold = 0.06
    
    print(f"Thresholds: Peak={peak_threshold}, Valley={valley_threshold}, Neutral={neutral_threshold}")
    print(f"Smoothed Y range: {np.min(smoothed_y):.3f} to {np.max(smoothed_y):.3f}")
    
    # Count threshold crossings
    peaks = np.sum(smoothed_y > peak_threshold)
    valleys = np.sum(smoothed_y < valley_threshold)
    
    print(f"Samples above peak threshold: {peaks}")
    print(f"Samples below valley threshold: {valleys}")
    
    # Check if motion ever reaches exercise state
    exercise_samples = sum(1 for mt in motion_types if mt == "exercise")
    print(f"Samples classified as exercise motion: {exercise_samples}")
    
    return timestamps, acc_x, acc_y, acc_z, smoothed_y, motion_types

def plot_analysis(timestamps, acc_x, acc_y, acc_z, smoothed_y, motion_types):
    """Create diagnostic plots"""
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Plot raw accelerations
    axes[0].plot(timestamps, acc_x, label='X (lateral)', alpha=0.7)
    axes[0].plot(timestamps, acc_y, label='Y (vertical)', alpha=0.7)
    axes[0].plot(timestamps, acc_z, label='Z (forward)', alpha=0.7)
    axes[0].set_ylabel('Acceleration (g)')
    axes[0].set_title('Raw IMU Acceleration Data')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot smoothed Y with thresholds
    axes[1].plot(timestamps, smoothed_y, 'b-', linewidth=2, label='Smoothed Y')
    axes[1].axhline(y=0.12, color='g', linestyle='--', label='Peak Threshold')
    axes[1].axhline(y=-0.10, color='r', linestyle='--', label='Valley Threshold')
    axes[1].axhline(y=0.06, color='orange', linestyle='--', label='Neutral Threshold')
    axes[1].axhline(y=-0.06, color='orange', linestyle='--')
    axes[1].set_ylabel('Smoothed Y (g)')
    axes[1].set_title('Smoothed Y-Axis with Rep Detection Thresholds')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot motion classification
    motion_colors = {
        'stable': 'green',
        'handling': 'orange', 
        'rustling': 'red',
        'exercise': 'blue',
        'unknown': 'gray',
        'insufficient_data': 'lightgray'
    }
    
    # Create motion type timeline
    motion_numeric = []
    for mt in motion_types:
        if mt == 'stable': motion_numeric.append(0)
        elif mt == 'handling': motion_numeric.append(1)
        elif mt == 'rustling': motion_numeric.append(2)
        elif mt == 'exercise': motion_numeric.append(3)
        elif mt == 'unknown': motion_numeric.append(4)
        else: motion_numeric.append(-1)
    
    axes[2].scatter(timestamps, motion_numeric, c=[motion_colors.get(mt, 'black') for mt in motion_types], 
                   s=1, alpha=0.6)
    axes[2].set_ylabel('Motion Type')
    axes[2].set_yticks([0, 1, 2, 3, 4])
    axes[2].set_yticklabels(['Stable', 'Handling', 'Rustling', 'Exercise', 'Unknown'])
    axes[2].set_title('Motion Classification Over Time')
    axes[2].grid(True, alpha=0.3)
    
    # Plot activity level
    buffer_size = 100
    activity_levels = []
    for i in range(len(acc_y)):
        if i < buffer_size:
            activity_levels.append(0)
        else:
            window_y = acc_y[i-buffer_size:i]
            activity = np.sqrt(np.mean(window_y**2))
            activity_levels.append(activity)
    
    axes[3].plot(timestamps, activity_levels, 'purple', linewidth=1)
    axes[3].axhline(y=0.08, color='blue', linestyle='--', label='Exercise Activity Threshold')
    axes[3].axhline(y=0.02, color='green', linestyle='--', label='Stable Activity Threshold')
    axes[3].set_ylabel('Activity Level')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].set_title('Motion Activity Level')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/jonathanschwartz/Documents/Social_Content/QuantumLeapValidator/failed_session_analysis.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

def main():
    session_path = "/Users/jonathanschwartz/Downloads/session_20250904_075342_0EACD0B7"
    
    print("=== Failed Session Analysis ===")
    print(f"Analyzing session: {session_path}")
    
    # Load data
    imu_data, metadata = load_session_data(session_path)
    
    print(f"\nSession Metadata:")
    print(f"  Exercise Type: {metadata['exerciseType']}")
    print(f"  Duration: {metadata['duration']:.2f} seconds")
    print(f"  Recorded Reps: {metadata['repCount']}")
    print(f"  Expected Reps: ~5 (user reported)")
    print(f"  IMU Samples: {metadata['imuSampleCount']}")
    print(f"  Average IMU Rate: {metadata['averageIMURate']:.1f} Hz")
    
    # Analyze motion patterns
    timestamps, acc_x, acc_y, acc_z, smoothed_y, motion_types = analyze_motion_patterns(imu_data)
    
    # Create diagnostic plots
    plot_analysis(timestamps, acc_x, acc_y, acc_z, smoothed_y, motion_types)
    
    print(f"\n=== DIAGNOSIS ===")
    print("Potential issues:")
    
    # Check if motion reaches exercise thresholds
    exercise_samples = sum(1 for mt in motion_types if mt == "exercise")
    if exercise_samples == 0:
        print("❌ CRITICAL: No motion was classified as 'exercise'")
        print("   - SmartExerciseDetector may be too restrictive")
        print("   - Motion thresholds may need adjustment")
    
    # Check rep detection thresholds
    smoothed_y_array = np.array(smoothed_y)
    if np.max(smoothed_y_array) < 0.12:
        print("❌ CRITICAL: Motion never exceeded peak threshold (0.12g)")
        print(f"   - Max smoothed Y: {np.max(smoothed_y_array):.3f}g")
        print("   - Peak threshold may be too high")
    
    if np.min(smoothed_y_array) > -0.10:
        print("❌ CRITICAL: Motion never went below valley threshold (-0.10g)")
        print(f"   - Min smoothed Y: {np.min(smoothed_y_array):.3f}g")
        print("   - Valley threshold may be too low")
    
    # Check motion range
    motion_range = np.max(smoothed_y_array) - np.min(smoothed_y_array)
    if motion_range < 0.15:
        print(f"⚠️  WARNING: Low motion range ({motion_range:.3f}g)")
        print("   - User may not be performing full range squats")
        print("   - Phone orientation may be incorrect")

if __name__ == "__main__":
    main()
