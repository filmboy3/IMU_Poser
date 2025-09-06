#!/usr/bin/env python3
"""
Deep analysis of session_20250903_161849_A058DD12 to understand rep counting failures
Focus: Why 5+ reps were missed between detected reps 10-11
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_and_analyze_session():
    """Load and perform deep analysis of the problematic session"""
    
    session_dir = "/Users/jonathanschwartz/Downloads/session_20250903_161849_A058DD12"
    
    # Load metadata
    with open(f"{session_dir}/session_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Load IMU data
    with open(f"{session_dir}/imu_data.json", 'r') as f:
        imu_data = json.load(f)
    
    print("=== SESSION ANALYSIS ===")
    print(f"Exercise: {metadata['exerciseType']}")
    print(f"Duration: {metadata['duration']:.2f} seconds")
    print(f"Detected Reps: {metadata['repCount']} (11 total)")
    print(f"IMU Samples: {metadata['imuSampleCount']}")
    print(f"Sample Rate: {metadata['averageIMURate']:.1f} Hz")
    print(f"PROBLEM: 5+ actual reps missed between detected reps 10-11")
    print()
    
    return metadata, imu_data

def extract_acceleration_data(imu_data):
    """Extract and normalize acceleration data"""
    
    timestamps = np.array([sample['timestamp'] for sample in imu_data])
    acc_x = np.array([sample['acceleration']['x'] for sample in imu_data])
    acc_y = np.array([sample['acceleration']['y'] for sample in imu_data])
    acc_z = np.array([sample['acceleration']['z'] for sample in imu_data])
    
    # Normalize timestamps to start from 0
    timestamps = timestamps - timestamps[0]
    
    print("=== ACCELERATION DATA ANALYSIS ===")
    print(f"Y-axis (vertical) range: {acc_y.min():.3f} to {acc_y.max():.3f}")
    print(f"Y-axis mean: {acc_y.mean():.3f}, std: {acc_y.std():.3f}")
    print(f"X-axis range: {acc_x.min():.3f} to {acc_x.max():.3f}")
    print(f"Z-axis range: {acc_z.min():.3f} to {acc_z.max():.3f}")
    print()
    
    return timestamps, acc_x, acc_y, acc_z

def simulate_current_algorithm(timestamps, acc_y):
    """Simulate the current rep counting algorithm with detailed logging"""
    
    # Current algorithm parameters (enhanced version)
    peak_threshold = 0.16
    valley_threshold = -0.14
    neutral_threshold = 0.08
    buffer_size = 10
    
    # State machine
    state = 'neutral'
    rep_count = 0
    buffer = []
    
    detected_reps = []
    state_changes = []
    all_states = []
    smoothed_values = []
    
    print("=== DETAILED ALGORITHM SIMULATION ===")
    print(f"Thresholds: peak={peak_threshold}, valley={valley_threshold}, neutral={neutral_threshold}")
    print()
    
    for i, acc in enumerate(acc_y):
        buffer.append(acc)
        if len(buffer) > buffer_size:
            buffer.pop(0)
        
        smoothed_acc = sum(buffer) / len(buffer)
        smoothed_values.append(smoothed_acc)
        all_states.append(state)
        
        prev_state = state
        
        if state == 'neutral':
            if smoothed_acc < valley_threshold:
                state = 'descending'
        elif state == 'descending':
            if smoothed_acc > peak_threshold:
                state = 'ascending'
        elif state == 'ascending':
            if abs(smoothed_acc) < neutral_threshold:
                rep_count += 1
                state = 'neutral'
                detected_reps.append((timestamps[i], rep_count, smoothed_acc))
                print(f"REP #{rep_count}: {timestamps[i]:.2f}s (smoothed_acc: {smoothed_acc:.3f})")
        
        if state != prev_state:
            state_changes.append((timestamps[i], state, smoothed_acc))
    
    print(f"\nFinal rep count: {rep_count}")
    print(f"State changes: {len(state_changes)}")
    print()
    
    return detected_reps, state_changes, smoothed_values, all_states

def analyze_missing_reps(timestamps, acc_y, detected_reps):
    """Analyze the gap where 5+ reps were missed"""
    
    print("=== MISSING REPS ANALYSIS ===")
    
    if len(detected_reps) >= 2:
        # Find the gap between reps 10 and 11 (assuming they exist)
        if len(detected_reps) >= 11:
            rep_10_time = detected_reps[9][0]  # 10th rep (0-indexed)
            rep_11_time = detected_reps[10][0]  # 11th rep
            
            print(f"Rep 10 detected at: {rep_10_time:.2f}s")
            print(f"Rep 11 detected at: {rep_11_time:.2f}s")
            print(f"Gap duration: {rep_11_time - rep_10_time:.2f}s")
            print(f"Expected reps in gap: ~{(rep_11_time - rep_10_time) / 3:.1f} (assuming 3s per rep)")
            
            # Analyze the gap period
            gap_mask = (timestamps >= rep_10_time) & (timestamps <= rep_11_time)
            gap_timestamps = timestamps[gap_mask]
            gap_acc_y = acc_y[gap_mask]
            
            print(f"\nGap period analysis:")
            print(f"- Samples in gap: {len(gap_acc_y)}")
            print(f"- Y-acceleration range: {gap_acc_y.min():.3f} to {gap_acc_y.max():.3f}")
            print(f"- Y-acceleration std: {gap_acc_y.std():.3f}")
            
            return gap_timestamps, gap_acc_y
    
    print("Could not identify specific gap between reps 10-11")
    return None, None

def detect_potential_reps_manually(timestamps, acc_y, smoothed_values):
    """Manual peak detection to find missed reps"""
    
    print("=== MANUAL REP DETECTION ===")
    
    # Convert to numpy arrays
    smoothed_values = np.array(smoothed_values)
    
    # Find peaks and valleys using a different approach
    from scipy.signal import find_peaks
    
    # Find peaks (upward motion)
    peaks, _ = find_peaks(smoothed_values, height=0.05, distance=30)  # ~0.3s apart
    valleys, _ = find_peaks(-smoothed_values, height=0.05, distance=30)
    
    print(f"Potential peaks found: {len(peaks)}")
    print(f"Potential valleys found: {len(valleys)}")
    
    # Analyze peak-valley pairs
    potential_reps = []
    for i, peak_idx in enumerate(peaks):
        # Find nearest valley before this peak
        before_valleys = valleys[valleys < peak_idx]
        if len(before_valleys) > 0:
            valley_idx = before_valleys[-1]
            
            peak_time = timestamps[peak_idx] if peak_idx < len(timestamps) else timestamps[-1]
            valley_time = timestamps[valley_idx] if valley_idx < len(timestamps) else timestamps[-1]
            
            # Check if this looks like a rep (reasonable duration)
            duration = peak_time - valley_time
            if 0.5 < duration < 5.0:  # Reasonable rep duration
                potential_reps.append({
                    'valley_time': valley_time,
                    'peak_time': peak_time,
                    'duration': duration,
                    'valley_acc': smoothed_values[valley_idx],
                    'peak_acc': smoothed_values[peak_idx]
                })
    
    print(f"\nPotential reps identified: {len(potential_reps)}")
    for i, rep in enumerate(potential_reps[:15]):  # Show first 15
        print(f"Rep {i+1}: {rep['valley_time']:.1f}s-{rep['peak_time']:.1f}s "
              f"(dur: {rep['duration']:.1f}s, range: {rep['valley_acc']:.3f} to {rep['peak_acc']:.3f})")
    
    return potential_reps

def analyze_motion_patterns(timestamps, acc_x, acc_y, acc_z):
    """Analyze different types of motion to distinguish setup vs exercise"""
    
    print("=== MOTION PATTERN ANALYSIS ===")
    
    # Calculate acceleration magnitude
    magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    # Analyze different time segments
    total_duration = timestamps[-1]
    segment_duration = total_duration / 10  # 10 segments
    
    print("Motion intensity by time segment:")
    for i in range(10):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        
        mask = (timestamps >= start_time) & (timestamps < end_time)
        segment_y = acc_y[mask]
        segment_mag = magnitude[mask]
        
        if len(segment_y) > 0:
            y_std = segment_y.std()
            mag_std = segment_mag.std()
            print(f"Segment {i+1:2d} ({start_time:4.1f}-{end_time:4.1f}s): "
                  f"Y_std={y_std:.3f}, Mag_std={mag_std:.3f}")
    
    # Identify high-activity periods
    window_size = 100  # ~1 second windows
    activity_levels = []
    
    for i in range(0, len(acc_y) - window_size, window_size//2):
        window_y = acc_y[i:i+window_size]
        activity = window_y.std()
        activity_levels.append(activity)
    
    activity_threshold = np.percentile(activity_levels, 75)  # Top 25% activity
    print(f"\nHigh activity threshold: {activity_threshold:.3f}")
    print(f"High activity windows: {np.sum(np.array(activity_levels) > activity_threshold)}")

def create_detailed_plots(timestamps, acc_x, acc_y, acc_z, detected_reps, smoothed_values, all_states):
    """Create comprehensive plots for analysis"""
    
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Raw acceleration with detected reps
    plt.subplot(4, 1, 1)
    plt.plot(timestamps, acc_y, alpha=0.5, label='Raw Y', color='blue')
    plt.plot(timestamps[:len(smoothed_values)], smoothed_values, label='Smoothed Y', color='red', linewidth=2)
    
    # Mark detected reps
    for rep_time, rep_num, _ in detected_reps:
        plt.axvline(x=rep_time, color='green', linestyle='--', alpha=0.7)
        plt.text(rep_time, plt.ylim()[1]*0.9, f'Rep {rep_num}', rotation=90, fontsize=8)
    
    plt.axhline(y=0.16, color='red', linestyle=':', alpha=0.5, label='Peak threshold')
    plt.axhline(y=-0.14, color='blue', linestyle=':', alpha=0.5, label='Valley threshold')
    plt.axhline(y=0.08, color='green', linestyle=':', alpha=0.5, label='Neutral threshold')
    plt.axhline(y=-0.08, color='green', linestyle=':', alpha=0.5)
    
    plt.ylabel('Acceleration (g)')
    plt.title('Y-Axis Acceleration with Rep Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: All acceleration components
    plt.subplot(4, 1, 2)
    plt.plot(timestamps, acc_x, alpha=0.7, label='X (lateral)')
    plt.plot(timestamps, acc_y, alpha=0.7, label='Y (vertical)')
    plt.plot(timestamps, acc_z, alpha=0.7, label='Z (forward/back)')
    plt.ylabel('Acceleration (g)')
    plt.title('All Acceleration Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Acceleration magnitude
    plt.subplot(4, 1, 3)
    magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    plt.plot(timestamps, magnitude, color='purple', alpha=0.7)
    plt.ylabel('Magnitude (g)')
    plt.title('Acceleration Magnitude')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Algorithm state over time
    plt.subplot(4, 1, 4)
    state_values = []
    for state in all_states:
        if state == 'neutral':
            state_values.append(0)
        elif state == 'descending':
            state_values.append(-1)
        elif state == 'ascending':
            state_values.append(1)
    
    plt.plot(timestamps[:len(state_values)], state_values, linewidth=2)
    plt.ylabel('Algorithm State')
    plt.xlabel('Time (seconds)')
    plt.title('Algorithm State Over Time (-1=Descending, 0=Neutral, 1=Ascending)')
    plt.yticks([-1, 0, 1], ['Descending', 'Neutral', 'Ascending'])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/jonathanschwartz/Documents/Social_Content/QuantumLeapValidator/deep_analysis_session2.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Load data
    metadata, imu_data = load_and_analyze_session()
    
    # Extract acceleration
    timestamps, acc_x, acc_y, acc_z = extract_acceleration_data(imu_data)
    
    # Simulate current algorithm
    detected_reps, state_changes, smoothed_values, all_states = simulate_current_algorithm(timestamps, acc_y)
    
    # Analyze missing reps
    gap_timestamps, gap_acc_y = analyze_missing_reps(timestamps, acc_y, detected_reps)
    
    # Manual rep detection
    try:
        potential_reps = detect_potential_reps_manually(timestamps, acc_y, smoothed_values)
    except ImportError:
        print("scipy not available, skipping manual peak detection")
        potential_reps = []
    
    # Motion pattern analysis
    analyze_motion_patterns(timestamps, acc_x, acc_y, acc_z)
    
    # Create plots
    create_detailed_plots(timestamps, acc_x, acc_y, acc_z, detected_reps, smoothed_values, all_states)
    
    print("\n=== CONCLUSIONS ===")
    print("1. Check if detected reps correspond to phone placement rather than actual squats")
    print("2. Look for patterns in the gap where 5+ reps were missed")
    print("3. Consider if algorithm is too sensitive to setup motions")
    print("4. May need to distinguish exercise motion from handling/placement motion")
