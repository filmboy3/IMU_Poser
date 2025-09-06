#!/usr/bin/env python3
"""
Analyze motion patterns to distinguish phone handling/rustling from actual exercise
Focus: 28 detected reps but only 13 actual - need to filter out handling motion
"""

import json
import numpy as np

def analyze_handling_vs_exercise():
    """Analyze the latest session to understand handling patterns"""
    
    session_dir = "/Users/jonathanschwartz/Downloads/session_20250903_162814_24ECED46"
    
    # Load data
    with open(f"{session_dir}/session_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    with open(f"{session_dir}/imu_data.json", 'r') as f:
        imu_data = json.load(f)
    
    print("=== MOTION PATTERN ANALYSIS ===")
    print(f"Duration: {metadata['duration']:.1f}s")
    print(f"Detected Reps: {metadata['repCount']} (but only 13 actual)")
    print(f"Problem: ~15 false positives from phone handling")
    print()
    
    # Extract acceleration data
    timestamps = np.array([sample['timestamp'] for sample in imu_data])
    acc_x = np.array([sample['acceleration']['x'] for sample in imu_data])
    acc_y = np.array([sample['acceleration']['y'] for sample in imu_data])
    acc_z = np.array([sample['acceleration']['z'] for sample in imu_data])
    
    # Normalize timestamps
    timestamps = timestamps - timestamps[0]
    
    print("=== ACCELERATION CHARACTERISTICS ===")
    print(f"Y-axis range: {acc_y.min():.3f} to {acc_y.max():.3f}")
    print(f"X-axis range: {acc_x.min():.3f} to {acc_x.max():.3f}")
    print(f"Z-axis range: {acc_z.min():.3f} to {acc_z.max():.3f}")
    
    # Calculate acceleration magnitude and jerk
    magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    jerk_y = np.diff(acc_y)  # Rate of change of acceleration
    jerk_mag = np.diff(magnitude)
    
    print(f"Magnitude range: {magnitude.min():.3f} to {magnitude.max():.3f}")
    print(f"Y-jerk std: {jerk_y.std():.3f}")
    print(f"Magnitude jerk std: {jerk_mag.std():.3f}")
    print()
    
    # Analyze motion patterns by time segments
    analyze_motion_segments(timestamps, acc_x, acc_y, acc_z, magnitude)
    
    # Detect handling vs exercise patterns
    detect_motion_types(timestamps, acc_x, acc_y, acc_z, magnitude)
    
    # Simulate rep detection with timeline
    simulate_rep_detection_timeline(timestamps, acc_y)

def analyze_motion_segments(timestamps, acc_x, acc_y, acc_z, magnitude):
    """Analyze motion characteristics in time segments"""
    
    print("=== MOTION SEGMENT ANALYSIS ===")
    total_duration = timestamps[-1]
    num_segments = 20  # More granular analysis
    segment_duration = total_duration / num_segments
    
    handling_segments = []
    exercise_segments = []
    
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        
        mask = (timestamps >= start_time) & (timestamps < end_time)
        if np.sum(mask) < 10:  # Skip segments with too few samples
            continue
            
        seg_x = acc_x[mask]
        seg_y = acc_y[mask]
        seg_z = acc_z[mask]
        seg_mag = magnitude[mask]
        
        # Calculate motion characteristics
        y_std = seg_y.std()
        x_std = seg_x.std()
        z_std = seg_z.std()
        mag_std = seg_mag.std()
        
        # Multi-axis variation (handling tends to affect all axes)
        multi_axis_activity = x_std + y_std + z_std
        
        # Jerkiness (handling tends to be more erratic)
        jerkiness = np.std(np.diff(seg_mag)) if len(seg_mag) > 1 else 0
        
        # Dominant axis (exercise tends to be Y-dominant)
        y_dominance = y_std / (x_std + z_std + 0.001)  # Avoid division by zero
        
        print(f"Seg {i+1:2d} ({start_time:4.1f}-{end_time:4.1f}s): "
              f"Y_std={y_std:.3f}, Multi_axis={multi_axis_activity:.3f}, "
              f"Jerk={jerkiness:.3f}, Y_dom={y_dominance:.2f}")
        
        # Classify segment
        if multi_axis_activity > 0.15 and jerkiness > 0.08:  # High multi-axis + jerky
            handling_segments.append(i)
            print(f"    -> HANDLING MOTION")
        elif y_dominance > 2.0 and y_std > 0.08:  # Y-dominant with good activity
            exercise_segments.append(i)
            print(f"    -> EXERCISE MOTION")
    
    print(f"\nSummary:")
    print(f"Handling segments: {len(handling_segments)} ({len(handling_segments)/num_segments*100:.1f}%)")
    print(f"Exercise segments: {len(exercise_segments)} ({len(exercise_segments)/num_segments*100:.1f}%)")
    print()

def detect_motion_types(timestamps, acc_x, acc_y, acc_z, magnitude):
    """Detect different types of motion patterns"""
    
    print("=== MOTION TYPE DETECTION ===")
    
    window_size = 100  # ~1 second windows
    motion_types = []
    
    for i in range(0, len(acc_y) - window_size, window_size//2):
        window_x = acc_x[i:i+window_size]
        window_y = acc_y[i:i+window_size]
        window_z = acc_z[i:i+window_size]
        window_mag = magnitude[i:i+window_size]
        window_time = timestamps[i + window_size//2]
        
        # Motion characteristics
        y_activity = window_y.std()
        x_activity = window_x.std()
        z_activity = window_z.std()
        total_activity = y_activity + x_activity + z_activity
        
        # Frequency analysis (rough)
        y_crossings = np.sum(np.diff(np.sign(window_y - window_y.mean())) != 0)
        
        # Jerkiness
        jerkiness = np.std(np.diff(window_mag))
        
        # Orientation consistency (handling changes phone orientation)
        orientation_stability = 1.0 / (1.0 + np.std([x_activity, y_activity, z_activity]))
        
        # Classification
        motion_type = "unknown"
        
        if total_activity < 0.05:
            motion_type = "stable"
        elif jerkiness > 0.12 and total_activity > 0.2:
            motion_type = "handling"
        elif y_activity > 0.08 and orientation_stability > 0.4 and y_crossings < 20:
            motion_type = "exercise"
        elif total_activity > 0.15:
            motion_type = "rustling"
        
        motion_types.append({
            'time': window_time,
            'type': motion_type,
            'y_activity': y_activity,
            'total_activity': total_activity,
            'jerkiness': jerkiness,
            'orientation_stability': orientation_stability
        })
    
    # Count motion types
    type_counts = {}
    for motion in motion_types:
        motion_type = motion['type']
        type_counts[motion_type] = type_counts.get(motion_type, 0) + 1
    
    print("Motion type distribution:")
    for motion_type, count in type_counts.items():
        percentage = count / len(motion_types) * 100
        print(f"  {motion_type}: {count} windows ({percentage:.1f}%)")
    
    # Show timeline of motion types
    print(f"\nMotion timeline (first 20 windows):")
    for i, motion in enumerate(motion_types[:20]):
        print(f"  {motion['time']:5.1f}s: {motion['type']:10s} "
              f"(y_act={motion['y_activity']:.3f}, jerk={motion['jerkiness']:.3f})")
    
    return motion_types

def simulate_rep_detection_timeline(timestamps, acc_y):
    """Simulate rep detection with detailed timeline"""
    
    print("=== REP DETECTION SIMULATION ===")
    
    # Current algorithm parameters
    peak_threshold = 0.12
    valley_threshold = -0.10
    neutral_threshold = 0.06
    buffer_size = 8
    
    state = 'neutral'
    rep_count = 0
    buffer = []
    
    print("Rep detection timeline:")
    
    for i, acc in enumerate(acc_y):
        buffer.append(acc)
        if len(buffer) > buffer_size:
            buffer.pop(0)
        
        smoothed_acc = sum(buffer) / len(buffer)
        
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
                print(f"  Rep {rep_count:2d}: {timestamps[i]:6.1f}s (smoothed: {smoothed_acc:6.3f})")
    
    print(f"\nTotal reps detected: {rep_count}")
    
    # Identify problematic periods
    if rep_count > 15:  # More than expected
        print("\n*** EXCESSIVE REP DETECTION - LIKELY HANDLING INTERFERENCE ***")

def suggest_improvements():
    """Suggest algorithm improvements based on analysis"""
    
    print("\n=== ALGORITHM IMPROVEMENT SUGGESTIONS ===")
    print("1. MOTION STATE DETECTION:")
    print("   - Add 'handling' and 'stable' states before exercise")
    print("   - Require phone to be stable for 3+ seconds before counting")
    print("   - Audio cue when ready to exercise")
    print()
    print("2. MOTION PATTERN FILTERING:")
    print("   - Filter out high-jerkiness motion (jerk > 0.12)")
    print("   - Require Y-axis dominance for valid reps")
    print("   - Ignore multi-axis chaotic motion")
    print()
    print("3. EXERCISE READINESS:")
    print("   - Detect when phone is placed in pocket (stable + oriented)")
    print("   - 5-second countdown before exercise tracking begins")
    print("   - Audio feedback for ready state")
    print()
    print("4. SMART REP FILTERING:")
    print("   - Minimum rep duration (1.5-4 seconds)")
    print("   - Maximum rep frequency (prevent rapid false positives)")
    print("   - Orientation consistency check")

if __name__ == "__main__":
    analyze_handling_vs_exercise()
    suggest_improvements()
