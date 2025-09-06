#!/usr/bin/env python3
"""
Debug tokenization pipeline to fix the None return issue
"""

import time
import numpy as np
from tokenization_pipeline import MultiModalTokenizer
from realtime_inference_pipeline import SensorData

def debug_tokenization():
    """Debug tokenization step by step"""
    print("=== Tokenization Debug Session ===")
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = MultiModalTokenizer()
    tokenizer.load_tokenizer("/Users/jonathanschwartz/Documents/Social_Content/QuantumLeapValidator/trained_tokenizer.pkl")
    print("✅ Tokenizer loaded")
    print(f"   IMU tokenizer trained: {tokenizer.imu_tokenizer.is_trained}")
    print(f"   Codebook size: {tokenizer.imu_tokenizer.codebook_size}")
    
    # Test sensor data
    print("\n2. Creating test sensor data...")
    sensor_data = SensorData(
        timestamp=time.time(),
        imu_data={'x': 1.5, 'y': 1.0, 'z': 9.8},
        session_id="debug"
    )
    print(f"✅ Sensor data: {sensor_data.imu_data}")
    
    # Test format conversion
    print("\n3. Testing format conversion...")
    imu_sample = {
        'acceleration': {
            'x': sensor_data.imu_data.get('x', 0.0),
            'y': sensor_data.imu_data.get('y', 0.0), 
            'z': sensor_data.imu_data.get('z', 0.0)
        },
        'rotationRate': {
            'x': 0.0,
            'y': 0.0,
            'z': 0.0
        },
        'timestamp': sensor_data.timestamp
    }
    print(f"✅ Formatted sample: {imu_sample}")
    
    # Test feature extraction
    print("\n4. Testing feature extraction...")
    try:
        features = tokenizer.imu_tokenizer.extract_features([imu_sample])
        print(f"✅ Features extracted: shape {features.shape}")
        print(f"   Features: {features}")
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return
    
    # Test window creation
    print("\n5. Testing window creation...")
    try:
        windows = tokenizer.imu_tokenizer.create_windows(features)
        print(f"✅ Windows created: shape {windows.shape}")
        print(f"   Windows: {windows}")
    except Exception as e:
        print(f"❌ Window creation failed: {e}")
        return
    
    # Test normalization
    print("\n6. Testing normalization...")
    try:
        if len(windows) > 0:
            windows_norm = tokenizer.imu_tokenizer.scaler.transform(windows)
            print(f"✅ Windows normalized: shape {windows_norm.shape}")
        else:
            print("❌ No windows to normalize")
            return
    except Exception as e:
        print(f"❌ Normalization failed: {e}")
        return
    
    # Test clustering prediction
    print("\n7. Testing clustering prediction...")
    try:
        tokens = tokenizer.imu_tokenizer.model.predict(windows_norm)
        print(f"✅ Tokens predicted: {tokens}")
        offset_tokens = tokens + 1024
        print(f"✅ Offset tokens: {offset_tokens}")
    except Exception as e:
        print(f"❌ Clustering prediction failed: {e}")
        return
    
    # Test full tokenization method
    print("\n8. Testing full tokenization method...")
    try:
        full_tokens = tokenizer.imu_tokenizer.tokenize([imu_sample])
        print(f"✅ Full tokenization result: {full_tokens}")
    except Exception as e:
        print(f"❌ Full tokenization failed: {e}")
        print(f"   Error details: {type(e).__name__}: {e}")
        
        # Try to understand the issue
        print("\n   Debugging tokenize method...")
        if hasattr(tokenizer.imu_tokenizer, 'is_trained') and not tokenizer.imu_tokenizer.is_trained:
            print("   Issue: Tokenizer not trained")
        else:
            print("   Tokenizer appears trained, checking method...")
    
    # Test with multiple samples to see if windowing is the issue
    print("\n9. Testing with multiple samples...")
    multi_samples = []
    for i in range(10):  # Create 10 samples
        sample = {
            'acceleration': {
                'x': 1.0 + i * 0.1,
                'y': 1.0 + np.sin(i) * 0.5,
                'z': 9.8 + np.cos(i) * 0.2
            },
            'rotationRate': {
                'x': 0.0,
                'y': 0.0,
                'z': 0.0
            },
            'timestamp': time.time() + i * 0.1
        }
        multi_samples.append(sample)
    
    try:
        multi_tokens = tokenizer.imu_tokenizer.tokenize(multi_samples)
        print(f"✅ Multi-sample tokenization: {len(multi_tokens)} tokens")
        print(f"   Tokens: {multi_tokens}")
    except Exception as e:
        print(f"❌ Multi-sample tokenization failed: {e}")

if __name__ == "__main__":
    debug_tokenization()
