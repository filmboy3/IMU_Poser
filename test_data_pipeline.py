#!/usr/bin/env python3
"""
Test script for QuantumLeap Validator data processing pipeline.
Validates the data ingestion and processing functionality.
"""

import json
import zipfile
import tempfile
import numpy as np
from pathlib import Path
import h5py
from datetime import datetime
import logging

from data_ingestion_pipeline import QuantumLeapDataProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_session_data(session_dir: Path, exercise_type: str = "Squat", duration: float = 30.0):
    """Create mock session data for testing."""
    
    session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create session metadata
    metadata = {
        "sessionId": session_id,
        "exerciseType": exercise_type,
        "startTime": "2025-09-03T08:30:00Z",
        "endTime": "2025-09-03T08:30:30Z",
        "duration": duration,
        "imuSampleCount": int(duration * 100),  # 100Hz
        "videoFrameCount": int(duration * 30),   # 30fps
        "averageIMURate": 100.0,
        "averageVideoFPS": 30.0
    }
    
    with open(session_dir / "session_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create mock IMU data (100Hz for 30 seconds = 3000 samples)
    imu_data = []
    for i in range(int(duration * 100)):
        timestamp = i / 100.0
        
        # Simulate squat motion with sine waves
        squat_phase = np.sin(2 * np.pi * timestamp / 3.0)  # 3-second squat cycle
        
        imu_sample = {
            "timestamp": timestamp,
            "acceleration": {
                "x": 0.1 * np.random.randn() + 0.2 * squat_phase,
                "y": -9.8 + 0.5 * np.random.randn() + squat_phase,
                "z": 0.1 * np.random.randn()
            },
            "rotationRate": {
                "x": 0.1 * np.random.randn(),
                "y": 0.1 * np.random.randn() + 0.3 * squat_phase,
                "z": 0.1 * np.random.randn()
            }
        }
        imu_data.append(imu_sample)
    
    with open(session_dir / "imu_data.json", 'w') as f:
        json.dump(imu_data, f)
    
    # Create mock pose data (30fps for 30 seconds = 900 frames)
    pose_data = []
    for i in range(int(duration * 30)):
        timestamp = i / 30.0
        
        # Simulate basic pose landmarks
        frame_data = {
            "timestamp": timestamp,
            "nose": {"x": 0.5 + 0.01 * np.random.randn(), "y": 0.2 + 0.01 * np.random.randn(), "confidence": 0.95},
            "neck": {"x": 0.5 + 0.01 * np.random.randn(), "y": 0.3 + 0.01 * np.random.randn(), "confidence": 0.92},
            "left_shoulder": {"x": 0.4 + 0.02 * np.random.randn(), "y": 0.35 + 0.02 * np.random.randn(), "confidence": 0.88},
            "right_shoulder": {"x": 0.6 + 0.02 * np.random.randn(), "y": 0.35 + 0.02 * np.random.randn(), "confidence": 0.87},
            "left_hip": {"x": 0.45 + 0.02 * np.random.randn(), "y": 0.6 + 0.05 * np.sin(2 * np.pi * timestamp / 3.0), "confidence": 0.85},
            "right_hip": {"x": 0.55 + 0.02 * np.random.randn(), "y": 0.6 + 0.05 * np.sin(2 * np.pi * timestamp / 3.0), "confidence": 0.84},
            "left_knee": {"x": 0.45 + 0.03 * np.random.randn(), "y": 0.75 + 0.1 * np.sin(2 * np.pi * timestamp / 3.0), "confidence": 0.80},
            "right_knee": {"x": 0.55 + 0.03 * np.random.randn(), "y": 0.75 + 0.1 * np.sin(2 * np.pi * timestamp / 3.0), "confidence": 0.79},
            "left_ankle": {"x": 0.45 + 0.02 * np.random.randn(), "y": 0.9, "confidence": 0.75},
            "right_ankle": {"x": 0.55 + 0.02 * np.random.randn(), "y": 0.9, "confidence": 0.74}
        }
        pose_data.append(frame_data)
    
    with open(session_dir / "pose_data.json", 'w') as f:
        json.dump(pose_data, f)
    
    logger.info(f"Created mock session data: {session_id}")
    return session_id

def create_mock_session_zip(output_path: Path, exercise_type: str = "Squat") -> Path:
    """Create a mock session zip file for testing."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        session_id = create_mock_session_data(temp_path, exercise_type)
        
        # Create zip file
        zip_path = output_path / f"{session_id}.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            for file_path in temp_path.glob("*.json"):
                zip_file.write(file_path, file_path.name)
        
        logger.info(f"Created mock session zip: {zip_path}")
        return zip_path

def test_single_session_processing():
    """Test processing a single session zip file."""
    logger.info("Testing single session processing...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock session zip
        zip_path = create_mock_session_zip(temp_path, "Squat")
        
        # Process the session
        processor = QuantumLeapDataProcessor(temp_path / "processed")
        result = processor.process_session_zip(str(zip_path))
        
        # Validate results
        assert result['status'] == 'success', f"Processing failed: {result}"
        assert result['imu_samples'] > 0, "No IMU samples processed"
        assert Path(result['output_file']).exists(), "Output file not created"
        
        # Validate HDF5 output
        with h5py.File(result['output_file'], 'r') as f:
            assert 'metadata' in f, "Metadata group missing"
            assert 'imu_data' in f, "IMU data group missing"
            assert 'pose_data' in f, "Pose data group missing"
            
            # Check data shapes
            imu_timestamps = f['imu_data/timestamps'][:]
            imu_acceleration = f['imu_data/acceleration'][:]
            assert len(imu_timestamps) == len(imu_acceleration), "IMU data shape mismatch"
            
            logger.info(f"IMU samples: {len(imu_timestamps)}")
            logger.info(f"IMU data shape: {imu_acceleration.shape}")
        
        logger.info("✅ Single session processing test passed")
        return result

def test_batch_processing():
    """Test batch processing of multiple sessions."""
    logger.info("Testing batch processing...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create multiple mock sessions
        exercises = ["Squat", "Bicep Curl", "Lateral Raise"]
        zip_files = []
        
        for exercise in exercises:
            zip_path = create_mock_session_zip(temp_path, exercise)
            zip_files.append(zip_path)
        
        # Process all sessions
        processor = QuantumLeapDataProcessor(temp_path / "processed")
        results = processor.batch_process_sessions(str(temp_path))
        
        # Validate results
        successful_results = [r for r in results if r['status'] == 'success']
        assert len(successful_results) == len(exercises), f"Expected {len(exercises)} successful results, got {len(successful_results)}"
        
        # Test dataset creation
        output_files = [r['output_file'] for r in successful_results]
        dataset_path = processor.create_training_dataset(output_files, "test_dataset.h5")
        
        # Validate combined dataset
        with h5py.File(dataset_path, 'r') as f:
            assert 'imu_sequences' in f, "Combined IMU data missing"
            assert f.attrs['num_sessions'] == len(exercises), f"Expected {len(exercises)} sessions in dataset"
            
            logger.info(f"Combined dataset sessions: {f.attrs['num_sessions']}")
        
        logger.info("✅ Batch processing test passed")
        return results

def test_data_synchronization():
    """Test data synchronization functionality."""
    logger.info("Testing data synchronization...")
    
    # Create test data with known timing
    imu_data = [
        {"timestamp": i/100.0, "acceleration": {"x": 0, "y": 0, "z": 0}, "rotationRate": {"x": 0, "y": 0, "z": 0}}
        for i in range(1000)  # 10 seconds at 100Hz
    ]
    
    pose_data = [
        {"timestamp": i/30.0, "landmarks": {"nose": {"x": 0.5, "y": 0.2, "confidence": 0.9}}}
        for i in range(300)   # 10 seconds at 30fps
    ]
    
    metadata = {"sessionId": "test_sync", "exerciseType": "Test", "duration": 10.0}
    
    processor = QuantumLeapDataProcessor()
    synchronized_data = processor._synchronize_data_streams(imu_data, pose_data, None, metadata)
    
    # Validate synchronization
    sync_info = synchronized_data['synchronization_info']
    
    assert sync_info['imu_sample_rate'] > 90, f"IMU sample rate too low: {sync_info['imu_sample_rate']}"
    assert sync_info['pose_frame_rate'] > 25, f"Pose frame rate too low: {sync_info['pose_frame_rate']}"
    
    logger.info(f"IMU sample rate: {sync_info['imu_sample_rate']:.1f} Hz")
    logger.info(f"Pose frame rate: {sync_info['pose_frame_rate']:.1f} fps")
    logger.info("✅ Data synchronization test passed")

def run_all_tests():
    """Run all test functions."""
    logger.info("🧪 Starting QuantumLeap Validator data pipeline tests...")
    
    try:
        # Test individual components
        test_data_synchronization()
        
        # Test processing pipeline
        single_result = test_single_session_processing()
        batch_results = test_batch_processing()
        
        logger.info("🎉 All tests passed successfully!")
        
        return {
            'single_session': single_result,
            'batch_processing': batch_results,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    run_all_tests()
