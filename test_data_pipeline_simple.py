#!/usr/bin/env python3
"""
Simplified test script for QuantumLeap Validator data processing pipeline.
Tests core functionality without external dependencies.
"""

import json
import zipfile
import tempfile
import numpy as np
from pathlib import Path
import h5py
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDataProcessor:
    """Simplified data processor for testing without external dependencies."""
    
    def __init__(self, output_dir: str = "processed_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def process_session_zip(self, zip_path: str) -> dict:
        """Process a session zip file."""
        logger.info(f"Processing session: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            temp_dir = self.output_dir / "temp_extraction"
            temp_dir.mkdir(exist_ok=True)
            zip_ref.extractall(temp_dir)
            
            try:
                # Load session data
                metadata = self._load_json(temp_dir / "session_metadata.json")
                imu_data = self._load_json(temp_dir / "imu_data.json")
                pose_data = self._load_json(temp_dir / "pose_data.json")
                
                # Process and save
                output_file = self._save_processed_data(metadata, imu_data, pose_data)
                
                return {
                    'status': 'success',
                    'session_id': metadata['sessionId'],
                    'output_file': str(output_file),
                    'imu_samples': len(imu_data),
                    'pose_frames': len(pose_data)
                }
                
            finally:
                import shutil
                shutil.rmtree(temp_dir)
    
    def _load_json(self, file_path: Path):
        """Load JSON file."""
        if not file_path.exists():
            return []
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def _save_processed_data(self, metadata, imu_data, pose_data) -> Path:
        """Save processed data to HDF5."""
        session_id = metadata['sessionId']
        output_file = self.output_dir / f"{session_id}_processed.h5"
        
        with h5py.File(output_file, 'w') as f:
            # Metadata
            meta_group = f.create_group('metadata')
            for key, value in metadata.items():
                if isinstance(value, (str, int, float)):
                    meta_group.attrs[key] = value
            
            # IMU data
            if imu_data:
                imu_group = f.create_group('imu_data')
                timestamps = [sample['timestamp'] for sample in imu_data]
                accelerations = [[sample['acceleration']['x'], 
                                sample['acceleration']['y'], 
                                sample['acceleration']['z']] for sample in imu_data]
                rotations = [[sample['rotationRate']['x'], 
                            sample['rotationRate']['y'], 
                            sample['rotationRate']['z']] for sample in imu_data]
                
                imu_group.create_dataset('timestamps', data=timestamps)
                imu_group.create_dataset('acceleration', data=accelerations)
                imu_group.create_dataset('rotation_rate', data=rotations)
            
            # Pose data
            if pose_data:
                pose_group = f.create_group('pose_data')
                pose_timestamps = [frame['timestamp'] for frame in pose_data]
                pose_group.create_dataset('timestamps', data=pose_timestamps)
        
        logger.info(f"Saved processed data to: {output_file}")
        return output_file

def create_mock_session_data(session_dir: Path, exercise_type: str = "Squat", duration: float = 10.0):
    """Create mock session data for testing."""
    
    session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create session metadata
    metadata = {
        "sessionId": session_id,
        "exerciseType": exercise_type,
        "startTime": "2025-09-03T08:30:00Z",
        "endTime": "2025-09-03T08:30:10Z",
        "duration": duration
    }
    
    with open(session_dir / "session_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create mock IMU data
    imu_data = []
    for i in range(int(duration * 100)):  # 100Hz
        timestamp = i / 100.0
        squat_phase = np.sin(2 * np.pi * timestamp / 3.0)
        
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
    
    # Create mock pose data
    pose_data = []
    for i in range(int(duration * 30)):  # 30fps
        timestamp = i / 30.0
        
        frame_data = {
            "timestamp": timestamp,
            "nose": {"x": 0.5, "y": 0.2, "confidence": 0.95},
            "left_shoulder": {"x": 0.4, "y": 0.35, "confidence": 0.88},
            "right_shoulder": {"x": 0.6, "y": 0.35, "confidence": 0.87}
        }
        pose_data.append(frame_data)
    
    with open(session_dir / "pose_data.json", 'w') as f:
        json.dump(pose_data, f)
    
    logger.info(f"Created mock session data: {session_id}")
    return session_id

def create_mock_session_zip(output_path: Path, exercise_type: str = "Squat") -> Path:
    """Create a mock session zip file."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        session_id = create_mock_session_data(temp_path, exercise_type)
        
        zip_path = output_path / f"{session_id}.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            for file_path in temp_path.glob("*.json"):
                zip_file.write(file_path, file_path.name)
        
        logger.info(f"Created mock session zip: {zip_path}")
        return zip_path

def test_processing():
    """Test the data processing pipeline."""
    logger.info("🧪 Testing QuantumLeap Validator data processing...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock session
        zip_path = create_mock_session_zip(temp_path, "Squat")
        
        # Process the session
        processor = SimpleDataProcessor(temp_path / "processed")
        result = processor.process_session_zip(str(zip_path))
        
        # Validate results
        assert result['status'] == 'success', f"Processing failed: {result}"
        assert result['imu_samples'] > 0, "No IMU samples processed"
        assert result['pose_frames'] > 0, "No pose frames processed"
        assert Path(result['output_file']).exists(), "Output file not created"
        
        # Validate HDF5 output
        with h5py.File(result['output_file'], 'r') as f:
            assert 'metadata' in f, "Metadata group missing"
            assert 'imu_data' in f, "IMU data group missing"
            assert 'pose_data' in f, "Pose data group missing"
            
            # Check data
            imu_timestamps = f['imu_data/timestamps'][:]
            imu_acceleration = f['imu_data/acceleration'][:]
            pose_timestamps = f['pose_data/timestamps'][:]
            
            logger.info(f"✅ IMU samples: {len(imu_timestamps)}")
            logger.info(f"✅ IMU acceleration shape: {imu_acceleration.shape}")
            logger.info(f"✅ Pose frames: {len(pose_timestamps)}")
            
            assert len(imu_timestamps) == len(imu_acceleration), "IMU data shape mismatch"
        
        logger.info("🎉 All tests passed successfully!")
        return result

if __name__ == "__main__":
    test_processing()
