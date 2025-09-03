#!/usr/bin/env python3
"""
QuantumLeap Validator - Data Ingestion Pipeline
Processes exported iOS session data for model training and validation.
"""

import json
import zipfile
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import h5py
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuantumLeapDataProcessor:
    """Process iOS QuantumLeap Validator session data for model training."""
    
    def __init__(self, output_dir: str = "processed_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def process_session_zip(self, zip_path: str) -> Dict:
        """Process a single session zip file from iOS app."""
        logger.info(f"Processing session: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract to temporary directory
            temp_dir = self.output_dir / "temp_extraction"
            temp_dir.mkdir(exist_ok=True)
            zip_ref.extractall(temp_dir)
            
            try:
                # Load session metadata
                metadata = self._load_session_metadata(temp_dir)
                
                # Load IMU data
                imu_data = self._load_imu_data(temp_dir)
                
                # Process video and extract poses
                pose_data, video_metadata = self._process_video(temp_dir)
                
                # Load existing pose data if available
                existing_pose_data = self._load_pose_data(temp_dir)
                
                # Synchronize data streams
                synchronized_data = self._synchronize_data_streams(
                    imu_data, pose_data, existing_pose_data, metadata
                )
                
                # Save processed data
                output_file = self._save_processed_data(synchronized_data, metadata)
                
                logger.info(f"Successfully processed session: {metadata['sessionId']}")
                return {
                    'status': 'success',
                    'session_id': metadata['sessionId'],
                    'output_file': str(output_file),
                    'imu_samples': len(imu_data),
                    'pose_frames': len(pose_data) if pose_data else 0,
                    'duration': metadata.get('duration', 0)
                }
                
            finally:
                # Clean up temporary files
                import shutil
                shutil.rmtree(temp_dir)
    
    def _load_session_metadata(self, session_dir: Path) -> Dict:
        """Load session metadata from JSON file."""
        metadata_file = session_dir / "session_metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Session metadata not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded metadata for session: {metadata['sessionId']}")
        return metadata
    
    def _load_imu_data(self, session_dir: Path) -> List[Dict]:
        """Load IMU data from JSON file."""
        imu_file = session_dir / "imu_data.json"
        
        if not imu_file.exists():
            logger.warning(f"IMU data not found: {imu_file}")
            return []
        
        with open(imu_file, 'r') as f:
            imu_data = json.load(f)
        
        logger.info(f"Loaded {len(imu_data)} IMU samples")
        return imu_data
    
    def _load_pose_data(self, session_dir: Path) -> Optional[List[Dict]]:
        """Load existing pose data from JSON file if available."""
        pose_file = session_dir / "pose_data.json"
        
        if not pose_file.exists():
            logger.info("No existing pose data found")
            return None
        
        with open(pose_file, 'r') as f:
            pose_data = json.load(f)
        
        logger.info(f"Loaded {len(pose_data)} pose frames")
        return pose_data
    
    def _process_video(self, session_dir: Path) -> Tuple[List[Dict], Dict]:
        """Process video file and extract pose data using MediaPipe."""
        video_file = session_dir / "video.mp4"
        
        if not video_file.exists():
            logger.warning(f"Video file not found: {video_file}")
            return [], {}
        
        try:
            import mediapipe as mp
            
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            cap = cv2.VideoCapture(str(video_file))
            
            if not cap.isOpened():
                logger.error(f"Could not open video file: {video_file}")
                return [], {}
            
            # Get video metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            video_metadata = {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': frame_count / fps if fps > 0 else 0
            }
            
            pose_data = []
            frame_idx = 0
            
            logger.info(f"Processing video: {frame_count} frames at {fps} FPS")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with MediaPipe
                results = pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    # Extract landmark data
                    landmarks = {}
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        landmark_name = mp_pose.PoseLandmark(idx).name.lower()
                        landmarks[landmark_name] = {
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': landmark.visibility
                        }
                    
                    pose_data.append({
                        'frame_index': frame_idx,
                        'timestamp': frame_idx / fps,
                        'landmarks': landmarks
                    })
                
                frame_idx += 1
                
                # Progress logging
                if frame_idx % 100 == 0:
                    logger.info(f"Processed {frame_idx}/{frame_count} frames")
            
            cap.release()
            pose.close()
            
            logger.info(f"Extracted pose data from {len(pose_data)} frames")
            return pose_data, video_metadata
            
        except ImportError:
            logger.warning("MediaPipe not available. Install with: pip install mediapipe")
            return [], {}
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return [], {}
    
    def _synchronize_data_streams(self, imu_data: List[Dict], 
                                video_pose_data: List[Dict],
                                existing_pose_data: Optional[List[Dict]],
                                metadata: Dict) -> Dict:
        """Synchronize IMU and pose data streams using timestamps."""
        
        # Convert IMU data to DataFrame for easier processing
        if imu_data:
            imu_df = pd.DataFrame([
                {
                    'timestamp': sample['timestamp'],
                    'acc_x': sample['acceleration']['x'],
                    'acc_y': sample['acceleration']['y'],
                    'acc_z': sample['acceleration']['z'],
                    'gyro_x': sample['rotationRate']['x'],
                    'gyro_y': sample['rotationRate']['y'],
                    'gyro_z': sample['rotationRate']['z']
                }
                for sample in imu_data
            ])
        else:
            imu_df = pd.DataFrame()
        
        # Use video pose data if available, otherwise existing pose data
        pose_data = video_pose_data if video_pose_data else (existing_pose_data or [])
        
        # Convert pose data to structured format
        if pose_data:
            pose_df = pd.DataFrame([
                {
                    'timestamp': frame.get('timestamp', frame.get('frame_index', 0) * (1/30)),  # Assume 30fps if no timestamp
                    'pose_landmarks': frame.get('landmarks', frame)
                }
                for frame in pose_data
            ])
        else:
            pose_df = pd.DataFrame()
        
        # Synchronize based on overlapping time ranges
        synchronized_data = {
            'metadata': metadata,
            'imu_data': imu_df.to_dict('records') if not imu_df.empty else [],
            'pose_data': pose_df.to_dict('records') if not pose_df.empty else [],
            'synchronization_info': {
                'imu_start_time': imu_df['timestamp'].min() if not imu_df.empty else None,
                'imu_end_time': imu_df['timestamp'].max() if not imu_df.empty else None,
                'pose_start_time': pose_df['timestamp'].min() if not pose_df.empty else None,
                'pose_end_time': pose_df['timestamp'].max() if not pose_df.empty else None,
                'imu_sample_rate': len(imu_df) / (imu_df['timestamp'].max() - imu_df['timestamp'].min()) if len(imu_df) > 1 else 0,
                'pose_frame_rate': len(pose_df) / (pose_df['timestamp'].max() - pose_df['timestamp'].min()) if len(pose_df) > 1 else 0
            }
        }
        
        logger.info(f"Synchronized data: {len(imu_df)} IMU samples, {len(pose_df)} pose frames")
        return synchronized_data
    
    def _save_processed_data(self, synchronized_data: Dict, metadata: Dict) -> Path:
        """Save processed data in HDF5 format compatible with QuantumLeap model."""
        
        session_id = metadata['sessionId']
        output_file = self.output_dir / f"{session_id}_processed.h5"
        
        with h5py.File(output_file, 'w') as f:
            # Save metadata
            meta_group = f.create_group('metadata')
            for key, value in metadata.items():
                if isinstance(value, str):
                    meta_group.attrs[key] = value
                elif isinstance(value, (int, float)):
                    meta_group.attrs[key] = value
                elif isinstance(value, datetime):
                    meta_group.attrs[key] = value.isoformat()
            
            # Save IMU data
            if synchronized_data['imu_data']:
                imu_group = f.create_group('imu_data')
                imu_df = pd.DataFrame(synchronized_data['imu_data'])
                
                imu_group.create_dataset('timestamps', data=imu_df['timestamp'].values)
                imu_group.create_dataset('acceleration', data=imu_df[['acc_x', 'acc_y', 'acc_z']].values)
                imu_group.create_dataset('rotation_rate', data=imu_df[['gyro_x', 'gyro_y', 'gyro_z']].values)
            
            # Save pose data
            if synchronized_data['pose_data']:
                pose_group = f.create_group('pose_data')
                pose_df = pd.DataFrame(synchronized_data['pose_data'])
                
                pose_group.create_dataset('timestamps', data=pose_df['timestamp'].values)
                
                # Convert pose landmarks to array format
                if 'pose_landmarks' in pose_df.columns:
                    landmarks_list = []
                    for landmarks in pose_df['pose_landmarks']:
                        if isinstance(landmarks, dict):
                            # Convert to array format expected by model
                            landmark_array = np.zeros((33, 3))  # 33 MediaPipe landmarks, 3D coordinates
                            for name, coords in landmarks.items():
                                # Map landmark names to indices (simplified)
                                if isinstance(coords, dict) and 'x' in coords:
                                    # This would need proper mapping based on MediaPipe landmark indices
                                    pass
                        landmarks_list.append(landmark_array)
                    
                    if landmarks_list:
                        pose_group.create_dataset('landmarks', data=np.array(landmarks_list))
            
            # Save synchronization info
            sync_group = f.create_group('synchronization')
            for key, value in synchronized_data['synchronization_info'].items():
                if value is not None:
                    sync_group.attrs[key] = value
        
        logger.info(f"Saved processed data to: {output_file}")
        return output_file
    
    def batch_process_sessions(self, session_dir: str) -> List[Dict]:
        """Process all session zip files in a directory."""
        session_path = Path(session_dir)
        zip_files = list(session_path.glob("*.zip"))
        
        if not zip_files:
            logger.warning(f"No zip files found in: {session_dir}")
            return []
        
        results = []
        for zip_file in zip_files:
            try:
                result = self.process_session_zip(str(zip_file))
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {zip_file}: {e}")
                results.append({
                    'status': 'error',
                    'file': str(zip_file),
                    'error': str(e)
                })
        
        return results
    
    def create_training_dataset(self, processed_files: List[str], output_file: str = "quantumleap_real_world_dataset.h5"):
        """Combine multiple processed sessions into a training dataset."""
        
        output_path = self.output_dir / output_file
        
        all_imu_data = []
        all_pose_data = []
        all_metadata = []
        
        for file_path in processed_files:
            try:
                with h5py.File(file_path, 'r') as f:
                    # Load data from each session
                    if 'imu_data' in f:
                        imu_timestamps = f['imu_data/timestamps'][:]
                        imu_acceleration = f['imu_data/acceleration'][:]
                        imu_rotation = f['imu_data/rotation_rate'][:]
                        
                        # Combine IMU data
                        session_imu = np.column_stack([imu_acceleration, imu_rotation])
                        all_imu_data.append(session_imu)
                    
                    if 'pose_data' in f and 'landmarks' in f['pose_data']:
                        pose_landmarks = f['pose_data/landmarks'][:]
                        all_pose_data.append(pose_landmarks)
                    
                    # Collect metadata
                    metadata = dict(f['metadata'].attrs)
                    all_metadata.append(metadata)
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        # Save combined dataset
        with h5py.File(output_path, 'w') as f:
            if all_imu_data:
                f.create_dataset('imu_sequences', data=all_imu_data)
            
            if all_pose_data:
                f.create_dataset('pose_sequences', data=all_pose_data)
            
            # Save metadata as JSON string
            f.attrs['metadata'] = json.dumps(all_metadata)
            f.attrs['num_sessions'] = len(all_metadata)
            f.attrs['creation_date'] = datetime.now().isoformat()
        
        logger.info(f"Created training dataset: {output_path}")
        logger.info(f"Combined {len(all_metadata)} sessions")
        return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Process QuantumLeap Validator session data")
    parser.add_argument("input", help="Input zip file or directory containing session zips")
    parser.add_argument("--output-dir", default="processed_data", help="Output directory for processed data")
    parser.add_argument("--create-dataset", action="store_true", help="Create combined training dataset")
    
    args = parser.parse_args()
    
    processor = QuantumLeapDataProcessor(args.output_dir)
    
    input_path = Path(args.input)
    
    if input_path.is_file() and input_path.suffix == '.zip':
        # Process single session
        result = processor.process_session_zip(str(input_path))
        print(f"Processing result: {result}")
        
    elif input_path.is_dir():
        # Process all sessions in directory
        results = processor.batch_process_sessions(str(input_path))
        
        successful_files = [r['output_file'] for r in results if r['status'] == 'success']
        
        print(f"Processed {len(successful_files)} sessions successfully")
        
        if args.create_dataset and successful_files:
            dataset_file = processor.create_training_dataset(successful_files)
            print(f"Created training dataset: {dataset_file}")
    
    else:
        print(f"Invalid input path: {input_path}")


if __name__ == "__main__":
    main()
