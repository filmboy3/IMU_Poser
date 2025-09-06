#!/usr/bin/env python3
"""
Multi-Modal Data Tokenizer for Project Chimera v2
Converts raw audio and IMU streams into unified token vocabulary for transformer training.

Phase 1.1: Multi-Modal Data Tokenizer Implementation
- Audio tokenization using EnCodec or similar RVQ model
- IMU tokenization using VQ-VAE for motion primitives
- Interleaved stream generation for unified perception training
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import librosa
from sklearn.preprocessing import StandardScaler
import pickle

@dataclass
class TokenizedSequence:
    """Represents a tokenized multimodal sequence"""
    tokens: List[int]
    token_types: List[str]  # 'audio' or 'motion'
    timestamps: List[float]
    metadata: Dict

class IMUTokenizer:
    """VQ-VAE based tokenizer for IMU motion data"""
    
    def __init__(self, codebook_size=64, embedding_dim=32, window_size=100):
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size  # 1 second at 100Hz
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        
    def extract_features(self, imu_data: List[Dict]) -> np.ndarray:
        """Extract features from raw IMU data"""
        features = []
        
        for sample in imu_data:
            # Extract acceleration and rotation rate
            acc = sample['acceleration']
            rot = sample['rotationRate']
            
            # Create feature vector: [acc_x, acc_y, acc_z, rot_x, rot_y, rot_z]
            feature_vec = [
                acc['x'], acc['y'], acc['z'],
                rot['x'], rot['y'], rot['z']
            ]
            features.append(feature_vec)
            
        return np.array(features)
    
    def create_windows(self, features: np.ndarray) -> np.ndarray:
        """Create sliding windows from feature sequence"""
        windows = []
        step_size = self.window_size // 2  # 50% overlap
        
        for i in range(0, len(features) - self.window_size + 1, step_size):
            window = features[i:i + self.window_size]
            windows.append(window.flatten())  # Flatten to 1D
            
        return np.array(windows)
    
    def train_vqvae(self, training_sessions: List[str]):
        """Train VQ-VAE on collected IMU data"""
        print("Training IMU VQ-VAE tokenizer...")
        
        # Collect all training data
        all_windows = []
        
        for session_path in training_sessions:
            session_dir = Path(session_path)
            imu_file = session_dir / "imu_data.json"
            
            if imu_file.exists():
                with open(imu_file, 'r') as f:
                    imu_data = json.load(f)
                
                features = self.extract_features(imu_data)
                windows = self.create_windows(features)
                all_windows.append(windows)
        
        # Combine all windows and normalize
        all_windows = np.vstack(all_windows)
        all_windows = self.scaler.fit_transform(all_windows)
        
        # Simple K-means clustering as VQ-VAE approximation
        from sklearn.cluster import KMeans
        
        self.model = KMeans(n_clusters=self.codebook_size, random_state=42)
        self.model.fit(all_windows)
        self.is_trained = True
        
        print(f"IMU tokenizer trained with {self.codebook_size} motion primitives")
        
    def tokenize(self, imu_data: List[Dict]) -> List[int]:
        """Convert IMU sequence to motion tokens"""
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained first")
            
        features = self.extract_features(imu_data)
        windows = self.create_windows(features)
        
        if len(windows) == 0:
            return []
            
        windows_normalized = self.scaler.transform(windows)
        tokens = self.model.predict(windows_normalized)
        
        # Offset by audio vocab size (assuming 1024 audio tokens)
        return (tokens + 1024).tolist()

class AudioTokenizer:
    """Audio tokenizer using pre-trained models or simple spectral features"""
    
    def __init__(self, vocab_size=1024, sample_rate=16000):
        self.vocab_size = vocab_size
        self.sample_rate = sample_rate
        
    def tokenize_audio_file(self, audio_path: str) -> List[int]:
        """Tokenize audio file into discrete tokens"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=512)
            
            # Quantize MFCC features to discrete tokens
            # Simple quantization - in production would use EnCodec
            mfccs_flat = mfccs.T.flatten()
            
            # Normalize and quantize
            mfccs_norm = (mfccs_flat - mfccs_flat.min()) / (mfccs_flat.max() - mfccs_flat.min())
            tokens = (mfccs_norm * (self.vocab_size - 1)).astype(int)
            
            # Downsample to ~50 tokens/second
            target_length = int(len(audio) / sr * 50)
            if len(tokens) > target_length:
                indices = np.linspace(0, len(tokens) - 1, target_length, dtype=int)
                tokens = tokens[indices]
                
            return tokens.tolist()
            
        except Exception as e:
            print(f"Error tokenizing audio {audio_path}: {e}")
            return []

class MultiModalTokenizer:
    """Main tokenizer that combines audio and IMU streams"""
    
    def __init__(self):
        self.audio_tokenizer = AudioTokenizer()
        self.imu_tokenizer = IMUTokenizer()
        
    def train_imu_tokenizer(self, session_paths: List[str]):
        """Train the IMU tokenizer on session data"""
        self.imu_tokenizer.train_vqvae(session_paths)
        
    def tokenize_session(self, session_path: str, audio_path: Optional[str] = None) -> TokenizedSequence:
        """Tokenize a complete session into interleaved sequence"""
        session_dir = Path(session_path)
        
        # Load session metadata
        with open(session_dir / "session_metadata.json", 'r') as f:
            metadata = json.load(f)
            
        # Load IMU data
        with open(session_dir / "imu_data.json", 'r') as f:
            imu_data = json.load(f)
            
        # Tokenize IMU data
        motion_tokens = self.imu_tokenizer.tokenize(imu_data)
        
        # Create timestamps for motion tokens (every 0.1 seconds for 10Hz motion tokens)
        motion_timestamps = np.arange(0, len(motion_tokens) * 0.1, 0.1)
        
        # Tokenize audio if available
        audio_tokens = []
        audio_timestamps = []
        
        if audio_path and Path(audio_path).exists():
            audio_tokens = self.audio_tokenizer.tokenize_audio_file(audio_path)
            # Audio tokens at 50Hz
            audio_timestamps = np.arange(0, len(audio_tokens) * 0.02, 0.02)
        
        # Interleave tokens based on timestamps
        all_tokens = []
        all_types = []
        all_timestamps = []
        
        audio_idx = 0
        motion_idx = 0
        
        while audio_idx < len(audio_tokens) or motion_idx < len(motion_tokens):
            audio_time = audio_timestamps[audio_idx] if audio_idx < len(audio_tokens) else float('inf')
            motion_time = motion_timestamps[motion_idx] if motion_idx < len(motion_tokens) else float('inf')
            
            if audio_time <= motion_time:
                all_tokens.append(audio_tokens[audio_idx])
                all_types.append('audio')
                all_timestamps.append(audio_time)
                audio_idx += 1
            else:
                all_tokens.append(motion_tokens[motion_idx])
                all_types.append('motion')
                all_timestamps.append(motion_time)
                motion_idx += 1
                
        return TokenizedSequence(
            tokens=all_tokens,
            token_types=all_types,
            timestamps=all_timestamps,
            metadata=metadata
        )
    
    def save_tokenizer(self, path: str):
        """Save trained tokenizer"""
        tokenizer_data = {
            'imu_scaler': self.imu_tokenizer.scaler,
            'imu_model': self.imu_tokenizer.model,
            'imu_config': {
                'codebook_size': self.imu_tokenizer.codebook_size,
                'embedding_dim': self.imu_tokenizer.embedding_dim,
                'window_size': self.imu_tokenizer.window_size
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(tokenizer_data, f)
            
    def load_tokenizer(self, path: str):
        """Load trained tokenizer"""
        with open(path, 'rb') as f:
            tokenizer_data = pickle.load(f)
            
        self.imu_tokenizer.scaler = tokenizer_data['imu_scaler']
        self.imu_tokenizer.model = tokenizer_data['imu_model']
        self.imu_tokenizer.is_trained = True
        
        config = tokenizer_data['imu_config']
        self.imu_tokenizer.codebook_size = config['codebook_size']
        self.imu_tokenizer.embedding_dim = config['embedding_dim']
        self.imu_tokenizer.window_size = config['window_size']

def collect_training_sessions(downloads_dir: str) -> List[str]:
    """Collect all session directories for training"""
    downloads_path = Path(downloads_dir)
    session_dirs = []
    
    for item in downloads_path.iterdir():
        if item.is_dir() and item.name.startswith('session_'):
            session_dirs.append(str(item))
            
    return session_dirs

def main():
    """Main training and tokenization pipeline"""
    print("=== Project Chimera v2: Multi-Modal Tokenizer ===")
    
    # Collect training sessions
    downloads_dir = "/Users/jonathanschwartz/Downloads"
    session_paths = collect_training_sessions(downloads_dir)
    
    print(f"Found {len(session_paths)} training sessions:")
    for path in session_paths:
        print(f"  - {Path(path).name}")
    
    # Initialize tokenizer
    tokenizer = MultiModalTokenizer()
    
    # Train IMU tokenizer
    print("\nTraining IMU tokenizer...")
    tokenizer.train_imu_tokenizer(session_paths)
    
    # Save trained tokenizer
    tokenizer_path = "/Users/jonathanschwartz/Documents/Social_Content/QuantumLeapValidator/trained_tokenizer.pkl"
    tokenizer.save_tokenizer(tokenizer_path)
    print(f"Tokenizer saved to: {tokenizer_path}")
    
    # Tokenize a sample session
    if session_paths:
        sample_session = session_paths[0]
        print(f"\nTokenizing sample session: {Path(sample_session).name}")
        
        tokenized = tokenizer.tokenize_session(sample_session)
        
        print(f"Generated {len(tokenized.tokens)} tokens:")
        print(f"  - Audio tokens: {tokenized.token_types.count('audio')}")
        print(f"  - Motion tokens: {tokenized.token_types.count('motion')}")
        print(f"  - Duration: {tokenized.timestamps[-1]:.2f} seconds")
        
        # Show sample token sequence
        print("\nSample token sequence (first 20 tokens):")
        for i in range(min(20, len(tokenized.tokens))):
            token_type = tokenized.token_types[i]
            token_val = tokenized.tokens[i]
            timestamp = tokenized.timestamps[i]
            print(f"  {timestamp:6.2f}s: {token_type:6} token {token_val}")
    
    print("\n=== Tokenization Pipeline Complete ===")

if __name__ == "__main__":
    main()
