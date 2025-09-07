"""
Audio Tokenization Pipeline for Chimera v2 Unified Perception
Implements EnCodec-style audio tokenization for multimodal transformer
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
from sklearn.cluster import KMeans
import pickle
import os
from typing import List, Tuple, Optional, Dict, Any

class AudioTokenizer:
    """
    Audio tokenization using MFCC features and K-means clustering
    Designed to complement IMU tokenization in unified perception system
    """
    
    def __init__(self, 
                 n_mfcc: int = 13,
                 n_clusters: int = 256,
                 sample_rate: int = 16000,
                 hop_length: int = 512,
                 n_fft: int = 2048):
        """
        Initialize audio tokenizer
        
        Args:
            n_mfcc: Number of MFCC coefficients
            n_clusters: Number of audio tokens (vocabulary size)
            sample_rate: Audio sample rate
            hop_length: STFT hop length
            n_fft: FFT window size
        """
        self.n_mfcc = n_mfcc
        self.n_clusters = n_clusters
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        
        # K-means clusterer for audio tokens
        self.clusterer = None
        self.is_fitted = False
        
        # Audio preprocessing parameters
        self.frame_length = 0.025  # 25ms frames
        self.frame_shift = 0.010   # 10ms shift
        
        print(f"🎵 AudioTokenizer initialized:")
        print(f"   - MFCC coefficients: {n_mfcc}")
        print(f"   - Token vocabulary: {n_clusters}")
        print(f"   - Sample rate: {sample_rate} Hz")
    
    def extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio signal
        
        Args:
            audio: Raw audio signal
            
        Returns:
            MFCC feature matrix (n_frames, n_mfcc)
        """
        try:
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            
            # Transpose to (time, features)
            mfcc = mfcc.T
            
            # Add delta and delta-delta features
            delta = librosa.feature.delta(mfcc.T).T
            delta2 = librosa.feature.delta(mfcc.T, order=2).T
            
            # Concatenate features
            features = np.concatenate([mfcc, delta, delta2], axis=1)
            
            return features
            
        except Exception as e:
            print(f"❌ MFCC extraction failed: {e}")
            return np.zeros((1, self.n_mfcc * 3))
    
    def fit(self, audio_files: List[str]) -> None:
        """
        Fit K-means clusterer on audio data
        
        Args:
            audio_files: List of audio file paths for training
        """
        print(f"🎵 Training audio tokenizer on {len(audio_files)} files...")
        
        all_features = []
        
        for i, audio_file in enumerate(audio_files):
            try:
                # Load audio
                audio, sr = librosa.load(audio_file, sr=self.sample_rate)
                
                # Extract features
                features = self.extract_mfcc_features(audio)
                all_features.append(features)
                
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(audio_files)} files")
                    
            except Exception as e:
                print(f"⚠️ Failed to process {audio_file}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid audio features extracted")
        
        # Concatenate all features
        features_matrix = np.vstack(all_features)
        print(f"   Total feature vectors: {features_matrix.shape[0]}")
        print(f"   Feature dimension: {features_matrix.shape[1]}")
        
        # Ensure float64 for sklearn compatibility
        features_matrix = features_matrix.astype(np.float64)
        
        # Fit K-means clusterer
        print(f"   Fitting K-means with {self.n_clusters} clusters...")
        self.clusterer = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        self.clusterer.fit(features_matrix)
        self.is_fitted = True
        
        print(f"✅ Audio tokenizer training complete")
        print(f"   Inertia: {self.clusterer.inertia_:.2f}")
    
    def tokenize(self, audio: np.ndarray) -> List[int]:
        """
        Tokenize audio signal into discrete tokens
        
        Args:
            audio: Raw audio signal
            
        Returns:
            List of audio token IDs
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        
        # Extract MFCC features
        features = self.extract_mfcc_features(audio)
        
        # Predict cluster assignments (ensure float64 for sklearn compatibility)
        features = features.astype(np.float64)
        tokens = self.clusterer.predict(features)
        
        return tokens.tolist()
    
    def tokenize_file(self, audio_file: str) -> List[int]:
        """
        Tokenize audio file
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            List of audio token IDs
        """
        # Load audio
        audio, sr = librosa.load(audio_file, sr=self.sample_rate)
        
        # Tokenize
        return self.tokenize(audio)
    
    def save(self, filepath: str) -> None:
        """Save trained tokenizer"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted tokenizer")
        
        tokenizer_data = {
            'clusterer': self.clusterer,
            'n_mfcc': self.n_mfcc,
            'n_clusters': self.n_clusters,
            'sample_rate': self.sample_rate,
            'hop_length': self.hop_length,
            'n_fft': self.n_fft,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        
        print(f"✅ Audio tokenizer saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load trained tokenizer"""
        with open(filepath, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        self.clusterer = tokenizer_data['clusterer']
        self.n_mfcc = tokenizer_data['n_mfcc']
        self.n_clusters = tokenizer_data['n_clusters']
        self.sample_rate = tokenizer_data['sample_rate']
        self.hop_length = tokenizer_data['hop_length']
        self.n_fft = tokenizer_data['n_fft']
        self.is_fitted = tokenizer_data['is_fitted']
        
        print(f"✅ Audio tokenizer loaded from {filepath}")

class MultiModalTokenizer:
    """
    Unified tokenizer for both audio and IMU data
    Combines audio and IMU tokens into interleaved sequences
    """
    
    def __init__(self, 
                 imu_tokenizer_path: str,
                 audio_tokenizer_path: Optional[str] = None):
        """
        Initialize multimodal tokenizer
        
        Args:
            imu_tokenizer_path: Path to trained IMU tokenizer
            audio_tokenizer_path: Path to trained audio tokenizer (optional)
        """
        # Load IMU tokenizer
        from tokenization_pipeline import SensorTokenizer
        self.imu_tokenizer = SensorTokenizer()
        self.imu_tokenizer.load(imu_tokenizer_path)
        
        # Load audio tokenizer if provided
        self.audio_tokenizer = None
        if audio_tokenizer_path and os.path.exists(audio_tokenizer_path):
            self.audio_tokenizer = AudioTokenizer()
            self.audio_tokenizer.load(audio_tokenizer_path)
        
        # Token offset to distinguish modalities
        self.imu_token_offset = 0
        self.audio_token_offset = 1000  # Audio tokens start at 1000
        
        print(f"🔄 MultiModalTokenizer initialized:")
        print(f"   - IMU tokenizer: ✅ Loaded")
        print(f"   - Audio tokenizer: {'✅ Loaded' if self.audio_tokenizer else '❌ Not available'}")
    
    def tokenize_multimodal(self, 
                           imu_data: np.ndarray,
                           audio_data: Optional[np.ndarray] = None,
                           interleave_ratio: float = 0.1) -> List[int]:
        """
        Tokenize multimodal data with interleaving
        
        Args:
            imu_data: IMU sensor data
            audio_data: Audio data (optional)
            interleave_ratio: Ratio of audio to IMU tokens
            
        Returns:
            Interleaved token sequence
        """
        # Tokenize IMU data
        imu_tokens = self.imu_tokenizer.tokenize_sample(imu_data)
        imu_tokens = [t + self.imu_token_offset for t in imu_tokens]
        
        # Tokenize audio data if available
        audio_tokens = []
        if audio_data is not None and self.audio_tokenizer is not None:
            audio_tokens = self.audio_tokenizer.tokenize(audio_data)
            audio_tokens = [t + self.audio_token_offset for t in audio_tokens]
        
        # Interleave tokens
        if not audio_tokens:
            return imu_tokens
        
        # Calculate interleaving pattern
        total_tokens = len(imu_tokens) + len(audio_tokens)
        audio_positions = set(np.linspace(0, total_tokens-1, 
                                        int(total_tokens * interleave_ratio), 
                                        dtype=int))
        
        # Create interleaved sequence
        interleaved = []
        imu_idx = 0
        audio_idx = 0
        
        for pos in range(total_tokens):
            if pos in audio_positions and audio_idx < len(audio_tokens):
                interleaved.append(audio_tokens[audio_idx])
                audio_idx += 1
            elif imu_idx < len(imu_tokens):
                interleaved.append(imu_tokens[imu_idx])
                imu_idx += 1
        
        return interleaved
    
    def get_vocab_size(self) -> int:
        """Get total vocabulary size for multimodal tokens"""
        base_size = self.imu_tokenizer.n_clusters
        if self.audio_tokenizer:
            base_size += self.audio_tokenizer.n_clusters
        return base_size + 100  # Add buffer for special tokens

def train_audio_tokenizer_demo():
    """
    Demo function to train audio tokenizer
    (In practice, would use real audio data from workout sessions)
    """
    print("🎵 Audio Tokenizer Training Demo")
    
    # Create synthetic audio data for demo
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    
    # Generate demo audio files
    demo_audio_files = []
    for i in range(10):
        # Create synthetic audio (mix of tones and noise)
        t = np.linspace(0, duration, int(sample_rate * duration))
        freq = 440 + i * 50  # Different frequencies
        audio = 0.5 * np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
        
        # Save to temporary file
        temp_file = f"/tmp/demo_audio_{i}.wav"
        import soundfile as sf
        sf.write(temp_file, audio, sample_rate)
        demo_audio_files.append(temp_file)
    
    # Train tokenizer
    tokenizer = AudioTokenizer(n_clusters=128)
    tokenizer.fit(demo_audio_files)
    
    # Save tokenizer
    tokenizer.save("trained_audio_tokenizer.pkl")
    
    # Test tokenization
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    tokens = tokenizer.tokenize(test_audio)
    
    print(f"✅ Demo complete:")
    print(f"   - Tokenized {len(test_audio)} audio samples")
    print(f"   - Generated {len(tokens)} tokens")
    print(f"   - Token range: {min(tokens)} - {max(tokens)}")
    
    # Cleanup
    for file in demo_audio_files:
        os.remove(file)

if __name__ == "__main__":
    # Run demo
    train_audio_tokenizer_demo()
    
    # Test multimodal tokenizer
    print("\n🔄 Testing MultiModal Tokenizer...")
    
    try:
        multimodal = MultiModalTokenizer(
            imu_tokenizer_path="trained_tokenizer.pkl",
            audio_tokenizer_path="trained_audio_tokenizer.pkl"
        )
        
        # Create dummy data
        imu_data = np.random.randn(100, 6)  # 100 samples, 6 features
        audio_data = np.random.randn(16000)  # 1 second of audio
        
        # Tokenize
        tokens = multimodal.tokenize_multimodal(imu_data, audio_data)
        
        print(f"✅ Multimodal tokenization successful:")
        print(f"   - Generated {len(tokens)} tokens")
        print(f"   - Vocabulary size: {multimodal.get_vocab_size()}")
        
    except Exception as e:
        print(f"⚠️ Multimodal test failed: {e}")
        print("   (This is expected if IMU tokenizer not available)")
