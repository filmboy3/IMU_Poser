#!/usr/bin/env python3
"""
Real-Time Inference Pipeline for Project Chimera v2
Optimized perception model inference with <200ms latency for live coaching.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import threading
import queue
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import deque
import json
from pathlib import Path

from tokenization_pipeline import MultiModalTokenizer
from perception_transformer import UnifiedPerceptionTransformer, TransformerConfig
from coaching_llm import GenerativeCoachingLLM, CoachingResponse

@dataclass
class InferenceConfig:
    """Configuration for real-time inference"""
    model_path: str
    tokenizer_path: str
    sequence_length: int = 32  # Shorter for real-time
    batch_size: int = 1
    max_latency_ms: float = 200.0
    imu_sample_rate: float = 10.0  # Hz
    audio_sample_rate: float = 16000  # Hz
    buffer_size: int = 64
    quantization: bool = True  # Enable model quantization
    device: str = 'mps'

@dataclass
class SensorData:
    """Real-time sensor input"""
    timestamp: float
    imu_data: Optional[Dict[str, float]] = None  # x, y, z acceleration
    audio_data: Optional[np.ndarray] = None
    session_id: str = ""

@dataclass
class InferenceResult:
    """Real-time inference output"""
    timestamp: float
    predicted_tokens: List[int]
    confidence: float
    rep_detected: bool
    motion_quality: float
    coaching_trigger: Optional[str] = None
    latency_ms: float = 0.0

class OptimizedPerceptionModel:
    """Optimized version of perception transformer for real-time inference"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.tokenizer = None
        self.sequence_buffer = deque(maxlen=config.sequence_length)
        self.last_inference_time = 0
        
    def load_model(self):
        """Load and optimize model for inference"""
        print(f"Loading optimized model from {self.config.model_path}")
        
        # Load tokenizer
        self.tokenizer = MultiModalTokenizer()
        self.tokenizer.load_tokenizer(self.config.tokenizer_path)
        
        # Load model
        checkpoint = torch.load(self.config.model_path, map_location=self.device, weights_only=False)
        model_config = checkpoint['config']
        
        self.model = UnifiedPerceptionTransformer(model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Apply optimizations (skip quantization on MPS due to compatibility issues)
        if self.config.quantization and hasattr(torch.quantization, 'quantize_dynamic') and self.device.type != 'mps':
            print("Applying dynamic quantization...")
            self.model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )
            
        # Compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            print("Compiling model for optimized inference...")
            self.model = torch.compile(self.model, mode='reduce-overhead')
            
        print("Model optimization complete")
        
    def preprocess_sensor_data(self, sensor_data: SensorData) -> Optional[int]:
        """Convert sensor data to token"""
        if not sensor_data.imu_data:
            return None
            
        # Convert to format expected by tokenizer (matches training data format)
        imu_sample = {
            'acceleration': {
                'x': sensor_data.imu_data.get('x', 0.0),
                'y': sensor_data.imu_data.get('y', 0.0), 
                'z': sensor_data.imu_data.get('z', 0.0)
            },
            'rotationRate': {
                'x': 0.0,  # Simulate rotation data
                'y': 0.0,
                'z': 0.0
            },
            'timestamp': sensor_data.timestamp
        }
        
        # For real-time processing, we need to handle single samples
        # Extract features directly and predict without windowing
        try:
            features = self.tokenizer.imu_tokenizer.extract_features([imu_sample])
            if len(features) == 0:
                return 1025  # Default stable token
                
            # For single sample, use the feature vector directly as a "window"
            feature_vector = features[0].reshape(1, -1)  # Shape: (1, 6)
            
            # Normalize using the trained scaler
            feature_norm = self.tokenizer.imu_tokenizer.scaler.transform(feature_vector)
            
            # Predict motion token
            motion_token = self.tokenizer.imu_tokenizer.model.predict(feature_norm)[0]
            return int(motion_token) + 1024  # Add offset for motion tokens
            
        except Exception as e:
            # Fallback to stable token if anything fails
            return 1025
        
    def inference(self, sensor_data: SensorData) -> Optional[InferenceResult]:
        """Perform real-time inference on sensor data"""
        start_time = time.time()
        
        # Preprocess input
        token = self.preprocess_sensor_data(sensor_data)
        if token is None:
            return None
            
        # Add to sequence buffer
        self.sequence_buffer.append(token)
        
        # For real-time processing, use shorter sequences or pad if needed
        min_length = min(self.config.sequence_length, 4)  # Allow shorter sequences
        if len(self.sequence_buffer) < min_length:
            return None
            
        # Prepare input tensor
        input_sequence = list(self.sequence_buffer)
        input_tensor = torch.tensor([input_sequence], dtype=torch.long).to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            
            # Get predictions
            last_logits = logits[0, -1, :]  # Last position predictions
            probabilities = torch.softmax(last_logits, dim=-1)
            
            # Get top prediction and confidence
            top_prob, top_token = torch.max(probabilities, dim=0)
            predicted_token = top_token.item()
            confidence = top_prob.item()
            
        # Analyze sequence for patterns
        rep_detected = self._detect_rep_pattern(input_sequence)
        motion_quality = self._assess_motion_quality(input_sequence)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Create result
        result = InferenceResult(
            timestamp=sensor_data.timestamp,
            predicted_tokens=[predicted_token],
            confidence=confidence,
            rep_detected=rep_detected,
            motion_quality=motion_quality,
            latency_ms=latency_ms
        )
        
        self.last_inference_time = time.time()
        return result
        
    def _detect_rep_pattern(self, sequence: List[int]) -> bool:
        """Detect rep completion from token sequence"""
        if len(sequence) < 8:
            return False
            
        # Look for variance pattern indicating rep completion
        recent_tokens = sequence[-8:]
        first_half_var = np.var(recent_tokens[:4])
        second_half_var = np.var(recent_tokens[4:])
        
        # Rep detected if high->low variance transition
        return first_half_var > second_half_var * 2 and second_half_var < 10
        
    def _assess_motion_quality(self, sequence: List[int]) -> float:
        """Assess motion quality from token sequence"""
        if len(sequence) < 5:
            return 0.5
            
        # Calculate token variance as quality metric
        variance = np.var(sequence[-10:])  # Recent motion
        
        # Normalize to 0-1 range (higher variance = more dynamic = better quality)
        # But too high variance = chaotic motion = poor quality
        if variance == 0:
            return 0.2  # Too static
        elif variance > 100:
            return 0.3  # Too chaotic
        else:
            # Optimal variance around 20-50
            optimal_variance = 35
            quality = 1.0 - abs(variance - optimal_variance) / optimal_variance
            return max(0.1, min(1.0, quality))

class RealTimeInferencePipeline:
    """Main real-time inference pipeline orchestrating all components"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.perception_model = OptimizedPerceptionModel(config)
        self.coaching_llm = GenerativeCoachingLLM("encouraging")
        
        # Threading components
        self.input_queue = queue.Queue(maxsize=config.buffer_size)
        self.output_queue = queue.Queue(maxsize=config.buffer_size)
        self.inference_thread = None
        self.coaching_thread = None
        self.running = False
        
        # State tracking
        self.current_rep = 0
        self.target_reps = 10
        self.session_active = False
        self.performance_stats = {
            'total_inferences': 0,
            'avg_latency_ms': 0.0,
            'max_latency_ms': 0.0,
            'dropped_frames': 0
        }
        
    def initialize(self):
        """Initialize all pipeline components"""
        print("Initializing real-time inference pipeline...")
        
        # Load models
        self.perception_model.load_model()
        
        print("Pipeline initialization complete")
        
    def start_session(self, target_reps: int = 10) -> CoachingResponse:
        """Start a new workout session"""
        self.session_active = True
        self.current_rep = 0
        self.target_reps = target_reps
        
        # Start coaching session
        coaching_response = self.coaching_llm.start_session()
        
        # Start processing threads
        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.coaching_thread = threading.Thread(target=self._coaching_worker, daemon=True)
        
        self.inference_thread.start()
        self.coaching_thread.start()
        
        print(f"Session started - Target: {target_reps} reps")
        return coaching_response
        
    def process_sensor_data(self, sensor_data: SensorData) -> bool:
        """Add sensor data to processing queue"""
        if not self.session_active:
            return False
            
        try:
            self.input_queue.put_nowait(sensor_data)
            return True
        except queue.Full:
            self.performance_stats['dropped_frames'] += 1
            return False
            
    def get_coaching_response(self) -> Optional[CoachingResponse]:
        """Get latest coaching response if available"""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None
            
    def end_session(self) -> CoachingResponse:
        """End current workout session"""
        self.session_active = False
        self.running = False
        
        # Wait for threads to finish
        if self.inference_thread:
            self.inference_thread.join(timeout=1.0)
        if self.coaching_thread:
            self.coaching_thread.join(timeout=1.0)
            
        # Generate final coaching
        final_coaching = self.coaching_llm.end_session(self.current_rep)
        
        print(f"Session ended - Completed: {self.current_rep} reps")
        return final_coaching
        
    def _inference_worker(self):
        """Background thread for running inference"""
        print("Inference worker thread started")
        while self.running:
            try:
                # Get sensor data with timeout
                sensor_data = self.input_queue.get(timeout=0.1)
                print(f"Processing sensor data: {sensor_data.timestamp}")
                
                # Run inference
                result = self.perception_model.inference(sensor_data)
                
                if result:
                    print(f"Inference result: latency={result.latency_ms:.2f}ms, confidence={result.confidence:.3f}")
                    # Update performance stats
                    self._update_performance_stats(result)
                    
                    # Check for rep completion
                    if result.rep_detected:
                        self.current_rep += 1
                        print(f"Rep detected! Total: {self.current_rep}")
                        
                    # Generate coaching if needed
                    self._process_coaching_triggers(result)
                else:
                    print("No inference result (sequence building)")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Inference error: {e}")
                continue
        print("Inference worker thread stopped")

    def _coaching_worker(self):
        """Background thread for coaching generation"""
        while self.running:
            try:
                time.sleep(0.1)  # Check every 100ms
                
                # Get recent motion data for coaching context
                if len(self.perception_model.sequence_buffer) > 0:
                    recent_tokens = list(self.perception_model.sequence_buffer)[-10:]
                    timestamps = [time.time() - i * 0.1 for i in range(len(recent_tokens))]
                    
                    # Generate coaching
                    coaching_response = self.coaching_llm.process_perception_output(
                        recent_tokens, timestamps, self.current_rep, self.target_reps
                    )
                    
                    if coaching_response:
                        try:
                            self.output_queue.put_nowait(coaching_response)
                        except queue.Full:
                            pass  # Skip if queue full
                            
            except Exception as e:
                print(f"Coaching error: {e}")
                
    def _process_coaching_triggers(self, result: InferenceResult):
        """Process inference result for coaching triggers"""
        # This is handled by the coaching worker thread
        pass
        
    def _update_performance_stats(self, result: InferenceResult):
        """Update performance statistics"""
        stats = self.performance_stats
        stats['total_inferences'] += 1
        
        # Update latency stats
        if stats['total_inferences'] == 1:
            stats['avg_latency_ms'] = result.latency_ms
        else:
            # Running average
            stats['avg_latency_ms'] = (
                (stats['avg_latency_ms'] * (stats['total_inferences'] - 1) + result.latency_ms) 
                / stats['total_inferences']
            )
            
        stats['max_latency_ms'] = max(stats['max_latency_ms'], result.latency_ms)
        
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        return self.performance_stats.copy()

def simulate_imu_stream(duration_seconds: float = 10.0, sample_rate: float = 10.0):
    """Simulate realistic IMU data stream for testing"""
    print(f"Simulating {duration_seconds}s of IMU data at {sample_rate}Hz")
    
    samples_per_second = int(sample_rate)
    total_samples = int(duration_seconds * samples_per_second)
    
    for i in range(total_samples):
        timestamp = time.time()
        
        # Simulate exercise motion pattern
        t = i / samples_per_second
        
        # Base motion with exercise pattern
        if i % 20 < 10:  # Active phase
            x = np.sin(t * 2) * 2.0 + np.random.normal(0, 0.1)
            y = np.cos(t * 2) * 1.5 + np.random.normal(0, 0.1) 
            z = 9.8 + np.sin(t * 4) * 0.5 + np.random.normal(0, 0.05)
        else:  # Rest phase
            x = np.random.normal(0, 0.05)
            y = np.random.normal(0, 0.05)
            z = 9.8 + np.random.normal(0, 0.02)
            
        imu_data = {'x': x, 'y': y, 'z': z}
        
        sensor_data = SensorData(
            timestamp=timestamp,
            imu_data=imu_data,
            session_id="test_session"
        )
        
        yield sensor_data
        time.sleep(1.0 / sample_rate)  # Maintain sample rate

def main():
    """Demo real-time inference pipeline"""
    print("=== Real-Time Inference Pipeline Demo ===")
    
    # Configuration
    config = InferenceConfig(
        model_path="/Users/jonathanschwartz/Documents/Social_Content/QuantumLeapValidator/perception_transformer.pt",
        tokenizer_path="/Users/jonathanschwartz/Documents/Social_Content/QuantumLeapValidator/trained_tokenizer.pkl",
        sequence_length=16,  # Shorter for real-time
        max_latency_ms=200.0,
        quantization=False  # Disable for demo
    )
    
    # Initialize pipeline
    pipeline = RealTimeInferencePipeline(config)
    pipeline.initialize()
    
    # Start session
    start_coaching = pipeline.start_session(target_reps=5)
    print(f"Coaching: {start_coaching.message}")
    
    # Simulate real-time data stream
    data_stream = simulate_imu_stream(duration_seconds=8.0, sample_rate=5.0)
    
    coaching_count = 0
    for sensor_data in data_stream:
        # Process sensor data
        success = pipeline.process_sensor_data(sensor_data)
        
        if not success:
            print("Warning: Dropped frame due to full buffer")
            
        # Check for coaching responses
        coaching = pipeline.get_coaching_response()
        if coaching:
            coaching_count += 1
            print(f"Coaching #{coaching_count}: [{coaching.trigger.value}] {coaching.message}")
            
        # Small delay to prevent overwhelming output
        time.sleep(0.1)
        
    # End session
    final_coaching = pipeline.end_session()
    print(f"Final: {final_coaching.message}")
    
    # Print performance stats
    stats = pipeline.get_performance_stats()
    print(f"\nPerformance Stats:")
    print(f"  Total inferences: {stats['total_inferences']}")
    print(f"  Average latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"  Max latency: {stats['max_latency_ms']:.2f}ms")
    print(f"  Dropped frames: {stats['dropped_frames']}")
    
    # Verify latency requirement
    if stats['avg_latency_ms'] < config.max_latency_ms:
        print(f"✅ Latency requirement met ({stats['avg_latency_ms']:.2f}ms < {config.max_latency_ms}ms)")
    else:
        print(f"❌ Latency requirement failed ({stats['avg_latency_ms']:.2f}ms >= {config.max_latency_ms}ms)")

if __name__ == "__main__":
    main()
