#!/usr/bin/env python3
"""
Test synchronous inference without threading to verify core functionality
"""

import time
from realtime_inference_pipeline import OptimizedPerceptionModel, InferenceConfig, SensorData
from perception_transformer import TransformerConfig

def test_synchronous_inference():
    """Test inference without threading complications"""
    print("=== Synchronous Inference Test ===")
    
    config = InferenceConfig(
        model_path="/Users/jonathanschwartz/Documents/Social_Content/QuantumLeapValidator/perception_transformer.pt",
        tokenizer_path="/Users/jonathanschwartz/Documents/Social_Content/QuantumLeapValidator/trained_tokenizer.pkl",
        sequence_length=4,  # Short sequence
        quantization=False
    )
    
    model = OptimizedPerceptionModel(config)
    model.load_model()
    
    # Test with multiple samples to build sequence
    print("\nBuilding sequence with multiple samples...")
    results = []
    
    for i in range(6):  # Send 6 samples
        sensor_data = SensorData(
            timestamp=time.time() + i * 0.1,
            imu_data={'x': 1.0 + i * 0.2, 'y': 1.0, 'z': 9.8},
            session_id="sync_test"
        )
        
        print(f"Sample {i+1}:")
        
        # Test preprocessing
        token = model.preprocess_sensor_data(sensor_data)
        print(f"  Token: {token}")
        
        # Test inference
        result = model.inference(sensor_data)
        if result:
            print(f"  ✅ Inference successful - Latency: {result.latency_ms:.2f}ms")
            print(f"     Confidence: {result.confidence:.3f}")
            print(f"     Rep detected: {result.rep_detected}")
            results.append(result)
        else:
            print(f"  ❌ No inference result (sequence too short)")
    
    print(f"\n=== Results ===")
    print(f"Total successful inferences: {len(results)}")
    if results:
        avg_latency = sum(r.latency_ms for r in results) / len(results)
        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"Confidence range: {min(r.confidence for r in results):.3f} - {max(r.confidence for r in results):.3f}")
        reps_detected = sum(1 for r in results if r.rep_detected)
        print(f"Reps detected: {reps_detected}")
        
        # Test passes if we get inferences with reasonable latency
        if avg_latency < 200:
            print("✅ PASS: Latency requirement met")
        else:
            print("❌ FAIL: Latency too high")
            
        if len(results) >= 2:  # Should get results after sequence builds up
            print("✅ PASS: Inference pipeline working")
        else:
            print("❌ FAIL: Insufficient inferences")
    else:
        print("❌ FAIL: No inferences completed")

if __name__ == "__main__":
    test_synchronous_inference()
