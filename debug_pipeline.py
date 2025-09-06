#!/usr/bin/env python3
"""
Debug script to identify why the real-time pipeline isn't processing data
"""

import time
from realtime_inference_pipeline import RealTimeInferencePipeline, InferenceConfig, SensorData
from perception_transformer import TransformerConfig  # Import for model loading

def debug_pipeline():
    """Debug the pipeline step by step"""
    print("=== Pipeline Debug Session ===")
    
    # Create config
    config = InferenceConfig(
        model_path="/Users/jonathanschwartz/Documents/Social_Content/QuantumLeapValidator/perception_transformer.pt",
        tokenizer_path="/Users/jonathanschwartz/Documents/Social_Content/QuantumLeapValidator/trained_tokenizer.pkl",
        sequence_length=8,
        quantization=False
    )
    
    # Test 1: Initialize pipeline
    print("\n1. Testing pipeline initialization...")
    pipeline = RealTimeInferencePipeline(config)
    pipeline.initialize()
    print("✅ Pipeline initialized")
    
    # Test 2: Start session
    print("\n2. Testing session start...")
    start_response = pipeline.start_session(target_reps=3)
    print(f"✅ Session started: {start_response.message}")
    print(f"   Session active: {pipeline.session_active}")
    print(f"   Threads running: {pipeline.running}")
    
    # Test 3: Test sensor data preprocessing
    print("\n3. Testing sensor data preprocessing...")
    test_sensor = SensorData(
        timestamp=time.time(),
        imu_data={'x': 1.5, 'y': 1.0, 'z': 9.8},
        session_id="debug"
    )
    
    # Test tokenization directly
    token = pipeline.perception_model.preprocess_sensor_data(test_sensor)
    print(f"✅ Tokenization result: {token}")
    
    # Test 4: Test inference directly
    print("\n4. Testing direct inference...")
    result = pipeline.perception_model.inference(test_sensor)
    if result:
        print(f"✅ Inference successful:")
        print(f"   Latency: {result.latency_ms:.2f}ms")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Rep detected: {result.rep_detected}")
    else:
        print("❌ Inference failed - no result")
    
    # Test 5: Test queue processing
    print("\n5. Testing queue processing...")
    success = pipeline.process_sensor_data(test_sensor)
    print(f"✅ Queue processing: {success}")
    print(f"   Input queue size: {pipeline.input_queue.qsize()}")
    
    # Wait for processing
    time.sleep(0.5)
    
    # Check stats
    stats = pipeline.get_performance_stats()
    print(f"\n6. Performance stats after processing:")
    print(f"   Total inferences: {stats['total_inferences']}")
    print(f"   Avg latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"   Dropped frames: {stats['dropped_frames']}")
    
    # Test 7: Send multiple data points
    print("\n7. Testing multiple data points...")
    for i in range(5):
        sensor_data = SensorData(
            timestamp=time.time(),
            imu_data={'x': i * 0.5, 'y': 1.0, 'z': 9.8},
            session_id="debug_batch"
        )
        pipeline.process_sensor_data(sensor_data)
        time.sleep(0.1)
    
    # Wait for processing
    time.sleep(1.0)
    
    # Final stats
    final_stats = pipeline.get_performance_stats()
    print(f"\n8. Final stats:")
    print(f"   Total inferences: {final_stats['total_inferences']}")
    print(f"   Avg latency: {final_stats['avg_latency_ms']:.2f}ms")
    print(f"   Dropped frames: {final_stats['dropped_frames']}")
    
    # Check for coaching responses
    coaching_count = 0
    while True:
        coaching = pipeline.get_coaching_response()
        if coaching:
            coaching_count += 1
            print(f"   Coaching #{coaching_count}: {coaching.message}")
        else:
            break
    
    print(f"   Total coaching messages: {coaching_count}")
    
    # End session
    end_response = pipeline.end_session()
    print(f"\n9. Session ended: {end_response.message}")
    
    # Diagnosis
    print(f"\n=== DIAGNOSIS ===")
    if final_stats['total_inferences'] == 0:
        print("❌ ISSUE: No inferences processed")
        print("   Possible causes:")
        print("   - Inference thread not processing queue")
        print("   - Tokenization failing silently")
        print("   - Model inference errors")
    else:
        print("✅ Pipeline processing data correctly")

if __name__ == "__main__":
    debug_pipeline()
