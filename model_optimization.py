#!/usr/bin/env python3
"""
Model Optimization and Quantization for Project Chimera v2
Optimizes perception transformer and tokenizers for mobile deployment.
"""

import torch
import torch.nn as nn
import torch.quantization as quant
import pickle
import numpy as np
from pathlib import Path
import time
from typing import Dict, Any, Tuple
import json

from perception_transformer import UnifiedPerceptionTransformer, TransformerConfig
from audio_tokenization_pipeline import AudioTokenizer, MultiModalTokenizer

class ModelOptimizer:
    """Optimizes models for mobile deployment"""
    
    def __init__(self):
        self.device = torch.device('cpu')  # Force CPU for mobile compatibility
        
    def quantize_perception_model(self, model_path: str, output_path: str) -> Dict[str, Any]:
        """Apply dynamic quantization to perception transformer"""
        print("🔧 Loading perception transformer for quantization...")
        
        # Load model
        config = TransformerConfig()
        model = UnifiedPerceptionTransformer(config)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"⚠️ Model loading failed, using random weights: {e}")
        
        model.eval()
        
        # Apply dynamic quantization
        print("🔧 Applying dynamic quantization...")
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.MultiheadAttention},  # Quantize linear layers and attention
            dtype=torch.qint8
        )
        
        # Save quantized model
        torch.save(quantized_model.state_dict(), output_path)
        
        # Calculate size reduction
        original_size = self._get_model_size(model)
        quantized_size = self._get_model_size(quantized_model)
        size_reduction = (original_size - quantized_size) / original_size * 100
        
        # Benchmark inference speed
        speed_results = self._benchmark_model_speed(model, quantized_model)
        
        results = {
            "original_size_mb": original_size / (1024 * 1024),
            "quantized_size_mb": quantized_size / (1024 * 1024),
            "size_reduction_percent": size_reduction,
            "speed_improvement": speed_results,
            "output_path": output_path
        }
        
        print(f"✅ Model quantized: {size_reduction:.1f}% size reduction")
        print(f"📊 Speed improvement: {speed_results['speedup']:.2f}x")
        
        return results
    
    def optimize_tokenizers(self, tokenizer_path: str, audio_tokenizer_path: str, 
                          output_dir: str) -> Dict[str, Any]:
        """Optimize tokenizers for mobile deployment"""
        print("🔧 Optimizing tokenizers for mobile deployment...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = {}
        
        # Optimize main tokenizer
        try:
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            
            optimized_tokenizer = self._compress_tokenizer(tokenizer)
            
            optimized_path = output_dir / "optimized_tokenizer.pkl"
            with open(optimized_path, 'wb') as f:
                pickle.dump(optimized_tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            original_size = Path(tokenizer_path).stat().st_size
            optimized_size = optimized_path.stat().st_size
            
            results["main_tokenizer"] = {
                "original_size_kb": original_size / 1024,
                "optimized_size_kb": optimized_size / 1024,
                "size_reduction_percent": (original_size - optimized_size) / original_size * 100,
                "output_path": str(optimized_path)
            }
            
        except Exception as e:
            print(f"⚠️ Main tokenizer optimization failed: {e}")
            results["main_tokenizer"] = {"error": str(e)}
        
        # Optimize audio tokenizer
        try:
            with open(audio_tokenizer_path, 'rb') as f:
                audio_tokenizer = pickle.load(f)
            
            optimized_audio = self._compress_audio_tokenizer(audio_tokenizer)
            
            audio_optimized_path = output_dir / "optimized_audio_tokenizer.pkl"
            with open(audio_optimized_path, 'wb') as f:
                pickle.dump(optimized_audio, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            original_size = Path(audio_tokenizer_path).stat().st_size
            optimized_size = audio_optimized_path.stat().st_size
            
            results["audio_tokenizer"] = {
                "original_size_kb": original_size / 1024,
                "optimized_size_kb": optimized_size / 1024,
                "size_reduction_percent": (original_size - optimized_size) / original_size * 100,
                "output_path": str(audio_optimized_path)
            }
            
        except Exception as e:
            print(f"⚠️ Audio tokenizer optimization failed: {e}")
            results["audio_tokenizer"] = {"error": str(e)}
        
        print("✅ Tokenizer optimization complete")
        return results
    
    def create_mobile_config(self, output_path: str) -> Dict[str, Any]:
        """Create optimized configuration for mobile deployment"""
        mobile_config = {
            "model": {
                "use_quantized": True,
                "max_sequence_length": 256,  # Reduced from 512
                "batch_size": 1,  # Single inference
                "num_attention_heads": 4,  # Reduced from 8
                "hidden_dim": 256,  # Reduced from 512
                "num_layers": 4,  # Reduced from 6
                "dropout": 0.0,  # Disable dropout for inference
            },
            "tokenizer": {
                "vocab_size": 1024,  # Reduced vocabulary
                "max_audio_tokens": 64,  # Reduced audio context
                "max_imu_tokens": 128,  # Reduced IMU context
            },
            "inference": {
                "enable_caching": True,
                "use_torch_jit": False,  # Disable JIT for iOS compatibility
                "threading": {
                    "num_threads": 2,  # Conservative for mobile
                    "enable_parallel": False
                }
            },
            "memory": {
                "max_buffer_size": 1024,  # Reduced memory footprint
                "enable_gradient_checkpointing": False,
                "clear_cache_interval": 100  # Clear cache more frequently
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(mobile_config, f, indent=2)
        
        print(f"✅ Mobile configuration saved to {output_path}")
        return mobile_config
    
    def _get_model_size(self, model: nn.Module) -> int:
        """Calculate model size in bytes"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return param_size + buffer_size
    
    def _benchmark_model_speed(self, original_model: nn.Module, 
                             quantized_model: nn.Module, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark inference speed comparison"""
        # Create dummy input
        batch_size = 1
        seq_length = 128
        hidden_dim = original_model.config.hidden_dim if hasattr(original_model, 'config') else 256
        
        dummy_input = torch.randn(batch_size, seq_length, hidden_dim)
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = original_model(dummy_input)
                _ = quantized_model(dummy_input)
        
        # Benchmark original model
        original_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = original_model(dummy_input)
                original_times.append(time.time() - start_time)
        
        # Benchmark quantized model
        quantized_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = quantized_model(dummy_input)
                quantized_times.append(time.time() - start_time)
        
        original_avg = np.mean(original_times)
        quantized_avg = np.mean(quantized_times)
        speedup = original_avg / quantized_avg
        
        return {
            "original_avg_ms": original_avg * 1000,
            "quantized_avg_ms": quantized_avg * 1000,
            "speedup": speedup
        }
    
    def _compress_tokenizer(self, tokenizer) -> Any:
        """Compress tokenizer by removing unnecessary data"""
        # Create a minimal version keeping only essential components
        if hasattr(tokenizer, 'vocab_size'):
            # Keep only essential attributes
            compressed = type(tokenizer)(
                vocab_size=getattr(tokenizer, 'vocab_size', 1024),
                sequence_length=getattr(tokenizer, 'sequence_length', 256)
            )
            
            # Copy essential methods and data
            if hasattr(tokenizer, 'encode'):
                compressed.encode = tokenizer.encode
            if hasattr(tokenizer, 'decode'):
                compressed.decode = tokenizer.decode
            if hasattr(tokenizer, 'token_to_id'):
                compressed.token_to_id = tokenizer.token_to_id
            if hasattr(tokenizer, 'id_to_token'):
                compressed.id_to_token = tokenizer.id_to_token
                
            return compressed
        
        return tokenizer
    
    def _compress_audio_tokenizer(self, audio_tokenizer) -> Any:
        """Compress audio tokenizer for mobile deployment"""
        if hasattr(audio_tokenizer, 'kmeans'):
            # Reduce precision of cluster centers
            if hasattr(audio_tokenizer.kmeans, 'cluster_centers_'):
                audio_tokenizer.kmeans.cluster_centers_ = audio_tokenizer.kmeans.cluster_centers_.astype(np.float32)
        
        # Remove training-specific attributes
        attrs_to_remove = ['training_data', 'fit_history', 'verbose']
        for attr in attrs_to_remove:
            if hasattr(audio_tokenizer, attr):
                delattr(audio_tokenizer, attr)
        
        return audio_tokenizer

def main():
    """Run model optimization pipeline"""
    print("=== Model Optimization for Mobile Deployment ===")
    
    optimizer = ModelOptimizer()
    
    # Paths
    model_path = "perception_model.pth"
    tokenizer_path = "trained_tokenizer.pkl"
    audio_tokenizer_path = "trained_audio_tokenizer.pkl"
    output_dir = "optimized_models"
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    results = {}
    
    # 1. Quantize perception model
    try:
        quantized_model_path = f"{output_dir}/quantized_perception_model.pth"
        model_results = optimizer.quantize_perception_model(model_path, quantized_model_path)
        results["model_quantization"] = model_results
    except Exception as e:
        print(f"⚠️ Model quantization failed: {e}")
        results["model_quantization"] = {"error": str(e)}
    
    # 2. Optimize tokenizers
    try:
        tokenizer_results = optimizer.optimize_tokenizers(
            tokenizer_path, audio_tokenizer_path, output_dir
        )
        results["tokenizer_optimization"] = tokenizer_results
    except Exception as e:
        print(f"⚠️ Tokenizer optimization failed: {e}")
        results["tokenizer_optimization"] = {"error": str(e)}
    
    # 3. Create mobile configuration
    try:
        config_path = f"{output_dir}/mobile_config.json"
        mobile_config = optimizer.create_mobile_config(config_path)
        results["mobile_config"] = {"path": config_path, "config": mobile_config}
    except Exception as e:
        print(f"⚠️ Mobile config creation failed: {e}")
        results["mobile_config"] = {"error": str(e)}
    
    # Save optimization report
    report_path = f"{output_dir}/optimization_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Optimization complete! Report saved to {report_path}")
    
    # Print summary
    print("\n📊 Optimization Summary:")
    if "model_quantization" in results and "error" not in results["model_quantization"]:
        model_res = results["model_quantization"]
        print(f"  Model: {model_res['size_reduction_percent']:.1f}% smaller, {model_res['speed_improvement']['speedup']:.2f}x faster")
    
    if "tokenizer_optimization" in results:
        tok_res = results["tokenizer_optimization"]
        if "main_tokenizer" in tok_res and "error" not in tok_res["main_tokenizer"]:
            print(f"  Main Tokenizer: {tok_res['main_tokenizer']['size_reduction_percent']:.1f}% smaller")
        if "audio_tokenizer" in tok_res and "error" not in tok_res["audio_tokenizer"]:
            print(f"  Audio Tokenizer: {tok_res['audio_tokenizer']['size_reduction_percent']:.1f}% smaller")

if __name__ == "__main__":
    main()
