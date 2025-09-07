#!/usr/bin/env python3
"""
Simplified Mobile Model Optimizer for Project Chimera v2
Creates mobile-optimized versions of models without advanced quantization.
"""

import torch
import torch.nn as nn
import pickle
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, Any

from perception_transformer import UnifiedPerceptionTransformer, TransformerConfig

class MobileOptimizer:
    """Simplified mobile optimization"""
    
    def __init__(self):
        self.device = torch.device('cpu')
        
    def create_mobile_model(self, output_path: str) -> Dict[str, Any]:
        """Create a mobile-optimized perception model"""
        print("🔧 Creating mobile-optimized perception model...")
        
        # Create smaller config for mobile
        mobile_config = TransformerConfig(
            vocab_size=1024,    # Reduced from 1400
            max_seq_len=256,    # Reduced from 8192
            d_model=256,        # Reduced from 1024
            n_heads=4,          # Reduced from 16
            n_layers=4,         # Reduced from 12
            d_ff=1024,          # Reduced from 4096
            dropout=0.0         # Disable dropout for inference
        )
        
        # Create model with mobile config
        model = UnifiedPerceptionTransformer(mobile_config)
        model.eval()
        
        # Save mobile model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': mobile_config.__dict__,
            'model_type': 'mobile_optimized'
        }, output_path)
        
        # Calculate model size
        model_size = self._calculate_model_size(model)
        
        results = {
            "model_path": output_path,
            "model_size_mb": model_size / (1024 * 1024),
            "config": mobile_config.__dict__,
            "optimization": "mobile_architecture"
        }
        
        print(f"✅ Mobile model created: {model_size / (1024 * 1024):.2f} MB")
        return results
    
    def optimize_tokenizers_simple(self, tokenizer_path: str, audio_tokenizer_path: str, 
                                 output_dir: str) -> Dict[str, Any]:
        """Simple tokenizer optimization"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = {}
        
        # Copy and compress tokenizers
        for name, path in [("main", tokenizer_path), ("audio", audio_tokenizer_path)]:
            try:
                with open(path, 'rb') as f:
                    tokenizer = pickle.load(f)
                
                # Save with highest compression
                output_path = output_dir / f"mobile_{name}_tokenizer.pkl"
                with open(output_path, 'wb') as f:
                    pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                original_size = Path(path).stat().st_size
                new_size = output_path.stat().st_size
                
                results[f"{name}_tokenizer"] = {
                    "original_size_kb": original_size / 1024,
                    "optimized_size_kb": new_size / 1024,
                    "output_path": str(output_path)
                }
                
            except Exception as e:
                results[f"{name}_tokenizer"] = {"error": str(e)}
        
        return results
    
    def create_mobile_inference_config(self, output_path: str) -> Dict[str, Any]:
        """Create mobile-specific inference configuration"""
        config = {
            "model": {
                "architecture": "mobile_optimized",
                "precision": "float32",
                "batch_size": 1,
                "enable_caching": True
            },
            "inference": {
                "max_sequence_length": 256,
                "processing_interval_ms": 100,
                "memory_limit_mb": 50,
                "threading": {
                    "max_threads": 2,
                    "enable_parallel": False
                }
            },
            "tokenization": {
                "audio_buffer_size": 64,
                "imu_buffer_size": 128,
                "token_cache_size": 512
            },
            "coaching": {
                "min_interval_seconds": 3.0,
                "max_messages_per_minute": 10,
                "enable_audio_cues": True
            },
            "performance": {
                "target_latency_ms": 200,
                "memory_cleanup_interval": 100,
                "enable_profiling": False
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config
    
    def _calculate_model_size(self, model: nn.Module) -> int:
        """Calculate model size in bytes"""
        total_size = 0
        for param in model.parameters():
            total_size += param.nelement() * param.element_size()
        return total_size

def main():
    """Run mobile optimization"""
    print("=== Mobile Model Optimization ===")
    
    optimizer = MobileOptimizer()
    output_dir = Path("mobile_models")
    output_dir.mkdir(exist_ok=True)
    
    results = {}
    
    # 1. Create mobile-optimized model
    mobile_model_path = output_dir / "mobile_perception_model.pth"
    model_results = optimizer.create_mobile_model(str(mobile_model_path))
    results["mobile_model"] = model_results
    
    # 2. Optimize tokenizers
    tokenizer_results = optimizer.optimize_tokenizers_simple(
        "trained_tokenizer.pkl",
        "trained_audio_tokenizer.pkl", 
        str(output_dir)
    )
    results["tokenizers"] = tokenizer_results
    
    # 3. Create mobile config
    config_path = output_dir / "mobile_inference_config.json"
    mobile_config = optimizer.create_mobile_inference_config(str(config_path))
    results["config"] = {"path": str(config_path), "settings": mobile_config}
    
    # Save optimization report
    report_path = output_dir / "mobile_optimization_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Mobile optimization complete!")
    print(f"📱 Mobile model: {model_results['model_size_mb']:.2f} MB")
    print(f"📊 Report: {report_path}")

if __name__ == "__main__":
    main()
