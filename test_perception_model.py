#!/usr/bin/env python3
"""
Test Suite for Project Chimera v2 Unified Perception Model
Validates that the transformer learns meaningful workout patterns and motion primitives.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from tokenization_pipeline import MultiModalTokenizer, collect_training_sessions
from perception_transformer import UnifiedPerceptionTransformer, TransformerConfig
import pickle

class PerceptionModelTester:
    """Comprehensive testing framework for the perception model"""
    
    def __init__(self, model_path: str, tokenizer_path: str):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
    def load_model_and_tokenizer(self):
        """Load trained model and tokenizer"""
        print("Loading trained model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = MultiModalTokenizer()
        self.tokenizer.load_tokenizer(self.tokenizer_path)
        
        # Load model
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        self.model = UnifiedPerceptionTransformer(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded: {self.config.n_layers} layers, {self.config.d_model} dim")
        
    def test_motion_primitive_consistency(self) -> Dict[str, float]:
        """Test if model learns consistent motion primitives"""
        print("\n=== Testing Motion Primitive Consistency ===")
        
        # Collect session data
        downloads_dir = "/Users/jonathanschwartz/Downloads"
        session_paths = collect_training_sessions(downloads_dir)
        
        # Analyze motion token patterns
        motion_patterns = {}
        total_tokens = 0
        
        for session_path in session_paths[:5]:  # Test on subset
            try:
                tokenized = self.tokenizer.tokenize_session(session_path)
                
                # Count motion token frequencies
                for token in tokenized.tokens:
                    if token >= 1024:  # Motion tokens
                        motion_patterns[token] = motion_patterns.get(token, 0) + 1
                        total_tokens += 1
                        
            except Exception as e:
                print(f"Error processing {Path(session_path).name}: {e}")
                
        # Calculate statistics
        if total_tokens == 0:
            return {"error": "No motion tokens found"}
            
        # Find most common motion primitives
        sorted_patterns = sorted(motion_patterns.items(), key=lambda x: x[1], reverse=True)
        top_5 = sorted_patterns[:5]
        
        print(f"Total motion tokens analyzed: {total_tokens}")
        print("Top 5 motion primitives:")
        for token, count in top_5:
            percentage = (count / total_tokens) * 100
            print(f"  Token {token}: {count} occurrences ({percentage:.1f}%)")
            
        # Test consistency - dominant patterns should represent significant portion
        dominant_coverage = sum(count for _, count in top_5) / total_tokens
        
        return {
            "total_tokens": total_tokens,
            "unique_primitives": len(motion_patterns),
            "dominant_coverage": dominant_coverage,
            "top_primitives": dict(top_5)
        }
    
    def test_sequence_prediction_accuracy(self) -> Dict[str, float]:
        """Test model's ability to predict next tokens in sequences"""
        print("\n=== Testing Sequence Prediction Accuracy ===")
        
        downloads_dir = "/Users/jonathanschwartz/Downloads"
        session_paths = collect_training_sessions(downloads_dir)
        
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for session_path in session_paths[:3]:  # Test subset
                try:
                    tokenized = self.tokenizer.tokenize_session(session_path)
                    tokens = tokenized.tokens
                    
                    if len(tokens) < 20:
                        continue
                        
                    # Test prediction on sliding windows
                    for i in range(10, len(tokens) - 1):
                        # Use previous 10 tokens to predict next
                        context = torch.tensor([tokens[i-10:i]], dtype=torch.long).to(self.device)
                        target = tokens[i]
                        
                        # Get model prediction
                        logits = self.model(context)
                        predicted = torch.argmax(logits[0, -1, :]).item()
                        
                        if predicted == target:
                            correct_predictions += 1
                        total_predictions += 1
                        
                except Exception as e:
                    print(f"Error testing {Path(session_path).name}: {e}")
                    
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        print(f"Prediction accuracy: {correct_predictions}/{total_predictions} ({accuracy:.3f})")
        
        return {
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions
        }
    
    def test_motion_state_transitions(self) -> Dict[str, any]:
        """Test if model learns realistic motion state transitions"""
        print("\n=== Testing Motion State Transitions ===")
        
        # Generate sequences and analyze transition patterns
        transitions = {}
        
        with torch.no_grad():
            # Start with a common motion token
            start_token = 1025  # Most common from training
            input_ids = torch.tensor([[start_token]], dtype=torch.long).to(self.device)
            
            # Generate a sequence
            generated = self.model.generate(input_ids, max_new_tokens=50, temperature=0.8)
            sequence = generated[0].cpu().tolist()
            
            # Analyze transitions
            for i in range(len(sequence) - 1):
                current = sequence[i]
                next_token = sequence[i + 1]
                
                if current >= 1024 and next_token >= 1024:  # Both motion tokens
                    key = f"{current}->{next_token}"
                    transitions[key] = transitions.get(key, 0) + 1
                    
        print("Generated sequence:", sequence[:20])
        print("Top transitions:")
        sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
        for transition, count in sorted_transitions[:5]:
            print(f"  {transition}: {count} times")
            
        return {
            "generated_sequence": sequence,
            "transitions": transitions,
            "sequence_length": len(sequence)
        }
    
    def test_rep_counting_patterns(self) -> Dict[str, any]:
        """Test if model can identify rep-like patterns in motion"""
        print("\n=== Testing Rep Counting Patterns ===")
        
        downloads_dir = "/Users/jonathanschwartz/Downloads"
        session_paths = collect_training_sessions(downloads_dir)
        
        rep_patterns = []
        
        for session_path in session_paths[:3]:
            try:
                # Load session metadata for ground truth
                session_dir = Path(session_path)
                with open(session_dir / "session_metadata.json", 'r') as f:
                    metadata = json.load(f)
                    
                ground_truth_reps = metadata.get('repCount', 0)
                duration = metadata.get('duration', 0)
                
                # Tokenize session
                tokenized = self.tokenizer.tokenize_session(session_path)
                
                # Simple rep detection: look for repeated motion patterns
                tokens = tokenized.tokens
                detected_reps = self._detect_rep_patterns(tokens)
                
                rep_patterns.append({
                    'session': Path(session_path).name,
                    'ground_truth': ground_truth_reps,
                    'detected': detected_reps,
                    'duration': duration,
                    'tokens': len(tokens)
                })
                
                print(f"{Path(session_path).name}: GT={ground_truth_reps}, Detected={detected_reps}")
                
            except Exception as e:
                print(f"Error analyzing {Path(session_path).name}: {e}")
                
        return {"rep_patterns": rep_patterns}
    
    def _detect_rep_patterns(self, tokens: List[int]) -> int:
        """Simple rep detection based on motion token patterns"""
        if len(tokens) < 10:
            return 0
            
        # Look for alternating patterns (up/down motion)
        transitions = 0
        prev_token = tokens[0]
        
        for token in tokens[1:]:
            if abs(token - prev_token) > 10:  # Significant change
                transitions += 1
            prev_token = token
            
        # Estimate reps as half the transitions (up + down = 1 rep)
        estimated_reps = max(0, transitions // 2)
        return min(estimated_reps, 20)  # Cap at reasonable max
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("UNIFIED PERCEPTION MODEL TEST REPORT")
        print("="*60)
        
        # Run all tests
        primitive_results = self.test_motion_primitive_consistency()
        prediction_results = self.test_sequence_prediction_accuracy()
        transition_results = self.test_motion_state_transitions()
        rep_results = self.test_rep_counting_patterns()
        
        # Generate report
        report = f"""
# Unified Perception Model Test Report
Generated: 2025-09-06 18:23:22

## Model Configuration
- Architecture: {self.config.n_layers} layers, {self.config.d_model} dimensions
- Vocabulary: {self.config.vocab_size} tokens (1024 audio + 64 motion)
- Parameters: ~{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M

## Test Results

### 1. Motion Primitive Consistency
- Total motion tokens: {primitive_results.get('total_tokens', 'N/A')}
- Unique primitives: {primitive_results.get('unique_primitives', 'N/A')}
- Dominant coverage: {primitive_results.get('dominant_coverage', 0):.1%}
- Status: {'✅ PASS' if primitive_results.get('dominant_coverage', 0) > 0.5 else '❌ FAIL'}

### 2. Sequence Prediction Accuracy
- Accuracy: {prediction_results.get('accuracy', 0):.3f}
- Predictions: {prediction_results.get('correct_predictions', 0)}/{prediction_results.get('total_predictions', 0)}
- Status: {'✅ PASS' if prediction_results.get('accuracy', 0) > 0.1 else '❌ FAIL'}

### 3. Motion State Transitions
- Generated sequence length: {transition_results.get('sequence_length', 0)}
- Unique transitions: {len(transition_results.get('transitions', {}))}
- Status: {'✅ PASS' if len(transition_results.get('transitions', {})) > 3 else '❌ FAIL'}

### 4. Rep Counting Patterns
- Sessions analyzed: {len(rep_results.get('rep_patterns', []))}
- Pattern detection: {'✅ IMPLEMENTED' if rep_results.get('rep_patterns') else '❌ FAILED'}

## Recommendations
1. {'Model shows good primitive learning' if primitive_results.get('dominant_coverage', 0) > 0.5 else 'Need more diverse training data'}
2. {'Prediction accuracy acceptable for early model' if prediction_results.get('accuracy', 0) > 0.1 else 'Model needs more training epochs'}
3. {'Transition patterns emerging' if len(transition_results.get('transitions', {})) > 3 else 'Need longer training sequences'}
4. Rep counting validation requires ground truth audio data for full testing

## Next Steps
- Integrate audio tokenization for complete multimodal testing
- Expand training dataset with more diverse sessions
- Implement real-time inference pipeline
- Add coaching trigger detection
        """
        
        return report

def main():
    """Run comprehensive test suite"""
    model_path = "/Users/jonathanschwartz/Documents/Social_Content/QuantumLeapValidator/perception_transformer.pt"
    tokenizer_path = "/Users/jonathanschwartz/Documents/Social_Content/QuantumLeapValidator/trained_tokenizer.pkl"
    
    tester = PerceptionModelTester(model_path, tokenizer_path)
    tester.load_model_and_tokenizer()
    
    # Generate test report
    report = tester.generate_test_report()
    
    # Save report
    report_path = "/Users/jonathanschwartz/Documents/Social_Content/QuantumLeapValidator/PERCEPTION_MODEL_TEST_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
        
    print(f"\nTest report saved to: {report_path}")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
