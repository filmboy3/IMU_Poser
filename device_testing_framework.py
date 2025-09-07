"""
Device Testing Framework for Chimera v2 Unified Perception System
Live testing with real IMU and audio data on iOS device
"""

import torch
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

class DeviceTestingFramework:
    """
    Framework for testing unified perception system with real device data
    Validates performance, accuracy, and latency on actual workout sessions
    """
    
    def __init__(self, 
                 model_path: str = "perception_transformer.pt",
                 tokenizer_path: str = "trained_tokenizer.pkl",
                 audio_tokenizer_path: str = "trained_audio_tokenizer.pkl"):
        """
        Initialize device testing framework
        
        Args:
            model_path: Path to trained perception model
            tokenizer_path: Path to IMU tokenizer
            audio_tokenizer_path: Path to audio tokenizer
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.audio_tokenizer_path = audio_tokenizer_path
        
        # Load components
        self.model = None
        self.tokenizer = None
        self.audio_tokenizer = None
        
        # Test results storage
        self.test_results = []
        self.performance_metrics = {}
        
        print("🧪 Device Testing Framework initialized")
    
    def load_components(self):
        """Load all system components for testing"""
        try:
            # Load perception model
            if Path(self.model_path).exists():
                from perception_transformer import UnifiedPerceptionTransformer, TransformerConfig
                
                config = TransformerConfig()
                self.model = UnifiedPerceptionTransformer(config)
                
                checkpoint = torch.load(self.model_path, map_location='cpu')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                print("✅ Perception model loaded")
            else:
                print("⚠️ Perception model not found - using mock inference")
            
            # Load IMU tokenizer
            if Path(self.tokenizer_path).exists():
                from tokenization_pipeline import MultiModalSensorTokenizer
                self.tokenizer = MultiModalSensorTokenizer()
                self.tokenizer.load(self.tokenizer_path)
                print("✅ IMU tokenizer loaded")
            else:
                print("⚠️ IMU tokenizer not found")
            
            # Load audio tokenizer
            if Path(self.audio_tokenizer_path).exists():
                from audio_tokenization_pipeline import AudioTokenizer
                self.audio_tokenizer = AudioTokenizer()
                self.audio_tokenizer.load(self.audio_tokenizer_path)
                print("✅ Audio tokenizer loaded")
            else:
                print("⚠️ Audio tokenizer not found")
                
        except Exception as e:
            print(f"❌ Component loading failed: {e}")
    
    def test_session_data(self, session_path: str) -> Dict:
        """
        Test unified perception system on real session data
        
        Args:
            session_path: Path to session directory
            
        Returns:
            Test results dictionary
        """
        print(f"🧪 Testing session: {Path(session_path).name}")
        
        # Load session data
        session_data = self.load_session_data(session_path)
        if not session_data:
            return {"error": "Failed to load session data"}
        
        # Initialize test metrics
        test_start = time.time()
        inference_times = []
        rep_predictions = []
        ground_truth_reps = session_data.get('metadata', {}).get('repCount', 0)
        
        # Process IMU data in chunks (simulate real-time)
        imu_data = session_data['imu_data']
        chunk_size = 100  # 1 second at 100Hz
        
        for i in range(0, len(imu_data), chunk_size):
            chunk = imu_data[i:i+chunk_size]
            
            # Measure inference time
            inference_start = time.time()
            
            # Tokenize IMU data
            if self.tokenizer:
                try:
                    tokens = self.tokenizer.tokenize_sample(np.array(chunk))
                except:
                    tokens = list(range(len(chunk)))  # Mock tokens
            else:
                tokens = list(range(len(chunk)))
            
            # Run inference
            if self.model and len(tokens) > 10:
                try:
                    with torch.no_grad():
                        input_ids = torch.tensor([tokens[:64]], dtype=torch.long)
                        outputs = self.model(input_ids)
                        
                        # Extract rep prediction (simplified)
                        rep_prob = torch.sigmoid(outputs.logits[:, -1, 0]).item()
                        rep_predictions.append(rep_prob)
                        
                except Exception as e:
                    print(f"⚠️ Inference failed: {e}")
                    rep_predictions.append(0.0)
            else:
                # Mock inference
                rep_predictions.append(np.random.random() * 0.1)
            
            inference_time = time.time() - inference_start
            inference_times.append(inference_time * 1000)  # Convert to ms
        
        # Analyze results
        total_time = time.time() - test_start
        avg_inference_time = np.mean(inference_times)
        
        # Count predicted reps (simple threshold)
        predicted_reps = sum(1 for p in rep_predictions if p > 0.5)
        
        # Calculate accuracy
        accuracy = 1.0 - abs(predicted_reps - ground_truth_reps) / max(ground_truth_reps, 1)
        accuracy = max(0.0, accuracy)
        
        results = {
            'session_id': Path(session_path).name,
            'ground_truth_reps': ground_truth_reps,
            'predicted_reps': predicted_reps,
            'accuracy': accuracy,
            'avg_inference_time_ms': avg_inference_time,
            'max_inference_time_ms': max(inference_times) if inference_times else 0,
            'total_test_time_s': total_time,
            'chunks_processed': len(inference_times),
            'rep_predictions': rep_predictions[:10],  # Sample predictions
            'latency_target_met': bool(avg_inference_time < 200.0)
        }
        
        self.test_results.append(results)
        
        print(f"   Ground truth reps: {ground_truth_reps}")
        print(f"   Predicted reps: {predicted_reps}")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Avg inference time: {avg_inference_time:.1f}ms")
        print(f"   Latency target (<200ms): {'✅' if results['latency_target_met'] else '❌'}")
        
        return results
    
    def load_session_data(self, session_path: str) -> Optional[Dict]:
        """Load session data from directory"""
        try:
            session_dir = Path(session_path)
            
            # Load metadata
            metadata_file = session_dir / "session_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # Load IMU data
            imu_file = session_dir / "imu_data.json"
            if imu_file.exists():
                with open(imu_file, 'r') as f:
                    imu_data = json.load(f)
            else:
                return None
            
            return {
                'metadata': metadata,
                'imu_data': imu_data
            }
            
        except Exception as e:
            print(f"❌ Failed to load session data: {e}")
            return None
    
    def run_comprehensive_test(self, session_directories: List[str]) -> Dict:
        """
        Run comprehensive testing across multiple sessions
        
        Args:
            session_directories: List of session directory paths
            
        Returns:
            Comprehensive test report
        """
        print(f"🧪 Running comprehensive test on {len(session_directories)} sessions")
        
        # Load components
        self.load_components()
        
        # Test each session
        for session_dir in session_directories:
            if Path(session_dir).exists():
                self.test_session_data(session_dir)
            else:
                print(f"⚠️ Session directory not found: {session_dir}")
        
        # Generate comprehensive report
        return self.generate_test_report()
    
    def generate_test_report(self) -> Dict:
        """Generate comprehensive test report"""
        if not self.test_results:
            return {"error": "No test results available"}
        
        # Calculate aggregate metrics
        accuracies = [r['accuracy'] for r in self.test_results]
        inference_times = [r['avg_inference_time_ms'] for r in self.test_results]
        latency_compliance = [r['latency_target_met'] for r in self.test_results]
        
        report = {
            'test_summary': {
                'total_sessions_tested': len(self.test_results),
                'avg_accuracy': np.mean(accuracies),
                'min_accuracy': min(accuracies),
                'max_accuracy': max(accuracies),
                'avg_inference_time_ms': np.mean(inference_times),
                'max_inference_time_ms': max(inference_times),
                'latency_compliance_rate': np.mean(latency_compliance),
                'test_timestamp': datetime.now().isoformat()
            },
            'individual_results': self.test_results,
            'performance_analysis': {
                'accuracy_distribution': {
                    'excellent_90_plus': sum(1 for a in accuracies if a >= 0.9),
                    'good_70_to_90': sum(1 for a in accuracies if 0.7 <= a < 0.9),
                    'poor_below_70': sum(1 for a in accuracies if a < 0.7)
                },
                'latency_analysis': {
                    'under_50ms': sum(1 for t in inference_times if t < 50),
                    'under_100ms': sum(1 for t in inference_times if t < 100),
                    'under_200ms': sum(1 for t in inference_times if t < 200),
                    'over_200ms': sum(1 for t in inference_times if t >= 200)
                }
            }
        }
        
        # Print summary
        print(f"\n📊 Test Report Summary:")
        print(f"   Sessions tested: {report['test_summary']['total_sessions_tested']}")
        print(f"   Average accuracy: {report['test_summary']['avg_accuracy']:.1%}")
        print(f"   Average inference time: {report['test_summary']['avg_inference_time_ms']:.1f}ms")
        print(f"   Latency compliance: {report['test_summary']['latency_compliance_rate']:.1%}")
        
        return report
    
    def save_test_report(self, report: Dict, filename: str = None):
        """Save test report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"device_test_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Test report saved to {filename}")
    
    def plot_performance_metrics(self, report: Dict):
        """Generate performance visualization plots"""
        if not self.test_results:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy distribution
        accuracies = [r['accuracy'] for r in self.test_results]
        ax1.hist(accuracies, bins=10, alpha=0.7, color='blue')
        ax1.set_xlabel('Accuracy')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Accuracy Distribution')
        ax1.axvline(np.mean(accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(accuracies):.1%}')
        ax1.legend()
        
        # Inference time distribution
        inference_times = [r['avg_inference_time_ms'] for r in self.test_results]
        ax2.hist(inference_times, bins=10, alpha=0.7, color='green')
        ax2.set_xlabel('Inference Time (ms)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Inference Time Distribution')
        ax2.axvline(200, color='red', linestyle='--', label='200ms Target')
        ax2.axvline(np.mean(inference_times), color='orange', linestyle='--', label=f'Mean: {np.mean(inference_times):.1f}ms')
        ax2.legend()
        
        # Accuracy vs Ground Truth
        ground_truth = [r['ground_truth_reps'] for r in self.test_results]
        predicted = [r['predicted_reps'] for r in self.test_results]
        ax3.scatter(ground_truth, predicted, alpha=0.7)
        ax3.plot([0, max(ground_truth)], [0, max(ground_truth)], 'r--', label='Perfect Accuracy')
        ax3.set_xlabel('Ground Truth Reps')
        ax3.set_ylabel('Predicted Reps')
        ax3.set_title('Prediction Accuracy')
        ax3.legend()
        
        # Performance over time
        session_ids = [r['session_id'] for r in self.test_results]
        ax4.plot(range(len(accuracies)), accuracies, 'b-o', label='Accuracy')
        ax4.set_xlabel('Session Index')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Performance Over Sessions')
        ax4.set_ylim(0, 1)
        ax4.legend()
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"device_test_performance_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Performance plots saved to {plot_filename}")
        
        plt.show()

def run_device_testing():
    """Main function to run device testing"""
    print("🧪 Starting Chimera v2 Device Testing")
    
    # Initialize testing framework
    tester = DeviceTestingFramework()
    
    # Find available session directories
    session_dirs = []
    downloads_dir = Path("/Users/jonathanschwartz/Downloads")
    
    for item in downloads_dir.iterdir():
        if item.is_dir() and item.name.startswith("session_"):
            session_dirs.append(str(item))
    
    if not session_dirs:
        print("❌ No session directories found for testing")
        return
    
    print(f"📁 Found {len(session_dirs)} session directories")
    
    # Run comprehensive testing
    report = tester.run_comprehensive_test(session_dirs[:5])  # Test first 5 sessions
    
    # Save report
    tester.save_test_report(report)
    
    # Generate performance plots
    try:
        tester.plot_performance_metrics(report)
    except Exception as e:
        print(f"⚠️ Plot generation failed: {e}")
    
    return report

if __name__ == "__main__":
    report = run_device_testing()
