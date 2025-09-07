import Foundation
import UIKit
import CoreMotion
import AVFoundation
import AudioToolbox
import PythonKit

/**
 * UnifiedPerceptionBridge - Full ML pipeline integration for Project Chimera v2
 * Real IMU processing, audio tokenization, and perception transformer inference
 */

@objc class UnifiedPerceptionBridge: NSObject, ObservableObject {
    
    // MARK: - Properties
    
    @Published var isActive: Bool = false
    @Published var currentRep: Int = 0
    @Published var targetReps: Int = 10
    @Published var coachingMessage: String = "Ready to start your AI-powered workout!"
    @Published var sessionDuration: TimeInterval = 0
    
    // Python integration
    private var pythonModule: PythonObject?
    private var pipeline: PythonObject?
    private var pythonInitialized: Bool = false
    
    // Core components
    private var motionManager: CMMotionManager
    private var sessionStartTime: Date?
    private var processingQueue: DispatchQueue
    private var sampleCount: Int = 0
    private var audioEngine: AVAudioEngine?
    private var audioInputNode: AVAudioInputNode?
    private var audioBuffer: [Float] = []
    
    // Configuration
    private let modelPath: String
    private let tokenizerPath: String
    private let audioTokenizerPath: String
    private let updateInterval: TimeInterval = 0.1 // 10Hz IMU sampling
    private let audioSampleRate: Double = 16000 // 16kHz for speech processing
    
    // MARK: - Initialization
    
    override init() {
        // Configure mobile-optimized model and tokenizer paths from app bundle
        self.modelPath = Bundle.main.path(forResource: "mobile_perception_model", ofType: "pth") ?? 
                        Bundle.main.path(forResource: "perception_model", ofType: "pth") ?? "perception_model.pth"
        
        // Try mobile tokenizers first, fallback to original
        self.tokenizerPath = Bundle.main.path(forResource: "mobile_main_tokenizer", ofType: "pkl") ??
                           Bundle.main.path(forResource: "trained_tokenizer", ofType: "pkl") ?? "trained_tokenizer.pkl"
        
        self.audioTokenizerPath = Bundle.main.path(forResource: "mobile_audio_tokenizer", ofType: "pkl") ??
                                Bundle.main.path(forResource: "trained_audio_tokenizer", ofType: "pkl") ?? "trained_audio_tokenizer.pkl"
        
        // Initialize motion manager
        self.motionManager = CMMotionManager()
        self.processingQueue = DispatchQueue(label: "perception.processing", qos: .userInitiated)
        
        super.init()
        
        setupPythonEnvironment()
        print("🧠 UnifiedPerceptionBridge initialized with full ML pipeline")
    }
    
    // MARK: - Python Environment Setup
    
    private func setupPythonEnvironment() {
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            
            do {
                try self.initializePythonEnvironment()
                print("✅ Full Python environment initialized successfully")
            } catch {
                print("⚠️ Python initialization failed: \(error). Using fallback mode.")
                self.pythonModule = self.createFallbackModule()
            }
        }
    }
    
    private func initializePythonEnvironment() throws {
        // Set up Python path with multiple fallback locations
        let pythonPaths = [
            "/opt/homebrew/lib/python3.11/site-packages",
            "/usr/local/lib/python3.11/site-packages", 
            "/opt/homebrew/lib/python3.10/site-packages",
            "/usr/local/lib/python3.10/site-packages"
        ]
        
        for path in pythonPaths {
            if FileManager.default.fileExists(atPath: path) {
                let sys = Python.import("sys")
                let pathExists = sys.path.__contains__(path)
                if !(Bool(pathExists) ?? false) {
                    sys.path.append(path)
                }
            }
        }
        
        // Import required Python modules
        let sys = Python.import("sys")
        let os = Python.import("os")
        
        // Add current directory to Python path for local modules
        let currentDir = FileManager.default.currentDirectoryPath
        sys.path.insert(0, currentDir)
        
        // Import the unified perception pipeline
        self.pythonModule = try Python.attemptImport("realtime_inference_pipeline")
        
        // Initialize the pipeline with model paths
        let pipelineClass = pythonModule!.UnifiedPerceptionPipeline
        self.pipeline = pipelineClass(
            model_path: modelPath,
            tokenizer_path: tokenizerPath,
            audio_tokenizer_path: audioTokenizerPath
        )
        
        self.pythonInitialized = true
        
        DispatchQueue.main.async {
            self.coachingMessage = "AI Coach ready! Advanced perception system loaded."
        }
    }
    
    private func createFallbackModule() -> PythonObject? {
        // Create a simple fallback that mimics the Python interface
        let fallback = Python.dict()
        // Return a simple fallback object
        return fallback
    }
    
    // MARK: - Session Management
    
    func startSession(targetReps: Int = 10) {
        guard !isActive else { return }
        
        self.targetReps = targetReps
        self.currentRep = 0
        self.sessionStartTime = Date()
        self.sampleCount = 0
        
        DispatchQueue.main.async {
            self.isActive = true
            self.coachingMessage = self.pythonInitialized ? 
                "AI Coach activated - Ready to analyze your workout!" : 
                "Fallback mode - Ready to count reps!"
        }
        
        startIMUMonitoring()
        startAudioCapture()
        
        // Initialize Python pipeline session
        if pythonInitialized, let pipeline = self.pipeline {
            pipeline.start_session(targetReps)
            print("🧠 Python perception pipeline session started")
        }
        
        print("🚀 Unified perception session started (target: \(targetReps) reps)")
    }
    
    func stopSession() {
        guard isActive else { return }
        
        stopIMUMonitoring()
        stopAudioCapture()
        
        // Stop Python pipeline session
        if pythonInitialized, let pipeline = self.pipeline {
            pipeline.stop_session()
            print("🧠 Python perception pipeline session stopped")
        }
        
        let duration = Date().timeIntervalSince(sessionStartTime ?? Date())
        
        DispatchQueue.main.async {
            self.isActive = false
            self.sessionDuration = duration
            self.coachingMessage = "Great workout! You completed \(self.currentRep) reps in \(String(format: "%.1f", duration/60)) minutes."
        }
        
        print("🏁 Session ended - Completed: \(currentRep) reps in \(String(format: "%.1f", duration)) seconds")
    }
    
    // MARK: - IMU Processing
    
    private func startIMUMonitoring() {
        guard motionManager.isDeviceMotionAvailable else {
            print("⚠️ Device motion not available")
            return
        }
        
        motionManager.deviceMotionUpdateInterval = updateInterval
        motionManager.startDeviceMotionUpdates(to: OperationQueue()) { [weak self] (motion, error) in
            guard let self = self, let motion = motion else { return }
            self.processIMUData(motion)
        }
        
        print("📱 IMU monitoring started with real sensor data")
    }
    
    private func stopIMUMonitoring() {
        motionManager.stopDeviceMotionUpdates()
        print("📱 IMU monitoring stopped")
    }
    
    private func processIMUData(_ motion: CMDeviceMotion) {
        // Extract comprehensive sensor data
        let acceleration = motion.userAcceleration
        let rotationRate = motion.rotationRate
        let attitude = motion.attitude
        let gravity = motion.gravity
        let timestamp = Date().timeIntervalSince1970
        
        self.sampleCount += 1
        
        // Process with unified perception pipeline if available
        if pythonInitialized, let pipeline = self.pipeline {
            do {
                // Create IMU data dictionary for Python
                let imuData = Python.dict([
                    "timestamp": timestamp,
                    "acceleration": [acceleration.x, acceleration.y, acceleration.z],
                    "rotation_rate": [rotationRate.x, rotationRate.y, rotationRate.z],
                    "attitude": [attitude.roll, attitude.pitch, attitude.yaw],
                    "gravity": [gravity.x, gravity.y, gravity.z]
                ])
                
                // Send to unified perception pipeline
                let result = pipeline.process_imu_data(imuData)
                
                // Process AI results
                if let repDetected = result["rep_detected"], Bool(repDetected) == true {
                    DispatchQueue.main.async {
                        self.currentRep += 1
                        self.coachingMessage = "Rep \(self.currentRep) detected by AI!"
                        
                        let impactFeedback = UIImpactFeedbackGenerator(style: .medium)
                        impactFeedback.impactOccurred()
                        
                        // Play rep completion sound
                        AudioServicesPlaySystemSound(1057) // Tink sound
                        
                        print("🧠 AI detected rep \(self.currentRep)")
                    }
                }
                
                // Update coaching based on AI analysis
                if let coaching = result["coaching_message"] {
                    DispatchQueue.main.async {
                        self.coachingMessage = String(describing: coaching)
                    }
                }
                
                // Handle coaching responses with audio cues
                if let coachingResponse = result["coaching_response"] {
                    self.handleCoachingResponse(coachingResponse)
                }
                
            } catch {
                print("⚠️ AI processing failed: \(error)")
                // Fallback to simple motion analysis
                self.fallbackIMUProcessing(motion)
            }
        } else {
            // Fallback processing without AI
            self.fallbackIMUProcessing(motion)
        }
        
        // Update session duration
        DispatchQueue.main.async {
            if let startTime = self.sessionStartTime {
                self.sessionDuration = Date().timeIntervalSince(startTime)
            }
        }
    }
    
    private func fallbackIMUProcessing(_ motion: CMDeviceMotion) {
        // Simple motion-based rep detection as fallback
        let totalAcceleration = sqrt(
            pow(motion.userAcceleration.x, 2) +
            pow(motion.userAcceleration.y, 2) +
            pow(motion.userAcceleration.z, 2)
        )
        
        // Simple threshold-based detection
        if totalAcceleration > 2.0 && sampleCount % 30 == 0 && currentRep < targetReps {
            DispatchQueue.main.async {
                self.currentRep += 1
                self.coachingMessage = "Rep \(self.currentRep) detected (fallback mode)"
                
                let impactFeedback = UIImpactFeedbackGenerator(style: .light)
                impactFeedback.impactOccurred()
                
                print("🔄 Fallback rep \(self.currentRep) detected")
            }
        }
    }
    
    // MARK: - Audio Processing
    
    private func startAudioCapture() {
        do {
            audioEngine = AVAudioEngine()
            guard let audioEngine = audioEngine else { return }
            
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker, .allowBluetooth])
            try audioSession.setActive(true)
            
            audioInputNode = audioEngine.inputNode
            let inputFormat = audioInputNode!.outputFormat(forBus: 0)
            let _ = AVAudioFormat(standardFormatWithSampleRate: audioSampleRate, channels: 1)!
            
            audioInputNode!.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) { [weak self] buffer, _ in
                self?.processAudioBuffer(buffer)
            }
            
            try audioEngine.start()
            print("🎤 Audio capture started with real microphone data")
            
        } catch {
            print("⚠️ Audio setup failed: \(error)")
        }
    }
    
    private func stopAudioCapture() {
        audioEngine?.stop()
        audioInputNode?.removeTap(onBus: 0)
        audioEngine = nil
        audioInputNode = nil
        print("🎤 Audio capture stopped")
    }
    
    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData?[0] else { return }
        let frameLength = Int(buffer.frameLength)
        
        // Convert audio data to array
        let audioData = Array(UnsafeBufferPointer(start: channelData, count: frameLength))
        
        // Send to Python pipeline for audio tokenization
        if pythonInitialized, let pipeline = self.pipeline {
            let audioDict = Python.dict(["audio_data": audioData, "sample_rate": audioSampleRate])
            pipeline.process_audio_data(audioDict)
        }
    }
    
    // MARK: - Coaching System
    
    private func handleCoachingResponse(_ response: PythonObject) {
        do {
            let message = String(describing: response["message"] ?? "Keep going!")
            let urgency = String(describing: response["urgency"] ?? "normal")
            let audioCue = Bool(response["audio_cue"]) ?? false
            let hapticFeedback = Bool(response["haptic_feedback"]) ?? false
            
            DispatchQueue.main.async {
                self.coachingMessage = message
                
                if hapticFeedback {
                    let feedback = UIImpactFeedbackGenerator(style: urgency == "high" ? .heavy : .medium)
                    feedback.impactOccurred()
                }
                
                if audioCue {
                    let soundId: SystemSoundID = urgency == "high" ? 1016 : 1057 // Different sounds for urgency
                    AudioServicesPlaySystemSound(soundId)
                }
            }
            
        } catch {
            print("⚠️ Coaching response handling failed: \(error)")
        }
    }
    
    func getSessionStats() -> [String: Any] {
        let stats: [String: Any] = [
            "total_reps": currentRep,
            "target_reps": targetReps,
            "session_duration": sessionDuration,
            "reps_per_minute": sessionDuration > 0 ? Double(currentRep) / (sessionDuration / 60) : 0,
            "completion_rate": targetReps > 0 ? Double(currentRep) / Double(targetReps) * 100 : 0,
            "ai_enabled": pythonInitialized,
            "sample_count": sampleCount
        ]
        
        // Get additional stats from Python pipeline if available
        if pythonInitialized, let pipeline = self.pipeline {
            let _ = pipeline.get_session_stats()
            // Merge Python stats with Swift stats
            return stats // For now, return Swift stats
        }
        
        return stats
    }
}

// MARK: - Error Handling

enum PerceptionError: Error {
    case modelLoadingFailed
    case pythonInitializationFailed
    case audioSessionFailed
    case motionDataUnavailable
}
