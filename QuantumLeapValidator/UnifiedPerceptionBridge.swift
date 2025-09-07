import Foundation
import UIKit
import CoreMotion
import AVFoundation
import AudioToolbox

/**
 * UnifiedPerceptionBridge - Hybrid approach with real IMU processing and intelligent rep detection
 * Uses actual sensor data with sophisticated motion analysis for rep counting
 */

@objc class UnifiedPerceptionBridge: NSObject, ObservableObject {
    
    // MARK: - Properties
    
    @Published var isActive: Bool = false
    @Published var currentRep: Int = 0
    @Published var targetReps: Int = 10
    @Published var coachingMessage: String = "Ready to start your AI-powered workout!"
    @Published var sessionDuration: TimeInterval = 0
    
    // Core components
    private var motionManager: CMMotionManager
    private var sessionStartTime: Date?
    private var processingQueue: DispatchQueue
    private var sampleCount: Int = 0
    private var audioEngine: AVAudioEngine?
    private var audioInputNode: AVAudioInputNode?
    private var audioBuffer: [Float] = []
    
    // Motion analysis state
    private var motionHistory: [MotionSample] = []
    private var lastRepTime: TimeInterval = 0
    private var isInRepMotion: Bool = false
    private var motionBaseline: MotionBaseline?
    private var repDetectionState: RepDetectionState = .ready
    private var sessionMetrics: SessionMetrics = SessionMetrics()
    private var motionDetectedTime: TimeInterval = 0
    private var sustainedMotionSamples: Int = 0
    
    // Session metrics tracking
    struct SessionMetrics {
        var totalSamples: Int = 0
        var motionEvents: [MotionEvent] = []
        var thresholdCrossings: Int = 0
        var peakDetections: Int = 0
        var maxAcceleration: Double = 0
        var avgAcceleration: Double = 0
        var motionIntensityHistory: [Double] = []
    }
    
    struct MotionEvent {
        let timestamp: TimeInterval
        let acceleration: Double
        let eventType: String
        let state: RepDetectionState
    }
    
    // Configuration
    private let updateInterval: TimeInterval = 0.05 // 20Hz IMU sampling for better precision
    private let audioSampleRate: Double = 16000
    private let motionHistorySize: Int = 100 // Keep last 5 seconds at 20Hz
    private let minRepInterval: TimeInterval = 1.0 // Minimum time between reps
    
    // MARK: - Motion Analysis Types
    
    struct MotionSample {
        let timestamp: TimeInterval
        let acceleration: CMAcceleration
        let rotationRate: CMRotationRate
        let attitude: CMAttitude
        let gravity: CMAcceleration
        let totalAcceleration: Double
        let motionIntensity: Double
    }
    
    struct MotionBaseline {
        let averageAcceleration: Double
        let accelerationVariance: Double
        let dominantAxis: String
        let calibrationTime: TimeInterval
    }
    
    enum RepDetectionState {
        case ready
        case motionDetected
        case peakReached
        case returning
        case repCompleted
    }
    
    // MARK: - Initialization
    
    override init() {
        self.motionManager = CMMotionManager()
        self.processingQueue = DispatchQueue(label: "perception.processing", qos: .userInitiated)
        
        super.init()
        
        print("🧠 UnifiedPerceptionBridge initialized with real IMU analysis")
    }
    
    // MARK: - Session Management
    
    func startSession(targetReps: Int = 10) {
        guard !isActive else { return }
        
        self.targetReps = targetReps
        self.currentRep = 0
        self.sessionStartTime = Date()
        self.sampleCount = 0
        self.lastRepTime = 0
        self.motionHistory.removeAll()
        self.motionBaseline = nil
        self.repDetectionState = .ready
        
        DispatchQueue.main.async {
            self.isActive = true
            self.coachingMessage = "Starting workout session with real motion analysis..."
        }
        
        startIMUMonitoring()
        startAudioCapture()
        
        print("🚀 Unified perception session started with real IMU analysis (target: \(targetReps) reps)")
    }
    
    func stopSession() {
        guard isActive else { return }
        
        stopIMUMonitoring()
        stopAudioCapture()
        
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
        
        print("📱 IMU monitoring started with real sensor data at \(Int(1.0/updateInterval))Hz")
    }
    
    private func stopIMUMonitoring() {
        motionManager.stopDeviceMotionUpdates()
        print("📱 IMU monitoring stopped")
    }
    
    private func processIMUData(_ motion: CMDeviceMotion) {
        let timestamp = Date().timeIntervalSince1970
        let acceleration = motion.userAcceleration
        let rotationRate = motion.rotationRate
        let attitude = motion.attitude
        let gravity = motion.gravity
        
        // Calculate motion metrics
        let totalAcceleration = sqrt(
            pow(acceleration.x, 2) +
            pow(acceleration.y, 2) +
            pow(acceleration.z, 2)
        )
        
        let motionIntensity = sqrt(
            pow(rotationRate.x, 2) +
            pow(rotationRate.y, 2) +
            pow(rotationRate.z, 2)
        ) + totalAcceleration
        
        // Create motion sample
        let sample = MotionSample(
            timestamp: timestamp,
            acceleration: acceleration,
            rotationRate: rotationRate,
            attitude: attitude,
            gravity: gravity,
            totalAcceleration: totalAcceleration,
            motionIntensity: motionIntensity
        )
        
        // Add to history
        motionHistory.append(sample)
        if motionHistory.count > motionHistorySize {
            motionHistory.removeFirst()
        }
        
        self.sampleCount += 1
        
        // Update session metrics
        sessionMetrics.totalSamples += 1
        sessionMetrics.maxAcceleration = max(sessionMetrics.maxAcceleration, sample.totalAcceleration)
        sessionMetrics.motionIntensityHistory.append(sample.motionIntensity)
        if sessionMetrics.motionIntensityHistory.count > 200 { // Keep last 10 seconds
            sessionMetrics.motionIntensityHistory.removeFirst()
        }
        sessionMetrics.avgAcceleration = motionHistory.map { $0.totalAcceleration }.reduce(0, +) / Double(motionHistory.count)
        
        // Establish baseline after initial samples
        if motionBaseline == nil && motionHistory.count >= 40 { // 2 seconds of data
            establishMotionBaseline()
        }
        
        // Perform rep detection if baseline is established
        if let baseline = motionBaseline {
            performRepDetection(sample: sample, baseline: baseline)
        }
        
        // Update session duration
        DispatchQueue.main.async {
            if let startTime = self.sessionStartTime {
                self.sessionDuration = Date().timeIntervalSince(startTime)
            }
        }
        
        // Provide periodic coaching based on motion analysis
        if sampleCount % 100 == 0 { // Every 5 seconds
            provideMotionBasedCoaching(sample: sample)
        }
    }
    
    private func establishMotionBaseline() {
        guard motionHistory.count >= 40 else { return }
        
        let recentSamples = Array(motionHistory.suffix(40))
        let accelerations = recentSamples.map { $0.totalAcceleration }
        
        let averageAcceleration = accelerations.reduce(0, +) / Double(accelerations.count)
        let variance = accelerations.map { pow($0 - averageAcceleration, 2) }.reduce(0, +) / Double(accelerations.count)
        
        // Determine dominant motion axis
        let xVariance = recentSamples.map { $0.acceleration.x }.map { pow($0, 2) }.reduce(0, +)
        let yVariance = recentSamples.map { $0.acceleration.y }.map { pow($0, 2) }.reduce(0, +)
        let zVariance = recentSamples.map { $0.acceleration.z }.map { pow($0, 2) }.reduce(0, +)
        
        let dominantAxis: String
        if yVariance > xVariance && yVariance > zVariance {
            dominantAxis = "Y" // Vertical movement (squats, etc.)
        } else if xVariance > zVariance {
            dominantAxis = "X" // Side-to-side
        } else {
            dominantAxis = "Z" // Forward-backward
        }
        
        motionBaseline = MotionBaseline(
            averageAcceleration: averageAcceleration,
            accelerationVariance: variance,
            dominantAxis: dominantAxis,
            calibrationTime: Date().timeIntervalSince1970
        )
        
        DispatchQueue.main.async {
            self.coachingMessage = "Motion baseline established! Dominant axis: \(dominantAxis). Ready to count reps!"
        }
        
        print("📊 Motion baseline established - Avg: \(String(format: "%.3f", averageAcceleration)), Variance: \(String(format: "%.3f", variance)), Axis: \(dominantAxis)")
    }
    
    private func performRepDetection(sample: MotionSample, baseline: MotionBaseline) {
        let currentTime = sample.timestamp
        let timeSinceLastRep = currentTime - lastRepTime
        
        // Skip if too soon after last rep
        guard timeSinceLastRep > minRepInterval else { return }
        
        // Calculate motion thresholds based on baseline (much more sensitive)
        let motionThreshold = max(baseline.averageAcceleration + 0.5 * sqrt(baseline.accelerationVariance), 0.15)
        let peakThreshold = max(baseline.averageAcceleration + 0.8 * sqrt(baseline.accelerationVariance), 0.25)
        
        // Debug logging for threshold analysis
        if sampleCount % 40 == 0 { // Every 2 seconds
            print("🔍 Motion Analysis - Current: \(String(format: "%.3f", sample.totalAcceleration)), Threshold: \(String(format: "%.3f", motionThreshold)), Peak: \(String(format: "%.3f", peakThreshold)), State: \(repDetectionState)")
        }
        
        // State machine for rep detection
        switch repDetectionState {
        case .ready:
            if sample.totalAcceleration > motionThreshold {
                repDetectionState = .motionDetected
                isInRepMotion = true
                motionDetectedTime = currentTime
                sustainedMotionSamples = 1
                sessionMetrics.thresholdCrossings += 1
                sessionMetrics.motionEvents.append(MotionEvent(
                    timestamp: currentTime,
                    acceleration: sample.totalAcceleration,
                    eventType: "threshold_crossed",
                    state: .motionDetected
                ))
                print("🎯 Motion threshold crossed: \(String(format: "%.3f", sample.totalAcceleration)) > \(String(format: "%.3f", motionThreshold))")
            }
            
        case .motionDetected:
            sustainedMotionSamples += 1
            
            if sample.totalAcceleration > peakThreshold {
                repDetectionState = .peakReached
                sessionMetrics.peakDetections += 1
                sessionMetrics.motionEvents.append(MotionEvent(
                    timestamp: currentTime,
                    acceleration: sample.totalAcceleration,
                    eventType: "peak_reached",
                    state: .peakReached
                ))
                print("⛰️ Peak reached: \(String(format: "%.3f", sample.totalAcceleration)) > \(String(format: "%.3f", peakThreshold))")
            } else if sample.totalAcceleration < motionThreshold * 0.5 && sustainedMotionSamples < 5 {
                // Only reset if motion drops quickly AND we haven't sustained motion for at least 5 samples (0.25 seconds)
                repDetectionState = .ready
                isInRepMotion = false
                sustainedMotionSamples = 0
                print("❌ False start detected (insufficient sustained motion)")
                SoundManager.shared.play(.falseStart)
            } else if sustainedMotionSamples >= 10 {
                // If we've sustained motion for 10 samples (0.5 seconds), treat as a rep even without peak
                repDetectionState = .returning
                print("✅ Sustained motion detected - treating as rep")
            }
            
        case .peakReached:
            if sample.totalAcceleration < motionThreshold {
                repDetectionState = .returning
            }
            
        case .returning:
            if sample.totalAcceleration < baseline.averageAcceleration * 2.0 { // Very lenient return threshold
                // Rep completed!
                repDetectionState = .repCompleted
                sustainedMotionSamples = 0
                sessionMetrics.motionEvents.append(MotionEvent(
                    timestamp: currentTime,
                    acceleration: sample.totalAcceleration,
                    eventType: "rep_completed",
                    state: .repCompleted
                ))
                detectRep(at: currentTime, intensity: sample.motionIntensity)
            } else if sample.totalAcceleration > peakThreshold {
                repDetectionState = .peakReached // Another peak
            }
            
        case .repCompleted:
            if sample.totalAcceleration < motionThreshold * 0.6 {
                repDetectionState = .ready
                isInRepMotion = false
                sustainedMotionSamples = 0
            }
        }
    }
    
    private func detectRep(at timestamp: TimeInterval, intensity: Double) {
        guard currentRep < targetReps else { return }
        
        lastRepTime = timestamp
        
        DispatchQueue.main.async {
            self.currentRep += 1
            
            // Provide intensity-based feedback
            let qualityFeedback = self.getRepQualityFeedback(intensity: intensity)
            self.coachingMessage = "Rep \(self.currentRep) detected! \(qualityFeedback)"
            
            // Haptic feedback intensity based on rep quality
            let feedbackStyle: UIImpactFeedbackGenerator.FeedbackStyle = intensity > 3.0 ? .heavy : .medium
            let impactFeedback = UIImpactFeedbackGenerator(style: feedbackStyle)
            impactFeedback.impactOccurred()
            
            // Audio feedback with modern SoundManager
            print("🔊 SoundManager: Requesting rep completion feedback.")
            SoundManager.shared.play(.repComplete)
            
            print("🏋️ Rep \(self.currentRep) detected with intensity \(String(format: "%.2f", intensity))")
        }
    }
    
    private func getRepQualityFeedback(intensity: Double) -> String {
        if intensity > 4.0 {
            return "Excellent form! 💪"
        } else if intensity > 2.5 {
            return "Good rep! Keep it up! 👍"
        } else if intensity > 1.5 {
            return "Nice work! Try for more power! ⚡"
        } else {
            return "Rep counted. Focus on full range of motion! 🎯"
        }
    }
    
    private func provideMotionBasedCoaching(sample: MotionSample) {
        guard let baseline = motionBaseline else { return }
        
        let recentSamples = Array(motionHistory.suffix(20)) // Last second
        let avgRecentIntensity = recentSamples.map { $0.motionIntensity }.reduce(0, +) / Double(recentSamples.count)
        
        let coachingMessages: [String]
        
        if avgRecentIntensity < baseline.averageAcceleration * 0.5 {
            coachingMessages = [
                "I notice you've slowed down. Keep pushing! 🔥",
                "Maintain your energy! You've got this! 💪",
                "Focus on consistent movement! ⚡"
            ]
        } else if avgRecentIntensity > baseline.averageAcceleration * 2.0 {
            coachingMessages = [
                "Great intensity! Control the movement! 🎯",
                "Powerful work! Focus on form! 👌",
                "Excellent energy! Keep it controlled! ⭐"
            ]
        } else {
            coachingMessages = [
                "Perfect rhythm! Keep it up! 🎵",
                "Consistent form looking great! ✨",
                "You're in the zone! Stay focused! 🎯"
            ]
        }
        
        DispatchQueue.main.async {
            self.coachingMessage = coachingMessages.randomElement() ?? "Keep going strong!"
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
        
        // Calculate audio level for basic analysis
        let audioLevel = audioData.map { abs($0) }.reduce(0, +) / Float(audioData.count)
        
        // Simple audio-based coaching (breathing detection, etc.)
        if audioLevel > 0.01 { // Breathing or vocalization detected
            // Could be used for breathing pattern analysis or vocal encouragement detection
        }
    }
    
    // MARK: - Session Stats
    
    func getSessionStats() -> [String: Any] {
        let repsPerMinute = sessionDuration > 0 ? Double(currentRep) / (sessionDuration / 60) : 0
        let completionRate = targetReps > 0 ? Double(currentRep) / Double(targetReps) * 100 : 0
        
        return [
            "total_reps": currentRep,
            "target_reps": targetReps,
            "session_duration": sessionDuration,
            "reps_per_minute": repsPerMinute,
            "completion_rate": completionRate,
            "sample_count": sampleCount,
            "motion_analysis": "Real IMU-based detection",
            "dominant_axis": motionBaseline?.dominantAxis ?? "Not established",
            "baseline_acceleration": motionBaseline?.averageAcceleration ?? 0,
            "baseline_variance": motionBaseline?.accelerationVariance ?? 0,
            "max_acceleration": sessionMetrics.maxAcceleration,
            "avg_acceleration": sessionMetrics.avgAcceleration,
            "threshold_crossings": sessionMetrics.thresholdCrossings,
            "peak_detections": sessionMetrics.peakDetections,
            "motion_events_count": sessionMetrics.motionEvents.count,
            "sampling_rate": Double(sampleCount) / sessionDuration
        ]
    }
    
    func exportSessionData() -> String {
        let stats = getSessionStats()
        let motionEvents = sessionMetrics.motionEvents.map { event in
            "\(event.timestamp),\(event.acceleration),\(event.eventType),\(event.state)"
        }.joined(separator: "\n")
        
        let csvData = """
        Session Statistics:
        \(stats.map { "\($0.key): \($0.value)" }.joined(separator: "\n"))
        
        Motion Events (timestamp,acceleration,event_type,state):
        \(motionEvents)
        
        Motion Intensity History:
        \(sessionMetrics.motionIntensityHistory.enumerated().map { "\($0.offset),\($0.element)" }.joined(separator: "\n"))
        """
        
        return csvData
    }
}

// MARK: - Error Handling

enum PerceptionError: Error {
    case motionDataUnavailable
    case audioSessionFailed
    case baselineNotEstablished
}
