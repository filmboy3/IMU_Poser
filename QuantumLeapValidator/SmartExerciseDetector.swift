import CoreMotion
import AVFoundation
import Foundation

class SmartExerciseDetector: ObservableObject {
    enum ExerciseState {
        case setup          // Phone being handled/positioned
        case stabilizing    // Phone settling into position
        case ready          // Phone stable, ready for exercise
        case exercising     // Active exercise tracking
        case completed      // Exercise session ended
    }
    
    enum MotionType {
        case handling       // Erratic multi-axis motion
        case rustling       // High jerkiness, chaotic
        case stable         // Minimal motion
        case exercise       // Rhythmic Y-axis dominant motion
        case unknown
    }
    
    @Published var currentState: ExerciseState = .setup
    @Published var isReadyForExercise = false
    @Published var countdownSeconds = 0
    
    // Motion analysis buffers
    private var motionBuffer: [CMAcceleration] = []
    private let motionBufferSize = 200  // ~2 seconds at 100Hz
    
    // Stability tracking
    private var stableStartTime: Date?
    private let requiredStabilityDuration: TimeInterval = 1.5  // Reduced from 3.0 seconds
    
    // Fallback mechanism
    private var sessionStartTime: Date?
    private let maxSetupDuration: TimeInterval = 10.0  // Auto-start after 10 seconds
    
    // Audio feedback
    private var audioPlayer: AVAudioPlayer?
    
    // Thresholds for motion classification (made more permissive)
    private let stableActivityThreshold: Double = 0.02
    private let handlingJerkThreshold: Double = 0.20  // Increased to be less sensitive
    private let exerciseYDominanceThreshold: Double = 1.2  // Lowered to be more permissive
    private let exerciseActivityThreshold: Double = 0.06  // Lowered to detect lighter motion
    
    init() {
        setupAudioFeedback()
    }
    
    private func setupAudioFeedback() {
        // Setup system sounds for audio cues
        // We'll use system sounds for simplicity
    }
    
    func processMotion(_ acceleration: CMAcceleration) -> (ExerciseState, MotionType) {
        // Initialize session start time
        if sessionStartTime == nil {
            sessionStartTime = Date()
        }
        
        // Update motion buffer
        motionBuffer.append(acceleration)
        if motionBuffer.count > motionBufferSize {
            motionBuffer.removeFirst()
        }
        
        // Need sufficient data for analysis
        guard motionBuffer.count >= 50 else {
            return (currentState, .unknown)
        }
        
        // Analyze current motion pattern
        let motionType = analyzeMotionType()
        let previousState = currentState
        
        // Fallback: Auto-start exercising after max setup duration
        if let startTime = sessionStartTime,
           Date().timeIntervalSince(startTime) >= maxSetupDuration,
           currentState != .exercising {
            print("🔄 Auto-starting exercise after \(maxSetupDuration)s timeout")
            currentState = .exercising
            isReadyForExercise = true
            playAudioCue(.exerciseStart)
        }
        
        // State machine
        switch currentState {
        case .setup:
            // More permissive transition conditions
            if motionType == .stable {
                // Start stability timer
                if stableStartTime == nil {
                    stableStartTime = Date()
                    playAudioCue(.stabilizing)
                }
                
                // Check if stable long enough (reduced duration)
                if let startTime = stableStartTime,
                   Date().timeIntervalSince(startTime) >= requiredStabilityDuration {
                    currentState = .ready
                    isReadyForExercise = true
                    startReadyCountdown()
                }
            } else if motionType == .exercise {
                // Direct transition to exercising if exercise motion detected
                currentState = .exercising
                isReadyForExercise = true
                playAudioCue(.exerciseStart)
            } else {
                // Reset stability timer if motion detected
                stableStartTime = nil
            }
            
        case .stabilizing:
            // This state is handled in .setup
            break
            
        case .ready:
            if motionType == .exercise {
                currentState = .exercising
                playAudioCue(.exerciseStart)
            } else if motionType == .handling || motionType == .rustling {
                // Phone picked up again
                currentState = .setup
                isReadyForExercise = false
                stableStartTime = nil
            }
            
        case .exercising:
            if motionType == .handling || motionType == .rustling {
                // Exercise interrupted by handling
                currentState = .setup
                isReadyForExercise = false
                stableStartTime = nil
                playAudioCue(.exerciseInterrupted)
            }
            
        case .completed:
            // Stay in completed state
            break
        }
        
        // Log state changes
        if currentState != previousState {
            print("🔄 Exercise State: \(previousState) → \(currentState)")
        }
        
        return (currentState, motionType)
    }
    
    private func analyzeMotionType() -> MotionType {
        guard motionBuffer.count >= 50 else { return .unknown }
        
        // Extract recent motion data
        let recentMotion = Array(motionBuffer.suffix(50))  // Last 0.5 seconds
        
        let x_values = recentMotion.map { $0.x }
        let y_values = recentMotion.map { $0.y }
        let z_values = recentMotion.map { $0.z }
        
        // Calculate motion characteristics
        let y_activity = standardDeviation(y_values)
        let x_activity = standardDeviation(x_values)
        let z_activity = standardDeviation(z_values)
        let total_activity = y_activity + x_activity + z_activity
        
        // Calculate jerkiness (rate of change)
        let magnitudes = recentMotion.map { sqrt($0.x*$0.x + $0.y*$0.y + $0.z*$0.z) }
        let jerkiness = magnitudes.count > 1 ? standardDeviation(Array(zip(magnitudes, magnitudes.dropFirst()).map { $1 - $0 })) : 0.0
        
        // Y-axis dominance
        let y_dominance = y_activity / (x_activity + z_activity + 0.001)
        
        // Classification logic
        if total_activity < stableActivityThreshold {
            return .stable
        } else if jerkiness > handlingJerkThreshold && total_activity > 0.15 {
            return .handling
        } else if total_activity > 0.15 && y_dominance < 0.5 {
            return .rustling
        } else if y_activity > exerciseActivityThreshold && y_dominance > exerciseYDominanceThreshold {
            return .exercise
        } else {
            return .unknown
        }
    }
    
    private func standardDeviation(_ values: [Double]) -> Double {
        guard values.count > 1 else { return 0.0 }
        let mean = values.reduce(0, +) / Double(values.count)
        let squaredDifferences = values.map { ($0 - mean) * ($0 - mean) }
        let variance = squaredDifferences.reduce(0, +) / Double(values.count)
        return sqrt(variance)
    }
    
    private func startReadyCountdown() {
        countdownSeconds = 5
        
        // Start countdown timer
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] timer in
            guard let self = self else {
                timer.invalidate()
                return
            }
            
            self.countdownSeconds -= 1
            
            if self.countdownSeconds > 0 {
                self.playAudioCue(.countdown)
            } else {
                timer.invalidate()
                self.playAudioCue(.exerciseReady)
                print("🎯 EXERCISE READY - Start your workout!")
            }
        }
    }
    
    private enum AudioCue {
        case stabilizing
        case exerciseStart
        case exerciseReady
        case exerciseInterrupted
        case countdown
    }
    
    private func playAudioCue(_ cue: AudioCue) {
        let soundID: SystemSoundID
        
        switch cue {
        case .stabilizing:
            soundID = 1057  // Pop sound
        case .exerciseStart:
            soundID = 1016  // Swoosh
        case .exerciseReady:
            soundID = 1013  // Success sound
        case .exerciseInterrupted:
            soundID = 1006  // Warning sound
        case .countdown:
            soundID = 1104  // Tick sound
        }
        
        AudioServicesPlaySystemSound(soundID)
    }
    
    func reset() {
        currentState = .setup
        isReadyForExercise = false
        countdownSeconds = 0
        stableStartTime = nil
        motionBuffer.removeAll()
    }
    
    func forceStartExercise() {
        currentState = .exercising
        isReadyForExercise = true
        playAudioCue(.exerciseStart)
    }
    
    func endExercise() {
        currentState = .completed
        isReadyForExercise = false
    }
    
    // Public getters for UI
    func getMotionTypeString(_ motionType: MotionType) -> String {
        switch motionType {
        case .handling: return "Handling"
        case .rustling: return "Rustling"
        case .stable: return "Stable"
        case .exercise: return "Exercise"
        case .unknown: return "Unknown"
        }
    }
    
    func getStateString() -> String {
        switch currentState {
        case .setup: return "Setup"
        case .stabilizing: return "Stabilizing"
        case .ready: return "Ready"
        case .exercising: return "Exercising"
        case .completed: return "Completed"
        }
    }
    
    func shouldCountReps() -> Bool {
        return currentState == .exercising
    }
}
