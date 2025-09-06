import Foundation
import CoreMotion
import Combine
import AVFoundation
import Speech
import QuartzCore

// Notification names for voice commands
extension Notification.Name {
    static let voiceStartCommand = Notification.Name("voiceStartCommand")
    static let voiceStopCommand = Notification.Name("voiceStopCommand")
}

// Custom struct to represent CMAttitude data for encoding
struct AttitudeData: Codable {
    let roll: Double
    let pitch: Double
    let yaw: Double
    let quaternionX: Double
    let quaternionY: Double
    let quaternionZ: Double
    let quaternionW: Double
    
    init(from attitude: CMAttitude) {
        self.roll = attitude.roll
        self.pitch = attitude.pitch
        self.yaw = attitude.yaw
        self.quaternionX = attitude.quaternion.x
        self.quaternionY = attitude.quaternion.y
        self.quaternionZ = attitude.quaternion.z
        self.quaternionW = attitude.quaternion.w
    }
}

struct IMUData: Codable {
    let timestamp: TimeInterval
    let acceleration: CMAcceleration
    let rotationRate: CMRotationRate
    let attitude: AttitudeData?
    
    init(timestamp: TimeInterval, acceleration: CMAcceleration, rotationRate: CMRotationRate, attitude: CMAttitude? = nil) {
        self.timestamp = timestamp
        self.acceleration = acceleration
        self.rotationRate = rotationRate
        self.attitude = attitude != nil ? AttitudeData(from: attitude!) : nil
    }
}

extension CMAcceleration: Codable {
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(x, forKey: .x)
        try container.encode(y, forKey: .y)
        try container.encode(z, forKey: .z)
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let x = try container.decode(Double.self, forKey: .x)
        let y = try container.decode(Double.self, forKey: .y)
        let z = try container.decode(Double.self, forKey: .z)
        self.init(x: x, y: y, z: z)
    }
    
    private enum CodingKeys: String, CodingKey {
        case x, y, z
    }
}

extension CMRotationRate: Codable {
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(x, forKey: .x)
        try container.encode(y, forKey: .y)
        try container.encode(z, forKey: .z)
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let x = try container.decode(Double.self, forKey: .x)
        let y = try container.decode(Double.self, forKey: .y)
        let z = try container.decode(Double.self, forKey: .z)
        self.init(x: x, y: y, z: z)
    }
    
    private enum CodingKeys: String, CodingKey {
        case x, y, z
    }
}

class MotionManager: ObservableObject {
    static let shared = MotionManager()
    private let motionManager = CMMotionManager()
    
    private let imuDataSubject = PassthroughSubject<IMUData, Never>()
    var imuDataPublisher: AnyPublisher<IMUData, Never> {
        imuDataSubject.eraseToAnyPublisher()
    }
    
    @Published var isActive = false
    @Published var dataRate: Double = 0.0
    
    // Smart exercise detection
    private let smartDetector = SmartExerciseDetector()
    private let repCounter = RepCounter()
    
    @Published var exerciseState: SmartExerciseDetector.ExerciseState = .setup
    @Published var currentMotionType: SmartExerciseDetector.MotionType = .unknown
    @Published var repCount: Int = 0
    @Published var smoothedAcceleration: Double = 0.0
    
    // Fallback mechanism
    private var motionStartTime: Date?
    
    private var detectorUpdateCounter = 0
    private let detectorUpdateInterval = 10  // Process detector every 10th sample (10Hz instead of 100Hz)
    
    // Voice control integration
    private let voiceController = VoiceController()
    @Published var isVoiceControlActive = false
    @Published var voiceRecognizedText = ""
    
    // AI Voice Coach integration
    private let aiCoach = AIVoiceCoach.shared
    @Published var isCoachingActive = false
    
    private var lastUpdateTime: TimeInterval = 0
    private var updateCount = 0
    
    private init() {
        // Configure for high-frequency updates
        guard motionManager.isDeviceMotionAvailable else {
            print("Error: Device motion is not available.")
            return
        }
        motionManager.deviceMotionUpdateInterval = 1.0 / 100.0 // 100Hz
        
        // Setup voice control callbacks
        setupVoiceControl()
    }
    
    private func setupVoiceControl() {
        voiceController.onStartCommand = { [weak self] in
            guard let self = self else { return }
            print("🎤 Voice START command received")
            // Initialize rep counter session with stabilization period
            self.repCounter.startSession()
            // Start AI Voice Coach
            self.aiCoach.startCoaching()
            DispatchQueue.main.async {
                self.isCoachingActive = true
            }
            // Start session recording automatically
            DispatchQueue.main.async {
                NotificationCenter.default.post(name: .voiceStartCommand, object: nil)
            }
        }
        
        voiceController.onStopCommand = { [weak self] in
            guard let self = self else { return }
            print("🎤 Voice STOP command received")
            // Stop AI Voice Coach
            self.aiCoach.stopCoaching()
            DispatchQueue.main.async {
                self.isCoachingActive = false
            }
            // Stop session and trigger export
            DispatchQueue.main.async {
                NotificationCenter.default.post(name: .voiceStopCommand, object: nil)
            }
            // Reset rep counter for next session
            self.repCounter.reset()
        }
    }
    
    func startUpdates() {
        guard !isActive else { return }
        
        // Use a background queue for performance
        let queue = OperationQueue()
        queue.name = "com.quantumleap.CoreMotionQueue"
        queue.qualityOfService = .userInitiated
        
        motionManager.startDeviceMotionUpdates(to: queue) { [weak self] (motion, error) in
            guard let self = self, let motion = motion else {
                if let error = error {
                    print("Motion update error: \(error)")
                }
                return
            }
            
            // Skip processing if not needed to reduce CPU load
            guard self.voiceController.isRepCountingActive || Date().timeIntervalSince(self.motionStartTime ?? Date()) > 2.0 else {
                return
            }
            
            let data = IMUData(
                timestamp: motion.timestamp,
                acceleration: motion.userAcceleration,
                rotationRate: motion.rotationRate,
                attitude: motion.attitude
            )

            // Re-enable smart detection for robust gating
            let (state, motionType) = self.smartDetector.processMotion(motion.userAcceleration)
            var detectedState: SmartExerciseDetector.ExerciseState = state
            let detectedMotionType: SmartExerciseDetector.MotionType = motionType

            // Only count reps if voice control has activated rep counting AND detector says exercising
            var currentRepCount = self.repCount
            if self.voiceController.isRepCountingActive && state == .exercising {
                let previousRepCount = self.repCount
                currentRepCount = self.repCounter.process(acceleration: motion.userAcceleration)

                if currentRepCount > previousRepCount {
                    self.aiCoach.onRepDetected(repNumber: currentRepCount)
                    AudioServicesPlaySystemSound(1057) // Bell sound
                }
            }
            
            // Update data rate calculation
            self.updateDataRate()
            
            // Publish data on main thread for UI updates
            DispatchQueue.main.async {
                self.imuDataSubject.send(data)
                self.exerciseState = detectedState
                self.currentMotionType = detectedMotionType
                self.repCount = currentRepCount
                self.smoothedAcceleration = self.repCounter.getSmoothedY()
                self.isVoiceControlActive = self.voiceController.isListening
                self.voiceRecognizedText = self.voiceController.recognizedText
            }
        }
        
        DispatchQueue.main.async {
            self.isActive = true
        }
        
        print("MotionManager: Started IMU updates at 100Hz")
    }
    
    func stopUpdates() {
        motionManager.stopDeviceMotionUpdates()
        DispatchQueue.main.async {
            self.isActive = false
        }
        print("MotionManager: Exercise session reset")
    }
    
    func startVoiceControl() {
        voiceController.startListening()
    }
    
    func stopVoiceControl() {
        voiceController.stopListening()
    }
    
    private func updateDataRate() {
        updateCount += 1
        let currentTime = CACurrentMediaTime()
        
        if lastUpdateTime == 0 {
            lastUpdateTime = currentTime
            return
        }
        
        let timeDelta = currentTime - lastUpdateTime
        if timeDelta >= 1.0 { // Update rate every second
            let rate = Double(updateCount) / timeDelta
            
            DispatchQueue.main.async {
                self.dataRate = rate
            }
            
            updateCount = 0
            lastUpdateTime = currentTime
        }
    }
    
    var isDeviceMotionAvailable: Bool {
        return motionManager.isDeviceMotionAvailable
    }
    
    func resetExerciseSession() {
        repCounter.reset()
        DispatchQueue.main.async {
            self.repCount = 0
        }
        print("MotionManager: Exercise session reset")
    }
}

// MARK: - Rep Counter Class

class RepCounter {
    enum State {
        case neutral
        case descending
        case ascending
    }
    
    private var currentState: State = .neutral
    private var repCount = 0
    
    // Robust thresholds based on failure analysis
    private let peakThreshold: Double = 0.12    // Lowered from 0.16
    private let valleyThreshold: Double = -0.10  // Raised from -0.14
    private let neutralThreshold: Double = 0.06  // Lowered from 0.08
    
    // Dual detection system
    private var buffer: [Double] = []
    private let bufferSize = 8  // Reduced for faster response
    private var smoothedY: Double = 0.0
    
    // State timeout to prevent getting stuck
    private var stateStartTime: Date = Date()
    private let maxStateTime: TimeInterval = 20.0  // Increased to 20 seconds for natural rep pacing
    
    // Rep validation
    private var lastRepTime: Date = Date()
    private let minRepDuration: TimeInterval = 0.4  // Reduced to 0.4 seconds per rep
    private var sessionStartTime: Date?
    private let startupStabilizationTime: TimeInterval = 1.0  // Reduced to 1 second stabilization
    
    // Peak/valley detection for backup counting
    private var lastPeakTime: Date = Date()
    private var lastValleyTime: Date = Date()
    private var lastPeakValue: Double = 0.0
    private var lastValleyValue: Double = 0.0
    
    // Motion activity tracking
    private var recentValues: [Double] = []
    private let activityWindowSize = 50  // ~0.5 seconds at 100Hz

    func process(acceleration: CMAcceleration) -> Int {
        let verticalAcceleration = acceleration.y
        let currentTime = Date()
        
        // Update buffers
        buffer.append(verticalAcceleration)
        if buffer.count > bufferSize {
            buffer.removeFirst()
        }
        
        recentValues.append(verticalAcceleration)
        if recentValues.count > activityWindowSize {
            recentValues.removeFirst()
        }
        
        smoothedY = buffer.reduce(0, +) / Double(buffer.count)
        
        // Check for state timeout (prevents getting stuck)
        if currentTime.timeIntervalSince(stateStartTime) > maxStateTime {
            print("⚠️ State timeout - resetting to neutral")
            currentState = .neutral
            stateStartTime = currentTime
        }
        
        // Primary state machine
        
        switch currentState {
        case .neutral:
            // Skip detection during startup stabilization period
            if let startTime = sessionStartTime,
               currentTime.timeIntervalSince(startTime) < startupStabilizationTime {
                return repCount
            }
            
            if smoothedY < valleyThreshold {
                currentState = .descending
                stateStartTime = currentTime
                lastValleyTime = currentTime
                lastValleyValue = smoothedY
                print("🔽 Descending: \(String(format: "%.3f", smoothedY))")
            }
            
        case .descending:
            // Update valley if we go lower
            if smoothedY < lastValleyValue {
                lastValleyValue = smoothedY
                lastValleyTime = currentTime
            }
            
            if smoothedY > peakThreshold {
                currentState = .ascending
                stateStartTime = currentTime
                lastPeakTime = currentTime
                lastPeakValue = smoothedY
                print("🔼 Ascending: \(String(format: "%.3f", smoothedY))")
            }
            
        case .ascending:
            // Update peak if we go higher
            if smoothedY > lastPeakValue {
                lastPeakValue = smoothedY
                lastPeakTime = currentTime
            }
            
            if abs(smoothedY) < neutralThreshold {
                let repDuration = currentTime.timeIntervalSince(lastValleyTime)
                let timeSinceLastRep = currentTime.timeIntervalSince(lastRepTime)
                
                // Validate rep: minimum duration and time between reps
                if repDuration >= minRepDuration && timeSinceLastRep >= minRepDuration {
                    repCount += 1
                    currentState = .neutral
                    stateStartTime = currentTime
                    lastRepTime = currentTime
                    
                    print("✅ Rep #\(repCount) completed! Duration: \(String(format: "%.1f", repDuration))s")
                    
                    // Audio feedback
                    AudioServicesPlaySystemSound(1057)
                } else {
                    // Invalid rep - too fast, reset to neutral without counting
                    print("⚠️ Rep too fast (\(String(format: "%.1f", repDuration))s) - not counting")
                    currentState = .neutral
                    stateStartTime = currentTime
                }
            }
        }
        
        // Backup detection: Peak-valley analysis
        if buffer.count == bufferSize {
            detectBackupReps(currentTime: currentTime)
        }
        
        return repCount
    }
    
    func startSession() {
        sessionStartTime = Date()
        print("🏁 Rep counter session started - 1s stabilization period")
    }
    
    func reset() {
        currentState = .neutral
        repCount = 0
        sessionStartTime = nil
        lastRepTime = Date()
        stateStartTime = Date()
        buffer.removeAll()
        recentValues.removeAll()
        print("🔄 Rep counter reset")
    }
    
    private func detectBackupReps(currentTime: Date) {
        // Calculate motion activity
        let activity = recentValues.isEmpty ? 0.0 : 
            sqrt(recentValues.map { $0 * $0 }.reduce(0, +) / Double(recentValues.count))
        
        // If primary algorithm hasn't detected a rep in a while but there's activity
        let timeSinceLastRep = currentTime.timeIntervalSince(lastPeakTime)
        
        if timeSinceLastRep > 8.0 && activity > 0.08 {  // 8 seconds without rep but motion detected
            // Look for significant motion patterns
            let recentStd = recentValues.isEmpty ? 0.0 : 
                sqrt(recentValues.map { val in 
                    let mean = recentValues.reduce(0, +) / Double(recentValues.count)
                    return (val - mean) * (val - mean) 
                }.reduce(0, +) / Double(recentValues.count))
            
            if recentStd > 0.05 {  // Significant motion variation
                repCount += 1
                lastPeakTime = currentTime  // Reset timer
                print("🔄 Backup rep #\(repCount) detected (activity: \(String(format: "%.3f", activity)))")
                AudioServicesPlaySystemSound(1057)
            }
        }
    }
    
    func getSmoothedY() -> Double {
        return smoothedY
    }
}
