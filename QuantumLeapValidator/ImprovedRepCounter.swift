import CoreMotion
import Foundation

class ImprovedRepCounter {
    enum State {
        case neutral
        case descending
        case ascending
    }
    
    private var currentState: State = .neutral
    private var repCount = 0
    
    // More robust thresholds based on analysis
    private let peakThreshold: Double = 0.12    // Lowered from 0.16
    private let valleyThreshold: Double = -0.10  // Raised from -0.14
    private let neutralThreshold: Double = 0.06  // Lowered from 0.08
    
    // Dual detection system
    private var buffer: [Double] = []
    private let bufferSize = 8  // Reduced for faster response
    private var smoothedY: Double = 0.0
    
    // State timeout to prevent getting stuck
    private var stateStartTime: Date = Date()
    private let maxStateTime: TimeInterval = 5.0  // Max 5 seconds in any state
    
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
        let previousState = currentState
        
        switch currentState {
        case .neutral:
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
                // Valid rep completed
                repCount += 1
                currentState = .neutral
                stateStartTime = currentTime
                
                let repDuration = currentTime.timeIntervalSince(lastValleyTime)
                print("✅ Rep #\(repCount) completed! Duration: \(String(format: "%.1f", repDuration))s")
                
                // Audio feedback
                AudioServicesPlaySystemSound(1057)
            }
        }
        
        // Backup detection: Peak-valley analysis
        if buffer.count == bufferSize {
            detectBackupReps(currentTime: currentTime)
        }
        
        return repCount
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
    
    func reset() {
        repCount = 0
        currentState = .neutral
        buffer.removeAll()
        recentValues.removeAll()
        smoothedY = 0.0
        stateStartTime = Date()
        lastPeakTime = Date()
        lastValleyTime = Date()
    }
    
    func getSmoothedY() -> Double {
        return smoothedY
    }
    
    func getCurrentState() -> String {
        switch currentState {
        case .neutral: return "Neutral"
        case .descending: return "Descending"
        case .ascending: return "Ascending"
        }
    }
    
    func getActivityLevel() -> Double {
        guard !recentValues.isEmpty else { return 0.0 }
        let mean = recentValues.reduce(0, +) / Double(recentValues.count)
        let variance = recentValues.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(recentValues.count)
        return sqrt(variance)
    }
}
