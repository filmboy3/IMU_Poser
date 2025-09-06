import Foundation
import CoreMotion

/**
 * Comprehensive Testing Framework for Rep Counter Accuracy
 * 
 * This framework provides both automated code tests and human benchmark protocols
 * to validate rep counting accuracy and eliminate "gut feel" debugging.
 */

class RepCounterTestFramework {
    
    // MARK: - Test Configuration
    struct TestConfig {
        let expectedReps: Int
        let exerciseType: String
        let testDuration: TimeInterval
        let tolerance: Double = 0.1 // 10% tolerance for accuracy
    }
    
    // MARK: - Test Results
    struct TestResult {
        let testName: String
        let expectedReps: Int
        let detectedReps: Int
        let accuracy: Double
        let latency: TimeInterval
        let falsePositives: Int
        let falseNegatives: Int
        let passed: Bool
        
        var accuracyPercentage: Double {
            return accuracy * 100.0
        }
    }
    
    // MARK: - Human Benchmark Protocol
    
    /**
     * Human Benchmark Test Protocol
     * 
     * This provides a structured approach to validate rep counting accuracy
     * with human-performed exercises, eliminating subjective "gut feel" assessment.
     */
    static func createHumanBenchmarkProtocol() -> [TestConfig] {
        return [
            // Slow, controlled reps (2 seconds per rep)
            TestConfig(expectedReps: 5, exerciseType: "Squat_Slow", testDuration: 10.0),
            
            // Normal pace reps (1 second per rep)
            TestConfig(expectedReps: 10, exerciseType: "Squat_Normal", testDuration: 10.0),
            
            // Fast reps (0.5 seconds per rep)
            TestConfig(expectedReps: 20, exerciseType: "Squat_Fast", testDuration: 10.0),
            
            // Mixed pace with pauses
            TestConfig(expectedReps: 8, exerciseType: "Squat_Mixed", testDuration: 15.0),
            
            // Partial range of motion (should NOT count)
            TestConfig(expectedReps: 0, exerciseType: "Squat_Partial", testDuration: 10.0),
            
            // Phone handling test (should NOT count)
            TestConfig(expectedReps: 0, exerciseType: "Phone_Handling", testDuration: 5.0)
        ]
    }
    
    // MARK: - Automated Code Tests
    
    /**
     * Synthetic IMU Data Generator
     * 
     * Generates realistic IMU data patterns for automated testing
     */
    class SyntheticIMUGenerator {
        
        static func generateSquatPattern(repCount: Int, repDuration: Double = 1.0) -> [IMUData] {
            var data: [IMUData] = []
            let sampleRate = 100.0 // 100Hz
            let samplesPerRep = Int(repDuration * sampleRate)
            
            for rep in 0..<repCount {
                let repStartTime = Double(rep) * repDuration
                
                // Generate one complete squat cycle
                for sample in 0..<samplesPerRep {
                    let t = Double(sample) / sampleRate
                    let phase = (t / repDuration) * 2.0 * Double.pi
                    
                    // Simulate squat motion: down (negative Y) then up (positive Y)
                    let yAccel = -0.15 * sin(phase) // Peak at -0.15g going down, +0.15g going up
                    let xAccel = 0.02 * cos(phase * 2) // Small lateral movement
                    let zAccel = 0.01 * sin(phase * 3) // Minimal forward/back
                    
                    let acceleration = CMAcceleration(x: xAccel, y: yAccel, z: zAccel)
                    let rotationRate = CMRotationRate(x: 0, y: 0, z: 0)
                    
                    let imuData = IMUData(
                        timestamp: repStartTime + t,
                        acceleration: acceleration,
                        rotationRate: rotationRate,
                        attitude: nil
                    )
                    
                    data.append(imuData)
                }
            }
            
            return data
        }
        
        static func generateNoisePattern(duration: Double) -> [IMUData] {
            var data: [IMUData] = []
            let sampleRate = 100.0
            let sampleCount = Int(duration * sampleRate)
            
            for sample in 0..<sampleCount {
                let t = Double(sample) / sampleRate
                
                // Generate random noise within normal phone handling range
                let xAccel = Double.random(in: -0.05...0.05)
                let yAccel = Double.random(in: -0.05...0.05)
                let zAccel = Double.random(in: -0.05...0.05)
                
                let acceleration = CMAcceleration(x: xAccel, y: yAccel, z: zAccel)
                let rotationRate = CMRotationRate(x: 0, y: 0, z: 0)
                
                let imuData = IMUData(
                    timestamp: t,
                    acceleration: acceleration,
                    rotationRate: rotationRate,
                    attitude: nil
                )
                
                data.append(imuData)
            }
            
            return data
        }
    }
    
    // MARK: - Test Execution
    
    /**
     * Run Automated Tests
     * 
     * Executes a suite of automated tests using synthetic IMU data
     */
    static func runAutomatedTests() -> [TestResult] {
        var results: [TestResult] = []
        
        // Test 1: Perfect 10 reps
        let perfectReps = SyntheticIMUGenerator.generateSquatPattern(repCount: 10, repDuration: 1.0)
        let result1 = testRepCounter(
            data: perfectReps,
            config: TestConfig(expectedReps: 10, exerciseType: "Perfect_Squats", testDuration: 10.0)
        )
        results.append(result1)
        
        // Test 2: Fast reps
        let fastReps = SyntheticIMUGenerator.generateSquatPattern(repCount: 20, repDuration: 0.5)
        let result2 = testRepCounter(
            data: fastReps,
            config: TestConfig(expectedReps: 20, exerciseType: "Fast_Squats", testDuration: 10.0)
        )
        results.append(result2)
        
        // Test 3: Noise only (should detect 0 reps)
        let noiseOnly = SyntheticIMUGenerator.generateNoisePattern(duration: 5.0)
        let result3 = testRepCounter(
            data: noiseOnly,
            config: TestConfig(expectedReps: 0, exerciseType: "Noise_Only", testDuration: 5.0)
        )
        results.append(result3)
        
        return results
    }
    
    /**
     * Test Rep Counter with Synthetic Data
     */
    private static func testRepCounter(data: [IMUData], config: TestConfig) -> TestResult {
        let repCounter = RepCounter()
        var detectedReps = 0
        let startTime = Date()
        
        // Process each IMU sample
        for sample in data {
            let newRepCount = repCounter.process(acceleration: sample.acceleration)
            if newRepCount > detectedReps {
                detectedReps = newRepCount
            }
        }
        
        let endTime = Date()
        let latency = endTime.timeIntervalSince(startTime)
        
        // Calculate accuracy metrics
        let accuracy = config.expectedReps > 0 ? 
            Double(detectedReps) / Double(config.expectedReps) : 
            (detectedReps == 0 ? 1.0 : 0.0)
        
        let falsePositives = max(0, detectedReps - config.expectedReps)
        let falseNegatives = max(0, config.expectedReps - detectedReps)
        
        let passed = abs(accuracy - 1.0) <= config.tolerance
        
        return TestResult(
            testName: config.exerciseType,
            expectedReps: config.expectedReps,
            detectedReps: detectedReps,
            accuracy: accuracy,
            latency: latency,
            falsePositives: falsePositives,
            falseNegatives: falseNegatives,
            passed: passed
        )
    }
    
    // MARK: - Performance Benchmarks
    
    /**
     * Latency Benchmark
     * 
     * Measures the time between rep completion and detection
     */
    static func measureRepDetectionLatency() -> TimeInterval {
        let repCounter = RepCounter()
        let startTime = Date()
        
        // Simulate a single rep with clear peak
        let peakAcceleration = CMAcceleration(x: 0, y: 0.2, z: 0) // Strong upward motion
        _ = repCounter.process(acceleration: peakAcceleration)
        
        let endTime = Date()
        return endTime.timeIntervalSince(startTime)
    }
    
    /**
     * Memory Usage Benchmark
     */
    static func measureMemoryUsage() -> Int {
        let repCounter = RepCounter()
        let initialMemory = getMemoryUsage()
        
        // Process 1000 samples
        for _ in 0..<1000 {
            let acceleration = CMAcceleration(x: 0.1, y: 0.1, z: 0.1)
            _ = repCounter.process(acceleration: acceleration)
        }
        
        let finalMemory = getMemoryUsage()
        return finalMemory - initialMemory
    }
    
    private static func getMemoryUsage() -> Int {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        if kerr == KERN_SUCCESS {
            return Int(info.resident_size)
        } else {
            return 0
        }
    }
    
    // MARK: - Test Reporting
    
    /**
     * Generate Test Report
     */
    static func generateTestReport(results: [TestResult]) -> String {
        var report = """
        # Rep Counter Test Report
        Generated: \(Date())
        
        ## Summary
        """
        
        let passedTests = results.filter { $0.passed }.count
        let totalTests = results.count
        let overallAccuracy = results.map { $0.accuracy }.reduce(0, +) / Double(results.count)
        
        report += """
        
        - Tests Passed: \(passedTests)/\(totalTests)
        - Overall Accuracy: \(String(format: "%.1f", overallAccuracy * 100))%
        - Average Latency: \(String(format: "%.3f", results.map { $0.latency }.reduce(0, +) / Double(results.count)))ms
        
        ## Detailed Results
        
        """
        
        for result in results {
            let status = result.passed ? "✅ PASS" : "❌ FAIL"
            report += """
            ### \(result.testName) \(status)
            - Expected: \(result.expectedReps) reps
            - Detected: \(result.detectedReps) reps
            - Accuracy: \(String(format: "%.1f", result.accuracyPercentage))%
            - Latency: \(String(format: "%.3f", result.latency * 1000))ms
            - False Positives: \(result.falsePositives)
            - False Negatives: \(result.falseNegatives)
            
            """
        }
        
        return report
    }
}

// MARK: - Human Test Protocol Instructions

/**
 * HUMAN BENCHMARK TEST PROTOCOL
 * 
 * Follow these exact steps to eliminate subjective assessment:
 * 
 * 1. PREPARATION:
 *    - Use a metronome app for consistent timing
 *    - Record yourself performing the exercises
 *    - Count reps out loud in the video
 * 
 * 2. TEST EXECUTION:
 *    - Start voice control: Say "GO"
 *    - Perform exactly the specified number of reps
 *    - Maintain consistent form and timing
 *    - Say "STOP" immediately after last rep
 * 
 * 3. VALIDATION:
 *    - Compare detected reps vs. your counted reps
 *    - Review video to confirm your count was accurate
 *    - Note any discrepancies and timing issues
 * 
 * 4. TEST SCENARIOS:
 *    a) 5 Slow Squats (2 seconds per rep, 10 second total)
 *    b) 10 Normal Squats (1 second per rep, 10 second total)  
 *    c) 20 Fast Squats (0.5 seconds per rep, 10 second total)
 *    d) 8 Mixed Pace Squats with 2-second pauses between sets
 *    e) 10 Partial Squats (should detect 0 - validation test)
 *    f) 5 seconds of phone handling (should detect 0)
 * 
 * 5. SUCCESS CRITERIA:
 *    - Accuracy within 10% of expected count
 *    - No false positives during handling test
 *    - Consistent detection across different speeds
 *    - Rep detection latency < 200ms after completion
 */
