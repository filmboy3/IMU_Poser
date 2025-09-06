import Foundation
import ZIPFoundation
import CoreMotion
import SwiftUI
import UIKit
import Combine

struct SessionMetadata: Codable {
    let sessionId: String
    let exerciseType: String
    let startTime: Date
    let endTime: Date?
    let duration: TimeInterval
    let imuSampleCount: Int
    let averageIMURate: Double
    let repCount: Int
    
    init(sessionId: String, exerciseType: String, startTime: Date, endTime: Date? = nil) {
        self.sessionId = sessionId
        self.exerciseType = exerciseType
        self.startTime = startTime
        self.endTime = endTime
        self.duration = endTime?.timeIntervalSince(startTime) ?? 0
        self.imuSampleCount = 0
        self.averageIMURate = 0.0
        self.repCount = 0
    }
    
    init(sessionId: String, exerciseType: String, startTime: Date, endTime: Date?, duration: TimeInterval, imuSampleCount: Int, averageIMURate: Double, repCount: Int) {
        self.sessionId = sessionId
        self.exerciseType = exerciseType
        self.startTime = startTime
        self.endTime = endTime
        self.duration = duration
        self.imuSampleCount = imuSampleCount
        self.averageIMURate = averageIMURate
        self.repCount = repCount
    }
}

class SessionRecorder: ObservableObject {
    static let shared = SessionRecorder()
    
    // Recording state
    @Published var isRecording = false
    @Published var currentSession: SessionMetadata?
    @Published var finalRepCount = 0
    @Published var sessionHistory: [SessionMetadata] = []
    
    // Session data storage
    private var imuDataBuffer: [IMUData] = []
    
    // Subscribers
    private var imuSubscription: AnyCancellable?
    
    private init() {
        setupDirectories()
        loadSessionHistory()
    }
    
    private func setupDirectories() {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let sessionsPath = documentsPath.appendingPathComponent("QuantumLeapSessions")
        
        do {
            try FileManager.default.createDirectory(at: sessionsPath, withIntermediateDirectories: true, attributes: nil)
        } catch {
            print("Error creating sessions directory: \(error)")
        }
    }
    
    private func loadSessionHistory() {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let historyPath = documentsPath.appendingPathComponent("QuantumLeapSessions/session_history.json")
        
        guard FileManager.default.fileExists(atPath: historyPath.path) else {
            sessionHistory = []
            return
        }
        
        do {
            let data = try Data(contentsOf: historyPath)
            sessionHistory = try JSONDecoder().decode([SessionMetadata].self, from: data)
        } catch {
            print("Error loading session history: \(error)")
            sessionHistory = []
        }
    }
    
    private func saveSessionHistory() {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let historyPath = documentsPath.appendingPathComponent("QuantumLeapSessions/session_history.json")
        
        do {
            let data = try JSONEncoder().encode(sessionHistory)
            try data.write(to: historyPath)
        } catch {
            print("Error saving session history: \(error)")
        }
    }
    
    func startRecording(exerciseType: String) {
        guard !isRecording else { return }
        
        let sessionId = generateSessionId()
        currentSession = SessionMetadata(sessionId: sessionId, exerciseType: exerciseType, startTime: Date())
        
        // Clear buffers
        imuDataBuffer.removeAll()
        finalRepCount = 0
        
        // Subscribe to IMU data
        imuSubscription = MotionManager.shared.imuDataPublisher
            .sink { [weak self] imuData in
                self?.recordIMU(data: imuData)
            }
        
        isRecording = true
        print("📱 SessionRecorder: Started recording session \(sessionId) for \(exerciseType)")
    }
    
    func stopRecording() {
        guard isRecording else { return }
        
        // Stop subscriptions
        imuSubscription?.cancel()
        imuSubscription = nil
        
        // Finalize session
        if var session = currentSession {
            let endTime = Date()
            let duration = endTime.timeIntervalSince(session.startTime)
            let averageRate = duration > 0 ? Double(imuDataBuffer.count) / duration : 0.0
            
            session = SessionMetadata(
                sessionId: session.sessionId,
                exerciseType: session.exerciseType,
                startTime: session.startTime,
                endTime: endTime,
                duration: duration,
                imuSampleCount: imuDataBuffer.count,
                averageIMURate: averageRate,
                repCount: finalRepCount
            )
            
            currentSession = session
            sessionHistory.append(session)
            saveSessionHistory()
            
            print("📱 SessionRecorder: Stopped recording. Duration: \(String(format: "%.1f", duration))s, IMU samples: \(imuDataBuffer.count), Reps: \(finalRepCount)")
        }
        
        isRecording = false
    }
    
    private func recordIMU(data: IMUData) {
        imuDataBuffer.append(data)
    }
    
    func exportSession() -> URL? {
        guard let session = currentSession else {
            print("❌ No current session to export")
            return nil
        }
        
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let sessionDir = documentsPath.appendingPathComponent("QuantumLeapSessions/\(session.sessionId)")
        
        do {
            // Create session directory
            try FileManager.default.createDirectory(at: sessionDir, withIntermediateDirectories: true, attributes: nil)
            
            // Save session metadata
            let metadataPath = sessionDir.appendingPathComponent("session_metadata.json")
            let metadataData = try JSONEncoder().encode(session)
            try metadataData.write(to: metadataPath)
            
            // Save IMU data
            let imuPath = sessionDir.appendingPathComponent("imu_data.json")
            let imuData = try JSONEncoder().encode(imuDataBuffer)
            try imuData.write(to: imuPath)
            
            // Create empty pose data file for compatibility
            let posePath = sessionDir.appendingPathComponent("pose_data.json")
            let emptyPoseData = "[]".data(using: .utf8)!
            try emptyPoseData.write(to: posePath)
            
            // Create ZIP archive
            let zipPath = documentsPath.appendingPathComponent("QuantumLeapSessions/\(session.sessionId).zip")
            try FileManager.default.zipItem(at: sessionDir, to: zipPath)
            
            print("📦 SessionRecorder: Exported session to \(zipPath.path)")
            return zipPath
            
        } catch {
            print("❌ Error exporting session: \(error)")
            return nil
        }
    }
    
    private func generateSessionId() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = formatter.string(from: Date())
        let uuid = UUID().uuidString.prefix(8)
        return "session_\(timestamp)_\(uuid)"
    }
}

// MARK: - ActivityViewController for sharing files
struct ActivityViewController: UIViewControllerRepresentable {
    let activityItems: [Any]
    
    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
        return controller
    }
    
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {
        // No updates needed
    }
}
