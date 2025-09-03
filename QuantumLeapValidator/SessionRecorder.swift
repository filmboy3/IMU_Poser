import Foundation
import AVFoundation
import Combine
import CoreMotion

struct SessionMetadata: Codable {
    let sessionId: String
    let exerciseType: String
    let startTime: Date
    let endTime: Date?
    let duration: TimeInterval
    let imuSampleCount: Int
    let videoFrameCount: Int
    let averageIMURate: Double
    let averageVideoFPS: Double
    
    init(sessionId: String, exerciseType: String, startTime: Date, endTime: Date? = nil) {
        self.sessionId = sessionId
        self.exerciseType = exerciseType
        self.startTime = startTime
        self.endTime = endTime
        self.duration = endTime?.timeIntervalSince(startTime) ?? 0
        self.imuSampleCount = 0
        self.videoFrameCount = 0
        self.averageIMURate = 0
        self.averageVideoFPS = 0
    }
}

class SessionRecorder: ObservableObject {
    static let shared = SessionRecorder()
    
    // Recording state
    @Published var isRecording = false
    @Published var currentSession: SessionMetadata?
    
    // Data storage
    private var imuDataBuffer: [IMUData] = []
    private var videoFrameTimestamps: [TimeInterval] = []
    private var poseDataBuffer: [[String: [String: Any]]] = []
    
    // File management
    private var sessionDirectory: URL?
    private var videoWriter: AVAssetWriter?
    private var videoWriterInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    
    // Timing and synchronization
    private var sessionStartTime: Date?
    private var recordingStartTime: TimeInterval?
    
    // IMU data subscription
    private var imuSubscription: AnyCancellable?
    
    private init() {
        setupDocumentsDirectory()
    }
    
    private func setupDocumentsDirectory() {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let sessionsPath = documentsPath.appendingPathComponent("QuantumLeapSessions")
        
        try? FileManager.default.createDirectory(at: sessionsPath, withIntermediateDirectories: true)
    }
    
    // MARK: - Recording Control
    
    func startRecording(exerciseType: String) {
        guard !isRecording else { return }
        
        let sessionId = generateSessionId()
        let startTime = Date()
        
        // Create session metadata
        currentSession = SessionMetadata(
            sessionId: sessionId,
            exerciseType: exerciseType,
            startTime: startTime
        )
        
        // Create session directory
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        sessionDirectory = documentsPath
            .appendingPathComponent("QuantumLeapSessions")
            .appendingPathComponent(sessionId)
        
        do {
            try FileManager.default.createDirectory(at: sessionDirectory!, withIntermediateDirectories: true)
        } catch {
            print("Error creating session directory: \(error)")
            return
        }
        
        // Initialize data buffers
        imuDataBuffer.removeAll()
        videoFrameTimestamps.removeAll()
        poseDataBuffer.removeAll()
        
        // Setup video recording
        setupVideoRecording()
        
        // Start IMU data collection
        startIMUCollection()
        
        // Update state
        sessionStartTime = startTime
        recordingStartTime = CACurrentMediaTime()
        isRecording = true
        
        print("SessionRecorder: Started recording session \(sessionId) for \(exerciseType)")
    }
    
    func stopRecording() {
        guard isRecording else { return }
        
        let endTime = Date()
        
        // Stop IMU collection
        stopIMUCollection()
        
        // Finalize video recording
        finalizeVideoRecording()
        
        // Update session metadata
        if var session = currentSession {
            session = SessionMetadata(
                sessionId: session.sessionId,
                exerciseType: session.exerciseType,
                startTime: session.startTime,
                endTime: endTime
            )
            currentSession = session
        }
        
        // Save session data
        saveSessionData()
        
        // Update state
        isRecording = false
        
        print("SessionRecorder: Stopped recording session")
    }
    
    // MARK: - Video Recording Setup
    
    private func setupVideoRecording() {
        guard let sessionDirectory = sessionDirectory else { return }
        
        let videoURL = sessionDirectory.appendingPathComponent("video.mp4")
        
        do {
            videoWriter = try AVAssetWriter(outputURL: videoURL, fileType: .mp4)
            
            let videoSettings: [String: Any] = [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: 1080,
                AVVideoHeightKey: 1920,
                AVVideoCompressionPropertiesKey: [
                    AVVideoAverageBitRateKey: 2000000,
                    AVVideoMaxKeyFrameIntervalKey: 30
                ]
            ]
            
            videoWriterInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
            videoWriterInput?.expectsMediaDataInRealTime = true
            
            let pixelBufferAttributes: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
                kCVPixelBufferWidthKey as String: 1080,
                kCVPixelBufferHeightKey as String: 1920
            ]
            
            pixelBufferAdaptor = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: videoWriterInput!,
                sourcePixelBufferAttributes: pixelBufferAttributes
            )
            
            if let videoWriterInput = videoWriterInput,
               videoWriter!.canAdd(videoWriterInput) {
                videoWriter!.add(videoWriterInput)
            }
            
            videoWriter!.startWriting()
            videoWriter!.startSession(atSourceTime: .zero)
            
        } catch {
            print("Error setting up video recording: \(error)")
        }
    }
    
    private func finalizeVideoRecording() {
        guard let videoWriter = videoWriter else { return }
        
        videoWriterInput?.markAsFinished()
        
        videoWriter.finishWriting { [weak self] in
            DispatchQueue.main.async {
                if videoWriter.status == .completed {
                    print("Video recording completed successfully")
                } else if let error = videoWriter.error {
                    print("Video recording failed: \(error)")
                }
                
                self?.videoWriter = nil
                self?.videoWriterInput = nil
                self?.pixelBufferAdaptor = nil
            }
        }
    }
    
    // MARK: - IMU Data Collection
    
    private func startIMUCollection() {
        // Start motion manager if not already running
        if !MotionManager.shared.isActive {
            MotionManager.shared.startUpdates()
        }
        
        // Subscribe to IMU data
        imuSubscription = MotionManager.shared.imuDataPublisher
            .sink { [weak self] imuData in
                self?.processIMUData(imuData)
            }
    }
    
    private func stopIMUCollection() {
        imuSubscription?.cancel()
        imuSubscription = nil
    }
    
    private func processIMUData(_ data: IMUData) {
        guard isRecording else { return }
        imuDataBuffer.append(data)
    }
    
    // MARK: - Video Frame Processing
    
    func processSampleBuffer(_ sampleBuffer: CMSampleBuffer) {
        guard isRecording,
              let _ = videoWriter,
              let videoWriterInput = videoWriterInput,
              let pixelBufferAdaptor = pixelBufferAdaptor else { return }
        
        guard videoWriterInput.isReadyForMoreMediaData else { return }
        
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        videoFrameTimestamps.append(CMTimeGetSeconds(timestamp))
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let frameTime = CMTime(seconds: CMTimeGetSeconds(timestamp), preferredTimescale: 600)
        pixelBufferAdaptor.append(pixelBuffer, withPresentationTime: frameTime)
    }
    
    // MARK: - Pose Data Processing
    
    func processPoseData(_ poseData: [String: [String: Any]]) {
        guard isRecording else { return }
        poseDataBuffer.append(poseData)
    }
    
    // MARK: - Data Persistence
    
    private func saveSessionData() {
        guard let sessionDirectory = sessionDirectory,
              let session = currentSession else { return }
        
        // Save IMU data as JSON
        saveIMUData(to: sessionDirectory)
        
        // Save pose data as JSON
        savePoseData(to: sessionDirectory)
        
        // Save session metadata
        saveSessionMetadata(session, to: sessionDirectory)
        
        print("SessionRecorder: Saved session data to \(sessionDirectory.path)")
    }
    
    private func saveIMUData(to directory: URL) {
        let imuURL = directory.appendingPathComponent("imu_data.json")
        
        do {
            let jsonData = try JSONEncoder().encode(imuDataBuffer)
            try jsonData.write(to: imuURL)
            print("Saved \(imuDataBuffer.count) IMU samples")
        } catch {
            print("Error saving IMU data: \(error)")
        }
    }
    
    private func savePoseData(to directory: URL) {
        let poseURL = directory.appendingPathComponent("pose_data.json")
        
        do {
            let jsonData = try JSONSerialization.data(withJSONObject: poseDataBuffer, options: .prettyPrinted)
            try jsonData.write(to: poseURL)
            print("Saved \(poseDataBuffer.count) pose frames")
        } catch {
            print("Error saving pose data: \(error)")
        }
    }
    
    private func saveSessionMetadata(_ session: SessionMetadata, to directory: URL) {
        let metadataURL = directory.appendingPathComponent("session_metadata.json")
        
        // Create updated metadata with actual counts
        let updatedMetadata = SessionMetadata(
            sessionId: session.sessionId,
            exerciseType: session.exerciseType,
            startTime: session.startTime,
            endTime: session.endTime
        )
        
        do {
            let jsonData = try JSONEncoder().encode(updatedMetadata)
            try jsonData.write(to: metadataURL)
        } catch {
            print("Error saving session metadata: \(error)")
        }
    }
    
    // MARK: - Session Management
    
    func exportSession() -> URL? {
        guard let sessionDirectory = sessionDirectory,
              let session = currentSession else { return nil }
        
        let zipURL = sessionDirectory.appendingPathComponent("\(session.sessionId).zip")
        
        // Create zip file with all session data
        do {
            try createZipArchive(from: sessionDirectory, to: zipURL)
            return zipURL
        } catch {
            print("Error creating zip archive: \(error)")
            return nil
        }
    }
    
    private func createZipArchive(from sourceDirectory: URL, to destinationURL: URL) throws {
        // Use FileManager for iOS-compatible zip creation
        let fileManager = FileManager.default
        
        // For iOS, we'll create a simple archive by copying files
        // In a production app, consider using a third-party zip library like ZIPFoundation
        
        // Create destination directory if needed
        let destinationDir = destinationURL.deletingLastPathComponent()
        try fileManager.createDirectory(at: destinationDir, withIntermediateDirectories: true, attributes: nil)
        
        // For now, just copy the directory contents to a .zip extension
        // This is a simplified approach - in production use proper zip compression
        if fileManager.fileExists(atPath: destinationURL.path) {
            try fileManager.removeItem(at: destinationURL)
        }
        
        try fileManager.copyItem(at: sourceDirectory, to: destinationURL)
        
        print("Session exported to: \(destinationURL.path)")
    }
    
    func getAllSessions() -> [SessionMetadata] {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let sessionsPath = documentsPath.appendingPathComponent("QuantumLeapSessions")
        
        var sessions: [SessionMetadata] = []
        
        do {
            let sessionDirectories = try FileManager.default.contentsOfDirectory(at: sessionsPath, includingPropertiesForKeys: nil)
            
            for directory in sessionDirectories {
                let metadataURL = directory.appendingPathComponent("session_metadata.json")
                
                if let data = try? Data(contentsOf: metadataURL),
                   let session = try? JSONDecoder().decode(SessionMetadata.self, from: data) {
                    sessions.append(session)
                }
            }
        } catch {
            print("Error loading sessions: \(error)")
        }
        
        return sessions.sorted { $0.startTime > $1.startTime }
    }
    
    private func generateSessionId() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = formatter.string(from: Date())
        let uuid = UUID().uuidString.prefix(8)
        return "session_\(timestamp)_\(uuid)"
    }
}
