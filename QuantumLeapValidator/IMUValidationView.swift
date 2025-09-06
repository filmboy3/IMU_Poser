import SwiftUI

struct IMUValidationView: View {
    let selectedExercise: String
    @State private var isRecording = false
    @State private var showingExportSheet = false
    @State private var exportURL: URL?
    @State private var showingSessionSummary = false
    @Environment(\.dismiss) private var dismiss
    @StateObject private var sessionRecorder = SessionRecorder.shared
    @StateObject private var motionManager = MotionManager.shared
    
    // Real-time acceleration display
    @State private var currentAcceleration = (x: 0.0, y: 0.0, z: 0.0)
    
    // Computed properties for UI
    private var stateColor: Color {
        switch motionManager.exerciseState {
        case .setup: return .orange
        case .stabilizing: return .yellow
        case .ready: return .green
        case .exercising: return .blue
        case .completed: return .purple
        }
    }
    
    private var motionTypeDescription: String {
        switch motionManager.currentMotionType {
        case .stable: return "Stable"
        case .handling: return "Phone Handling"
        case .rustling: return "Rustling"
        case .exercise: return "Exercise Motion"
        case .unknown: return "Unknown"
        }
    }
    
    var body: some View {
        ZStack {
            // Dark background
            Color.black.ignoresSafeArea()
            
            VStack(spacing: 40) {
                // Header
                HStack {
                    Button("Back") {
                        dismiss()
                    }
                    .foregroundColor(.white)
                    
                    Spacer()
                    
                    Text("IMU Tracking")
                        .font(.headline)
                        .foregroundColor(.white)
                    
                    Spacer()
                    
                    // Recording indicator
                    if isRecording {
                        HStack(spacing: 8) {
                            Circle()
                                .fill(Color.red)
                                .frame(width: 8, height: 8)
                                .scaleEffect(isRecording ? 1.2 : 1.0)
                                .animation(.easeInOut(duration: 0.8).repeatForever(), value: isRecording)
                            Text("REC")
                                .font(.caption)
                                .fontWeight(.semibold)
                        }
                        .foregroundColor(.red)
                    }
                }
                .padding()
                
                // Exercise label
                VStack(spacing: 20) {
                    Text("IMU Rep Counter")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    // AI Coach and Voice control status
                    if motionManager.isCoachingActive {
                        VStack(spacing: 8) {
                            HStack {
                                Image(systemName: "person.wave.2.fill")
                                    .foregroundColor(.green)
                                Text("AI Coach Active")
                                    .font(.headline)
                                    .foregroundColor(.green)
                            }
                            
                            Text("Your AI coach is guiding you through the workout!")
                                .font(.subheadline)
                                .foregroundColor(.green.opacity(0.8))
                                .multilineTextAlignment(.center)
                        }
                        .padding()
                        .background(Color.green.opacity(0.1))
                        .cornerRadius(10)
                    } else if motionManager.isVoiceControlActive {
                        VStack(spacing: 5) {
                            HStack {
                                Image(systemName: "mic.fill")
                                    .foregroundColor(.blue)
                                Text("Voice Control Active")
                                    .font(.headline)
                                    .foregroundColor(.blue)
                            }
                            
                            if !motionManager.voiceRecognizedText.isEmpty {
                                Text("Heard: \"\(motionManager.voiceRecognizedText)\"")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            Text("Say 'GO' to start AI coaching, 'STOP' to stop")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .background(Color.blue.opacity(0.1))
                        .cornerRadius(10)
                    }
                    
                    // Exercise state and motion type display
                    VStack(spacing: 10) {
                        HStack {
                            Text("State:")
                                .font(.headline)
                            Text(stateDisplayText)
                                .font(.title2)
                                .fontWeight(.semibold)
                                .foregroundColor(stateColor)
                        }
                        
                        HStack {
                            Text("Motion:")
                                .font(.headline)
                            Text(motionTypeDisplayText)
                                .font(.body)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(10)
                    
                    // Large rep counter
                    VStack(spacing: 10) {
                        Text("\(motionManager.repCount)")
                            .font(.system(size: 120, weight: .bold, design: .rounded))
                            .foregroundColor(.green)
                            .shadow(radius: 10)
                        
                        Text("REPS")
                            .font(.title)
                            .fontWeight(.semibold)
                            .foregroundColor(.white.opacity(0.8))
                            .kerning(2.0)
                    }
                }
                
                // Real-time acceleration data
                VStack(spacing: 15) {
                    Text("Acceleration Data")
                        .font(.headline)
                        .foregroundColor(.white)
                    
                    VStack(spacing: 8) {
                        HStack {
                            Text("Y (Vertical):")
                                .foregroundColor(.white.opacity(0.7))
                            Spacer()
                            Text(String(format: "%.3f g", currentAcceleration.y))
                                .foregroundColor(abs(currentAcceleration.y) > 0.2 ? .yellow : .white)
                                .fontWeight(.semibold)
                        }
                        
                        HStack {
                            Text("Smoothed Y:")
                                .foregroundColor(.white.opacity(0.7))
                            Spacer()
                            Text(String(format: "%.3f g", motionManager.smoothedAcceleration))
                                .foregroundColor(abs(motionManager.smoothedAcceleration) > 0.16 ? .orange : .white)
                                .fontWeight(.semibold)
                        }
                        
                        HStack {
                            Text("X (Lateral):")
                                .foregroundColor(.white.opacity(0.7))
                            Spacer()
                            Text(String(format: "%.3f g", currentAcceleration.x))
                                .foregroundColor(.white)
                        }
                        
                        HStack {
                            Text("Z (Forward):")
                                .foregroundColor(.white.opacity(0.7))
                            Spacer()
                            Text(String(format: "%.3f g", currentAcceleration.z))
                                .foregroundColor(.white)
                        }
                    }
                    .font(.system(.body, design: .monospaced))
                }
                .padding()
                .background(Color.white.opacity(0.1))
                .cornerRadius(15)
                .padding(.horizontal)
                
                // AI Coach instructions
                VStack(spacing: 15) {
                    Text("AI Voice Coach")
                        .font(.headline)
                        .foregroundColor(.white)
                    
                    VStack(spacing: 8) {
                        HStack {
                            Image(systemName: "mic.fill")
                                .foregroundColor(.green)
                            Text("Say 'GO' to start AI-guided workout")
                                .foregroundColor(.white)
                        }
                        
                        HStack {
                            Image(systemName: "person.wave.2.fill")
                                .foregroundColor(.orange)
                            Text("Coach will count reps and guide your pace")
                                .foregroundColor(.white)
                        }
                        
                        HStack {
                            Image(systemName: "stop.fill")
                                .foregroundColor(.red)
                            Text("Say 'STOP' to finish and export session")
                                .foregroundColor(.white)
                        }
                    }
                    .font(.subheadline)
                    
                    // Export button (only when not recording)
                    if !isRecording && sessionRecorder.currentSession != nil {
                        Button(action: exportSession) {
                            HStack {
                                Image(systemName: "square.and.arrow.up")
                                Text("Export Session Data")
                            }
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(12)
                        }
                    }
                }
                .padding(.horizontal)
                .padding(.bottom, 100) // Increased bottom padding to prevent cutoff
            }
        }
        .sheet(isPresented: $showingExportSheet) {
            if let exportURL = exportURL {
                ActivityViewController(activityItems: [exportURL])
            }
        }
        .overlay {
            if showingSessionSummary {
                SessionSummaryView(
                    exerciseType: selectedExercise,
                    repCount: motionManager.repCount,
                    duration: sessionRecorder.currentSession?.duration ?? 0,
                    onDismiss: {
                        showingSessionSummary = false
                    },
                    onExport: {
                        showingSessionSummary = false
                        exportSession()
                    }
                )
            }
        }
        .onAppear {
            // Auto-start voice control and IMU when view appears
            motionManager.startUpdates()
            motionManager.startVoiceControl()
        }
        .onReceive(NotificationCenter.default.publisher(for: .voiceStartCommand)) { _ in
            sessionRecorder.startRecording(exerciseType: "Squat")
            isRecording = true
        }
        .onReceive(NotificationCenter.default.publisher(for: .voiceStopCommand)) { _ in
            sessionRecorder.finalRepCount = motionManager.repCount
            sessionRecorder.stopRecording()
            isRecording = false
            showingSessionSummary = true
        }
        .onDisappear {
            if isRecording {
                stopRecording()
            }
        }
        .onReceive(motionManager.imuDataPublisher) { imuData in
            // Update real-time display
            currentAcceleration = (
                x: imuData.acceleration.x,
                y: imuData.acceleration.y,
                z: imuData.acceleration.z
            )
        }
        .onChange(of: isRecording) { _, newValue in
            if newValue {
                motionManager.resetExerciseSession()
            }
        }
    }
    
    private func startRecording() {
        sessionRecorder.startRecording(exerciseType: selectedExercise)
        isRecording = true
    }
    
    private func stopRecording() {
        sessionRecorder.finalRepCount = motionManager.repCount
        sessionRecorder.stopRecording()
        isRecording = false
        showingSessionSummary = true
    }
    
    private func exportSession() {
        if let url = sessionRecorder.exportSession() {
            exportURL = url
            showingExportSheet = true
        }
    }
    
    // Computed properties for display
    private var stateDisplayText: String {
        switch motionManager.exerciseState {
        case .setup:
            return "Setup"
        case .stabilizing:
            return "Stabilizing"
        case .ready:
            return "Ready"
        case .exercising:
            return "Exercising"
        case .completed:
            return "Completed"
        }
    }
    
    private var motionTypeDisplayText: String {
        switch motionManager.currentMotionType {
        case .stable:
            return "Stable"
        case .handling:
            return "Phone Handling"
        case .rustling:
            return "Rustling"
        case .exercise:
            return "Exercise Motion"
        case .unknown:
            return "Unknown"
        }
    }
}
