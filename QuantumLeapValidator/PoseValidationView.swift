import SwiftUI
import UIKit

struct PoseValidationView: View {
    let selectedExercise: String
    @State private var isRecording = false
    @State private var showingExportSheet = false
    @State private var exportURL: URL?
    @Environment(\.dismiss) private var dismiss
    @StateObject private var sessionRecorder = SessionRecorder.shared
    @StateObject private var motionManager = MotionManager.shared
    
    var body: some View {
        ZStack {
            // Camera and pose overlay
            PoseViewControllerWrapper(selectedExercise: selectedExercise)
            .ignoresSafeArea()
            
            // Control overlay
            VStack {
                // Top controls
                HStack {
                    Button(action: {
                        if isRecording {
                            stopRecording()
                        }
                        dismiss()
                    }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title)
                            .foregroundColor(.white)
                            .background(Color.black.opacity(0.6))
                            .clipShape(Circle())
                    }
                    
                    Spacer()
                    
                    // Status indicators
                    VStack(alignment: .trailing, spacing: 4) {
                        if isRecording {
                            HStack(spacing: 4) {
                                Circle()
                                    .fill(Color.red)
                                    .frame(width: 8, height: 8)
                                Text("REC")
                                    .font(.caption)
                                    .fontWeight(.bold)
                            }
                            .foregroundColor(.red)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(Color.black.opacity(0.6))
                            .cornerRadius(15)
                        }
                        
                        // IMU status
                        HStack(spacing: 4) {
                            Circle()
                                .fill(motionManager.isActive ? Color.green : Color.red)
                                .frame(width: 6, height: 6)
                            Text("IMU: \(String(format: "%.0f Hz", motionManager.dataRate))")
                                .font(.caption2)
                        }
                        .foregroundColor(.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.black.opacity(0.6))
                        .cornerRadius(10)
                    }
                }
                .padding()
                
                Spacer()
                
                // Bottom controls
                VStack(spacing: 20) {
                    // Exercise label
                    Text(selectedExercise)
                        .font(.title2)
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 10)
                        .background(Color.black.opacity(0.6))
                        .cornerRadius(20)
                    
                    // Recording controls
                    HStack(spacing: 30) {
                        // Record/Stop button
                        Button(action: isRecording ? stopRecording : startRecording) {
                            VStack {
                                Image(systemName: isRecording ? "stop.circle.fill" : "record.circle")
                                    .font(.system(size: 40))
                                    .foregroundColor(isRecording ? .red : .white)
                                
                                Text(isRecording ? "Stop" : "Record")
                                    .font(.caption)
                                    .fontWeight(.semibold)
                                    .foregroundColor(.white)
                            }
                        }
                        .disabled(sessionRecorder.isRecording && !isRecording)
                        
                        // Export button (only show if we have a completed session)
                        if !isRecording && sessionRecorder.currentSession != nil {
                            Button(action: exportSession) {
                                VStack {
                                    Image(systemName: "square.and.arrow.up")
                                        .font(.system(size: 30))
                                        .foregroundColor(.blue)
                                    
                                    Text("Export")
                                        .font(.caption)
                                        .fontWeight(.semibold)
                                        .foregroundColor(.white)
                                }
                            }
                        }
                    }
                    .padding(.horizontal, 40)
                    
                    // Session info
                    if let session = sessionRecorder.currentSession {
                        VStack(spacing: 4) {
                            Text("Session: \(session.sessionId)")
                                .font(.caption2)
                                .foregroundColor(.white.opacity(0.8))
                            
                            if isRecording {
                                Text("Recording \(session.exerciseType)...")
                                    .font(.caption)
                                    .foregroundColor(.white)
                            } else if session.endTime != nil {
                                Text("Duration: \(String(format: "%.1f", session.duration))s")
                                    .font(.caption)
                                    .foregroundColor(.white)
                            }
                        }
                        .padding(.horizontal, 20)
                        .padding(.vertical, 8)
                        .background(Color.black.opacity(0.4))
                        .cornerRadius(12)
                    }
                }
                .padding(.bottom, 40)
            }
        }
        .sheet(isPresented: $showingExportSheet) {
            if let exportURL = exportURL {
                ActivityViewController(activityItems: [exportURL])
            }
        }
        .onAppear {
            // Start motion manager when view appears
            if !motionManager.isActive {
                motionManager.startUpdates()
            }
        }
        .onDisappear {
            // Stop recording if active when leaving
            if isRecording {
                stopRecording()
            }
        }
    }
    
    private func startRecording() {
        sessionRecorder.startRecording(exerciseType: selectedExercise)
        isRecording = true
    }
    
    private func stopRecording() {
        sessionRecorder.stopRecording()
        isRecording = false
    }
    
    private func exportSession() {
        if let url = sessionRecorder.exportSession() {
            exportURL = url
            showingExportSheet = true
        }
    }
}

// MARK: - Camera View Integration
// Note: PoseViewControllerWrapper is defined in PoseViewController.swift

// MARK: - Activity View Controller for Sharing

struct ActivityViewController: UIViewControllerRepresentable {
    let activityItems: [Any]
    
    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(
            activityItems: activityItems,
            applicationActivities: nil
        )
        return controller
    }
    
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {
        // No updates needed
    }
}

// MARK: - Preview

struct PoseValidationView_Previews: PreviewProvider {
    static var previews: some View {
        PoseValidationView(selectedExercise: "Squat")
    }
}
