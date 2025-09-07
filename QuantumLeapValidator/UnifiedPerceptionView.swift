import SwiftUI

/**
 * UnifiedPerceptionView - Simplified UI for testing without PythonKit
 */

struct UnifiedPerceptionView: View {
    @ObservedObject var perceptionBridge: UnifiedPerceptionBridge
    @State private var targetReps: Int = 10
    
    var body: some View {
        VStack(spacing: 20) {
            headerSection
            
            if perceptionBridge.isActive {
                activeWorkoutSection
            } else {
                setupSection
            }
            
            coachingSection
            controlButtons
        }
        .padding()
        .navigationTitle("AI Workout")
        .navigationBarTitleDisplayMode(.large)
    }
    
    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "brain.head.profile")
                .font(.system(size: 50))
                .foregroundColor(.blue)
            
            Text("Unified Perception System")
                .font(.title2)
                .fontWeight(.bold)
            
            Text(perceptionBridge.isActive ? "Workout Active" : "Ready to Start")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }
    
    private var setupSection: some View {
        VStack(spacing: 16) {
            Text("Target Reps")
                .font(.headline)
            
            Stepper(value: $targetReps, in: 1...50) {
                Text("\(targetReps) reps")
                    .font(.title2)
                    .fontWeight(.semibold)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private var activeWorkoutSection: some View {
        VStack(spacing: 16) {
            // Rep Counter
            HStack {
                VStack {
                    Text("Current")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(perceptionBridge.currentRep)")
                        .font(.system(size: 36, weight: .bold, design: .rounded))
                        .foregroundColor(.primary)
                }
                
                Text("/")
                    .font(.title)
                    .foregroundColor(.secondary)
                
                VStack {
                    Text("Target")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(perceptionBridge.targetReps)")
                        .font(.system(size: 24, weight: .medium, design: .rounded))
                        .foregroundColor(.secondary)
                }
            }
            
            // Progress Bar
            ProgressView(value: Double(perceptionBridge.currentRep), total: Double(perceptionBridge.targetReps))
                .progressViewStyle(LinearProgressViewStyle(tint: progressColor))
                .scaleEffect(x: 1, y: 2, anchor: .center)
            
            // Session Duration
            Text("Duration: \(formattedDuration)")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private var coachingSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .foregroundColor(.blue)
                Text("AI Coach")
                    .font(.headline)
                    .fontWeight(.semibold)
                Spacer()
            }
            
            Text(perceptionBridge.coachingMessage)
                .font(.body)
                .foregroundColor(.primary)
                .multilineTextAlignment(.leading)
                .padding()
                .background(Color.blue.opacity(0.1))
                .cornerRadius(8)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private var controlButtons: some View {
        VStack(spacing: 12) {
            if perceptionBridge.isActive {
                Button(action: {
                    perceptionBridge.stopSession()
                }) {
                    HStack {
                        Image(systemName: "stop.fill")
                        Text("Stop Workout")
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.red)
                    .cornerRadius(12)
                }
            } else {
                Button(action: {
                    perceptionBridge.startSession(targetReps: targetReps)
                }) {
                    HStack {
                        Image(systemName: "play.fill")
                        Text("Start Workout")
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.green)
                    .cornerRadius(12)
                }
            }
            
            if !perceptionBridge.isActive && perceptionBridge.sessionDuration > 0 {
                HStack(spacing: 12) {
                    Button(action: {
                        showSessionStats()
                    }) {
                        HStack {
                            Image(systemName: "chart.bar.fill")
                            Text("View Stats")
                        }
                        .font(.subheadline)
                        .foregroundColor(.blue)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue.opacity(0.1))
                        .cornerRadius(8)
                    }
                    
                    Button(action: {
                        exportSessionData()
                    }) {
                        HStack {
                            Image(systemName: "square.and.arrow.up")
                            Text("Export Data")
                        }
                        .font(.subheadline)
                        .foregroundColor(.green)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green.opacity(0.1))
                        .cornerRadius(8)
                    }
                }
            }
        }
    }
    
    private var progressColor: Color {
        let progress = Double(perceptionBridge.currentRep) / Double(perceptionBridge.targetReps)
        if progress < 0.3 {
            return .red
        } else if progress < 0.7 {
            return .orange
        } else {
            return .green
        }
    }
    
    private var formattedDuration: String {
        let minutes = Int(perceptionBridge.sessionDuration) / 60
        let seconds = Int(perceptionBridge.sessionDuration) % 60
        return String(format: "%02d:%02d", minutes, seconds)
    }
    
    private func showSessionStats() {
        let stats = perceptionBridge.getSessionStats()
        print("📊 Session Stats: \(stats)")
        // In a real app, this would show a stats modal
    }
    
    private func exportSessionData() {
        let sessionData = perceptionBridge.exportSessionData()
        let timestamp = DateFormatter().string(from: Date())
        
        // Create temporary file
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("workout_session_\(timestamp).csv")
        
        do {
            try sessionData.write(to: tempURL, atomically: true, encoding: .utf8)
            
            // Share the file
            let activityVC = UIActivityViewController(activityItems: [tempURL], applicationActivities: nil)
            
            if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
               let window = windowScene.windows.first {
                window.rootViewController?.present(activityVC, animated: true)
            }
            
            print("📤 Session data exported to: \(tempURL.path)")
        } catch {
            print("❌ Failed to export session data: \(error)")
        }
    }
}

struct UnifiedPerceptionView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            UnifiedPerceptionView(perceptionBridge: UnifiedPerceptionBridge())
        }
    }
}
