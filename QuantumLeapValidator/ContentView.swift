import SwiftUI

struct ContentView: View {
    @State private var selectedExercise = "Squat"
    @State private var showingPoseValidation = false
    @StateObject private var sessionRecorder = SessionRecorder.shared
    @State private var showingPoseView = false
    
    let exercises = ["Squat", "Bicep Curl", "Lateral Raise", "Push-up"]
    
    var body: some View {
        NavigationView {
            VStack(spacing: 30) {
                // Header
                VStack(spacing: 10) {
                    Text("QuantumLeap Validator")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                        .foregroundColor(.primary)
                    
                    Text("IMU-based rep counting & data capture")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding(.top, 40)
                
                // Exercise Selection
                VStack(alignment: .leading, spacing: 15) {
                    Text("Select Exercise")
                        .font(.headline)
                        .fontWeight(.semibold)
                    
                    Picker("Exercise", selection: $selectedExercise) {
                        ForEach(exercises, id: \.self) { exercise in
                            Text(exercise).tag(exercise)
                        }
                    }
                    .pickerStyle(SegmentedPickerStyle())
                }
                .padding(.horizontal, 20)
                
                // Status Card
                VStack(spacing: 15) {
                    HStack {
                        Image(systemName: "sensor.tag.radiowaves.forward")
                            .foregroundColor(.blue)
                            .font(.title2)
                        
                        VStack(alignment: .leading) {
                            Text("IMU Sensors")
                                .font(.headline)
                            Text("Ready for data capture")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                        
                        Circle()
                            .fill(Color.green)
                            .frame(width: 12, height: 12)
                    }
                    
                    HStack {
                        Image(systemName: "waveform.path.ecg")
                            .foregroundColor(.orange)
                            .font(.title2)
                        
                        VStack(alignment: .leading) {
                            Text("Rep Counter")
                                .font(.headline)
                            Text("Enhanced algorithm")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                        
                        Circle()
                            .fill(Color.green)
                            .frame(width: 12, height: 12)
                    }
                }
                .padding(20)
                .background(Color(.systemGray6))
                .cornerRadius(15)
                .padding(.horizontal, 20)
                
                // Session History
                VStack(alignment: .leading, spacing: 15) {
                    Text("Recent Sessions")
                        .font(.headline)
                        .fontWeight(.semibold)
                    
                    if sessionRecorder.sessionHistory.isEmpty {
                        Text("No sessions recorded yet")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .frame(maxWidth: .infinity, alignment: .center)
                            .padding()
                    } else {
                        ForEach(sessionRecorder.sessionHistory.suffix(5).reversed(), id: \.sessionId) { session in
                            HStack {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("\(session.exerciseType) Session")
                                        .font(.subheadline)
                                        .fontWeight(.medium)
                                    Text("\(session.repCount) reps • \(formatDuration(session.duration))")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                
                                Spacer()
                                
                                Text(formatRelativeTime(session.endTime ?? session.startTime))
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .padding()
                            .background(Color(.systemGray6))
                            .cornerRadius(12)
                        }
                    }
                }
                .padding(.horizontal, 20)
                
                Spacer()
                
                // Main Action Button
                Button(action: {
                    showingPoseView = true
                }) {
                    HStack {
                        Image(systemName: "play.circle.fill")
                            .font(.title2)
                        Text("Start \(selectedExercise) Session")
                            .font(.headline)
                            .fontWeight(.semibold)
                    }
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .frame(height: 60)
                    .background(
                        LinearGradient(
                            gradient: Gradient(colors: [Color.blue, Color.purple]),
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .cornerRadius(15)
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 40)
                
                // Navigation to Sessions (Future)
                NavigationLink(destination: Text("Sessions History - Coming Soon")) {
                    HStack {
                        Image(systemName: "folder.circle")
                        Text("View Sessions")
                    }
                    .foregroundColor(.blue)
                    .font(.headline)
                }
                .padding(.bottom, 20)
            }
            .navigationBarHidden(true)
        }
        .sheet(isPresented: $showingPoseView) {
            IMUValidationView(selectedExercise: selectedExercise)
        }
    }
    
    private func formatDuration(_ duration: TimeInterval) -> String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
    
    private func formatRelativeTime(_ date: Date) -> String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: date, relativeTo: Date())
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
