import SwiftUI

struct ContentView: View {
    @State private var selectedExercise = "Squat"
    @State private var isRecording = false
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
                    
                    Text("Real-time pose validation & data capture")
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
                        Image(systemName: "camera.viewfinder")
                            .foregroundColor(.purple)
                            .font(.title2)
                        
                        VStack(alignment: .leading) {
                            Text("Vision System")
                                .font(.headline)
                            Text("Real-time pose detection")
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
        .fullScreenCover(isPresented: $showingPoseView) {
            PoseValidationView(selectedExercise: selectedExercise)
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
