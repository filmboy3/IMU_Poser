import SwiftUI

struct SessionSummaryView: View {
    let exerciseType: String
    let repCount: Int
    let duration: TimeInterval
    let onDismiss: () -> Void
    let onExport: () -> Void
    
    private var formattedDuration: String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
    
    var body: some View {
        ZStack {
            Color.black.opacity(0.8)
                .ignoresSafeArea()
            
            VStack(spacing: 30) {
                // Header
                VStack(spacing: 10) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 80))
                        .foregroundColor(.green)
                    
                    Text("Session Complete!")
                        .font(.title)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                }
                
                // Stats Card
                VStack(spacing: 20) {
                    Text(exerciseType)
                        .font(.title2)
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                    
                    HStack(spacing: 40) {
                        // Rep Count
                        VStack(spacing: 8) {
                            Text("\(repCount)")
                                .font(.system(size: 60, weight: .bold, design: .rounded))
                                .foregroundColor(.green)
                            Text("REPS")
                                .font(.caption)
                                .fontWeight(.semibold)
                                .foregroundColor(.white.opacity(0.8))
                                .kerning(1.0)
                        }
                        
                        // Duration
                        VStack(spacing: 8) {
                            Text(formattedDuration)
                                .font(.system(size: 30, weight: .bold, design: .monospaced))
                                .foregroundColor(.blue)
                            Text("TIME")
                                .font(.caption)
                                .fontWeight(.semibold)
                                .foregroundColor(.white.opacity(0.8))
                                .kerning(1.0)
                        }
                    }
                }
                .padding(30)
                .background(Color.white.opacity(0.1))
                .cornerRadius(20)
                
                // Action Buttons
                VStack(spacing: 15) {
                    Button(action: onExport) {
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
                    
                    Button(action: onDismiss) {
                        Text("Done")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.gray.opacity(0.3))
                            .cornerRadius(12)
                    }
                }
                .padding(.horizontal, 20)
            }
            .padding(40)
        }
    }
}

#Preview {
    SessionSummaryView(
        exerciseType: "Squat",
        repCount: 15,
        duration: 125.0,
        onDismiss: {},
        onExport: {}
    )
}
