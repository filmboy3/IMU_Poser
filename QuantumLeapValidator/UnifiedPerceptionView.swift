import SwiftUI
import CoreMotion

/**
 * UnifiedPerceptionView - SwiftUI interface for the unified perception system
 * Replaces existing IMUValidationView with integrated AI coaching
 */

struct UnifiedPerceptionView: View {
    @StateObject private var perceptionBridge = UnifiedPerceptionBridge()
    @State private var showingStats = false
    @State private var targetReps = 10
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                
                // Header
                headerSection
                
                // Progress Section
                progressSection
                
                // Coaching Section
                coachingSection
                
                // Controls
                controlsSection
                
                Spacer()
                
                // Performance Stats Button
                statsButton
            }
            .padding()
            .navigationTitle("AI Fitness Coach")
            .navigationBarTitleDisplayMode(.large)
            .sheet(isPresented: $showingStats) {
                PerformanceStatsView(bridge: perceptionBridge)
            }
        }
    }
    
    // MARK: - View Components
    
    private var headerSection: some View {
        VStack(spacing: 8) {
            Text("Project Chimera v2")
                .font(.headline)
                .foregroundColor(.secondary)
            
            Text("Unified Perception System")
                .font(.title2)
                .fontWeight(.semibold)
        }
    }
    
    private var progressSection: some View {
        VStack(spacing: 16) {
            // Rep Counter
            HStack {
                VStack(alignment: .leading) {
                    Text("Reps Completed")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text("\(perceptionBridge.currentRep)")
                        .font(.system(size: 48, weight: .bold, design: .rounded))
                        .foregroundColor(.primary)
                }
                
                Spacer()
                
                VStack(alignment: .trailing) {
                    Text("Target")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text("\(perceptionBridge.targetReps)")
                        .font(.system(size: 24, weight: .medium, design: .rounded))
                        .foregroundColor(.secondary)
                }
            }
            
            // Progress Bar
            ProgressView(value: perceptionBridge.progressPercentage)
                .progressViewStyle(LinearProgressViewStyle(tint: progressColor))
                .scaleEffect(x: 1, y: 2, anchor: .center)
            
            // Session Duration
            Text("Duration: \(perceptionBridge.formattedDuration)")
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
            
            Text(perceptionBridge.coachingMessage.isEmpty ? "Ready to start your workout!" : perceptionBridge.coachingMessage)
                .font(.body)
                .foregroundColor(.primary)
                .multilineTextAlignment(.leading)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(Color(.systemBackground))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.blue.opacity(0.3), lineWidth: 1)
                )
        }
        .padding()
        .background(Color.blue.opacity(0.05))
        .cornerRadius(12)
    }
    
    private var controlsSection: some View {
        VStack(spacing: 16) {
            // Target Reps Selector
            if !perceptionBridge.isActive {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Target Reps")
                        .font(.headline)
                    
                    Picker("Target Reps", selection: $targetReps) {
                        ForEach([5, 10, 15, 20, 25], id: \.self) { reps in
                            Text("\(reps) reps").tag(reps)
                        }
                    }
                    .pickerStyle(SegmentedPickerStyle())
                }
            }
            
            // Main Action Button
            Button(action: toggleSession) {
                HStack {
                    Image(systemName: perceptionBridge.isActive ? "stop.circle.fill" : "play.circle.fill")
                        .font(.title2)
                    
                    Text(perceptionBridge.isActive ? "End Workout" : "Start Workout")
                        .font(.headline)
                        .fontWeight(.semibold)
                }
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding()
                .background(perceptionBridge.isActive ? Color.red : Color.green)
                .cornerRadius(12)
            }
            .disabled(!CMMotionManager().isDeviceMotionAvailable)
        }
    }
    
    private var statsButton: some View {
        Button(action: { showingStats = true }) {
            HStack {
                Image(systemName: "chart.bar.fill")
                Text("Performance Stats")
            }
            .font(.subheadline)
            .foregroundColor(.blue)
        }
    }
    
    // MARK: - Computed Properties
    
    private var progressColor: Color {
        if perceptionBridge.progressPercentage < 0.3 {
            return .red
        } else if perceptionBridge.progressPercentage < 0.7 {
            return .orange
        } else {
            return .green
        }
    }
    
    // MARK: - Actions
    
    private func toggleSession() {
        if perceptionBridge.isActive {
            perceptionBridge.endSession()
        } else {
            perceptionBridge.startSession(targetReps: targetReps)
        }
    }
}

// MARK: - Performance Stats View

struct PerformanceStatsView: View {
    @ObservedObject var bridge: UnifiedPerceptionBridge
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text("Performance Statistics")
                    .font(.title)
                    .fontWeight(.bold)
                
                let stats = bridge.getPerformanceStats()
                
                VStack(spacing: 16) {
                    StatRow(title: "Total Inferences", 
                           value: "\(stats["total_inferences"] as? Int ?? 0)")
                    
                    StatRow(title: "Average Latency", 
                           value: String(format: "%.1fms", stats["avg_latency_ms"] as? Double ?? 0.0))
                    
                    StatRow(title: "Dropped Frames", 
                           value: "\(stats["dropped_frames"] as? Int ?? 0)")
                    
                    StatRow(title: "Session Duration", 
                           value: String(format: "%.1fs", stats["session_duration"] as? Double ?? 0.0))
                    
                    StatRow(title: "Rep Accuracy", 
                           value: String(format: "%.1f%%", bridge.repCountingAccuracy * 100))
                    
                    StatRow(title: "Exercise State", 
                           value: bridge.exerciseState.capitalized)
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(12)
                
                Spacer()
            }
            .padding()
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(trailing: Button("Done") {
                presentationMode.wrappedValue.dismiss()
            })
        }
    }
}

struct StatRow: View {
    let title: String
    let value: String
    
    var body: some View {
        HStack {
            Text(title)
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            Spacer()
            
            Text(value)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(.primary)
        }
    }
}

// MARK: - Preview

struct UnifiedPerceptionView_Previews: PreviewProvider {
    static var previews: some View {
        UnifiedPerceptionView()
    }
}
