import SwiftUI

struct ContentView: View {
    @State private var useUnifiedSystem = true // Feature flag for gradual rollout
    @StateObject private var perceptionBridge = UnifiedPerceptionBridge()
    
    var body: some View {
        TabView {
            if useUnifiedSystem {
                UnifiedPerceptionView(perceptionBridge: perceptionBridge)
                    .tabItem {
                        Image(systemName: "brain.head.profile")
                        Text("AI Coach")
                    }
            } else {
                IMUValidationView(selectedExercise: "Squat")
                    .tabItem {
                        Image(systemName: "gyroscope")
                        Text("IMU Validation")
                    }
            }
            
            // Settings tab for system selection
            VStack {
                Text("System Configuration")
                    .font(.title)
                    .padding()
                
                Toggle("Use Unified Perception System", isOn: $useUnifiedSystem)
                    .padding()
                
                Text(useUnifiedSystem ? "Chimera v2 Active" : "Legacy System Active")
                    .foregroundColor(useUnifiedSystem ? .green : .orange)
                    .padding()
                
                if useUnifiedSystem {
                    Text("✅ Eliminates audio session failures\n✅ AI-driven rep counting\n✅ Contextual coaching\n✅ <200ms latency")
                        .multilineTextAlignment(.center)
                        .padding()
                } else {
                    Text("⚠️ Legacy system with known issues\n❌ Audio session conflicts\n❌ State machine thrashing")
                        .multilineTextAlignment(.center)
                        .padding()
                }
                
                Spacer()
            }
            .tabItem {
                Image(systemName: "gear")
                Text("Settings")
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
