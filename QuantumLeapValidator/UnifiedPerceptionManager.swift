import Foundation
import SwiftUI

/**
 * UnifiedPerceptionManager - Simplified manager for testing
 */

class UnifiedPerceptionManager: ObservableObject {
    
    static let shared = UnifiedPerceptionManager()
    
    // MARK: - Properties
    
    @Published var isSystemActive: Bool = false
    private var perceptionBridge: UnifiedPerceptionBridge?
    
    private init() {
        setupUnifiedSystem()
    }
    
    // MARK: - System Setup
    
    private func setupUnifiedSystem() {
        perceptionBridge = UnifiedPerceptionBridge()
        print("🧠 UnifiedPerceptionManager initialized")
    }
    
    // MARK: - Public Interface
    
    func startWorkout(targetReps: Int = 10) {
        guard !isSystemActive else { return }
        
        perceptionBridge?.startSession(targetReps: targetReps)
        isSystemActive = true
        
        print("🚀 Unified workout started")
    }
    
    func stopWorkout() {
        guard isSystemActive else { return }
        
        perceptionBridge?.stopSession()
        isSystemActive = false
        
        print("🏁 Unified workout stopped")
    }
    
    func getBridge() -> UnifiedPerceptionBridge? {
        return perceptionBridge
    }
}
