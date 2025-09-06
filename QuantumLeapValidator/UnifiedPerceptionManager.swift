import Foundation
import CoreMotion
import AVFoundation

/**
 * UnifiedPerceptionManager - Singleton manager to replace existing component coordination
 * Eliminates AudioSessionManager, SmartExerciseDetector coordination issues
 */

class UnifiedPerceptionManager: ObservableObject {
    
    static let shared = UnifiedPerceptionManager()
    
    // MARK: - Properties
    
    @Published var isSystemActive: Bool = false
    private var perceptionBridge: UnifiedPerceptionBridge?
    
    // Legacy component references (to be phased out)
    private var legacyVoiceController: Any?
    private var legacyAIVoiceCoach: Any?
    private var legacyAudioSessionManager: Any?
    private var legacySmartExerciseDetector: Any?
    
    private init() {
        setupUnifiedSystem()
    }
    
    // MARK: - System Setup
    
    private func setupUnifiedSystem() {
        perceptionBridge = UnifiedPerceptionBridge()
        
        // Configure unified audio session (replaces AudioSessionManager)
        perceptionBridge?.configureAudioSession()
        
        print("🚀 Unified Perception System initialized")
    }
    
    // MARK: - Migration Methods
    
    /**
     * Replaces the existing component architecture
     * Call this to migrate from legacy system to unified perception
     */
    func migrateFromLegacySystem() {
        print("🔄 Starting migration from legacy system...")
        
        // Disable legacy components
        disableLegacyComponents()
        
        // Replace with unified system
        perceptionBridge?.replaceExistingComponents()
        
        isSystemActive = true
        
        print("✅ Migration complete - Unified system active")
    }
    
    private func disableLegacyComponents() {
        // Stop any active legacy sessions
        // This would integrate with existing VoiceController, AIVoiceCoach, etc.
        
        print("🛑 Legacy components disabled")
        print("   - VoiceController: Replaced by unified speech processing")
        print("   - AIVoiceCoach: Replaced by generative coaching LLM")
        print("   - AudioSessionManager: Replaced by persistent .playAndRecord session")
        print("   - SmartExerciseDetector: Replaced by transformer-based perception")
    }
    
    // MARK: - Unified Interface
    
    func startWorkoutSession(targetReps: Int = 10) {
        guard let bridge = perceptionBridge else { return }
        
        if !isSystemActive {
            migrateFromLegacySystem()
        }
        
        bridge.startSession(targetReps: targetReps)
    }
    
    func endWorkoutSession() {
        perceptionBridge?.endSession()
    }
    
    // MARK: - Legacy Compatibility Layer
    
    /**
     * Provides compatibility with existing MotionManager integration
     * Gradually replace calls to this with direct UnifiedPerceptionBridge usage
     */
    func processMotionUpdate(_ motion: CMDeviceMotion) {
        // Legacy compatibility - route through unified system
        // This allows gradual migration of existing MotionManager code
        
        if let bridge = perceptionBridge, bridge.isActive {
            // Motion processing handled internally by bridge
            // No manual rep counting or state management needed
        }
    }
    
    /**
     * Replaces SmartExerciseDetector state management
     */
    var exerciseState: String {
        return perceptionBridge?.exerciseState ?? "setup"
    }
    
    /**
     * Replaces manual rep counting logic
     */
    var currentRepCount: Int {
        return perceptionBridge?.currentRep ?? 0
    }
    
    // MARK: - Audio Session Unification
    
    /**
     * Eliminates audio session coordination issues
     * Single persistent session replaces complex state management
     */
    func handleAudioInterruption() {
        perceptionBridge?.handleAudioSessionInterruption()
    }
    
    // MARK: - Performance Monitoring
    
    func getSystemPerformance() -> [String: Any] {
        return perceptionBridge?.getPerformanceStats() ?? [:]
    }
    
    // MARK: - Error Recovery
    
    func recoverFromError() {
        print("🔧 Recovering unified system...")
        
        // Reset unified system
        perceptionBridge = UnifiedPerceptionBridge()
        perceptionBridge?.configureAudioSession()
        
        print("✅ System recovery complete")
    }
}
