// QuantumLeapValidator/AudioSessionManager.swift

import Foundation
import AVFoundation

class AudioSessionManager: ObservableObject {
    static let shared = AudioSessionManager()

    enum AudioState {
        case idle
        case recording
        case speaking
    }

    @Published private(set) var currentState: AudioState = .idle
    private let audioSession = AVAudioSession.sharedInstance()
    private var pendingState: AudioState?

    private init() {
        // Interruptions handled as before
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAudioInterruption),
            name: AVAudioSession.interruptionNotification,
            object: nil
        )
    }

    // The key change: We now have explicit, separate functions.
    // This prevents one state from stomping on another.

    func activateRecordingSession(completion: ((Bool) -> Void)? = nil) {
        let session = AVAudioSession.sharedInstance()

        print("🎙️ AudioSessionManager: Activating recording session (current: \(currentState))")

        // If we're currently marked as speaking, attempt to deactivate first to allow recording to take ownership.
        if currentState == .speaking {
            do {
                try session.setActive(false, options: [.notifyOthersOnDeactivation])
                currentState = .idle
                print("🔄 AudioSessionManager: Pre-deactivated speaking to allow recording activation")
            } catch {
                print("⚠️ AudioSessionManager: Failed pre-deactivation before recording: \(error)")
                // Continue: attempt to set recording category anyway.
            }
        }

        do {
            try session.setCategory(.record, mode: .measurement, options: [])
            try session.setActive(true)
            currentState = .recording
            print("✅ AudioSessionManager: Recording session activated successfully")
            completion?(true)
        } catch {
            print("❌ AudioSessionManager: Failed to activate recording session: \(error)")
            completion?(false)
        }
    }
    
    func activateSpeakingSession(completion: ((Bool) -> Void)? = nil) {
        print("🔊 AudioSessionManager: Activating speaking session (current: \(currentState))")

        // If we're already in speaking, don't reconfigure; just report success.
        if currentState == .speaking {
            print("ℹ️ AudioSessionManager: Already in speaking state; skipping re-activation")
            completion?(true)
            return
        }

        // If we're currently recording, deactivate first to avoid category priority conflicts
        if currentState == .recording {
            do {
                try audioSession.setActive(false, options: [.notifyOthersOnDeactivation])
                currentState = .idle
                print("🔄 AudioSessionManager: Deactivated recording prior to speaking activation")
            } catch {
                print("⚠️ AudioSessionManager: Failed to pre-deactivate before speaking: \(error)")
                // Continue anyway; upstream should have paused input and we will attempt activation.
            }
        }

        do {
            try audioSession.setCategory(.playback, mode: .voicePrompt, options: [.duckOthers])
            try audioSession.setActive(true)
            DispatchQueue.main.async {
                self.currentState = .speaking
                print("✅ AudioSessionManager: Speaking session activated successfully")
                completion?(true)
            }
        } catch {
            print("❌ AudioSessionManager: Failed to activate speaking session: \(error)")
            completion?(false)
        }
    }

    func deactivateSession() {
        print("🔇 AudioSessionManager: Deactivating session (current: \(currentState)) → idle")
        do {
            try audioSession.setActive(false, options: .notifyOthersOnDeactivation)
            DispatchQueue.main.async {
                self.currentState = .idle
                print("✅ AudioSessionManager: Session deactivated, now idle")
                
                // If there was a pending state request, fulfill it now.
                if self.pendingState == .recording {
                    print("🔄 AudioSessionManager: Processing pending recording request")
                    self.pendingState = nil
                    self.activateRecordingSession()
                }
            }
        } catch {
            print("❌ AudioSessionManager: Failed to deactivate session: \(error)")
            // Force logical idle to avoid deadlock; attempt pending transitions.
            DispatchQueue.main.async {
                self.currentState = .idle
                if self.pendingState == .recording {
                    print("🔄 AudioSessionManager: Forced idle; processing pending recording request")
                    self.pendingState = nil
                    self.activateRecordingSession()
                }
            }
        }
    }
    
    // Legacy method for backward compatibility
    func requestState(_ requestedState: AudioState) {
        switch requestedState {
        case .idle:
            deactivateSession()
        case .recording:
            activateRecordingSession()
        case .speaking:
            activateSpeakingSession()
        }
    }
    
    @objc private func handleAudioInterruption(notification: Notification) {
        guard let userInfo = notification.userInfo,
              let typeValue = userInfo[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: typeValue) else {
            return
        }

        if type == .began {
            print("AudioSessionManager: Interruption began. Setting state to idle.")
            DispatchQueue.main.async {
                self.currentState = .idle
            }
        } else if type == .ended {
            print("AudioSessionManager: Interruption ended.")
            deactivateSession()
        }
    }
}
