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
        // Don't interrupt if we are speaking
        guard currentState != .speaking else {
            print("AudioSessionManager: Deferring recording request because TTS is active.")
            pendingState = .recording // Remember that we want to record next
            completion?(false)
            return
        }
        
        print("AudioSessionManager: Activating recording session...")
        do {
            try audioSession.setCategory(.record, mode: .measurement, options: [])
            try audioSession.setActive(true)
            DispatchQueue.main.async {
                self.currentState = .recording
                print("AudioSessionManager: ✅ Recording session active.")
                completion?(true)
            }
        } catch {
            print("AudioSessionManager: ❌ FAILED to activate recording session. Error: \(error.localizedDescription)")
            completion?(false)
        }
    }
    
    func activateSpeakingSession(completion: ((Bool) -> Void)? = nil) {
        print("AudioSessionManager: Activating speaking session...")
        do {
            try audioSession.setCategory(.playback, mode: .voicePrompt, options: [.duckOthers])
            try audioSession.setActive(true)
            DispatchQueue.main.async {
                self.currentState = .speaking
                print("AudioSessionManager: ✅ Speaking session active.")
                completion?(true)
            }
        } catch {
            print("AudioSessionManager: ❌ FAILED to activate speaking session. Error: \(error.localizedDescription)")
            completion?(false)
        }
    }

    func deactivateSession() {
        // This is called when we are done speaking or listening.
        print("AudioSessionManager: Deactivating session, returning to idle.")
        do {
            try audioSession.setActive(false, options: .notifyOthersOnDeactivation)
            DispatchQueue.main.async {
                self.currentState = .idle
                
                // If there was a pending state request, fulfill it now.
                if self.pendingState == .recording {
                    self.pendingState = nil
                    self.activateRecordingSession()
                }
            }
        } catch {
            print("AudioSessionManager: ❌ FAILED to deactivate session. Error: \(error.localizedDescription)")
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
