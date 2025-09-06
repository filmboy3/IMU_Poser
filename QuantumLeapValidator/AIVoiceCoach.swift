// QuantumLeapValidator/AIVoiceCoach.swift

import Foundation
import AVFoundation

class AIVoiceCoach: NSObject, AVSpeechSynthesizerDelegate {
    static let shared = AIVoiceCoach()
    
    private let synthesizer = AVSpeechSynthesizer()
    private let audioManager = AudioSessionManager.shared // We will fix the manager in the next step
    private let minAnnouncementInterval: TimeInterval = 0.6
    
    // --- NEW STATE MANAGEMENT PROPERTIES ---
    private var speechQueue: [String] = []
    private var isSpeaking = false
    private var lastSpokenRep = 0
    private var lastRepSpokenTime: Date?
    // ------------------------------------

    override init() {
        super.init()
        self.synthesizer.delegate = self
    }

    func startCoaching() {
        reset()
        queueSpeech("Workout started. Say 'GO' to begin reps.")
    }

    func stopCoaching() {
        synthesizer.stopSpeaking(at: .immediate)
        speechQueue.removeAll()
        isSpeaking = false
        audioManager.deactivateSession()
    }

    func reset() {
        lastSpokenRep = 0
        lastRepSpokenTime = nil
        speechQueue.removeAll()
        isSpeaking = false
    }

    func onRepDetected(repNumber: Int) {
        // Ensure all coach state changes happen on main to avoid data races
        if !Thread.isMainThread {
            DispatchQueue.main.async { [weak self] in self?.onRepDetected(repNumber: repNumber) }
            return
        }

        // --- CRITICAL FIX: DEBOUNCING LOGIC ---
        // 1. Only speak for new reps
        guard repNumber > lastSpokenRep else { return }

        // 2. Time-based debouncing to prevent spam from a noisy rep counter
        let now = Date()
        if let lastTime = lastRepSpokenTime, now.timeIntervalSince(lastTime) < minAnnouncementInterval {
            let delta = now.timeIntervalSince(lastTime)
            print("🚫 TTS Debounced: Only \(String(format: "%.2f", delta))s since last announcement (< \(minAnnouncementInterval)s)")
            return
        }
        
        lastSpokenRep = repNumber
        lastRepSpokenTime = now
        queueSpeech("That's \(repNumber)!")
    }

    private func queueSpeech(_ text: String) {
        if !Thread.isMainThread {
            DispatchQueue.main.async { [weak self] in self?.queueSpeech(text) }
            return
        }
        speechQueue.append(text)
        processSpeechQueue()
    }

    private func processSpeechQueue() {
        if !Thread.isMainThread {
            DispatchQueue.main.async { [weak self] in self?.processSpeechQueue() }
            return
        }
        // If we are already speaking, or the queue is empty, do nothing.
        guard !isSpeaking, !speechQueue.isEmpty else { return }
        
        guard let textToSpeak = speechQueue.first else { return }
        speechQueue.removeFirst()
        print("🗣️ AIVoiceCoach: Processing speech: '\(textToSpeak)' (queue remaining: \(speechQueue.count))")
        
        // Ask VoiceController to pause recognition and release the session first
        NotificationCenter.default.post(name: .qlvPauseRecognition, object: nil)

        // Give the OS a brief moment to settle teardown of the input graph
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) { [weak self] in
            guard let self = self else { return }

            self.audioManager.activateSpeakingSession { success in
                guard success else {
                    print("❌ AIVoiceCoach: Failed to activate speaking session")
                    return
                }
                self.isSpeaking = true
                print("✅ AIVoiceCoach: Speaking session activated, starting TTS")

                let utterance = AVSpeechUtterance(string: textToSpeak)
                utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
                utterance.rate = AVSpeechUtteranceDefaultSpeechRate

                self.synthesizer.speak(utterance)
            }
        }
    }

    // AVSpeechSynthesizerDelegate method
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        if !Thread.isMainThread {
            DispatchQueue.main.async { [weak self] in self?.speechSynthesizer(synthesizer, didFinish: utterance) }
            return
        }
        print("🔚 AIVoiceCoach: TTS finished, deactivating session")
        isSpeaking = false
        audioManager.deactivateSession() // Deactivate when done
        
        // Allow time to settle back to idle, then resume recognition and continue queue
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            NotificationCenter.default.post(name: .qlvResumeRecognition, object: nil)
            if !self.speechQueue.isEmpty {
                print("🔄 AIVoiceCoach: Processing next queued speech item")
            }
            self.processSpeechQueue()
        }
    }
}
