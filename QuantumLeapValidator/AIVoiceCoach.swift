// QuantumLeapValidator/AIVoiceCoach.swift

import Foundation
import AVFoundation

class AIVoiceCoach: NSObject, AVSpeechSynthesizerDelegate {
    
    private let synthesizer = AVSpeechSynthesizer()
    private let audioManager = AudioSessionManager.shared // We will fix the manager in the next step
    
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
        audioManager.requestState(.idle)
    }

    func reset() {
        lastSpokenRep = 0
        lastRepSpokenTime = nil
        speechQueue.removeAll()
        isSpeaking = false
    }

    func onRepDetected(repNumber: Int) {
        // --- CRITICAL FIX: DEBOUNCING LOGIC ---
        // 1. Only speak for new reps
        guard repNumber > lastSpokenRep else { return }

        // 2. Time-based debouncing to prevent spam from a noisy rep counter
        let now = Date()
        if let lastTime = lastRepSpokenTime, now.timeIntervalSince(lastTime) < 2.0 {
            print("🚫 TTS Debounced: Less than 2 seconds since last rep.")
            return
        }
        
        lastSpokenRep = repNumber
        lastRepSpokenTime = now
        queueSpeech("That's \(repNumber)!")
    }

    private func queueSpeech(_ text: String) {
        speechQueue.append(text)
        processSpeechQueue()
    }

    private func processSpeechQueue() {
        // If we are already speaking, or the queue is empty, do nothing.
        guard !isSpeaking, !speechQueue.isEmpty else { return }

        isSpeaking = true
        let textToSpeak = speechQueue.removeFirst()
        
        audioManager.requestState(.speaking) // Request playback state
        
        let utterance = AVSpeechUtterance(string: textToSpeak)
        // You can add your voice selection logic here
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        
        synthesizer.speak(utterance)
    }

    // AVSpeechSynthesizerDelegate method
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        isSpeaking = false
        
        // If there's more to say, process it. Otherwise, return to idle.
        if !speechQueue.isEmpty {
            processSpeechQueue()
        } else {
            audioManager.requestState(.idle)
        }
    }
}
