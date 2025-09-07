// QuantumLeapValidator/SoundManager.swift

import Foundation
import AVFoundation
import UIKit // For haptics

class SoundManager {
    static let shared = SoundManager()

    private var audioPlayer: AVAudioPlayer?
    private let hapticGenerator = UIImpactFeedbackGenerator(style: .medium)

    private init() {
        hapticGenerator.prepare()
    }

    enum FeedbackType {
        case repComplete
        case falseStart
    }

    func play(_ type: FeedbackType) {
        // We run this on the main thread to ensure it doesn't conflict
        // with other UI updates or audio session changes.
        DispatchQueue.main.async {
            switch type {
            case .repComplete:
                self.playSound(named: "rep_sound")
                self.triggerHaptic(style: .medium)
            case .falseStart:
                self.triggerHaptic(style: .light)
            }
        }
    }

    private func playSound(named soundName: String) {
        // Using AVAudioPlayer respects the app's main AVAudioSession.
        // This is the architecturally correct way to play short sounds.
        guard let soundURL = Bundle.main.url(forResource: soundName, withExtension: "wav") else {
            print("🔊 SoundManager Error: Could not find sound file named \(soundName).wav")
            return
        }

        do {
            // We create a new player instance each time to ensure sounds can overlap
            // if needed, and to avoid issues with player state.
            self.audioPlayer = try AVAudioPlayer(contentsOf: soundURL)
            self.audioPlayer?.play()
        } catch {
            print("🔊 SoundManager Error: Could not play sound file. Error: \(error.localizedDescription)")
        }
    }

    private func triggerHaptic(style: UIImpactFeedbackGenerator.FeedbackStyle) {
        let generator = UIImpactFeedbackGenerator(style: style)
        generator.prepare()
        generator.impactOccurred()
    }
}
