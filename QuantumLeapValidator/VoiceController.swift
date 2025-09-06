import Foundation
import Speech
import AVFoundation

extension Notification.Name {
    static let qlvPauseRecognition = Notification.Name("QLVPauseRecognition")
    static let qlvResumeRecognition = Notification.Name("QLVResumeRecognition")
}

class VoiceController: NSObject, ObservableObject {
    
    @Published var isListening = false
    @Published var recognizedText = ""
    @Published var isRepCountingActive = false
    
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    private let audioManager = AudioSessionManager.shared
    
    // Callbacks for start/stop commands
    var onStartCommand: (() -> Void)?
    var onStopCommand: (() -> Void)?
    
    override init() {
        super.init()
        NotificationCenter.default.addObserver(self, selector: #selector(handlePauseRecognition), name: .qlvPauseRecognition, object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(handleResumeRecognition), name: .qlvResumeRecognition, object: nil)
        requestSpeechAuthorization()
    }
    
    private func requestSpeechAuthorization() {
        SFSpeechRecognizer.requestAuthorization { authStatus in
            DispatchQueue.main.async {
                switch authStatus {
                case .authorized:
                    print("✅ Speech recognition authorized")
                case .denied:
                    print("❌ Speech recognition denied")
                case .restricted:
                    print("❌ Speech recognition restricted")
                case .notDetermined:
                    print("⏳ Speech recognition not determined")
                @unknown default:
                    print("❓ Unknown speech recognition status")
                }
            }
        }
    }
    
    func startListening() {
        guard !isListening else { return }
        
        audioManager.activateRecordingSession { [weak self] success in
            guard success, let self = self else { return }
            
            do {
                try self.startRecording()
                DispatchQueue.main.async {
                    self.isListening = true
                    print("🎤 Voice control started - say 'go' to start reps, 'stop' to stop")
                }
            } catch {
                print("❌ Failed to start voice recognition: \(error)")
            }
        }
    }
    
    func stopListening() {
        audioEngine.stop()
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        
        recognitionRequest = nil
        recognitionTask = nil
        isListening = false
        
        audioManager.deactivateSession()
        
        print("🔇 Voice control stopped")
    }
    
    private func startRecording() throws {
        // Cancel any previous task
        recognitionTask?.cancel()
        recognitionTask = nil
        
        // Audio session is now managed by AudioSessionManager - no direct configuration needed
        
        // Create recognition request
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else {
            throw VoiceControlError.recognitionRequestFailed
        }
        
        recognitionRequest.shouldReportPartialResults = true
        
        // Configure audio engine
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            recognitionRequest.append(buffer)
        }
        
        audioEngine.prepare()
        try audioEngine.start()
        
        // Start recognition task
        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            guard let self = self else { return }
            
            if let result = result {
                let recognizedText = result.bestTranscription.formattedString.lowercased()
                
                DispatchQueue.main.async {
                    self.recognizedText = recognizedText
                    self.processVoiceCommand(recognizedText)
                }
            }
            
            if error != nil || result?.isFinal == true {
                self.audioEngine.stop()
                inputNode.removeTap(onBus: 0)
                self.recognitionRequest = nil
                self.recognitionTask = nil
            }
        }
    }
    
    private func processVoiceCommand(_ text: String) {
        let words = text.components(separatedBy: .whitespaces)
        
        // Look for "go" command to start rep counting
        if words.contains("go") && !isRepCountingActive {
            print("🎤 Voice command: GO - Starting rep counting")
            isRepCountingActive = true
            onStartCommand?()
            
            // Play confirmation sound
            AudioServicesPlaySystemSound(1054) // Pop sound
        }
        
        // Look for "stop" command to stop rep counting
        if words.contains("stop") && isRepCountingActive {
            print("🎤 Voice command: STOP - Stopping rep counting")
            isRepCountingActive = false
            onStopCommand?()
            
            // Play confirmation sound
            AudioServicesPlaySystemSound(1055) // Tock sound
        }
    }
    
    func reset() {
        isRepCountingActive = false
        recognizedText = ""
    }

    @objc private func handlePauseRecognition() {
        guard isListening else { return }
        print("🎧 VoiceController: Pausing recognition for TTS")
        audioEngine.stop()
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        // Remove any existing tap to avoid duplicate taps on resume
        let inputNode = audioEngine.inputNode
        inputNode.removeTap(onBus: 0)
        recognitionRequest = nil
        recognitionTask = nil
        isListening = false
        // Release the session so TTS can take exclusive control
        audioManager.deactivateSession()
    }

    @objc private func handleResumeRecognition() {
        print("🎧 VoiceController: Resuming recognition after TTS")
        startListening()
    }
}

enum VoiceControlError: Error {
    case recognitionRequestFailed
    case audioEngineFailed
}
