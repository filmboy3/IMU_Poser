# Fatal Crash: Index Out of Range During TTS Start

Date: 2025-09-06 10:32 EDT
Environment: iOS Simulator (iPhone 16), iOS 18.5 SDK via Xcode 16F6
Build: Debug-iphonesimulator

## Summary
After successfully detecting the “GO” command and coordinating audio session transitions to pause recognition and activate speaking, the app crashes with a fatal Swift runtime error:

```
Swift/ContiguousArrayBuffer.swift:690: Fatal error: Index out of range
```

The crash happens immediately after a speaking session is activated and `AVSpeechSynthesizer` begins speaking the first utterance ("Workout started..."), suggesting a race condition or out-of-bounds array operation triggered by our TTS start coordination.

## Relevant Logs (trimmed)
```
MotionManager: Started IMU updates at 100Hz
🎙️ AudioSessionManager: Activating recording session (current: idle)
✅ AudioSessionManager: Recording session activated successfully
✅ Speech recognition authorized
🎤 Voice control started - say 'go' to start reps, 'stop' to stop
🎤 Voice command: GO - Starting rep counting
🎤 Voice START command received
🏁 Rep counter session started - 1s stabilization period
🗣️ AIVoiceCoach: Processing speech: 'Workout started. Say 'GO' to begin reps.' (queue remaining: 0)
🎧 VoiceController: Pausing recognition for TTS
🔇 AudioSessionManager: Deactivating session (current: recording) → idle
✅ AudioSessionManager: Session deactivated, now idle
📱 SessionRecorder: Started recording session session_20250906_103117_90FFEB66 for Squat
🔄 Rep counter reset
MotionManager: Exercise session reset
🔊 AudioSessionManager: Activating speaking session (current: idle)
✅ AudioSessionManager: Speaking session activated successfully
✅ AIVoiceCoach: Speaking session activated, starting TTS
…
Swift/ContiguousArrayBuffer.swift:690: Fatal error: Index out of range
```

Other benign system messages around the time of the crash:
- Query for com.apple.MobileAsset.VoiceServices… failed: 2
- IPCAUClient: bundle display name is nil

These do not appear as root causes; they are typical on simulators when certain voices are not present and the TTS stack falls back.

## Repro Steps
1. Launch app in the simulator and open `IMUValidationView`.
2. The app starts IMU updates and voice control automatically.
3. Say “GO”.
4. Observe:
   - Voice controller pauses recognition (by design)
   - Recording session deactivates (idle)
   - SessionRecorder starts
   - `AudioSessionManager` activates speaking session
   - `AVSpeechSynthesizer` begins speaking the first utterance
5. Crash occurs immediately after TTS starts.

## Expected vs Actual
- Expected: First utterance plays; after completion, the system resumes recognition and rep counting continues.
- Actual: App crashes with Index out of range before utterance finishes; no recovery.

## Technical Analysis & Hypotheses

1) Race on speech queue state in `AIVoiceCoach`
- Prior to this fix, `AIVoiceCoach` methods could be called from non-main threads (e.g., `MotionManager` deviceMotion callback). This creates potential data races on `speechQueue` and `isSpeaking`.
- An empty queue combined with `removeFirst()` could cause Index out of range if a concurrent path cleared the queue in between checks.
- Mitigation implemented: Main-thread serialization for coach state transitions.
  - `onRepDetected`, `queueSpeech`, `processSpeechQueue`, and the `AVSpeechSynthesizer` delegate now marshal to the main queue and re-check conditions.
  - `processSpeechQueue` now uses `guard let first = speechQueue.first` followed by `removeFirst()` to avoid stale preconditions.

2) Category transition coordination
- We now pause recognition and deactivate recording before activating speaking to avoid '!pri' conflicts. This introduces strict ordering and small delays to let the OS settle routes.
- The crash occurs after activation; less likely related to category change itself (which now succeeds).

3) Other arrays under load in motion pipeline
- `RepCounter` and `SmartExerciseDetector` use ring-buffers with `removeFirst()` gated by size checks; no out-of-range risks detected.
- `SessionRecorder` appends IMU samples to arrays; no indexing operations identified.

Conclusion: The most probable cause is a previously unsynchronized access to `AIVoiceCoach.speechQueue` around the time we begin TTS. We have implemented serialization to the main thread to eliminate this class of crash.

## Code Points Reviewed
- `AIVoiceCoach.processSpeechQueue()` — queue mutation and TTS start
- `AIVoiceCoach.speechSynthesizer(_:didFinish:)` — queue draining and deactivation/resume
- `MotionManager.RepCounter` ring buffers — guarded `removeFirst()` calls
- `SmartExerciseDetector` motion buffers — guarded `removeFirst()` calls
- `SessionRecorder` — append-only, no unsafe indexing

## Fixes Implemented (Phase 1.4 + Crash Hardening)
- `AIVoiceCoach.swift`
  - Serialize all state mutations on the main queue (`onRepDetected`, `queueSpeech`, `processSpeechQueue`, `didFinish`).
  - Use `guard let first = speechQueue.first` + `removeFirst()` instead of pre-calculated index/state to avoid TOCTOU.
  - Continue coordinating: post `.qlvPauseRecognition` before speaking, `.qlvResumeRecognition` after deactivation.

- `VoiceController.swift`
  - Observe `.qlvPauseRecognition`/`.qlvResumeRecognition` notifications.
  - On pause: stop engine, end/cancel task, remove tap, release session via `deactivateSession()`.
  - On resume: call `startListening()` to re-acquire recording session cleanly.

- `AudioSessionManager.swift`
  - Defensive deactivation if `currentState == .recording` before activating speaking.
  - Logging of transitions for diagnosis.

## Validation Plan
- Run on simulator and device.
- Say “GO”; expect:
  - Pause recognition
  - Activate speaking
  - Audio plays; app does not crash
  - After TTS: deactivation → resume recognition
- Confirm no '!pri' / 4097 / 4099 and no fatal crash.

## Next Hardening (if needed)
- Add a short debounce between SessionRecorder start and TTS start (e.g., 200–300 ms) to reduce concurrent initialization pressure.
- Add a small state machine to `AIVoiceCoach` for `idle → pausing → speaking → resuming` to further serialize flows if more issues appear.
- Capture a full crash backtrace via Xcode to pinpoint any remaining edge cases if the crash persists.
