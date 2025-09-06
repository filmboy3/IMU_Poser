# Project Chimera v2: Unified Perception Engine Architecture

Date: 2025-09-06 11:09 EDT  
Status: **ARCHITECTURAL PIVOT** - Moving from decoupled state machines to unified multimodal perception  
Inspiration: Meta's "Sesame" Conversational Speech Model + IMU fusion for embodied AI

## Executive Summary

We are pivoting from our current brittle, decoupled architecture to a unified perception engine that processes audio and IMU data as a single interleaved token stream. This addresses the root cause of our session failures, state thrashing, and audio coordination issues by eliminating the need for separate components to fight over hardware resources.

**Core Innovation**: The first Embodied Conversational Agent that achieves both "voice presence" (like Sesame) and "physical presence" through IMU fusion.

## The Fundamental Problem with Current Architecture

Our current system treats perception as separate, competing processes:
- `VoiceController` owns microphone input
- `AIVoiceCoach` owns speaker output  
- `SmartExerciseDetector` owns motion classification
- `AudioSessionManager` frantically orchestrates hardware handoffs

This creates inevitable race conditions, session conflicts, and state synchronization failures. The logs show the symptoms: detector thrashing, audio session errors, speech recognition collapse.

## The Unified Perception Solution

### Core Principle: Single Model, Holistic Understanding

Instead of separate detectors, we build one transformer that understands the complete user context:
- Audio tokens (speech, breathing, grunts, ambient noise)
- Motion tokens (squat phases, phone handling, stability states)
- Temporal relationships between modalities

The model learns workout dynamics implicitly: after "GO" audio tokens, expect squat motion tokens. After rep completion tokens, coaching audio tokens follow.

## Technical Architecture

### Phase 1: Unified Data Pipeline & Perception Model

#### Milestone 1.1: Multi-Modal Data Tokenizer
**Objective**: Convert raw audio + IMU streams into unified token vocabulary

**Audio Tokenization**:
- Use pre-trained EnCodec or similar RVQ model
- Convert 16kHz audio → discrete acoustic tokens
- Captures speech, breathing, environmental audio
- ~50 tokens/second temporal resolution

**IMU Tokenization**:
- Train VQ-VAE on IMU windows (1s @ 100Hz = 100 samples)
- Learn codebook of fundamental motion primitives:
  - `squat_descending`, `squat_ascending`, `phone_rustling`
  - `arm_stable`, `transition_motion`, `handling_detected`
- ~10 motion tokens/second temporal resolution

**Interleaved Stream Format**:
```
[audio_001, audio_002, motion_squat_desc, audio_003, audio_004, motion_squat_asc, ...]
```

**Implementation**:
- Python pipeline for offline tokenization of session recordings
- Real-time tokenization pipeline for live inference
- Token vocabulary size: ~1024 audio + ~64 motion = ~1088 total

#### Milestone 1.2: Unified Perception Transformer
**Objective**: Single causal transformer predicts next token in interleaved sequence

**Architecture**:
- Llama-style decoder-only transformer
- Context window: ~8192 tokens (~3-4 minutes of workout)
- Embedding dimension: 1024
- Layers: 12-16 (targeting ~100M parameters)

**Training Data**:
- Existing session recordings tokenized as interleaved streams
- Self-supervised next-token prediction objective
- No manual labeling required - learns workout patterns implicitly

**Emergent Capabilities**:
- Rep counting: Model learns to predict rep completion tokens
- Form analysis: Learns motion quality patterns
- Coaching timing: Learns when to generate coaching tokens
- State management: No explicit state machine needed

### Phase 2: Generative Coaching & Real-Time Interaction

#### Milestone 2.1: Generative Coaching LLM
**Objective**: Replace static announcements with dynamic, contextual coaching

**Coach Brain Architecture**:
- Fine-tuned Llama 3 8B or Phi-3 model
- Specialized for fitness coaching domain
- Input: Special tokens from Perception Model (`[REP_5_GOOD_FORM]`, `[REP_3_KNEE_VALGUS]`)
- Output: Contextual coaching text

**Example Flow**:
1. Perception Model detects: `[REP_5_COMPLETED, FORM_ISSUE_KNEE_VALGUS]`
2. Coach Brain receives prompt: "User completed rep 5 with knee valgus. Provide corrective cue."
3. Generates: "Great job on five! Next rep, imagine pushing the floor apart with your feet."
4. Text → TTS → Audio output

#### Milestone 2.2: Real-Time Duplex Audio Pipeline
**Objective**: Eliminate audio session coordination issues entirely

**Single Persistent Session**:
- One `.playAndRecord` session for entire workout
- No session switching, no coordination failures
- Continuous bidirectional audio stream

**Streaming Architecture**:
- Microphone → Real-time tokenizer → Perception Model
- Coach Brain → TTS → Speaker (simultaneous with listening)
- Acoustic Echo Cancellation prevents feedback loops

**Technical Stack**:
- iOS: AVAudioEngine with real-time processing
- AEC: iOS built-in or SpeexDSP library
- Streaming inference: Core ML or ONNX Runtime

## Implementation Roadmap

### Phase 1 Timeline (4-6 weeks)
1. **Week 1-2**: Build tokenization pipeline, collect training data
2. **Week 3-4**: Train Perception Transformer on existing sessions
3. **Week 5-6**: Implement real-time inference pipeline

### Phase 2 Timeline (3-4 weeks)
1. **Week 1-2**: Fine-tune Coach Brain LLM, integrate with Perception Model
2. **Week 3-4**: Build duplex audio pipeline, end-to-end testing

## Competitive Advantages

### Technical Moats
1. **First Embodied Conversational Agent**: No competitor has IMU + audio fusion
2. **Proprietary Motion Vocabulary**: Our VQ-VAE learns exercise-specific motion primitives
3. **Workout-Specific Transformer**: Model understands fitness context deeply

### User Experience Breakthroughs
1. **Zero Configuration**: No manual exercise selection or setup
2. **Contextual Coaching**: Dynamic feedback based on actual performance
3. **Natural Interaction**: Conversational flow without artificial pauses

## Risk Mitigation

### Technical Risks
- **Model Size**: Target 100M parameters for on-device inference
- **Latency**: <200ms end-to-end for real-time feel
- **Accuracy**: Validate on diverse user populations and exercise types

### Fallback Strategy
- Keep simplified version of current architecture as backup
- Gradual rollout with A/B testing
- Progressive enhancement approach

## Success Metrics

### Phase 1 Success Criteria
- Perception Model achieves >95% accuracy on rep counting
- Zero audio session coordination failures
- <100ms tokenization latency for real-time use

### Phase 2 Success Criteria
- Coach Brain generates contextually appropriate responses >90% of time
- End-to-end latency <200ms from user action to coaching response
- User satisfaction scores improve 2x over current system

## Next Steps

1. **Immediate**: Begin collecting and tokenizing existing session data
2. **Week 1**: Start VQ-VAE training for motion tokenization
3. **Week 2**: Begin Perception Transformer architecture implementation
4. **Ongoing**: Document architecture decisions and maintain fallback compatibility

---

This architecture represents a fundamental paradigm shift from reactive state machines to proactive, intelligent perception. It positions us to build not just a better fitness app, but the foundation for embodied AI interactions across multiple domains.
