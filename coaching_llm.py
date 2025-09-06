#!/usr/bin/env python3
"""
Generative Coaching LLM for Project Chimera v2
Converts perception model outputs into contextual fitness coaching feedback.
"""

import torch
import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path

class CoachingTrigger(Enum):
    """Types of coaching interventions"""
    REP_COUNT = "rep_count"
    FORM_CORRECTION = "form_correction" 
    ENCOURAGEMENT = "encouragement"
    REST_REMINDER = "rest_reminder"
    WORKOUT_START = "workout_start"
    WORKOUT_END = "workout_end"
    PACE_ADJUSTMENT = "pace_adjustment"

@dataclass
class CoachingContext:
    """Context information for coaching decisions"""
    current_rep: int
    total_reps: int
    workout_duration: float  # seconds
    motion_quality_score: float  # 0-1
    pace_score: float  # 0-1 (0=too slow, 0.5=perfect, 1=too fast)
    fatigue_level: float  # 0-1
    last_coaching_time: float
    exercise_phase: str  # "setup", "active", "rest", "complete"
    motion_tokens: List[int]  # Recent motion token sequence

@dataclass
class CoachingResponse:
    """Generated coaching response"""
    trigger: CoachingTrigger
    message: str
    urgency: float  # 0-1 (0=casual, 1=immediate)
    audio_cue: Optional[str] = None  # System sound to play
    delay_seconds: float = 0.0  # Delay before delivery

class PerceptionAnalyzer:
    """Analyzes perception model outputs for coaching triggers"""
    
    def __init__(self):
        self.motion_history = []
        self.rep_pattern_buffer = []
        self.last_rep_time = 0
        self.baseline_motion_variance = None
        
    def analyze_motion_sequence(self, tokens: List[int], timestamps: List[float]) -> Dict[str, float]:
        """Analyze motion token sequence for coaching insights"""
        if len(tokens) < 5:
            return {"quality": 0.5, "pace": 0.5, "fatigue": 0.0}
            
        # Calculate motion variance (higher = more dynamic movement)
        token_variance = np.var(tokens)
        
        # Establish baseline on first analysis
        if self.baseline_motion_variance is None:
            self.baseline_motion_variance = token_variance
            
        # Quality score based on consistency vs chaos
        if token_variance == 0:
            quality_score = 0.3  # Too static
        else:
            # Good quality = moderate variance (controlled movement)
            variance_ratio = token_variance / (self.baseline_motion_variance + 1e-6)
            quality_score = max(0.1, min(1.0, 1.0 - abs(variance_ratio - 1.0)))
            
        # Pace analysis from timestamps
        if len(timestamps) > 1:
            time_diffs = np.diff(timestamps)
            avg_interval = np.mean(time_diffs)
            # Ideal rep interval ~2-4 seconds
            if 2.0 <= avg_interval <= 4.0:
                pace_score = 1.0
            elif avg_interval < 2.0:
                pace_score = 0.2  # Too fast
            else:
                pace_score = 0.8  # Slightly slow but ok
        else:
            pace_score = 0.5
            
        # Fatigue detection from motion degradation
        recent_variance = np.var(tokens[-10:]) if len(tokens) >= 10 else token_variance
        fatigue_score = max(0.0, (token_variance - recent_variance) / (token_variance + 1e-6))
        
        return {
            "quality": quality_score,
            "pace": pace_score, 
            "fatigue": min(1.0, fatigue_score)
        }
        
    def detect_rep_completion(self, tokens: List[int], timestamps: List[float]) -> bool:
        """Detect when a rep is completed based on motion patterns"""
        if len(tokens) < 8:
            return False
            
        # Look for motion cycle: active -> stable -> active pattern
        recent_tokens = tokens[-8:]
        
        # Check for transition from high variance to low variance (end of rep)
        first_half_var = np.var(recent_tokens[:4])
        second_half_var = np.var(recent_tokens[4:])
        
        # Rep completed if we see high->low variance transition
        if first_half_var > second_half_var * 2 and second_half_var < 10:
            current_time = timestamps[-1] if timestamps else time.time()
            # Prevent double-counting (min 1 second between reps)
            if current_time - self.last_rep_time > 1.0:
                self.last_rep_time = current_time
                return True
                
        return False

class CoachingPersonality:
    """Defines coaching personality and response templates"""
    
    def __init__(self, style: str = "encouraging"):
        self.style = style
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, List[str]]:
        """Load coaching message templates by trigger type"""
        return {
            CoachingTrigger.REP_COUNT.value: [
                "Great! That's {rep_count} reps. Keep it up!",
                "Nice work! {rep_count} down, {remaining} to go!",
                "Excellent form on rep {rep_count}!",
                "{rep_count} reps completed. You're doing amazing!",
                "Perfect! That's {rep_count}. Stay focused!"
            ],
            CoachingTrigger.FORM_CORRECTION.value: [
                "Focus on your form - slow and controlled movements work best.",
                "Remember to maintain good posture throughout the movement.",
                "Try to keep your movements smooth and deliberate.",
                "Great effort! Just focus on quality over speed.",
                "Take your time - perfect form is more important than speed."
            ],
            CoachingTrigger.ENCOURAGEMENT.value: [
                "You're doing fantastic! Keep pushing!",
                "Amazing work! Your consistency is paying off!",
                "I can see your improvement! Stay strong!",
                "You've got this! Every rep counts!",
                "Incredible dedication! You're crushing it!"
            ],
            CoachingTrigger.PACE_ADJUSTMENT.value: [
                "Try slowing down a bit - quality over speed!",
                "You can pick up the pace a little if you're feeling good!",
                "Perfect rhythm! Maintain this pace.",
                "Take a moment between reps to reset your form.",
                "Nice steady pace - this is exactly right!"
            ],
            CoachingTrigger.REST_REMINDER.value: [
                "Great set! Take a 30-second rest before continuing.",
                "Excellent work! Rest up and prepare for the next set.",
                "Perfect! Give yourself a moment to recover.",
                "Well done! Take a breather and stay hydrated.",
                "Outstanding set! Rest and get ready for more."
            ],
            CoachingTrigger.WORKOUT_START.value: [
                "Let's get started! I'll be here to guide you through each rep.",
                "Ready to crush this workout? I'm here to help!",
                "Time to get moving! I'll count your reps and keep you motivated.",
                "Let's do this! Focus on your form and I'll handle the counting.",
                "Workout time! I'm your personal coach - let's make it count!"
            ],
            CoachingTrigger.WORKOUT_END.value: [
                "Incredible workout! You completed {total_reps} reps in {duration} minutes!",
                "Amazing job! {total_reps} reps done with excellent form!",
                "Workout complete! {total_reps} reps - you should be proud!",
                "Fantastic session! {total_reps} quality reps in the books!",
                "Outstanding work! {total_reps} reps completed - well done!"
            ]
        }
        
    def generate_message(self, trigger: CoachingTrigger, context: CoachingContext) -> str:
        """Generate contextual coaching message"""
        templates = self.templates.get(trigger.value, ["Great job!"])
        
        # Select template based on context
        if trigger == CoachingTrigger.REP_COUNT:
            template = np.random.choice(templates)
            remaining = max(0, context.total_reps - context.current_rep)
            return template.format(rep_count=context.current_rep, remaining=remaining)
            
        elif trigger == CoachingTrigger.WORKOUT_END:
            template = np.random.choice(templates)
            duration_min = context.workout_duration / 60
            return template.format(total_reps=context.total_reps, duration=f"{duration_min:.1f}")
            
        elif trigger == CoachingTrigger.FORM_CORRECTION:
            # Choose message based on quality score
            if context.motion_quality_score < 0.3:
                return "Focus on your form - slow and controlled movements work best."
            elif context.motion_quality_score < 0.6:
                return "Try to keep your movements smooth and deliberate."
            else:
                return "Great effort! Just focus on quality over speed."
                
        elif trigger == CoachingTrigger.PACE_ADJUSTMENT:
            if context.pace_score < 0.3:
                return "Try slowing down a bit - quality over speed!"
            elif context.pace_score > 0.7:
                return "You can pick up the pace a little if you're feeling good!"
            else:
                return "Perfect rhythm! Maintain this pace."
                
        else:
            return np.random.choice(templates)

class GenerativeCoachingLLM:
    """Main coaching LLM that generates contextual fitness feedback"""
    
    def __init__(self, personality_style: str = "encouraging"):
        self.analyzer = PerceptionAnalyzer()
        self.personality = CoachingPersonality(personality_style)
        self.coaching_history = []
        self.session_start_time = None
        self.last_coaching_time = 0
        self.min_coaching_interval = 3.0  # Minimum seconds between coaching
        
    def start_session(self) -> CoachingResponse:
        """Initialize coaching session"""
        self.session_start_time = time.time()
        self.last_coaching_time = 0
        self.coaching_history = []
        
        context = CoachingContext(
            current_rep=0, total_reps=0, workout_duration=0,
            motion_quality_score=1.0, pace_score=0.5, fatigue_level=0.0,
            last_coaching_time=0, exercise_phase="setup", motion_tokens=[]
        )
        
        message = self.personality.generate_message(CoachingTrigger.WORKOUT_START, context)
        
        response = CoachingResponse(
            trigger=CoachingTrigger.WORKOUT_START,
            message=message,
            urgency=0.8,
            audio_cue="start_workout",
            delay_seconds=0.5
        )
        
        self.coaching_history.append(response)
        return response
        
    def process_perception_output(self, motion_tokens: List[int], timestamps: List[float], 
                                current_rep: int, target_reps: int = 10) -> Optional[CoachingResponse]:
        """Process perception model output and generate coaching if needed"""
        
        if not motion_tokens or not timestamps:
            return None
            
        current_time = time.time()
        
        # Skip if too soon since last coaching
        if current_time - self.last_coaching_time < self.min_coaching_interval:
            return None
            
        # Analyze motion patterns
        analysis = self.analyzer.analyze_motion_sequence(motion_tokens, timestamps)
        
        # Check for rep completion
        rep_completed = self.analyzer.detect_rep_completion(motion_tokens, timestamps)
        
        # Build context
        workout_duration = current_time - (self.session_start_time or current_time)
        context = CoachingContext(
            current_rep=current_rep,
            total_reps=target_reps,
            workout_duration=workout_duration,
            motion_quality_score=analysis["quality"],
            pace_score=analysis["pace"],
            fatigue_level=analysis["fatigue"],
            last_coaching_time=self.last_coaching_time,
            exercise_phase="active",
            motion_tokens=motion_tokens[-10:]  # Recent tokens
        )
        
        # Determine coaching trigger
        trigger = self._select_coaching_trigger(context, rep_completed)
        
        if not trigger:
            return None
            
        # Generate response
        message = self.personality.generate_message(trigger, context)
        urgency = self._calculate_urgency(trigger, context)
        
        response = CoachingResponse(
            trigger=trigger,
            message=message,
            urgency=urgency,
            audio_cue=self._get_audio_cue(trigger),
            delay_seconds=self._get_delay(trigger, context)
        )
        
        self.last_coaching_time = current_time
        self.coaching_history.append(response)
        
        return response
        
    def end_session(self, final_rep_count: int) -> CoachingResponse:
        """Generate session completion coaching"""
        workout_duration = time.time() - (self.session_start_time or time.time())
        
        context = CoachingContext(
            current_rep=final_rep_count,
            total_reps=final_rep_count,
            workout_duration=workout_duration,
            motion_quality_score=1.0,
            pace_score=0.5,
            fatigue_level=0.0,
            last_coaching_time=self.last_coaching_time,
            exercise_phase="complete",
            motion_tokens=[]
        )
        
        message = self.personality.generate_message(CoachingTrigger.WORKOUT_END, context)
        
        response = CoachingResponse(
            trigger=CoachingTrigger.WORKOUT_END,
            message=message,
            urgency=0.9,
            audio_cue="workout_complete",
            delay_seconds=1.0
        )
        
        self.coaching_history.append(response)
        return response
        
    def _select_coaching_trigger(self, context: CoachingContext, rep_completed: bool) -> Optional[CoachingTrigger]:
        """Select appropriate coaching trigger based on context"""
        
        # Rep completion has highest priority
        if rep_completed:
            return CoachingTrigger.REP_COUNT
            
        # Form correction for poor quality
        if context.motion_quality_score < 0.4:
            return CoachingTrigger.FORM_CORRECTION
            
        # Pace adjustment for extreme pace issues
        if context.pace_score < 0.3 or context.pace_score > 0.8:
            return CoachingTrigger.PACE_ADJUSTMENT
            
        # Encouragement for fatigue
        if context.fatigue_level > 0.6:
            return CoachingTrigger.ENCOURAGEMENT
            
        # Periodic encouragement (every 15 seconds)
        if context.workout_duration - context.last_coaching_time > 15.0:
            return CoachingTrigger.ENCOURAGEMENT
            
        return None
        
    def _calculate_urgency(self, trigger: CoachingTrigger, context: CoachingContext) -> float:
        """Calculate message urgency (0-1)"""
        urgency_map = {
            CoachingTrigger.REP_COUNT: 0.9,
            CoachingTrigger.FORM_CORRECTION: 0.7,
            CoachingTrigger.PACE_ADJUSTMENT: 0.6,
            CoachingTrigger.ENCOURAGEMENT: 0.4,
            CoachingTrigger.REST_REMINDER: 0.5,
            CoachingTrigger.WORKOUT_START: 0.8,
            CoachingTrigger.WORKOUT_END: 0.9
        }
        return urgency_map.get(trigger, 0.5)
        
    def _get_audio_cue(self, trigger: CoachingTrigger) -> Optional[str]:
        """Get system audio cue for trigger"""
        audio_map = {
            CoachingTrigger.REP_COUNT: "rep_complete",
            CoachingTrigger.WORKOUT_START: "start_workout", 
            CoachingTrigger.WORKOUT_END: "workout_complete",
            CoachingTrigger.FORM_CORRECTION: "attention",
        }
        return audio_map.get(trigger)
        
    def _get_delay(self, trigger: CoachingTrigger, context: CoachingContext) -> float:
        """Calculate delivery delay for natural timing"""
        if trigger == CoachingTrigger.REP_COUNT:
            return 0.2  # Quick feedback on reps
        elif trigger == CoachingTrigger.FORM_CORRECTION:
            return 0.5  # Slight delay for form feedback
        elif trigger == CoachingTrigger.ENCOURAGEMENT:
            return 1.0  # More natural timing for encouragement
        else:
            return 0.3
            
    def get_coaching_summary(self) -> Dict[str, any]:
        """Get summary of coaching session"""
        if not self.coaching_history:
            return {"total_messages": 0}
            
        trigger_counts = {}
        for response in self.coaching_history:
            trigger_counts[response.trigger.value] = trigger_counts.get(response.trigger.value, 0) + 1
            
        return {
            "total_messages": len(self.coaching_history),
            "trigger_breakdown": trigger_counts,
            "session_duration": time.time() - (self.session_start_time or time.time()),
            "avg_coaching_interval": len(self.coaching_history) / max(1, (time.time() - (self.session_start_time or time.time())) / 60)
        }

def main():
    """Demo the coaching LLM"""
    print("=== Generative Coaching LLM Demo ===")
    
    coach = GenerativeCoachingLLM("encouraging")
    
    # Start session
    start_response = coach.start_session()
    print(f"Session Start: {start_response.message}")
    
    # Simulate workout with motion tokens
    demo_tokens = [1025, 1025, 1055, 1058, 1078, 1030, 1025, 1025]  # Simulated motion pattern
    demo_timestamps = [i * 0.5 for i in range(len(demo_tokens))]  # 0.5s intervals
    
    current_rep = 0
    for i in range(5):  # Simulate 5 reps
        time.sleep(1)  # Simulate time passage
        
        # Add some variation to tokens
        varied_tokens = demo_tokens + [1025 + np.random.randint(-5, 5) for _ in range(3)]
        varied_timestamps = demo_timestamps + [demo_timestamps[-1] + j * 0.1 for j in range(1, 4)]
        
        response = coach.process_perception_output(varied_tokens, varied_timestamps, current_rep + 1, 10)
        
        if response:
            print(f"Rep {current_rep + 1}: [{response.trigger.value}] {response.message}")
            current_rep += 1
            
    # End session
    end_response = coach.end_session(current_rep)
    print(f"Session End: {end_response.message}")
    
    # Print summary
    summary = coach.get_coaching_summary()
    print(f"\nSession Summary: {summary}")

if __name__ == "__main__":
    main()
