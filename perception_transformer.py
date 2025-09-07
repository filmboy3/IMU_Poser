#!/usr/bin/env python3
"""
Unified Perception Transformer for Project Chimera v2
Single causal transformer that processes interleaved audio+IMU token streams
and learns workout dynamics implicitly through next-token prediction.

Phase 1.2: Unified Perception Transformer Implementation
- Llama-style decoder-only transformer architecture
- Processes unified token vocabulary (audio + motion)
- Learns rep counting, form analysis, and coaching timing emergently
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from tokenization_pipeline import MultiModalTokenizer, TokenizedSequence

@dataclass
class TransformerConfig:
    """Configuration for the Unified Perception Transformer"""
    vocab_size: int = 1400  # 256 audio + 256 IMU + special tokens + buffer
    max_seq_len: int = 8192  # ~3-4 minutes of workout context
    d_model: int = 1024  # Embedding dimension
    n_heads: int = 16  # Attention heads
    n_layers: int = 12  # Transformer layers
    d_ff: int = 4096  # Feed-forward dimension
    dropout: float = 0.1
    
    # Multimodal token ranges
    imu_token_start: int = 0
    imu_token_end: int = 256
    audio_token_start: int = 1000
    audio_token_end: int = 1256
    
    # Special tokens
    rep_start_token: int = 1300
    rep_end_token: int = 1301
    session_start_token: int = 1302
    session_end_token: int = 1303
    coaching_trigger_token: int = 1304
    
class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better sequence modeling"""
    
    def __init__(self, d_model: int, max_seq_len: int = 8192):
        super().__init__()
        self.d_model = d_model
        
        # Create frequency tensor
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Pre-compute positional encodings
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to query and key tensors"""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryPositionalEmbedding(self.d_head, config.max_seq_len)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rope(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.o_proj(out)

class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(W1 * x) * (W3 * x) -> W2
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.attention(self.ln1(x), mask)
        # Pre-norm feed-forward
        x = x + self.feed_forward(self.ln2(x))
        return x

class UnifiedPerceptionTransformer(nn.Module):
    """Main transformer model for unified perception"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
            
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, temperature: float = 1.0) -> torch.Tensor:
        """Generate new tokens autoregressively"""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get predictions for the last token
                logits = self.forward(input_ids)
                logits = logits[:, -1, :] / temperature
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if we exceed max sequence length
                if input_ids.shape[1] >= self.config.max_seq_len:
                    break
                    
        return input_ids
    
    def predict_rep_and_coaching(self, token_sequence: List[int], context_length: int = 128) -> Dict[str, any]:
        """
        Predict rep detection and coaching triggers from multimodal token sequence
        
        Args:
            token_sequence: List of interleaved IMU and audio tokens
            context_length: Number of recent tokens to consider
            
        Returns:
            Dictionary with rep detection and coaching predictions
        """
        self.eval()
        
        # Take recent context
        if len(token_sequence) > context_length:
            context_tokens = token_sequence[-context_length:]
        else:
            context_tokens = token_sequence
            
        if not context_tokens:
            return {"rep_detected": False, "coaching_trigger": False, "confidence": 0.0}
        
        # Convert to tensor
        input_ids = torch.tensor([context_tokens], dtype=torch.long)
        
        with torch.no_grad():
            # Get model predictions
            logits = self.forward(input_ids)
            last_logits = logits[0, -1, :]  # Last token predictions
            
            # Check for special token predictions
            rep_start_prob = torch.softmax(last_logits, dim=-1)[self.config.rep_start_token].item()
            coaching_prob = torch.softmax(last_logits, dim=-1)[self.config.coaching_trigger_token].item()
            
            # Analyze token patterns for rep detection
            imu_tokens = [t for t in context_tokens if self.config.imu_token_start <= t < self.config.imu_token_end]
            audio_tokens = [t for t in context_tokens if self.config.audio_token_start <= t < self.config.audio_token_end]
            
            # Simple pattern-based rep detection (enhanced by transformer predictions)
            rep_detected = False
            confidence = 0.0
            
            if len(imu_tokens) >= 10:  # Need sufficient IMU context
                # Look for repetitive patterns in IMU tokens
                token_variance = np.var(imu_tokens[-10:]) if len(imu_tokens) >= 10 else 0
                
                # Combine pattern analysis with transformer prediction
                pattern_score = min(1.0, token_variance / 100.0)  # Normalize variance
                transformer_score = rep_start_prob
                
                combined_score = 0.7 * pattern_score + 0.3 * transformer_score
                confidence = combined_score
                
                # Threshold for rep detection
                rep_detected = combined_score > 0.5
            
            # Coaching trigger based on model prediction and context
            coaching_trigger = coaching_prob > 0.3 or (len(audio_tokens) > 5 and coaching_prob > 0.1)
            
            return {
                "rep_detected": rep_detected,
                "coaching_trigger": coaching_trigger,
                "confidence": confidence,
                "rep_probability": rep_start_prob,
                "coaching_probability": coaching_prob,
                "imu_token_count": len(imu_tokens),
                "audio_token_count": len(audio_tokens)
            }
    
    def analyze_multimodal_context(self, imu_tokens: List[int], audio_tokens: List[int]) -> Dict[str, any]:
        """
        Analyze multimodal context for workout insights
        
        Args:
            imu_tokens: Recent IMU tokens
            audio_tokens: Recent audio tokens
            
        Returns:
            Analysis results with workout insights
        """
        # Interleave tokens (simple alternating pattern)
        interleaved = []
        max_len = max(len(imu_tokens), len(audio_tokens))
        
        for i in range(max_len):
            if i < len(imu_tokens):
                interleaved.append(imu_tokens[i])
            if i < len(audio_tokens):
                interleaved.append(audio_tokens[i] + self.config.audio_token_start)
        
        # Get predictions
        predictions = self.predict_rep_and_coaching(interleaved)
        
        # Add multimodal analysis
        predictions.update({
            "modality_balance": len(audio_tokens) / (len(imu_tokens) + 1e-6),
            "total_tokens": len(interleaved),
            "sequence_complexity": len(set(interleaved)) / len(interleaved) if interleaved else 0
        })
        
        return predictions

class WorkoutDataset(Dataset):
    """Dataset for training the perception transformer"""
    
    def __init__(self, tokenized_sessions: List[TokenizedSequence], seq_len: int = 64):
        self.sessions = tokenized_sessions
        self.seq_len = seq_len
        self.sequences = self._create_sequences()
        
    def _create_sequences(self) -> List[List[int]]:
        """Create training sequences from tokenized sessions"""
        sequences = []
        
        for session in self.sessions:
            tokens = session.tokens
            
            # Skip sessions that are too short
            if len(tokens) < self.seq_len + 1:
                continue
                
            # Create overlapping sequences
            step_size = max(1, self.seq_len // 4)  # 25% overlap
            for i in range(0, len(tokens) - self.seq_len, step_size):
                seq = tokens[i:i + self.seq_len + 1]  # +1 for target
                if len(seq) == self.seq_len + 1:
                    sequences.append(seq)
                    
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        
        return {
            'input_ids': torch.tensor(seq[:-1], dtype=torch.long),
            'labels': torch.tensor(seq[1:], dtype=torch.long)
        }

def train_perception_transformer(
    tokenized_sessions: List[TokenizedSequence],
    config: TransformerConfig,
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    device: str = 'mps'  # Use Metal Performance Shaders on Mac
) -> UnifiedPerceptionTransformer:
    """Train the unified perception transformer"""
    
    print(f"Training Unified Perception Transformer...")
    print(f"Config: {config.n_layers} layers, {config.d_model} dim, {config.n_heads} heads")
    print(f"Training on {len(tokenized_sessions)} sessions")
    
    # Create dataset and dataloader
    dataset = WorkoutDataset(tokenized_sessions, seq_len=64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Created {len(dataset)} training sequences")
    
    # Initialize model
    model = UnifiedPerceptionTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Training loop
    model.train()
    total_loss = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        avg_loss = epoch_loss / num_batches
        total_loss += avg_loss
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
    print(f"Training complete. Average loss: {total_loss / num_epochs:.4f}")
    return model

def load_tokenized_sessions(session_paths: List[str], tokenizer: MultiModalTokenizer) -> List[TokenizedSequence]:
    """Load and tokenize all training sessions"""
    tokenized_sessions = []
    
    print("Tokenizing training sessions...")
    for session_path in session_paths:
        try:
            tokenized = tokenizer.tokenize_session(session_path)
            if len(tokenized.tokens) > 50:  # Only use sessions with sufficient data
                tokenized_sessions.append(tokenized)
                print(f"  {Path(session_path).name}: {len(tokenized.tokens)} tokens")
        except Exception as e:
            print(f"  Error tokenizing {Path(session_path).name}: {e}")
            
    return tokenized_sessions

def main():
    """Main training pipeline"""
    print("=== Project Chimera v2: Unified Perception Transformer Training ===")
    
    # Load trained tokenizer
    tokenizer_path = "/Users/jonathanschwartz/Documents/Social_Content/QuantumLeapValidator/trained_tokenizer.pkl"
    tokenizer = MultiModalTokenizer()
    tokenizer.load_tokenizer(tokenizer_path)
    print(f"Loaded tokenizer from: {tokenizer_path}")
    
    # Collect session paths
    from tokenization_pipeline import collect_training_sessions
    downloads_dir = "/Users/jonathanschwartz/Downloads"
    session_paths = collect_training_sessions(downloads_dir)
    
    # Tokenize all sessions
    tokenized_sessions = load_tokenized_sessions(session_paths, tokenizer)
    print(f"Successfully tokenized {len(tokenized_sessions)} sessions")
    
    if len(tokenized_sessions) == 0:
        print("No valid sessions found. Exiting.")
        return
    
    # Configure model
    config = TransformerConfig(
        vocab_size=1088,
        max_seq_len=4096,  # Reduced for training
        d_model=512,       # Reduced for faster training
        n_heads=8,         # Reduced for faster training
        n_layers=6,        # Reduced for faster training
        d_ff=2048,         # Reduced for faster training
        dropout=0.1
    )
    
    # Train model
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Training on device: {device}")
    
    model = train_perception_transformer(
        tokenized_sessions=tokenized_sessions,
        config=config,
        num_epochs=5,  # Reduced for initial training
        batch_size=2,  # Small batch size for memory
        learning_rate=1e-4,
        device=device
    )
    
    # Save trained model
    model_path = "/Users/jonathanschwartz/Documents/Social_Content/QuantumLeapValidator/perception_transformer.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'vocab_size': config.vocab_size
    }, model_path)
    
    print(f"Model saved to: {model_path}")
    
    # Test generation
    print("\nTesting model generation...")
    model.eval()
    
    # Create a sample input (first few tokens from a session)
    if tokenized_sessions:
        sample_tokens = tokenized_sessions[0].tokens[:20]
        input_ids = torch.tensor([sample_tokens], dtype=torch.long).to(device)
        
        generated = model.generate(input_ids, max_new_tokens=10, temperature=0.8)
        
        print("Sample generation:")
        print(f"Input:  {sample_tokens}")
        print(f"Output: {generated[0].cpu().tolist()}")
    
    print("\n=== Perception Transformer Training Complete ===")

if __name__ == "__main__":
    main()
