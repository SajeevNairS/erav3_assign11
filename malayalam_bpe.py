import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import json

class MalayalamBPE:
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[str, str], str] = {}
        self.vocab: Set[str] = set()
        self.byte_encoder = {str(i): bytes([i]).decode('utf-8', errors='replace') for i in range(256)}
        
    def byte_encode(self, text: str) -> List[str]:
        """Convert text to sequence of utf-8 bytes."""
        return [self.byte_encoder[str(b)] for b in text.encode('utf-8')]
    
    def get_stats(self, sequences: List[List[str]]) -> Counter:
        """Count frequency of adjacent pairs in sequences."""
        pairs = Counter()
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                pairs[tuple(sequence[i:i+2])] += 1
        return pairs
    
    def merge_pair(self, pair: Tuple[str, str], sequences: List[List[str]]) -> List[List[str]]:
        """Merge all occurrences of pair in sequences."""
        first, second = pair
        merged = first + second
        new_sequences = []
        
        for sequence in sequences:
            i = 0
            new_sequence = []
            while i < len(sequence):
                if i < len(sequence) - 1 and sequence[i] == first and sequence[i+1] == second:
                    new_sequence.append(merged)
                    i += 2
                else:
                    new_sequence.append(sequence[i])
                    i += 1
            new_sequences.append(new_sequence)
        
        return new_sequences
    
    def train(self, texts: List[str], min_frequency: int = 2):
        """Train BPE on Malayalam texts."""
        # Initialize with byte vocabulary
        sequences = [self.byte_encode(text) for text in texts]
        self.vocab = set(token for sequence in sequences for token in sequence)
        
        while len(self.vocab) < self.vocab_size:
            pairs = self.get_stats(sequences)
            if not pairs:
                break
                
            # Get most frequent pair
            most_freq = max(pairs.items(), key=lambda x: x[1])
            pair, freq = most_freq
            
            if freq < min_frequency:
                break
                
            # Merge pair in all sequences
            sequences = self.merge_pair(pair, sequences)
            self.merges[pair] = pair[0] + pair[1]
            self.vocab.add(pair[0] + pair[1])
            
        return len(self.vocab)
    
    def encode(self, text: str) -> List[str]:
        """Encode text using learned BPE merges."""
        sequence = self.byte_encode(text)
        
        while True:
            pairs = [(tuple(sequence[i:i+2]), i) 
                    for i in range(len(sequence)-1)]
            if not pairs:
                break
                
            # Find mergeable pairs
            mergeable = [(pair, idx) for pair, idx in pairs if pair in self.merges]
            if not mergeable:
                break
                
            # Merge first found pair
            pair, idx = mergeable[0]
            sequence = sequence[:idx] + [self.merges[pair]] + sequence[idx+2:]
            
        return sequence
    
    def decode(self, tokens: List[str]) -> str:
        """Decode tokens back to text."""
        # Convert merged tokens back to bytes
        byte_sequence = []
        for token in tokens:
            # Split merged tokens back to original bytes
            bytes_tokens = [int(b) for b in token.encode('utf-8')]
            byte_sequence.extend(bytes_tokens)
            
        # Convert bytes back to text
        return bytes(byte_sequence).decode('utf-8', errors='replace')
    
    def save(self, path: str):
        """Save BPE model to file."""
        model_data = {
            'merges': {f"{k[0]}|{k[1]}": v for k, v in self.merges.items()},
            'vocab': list(self.vocab)
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """Load BPE model from file."""
        with open(path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        self.vocab = set(model_data['vocab'])
        self.merges = {tuple(k.split('|')): v for k, v in model_data['merges'].items()} 