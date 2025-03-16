from transformers import PreTrainedTokenizer
from typing import List, Optional, Dict
import json
from malayalam_bpe import MalayalamBPE

class MalayalamBPETokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )
        
        self.bpe = MalayalamBPE()
        if vocab_file:
            self.bpe.load(vocab_file)
        
        # Add special tokens to vocabulary
        self.special_tokens = [
            self.unk_token,
            self.sep_token,
            self.pad_token,
            self.cls_token,
            self.mask_token
        ]
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using Malayalam BPE."""
        return self.bpe.encode(text)
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to vocabulary id."""
        if token in self.special_tokens:
            return self.special_tokens.index(token)
        return len(self.special_tokens) + list(self.bpe.vocab).index(token)
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert vocabulary id to token."""
        if index < len(self.special_tokens):
            return self.special_tokens[index]
        return list(self.bpe.vocab)[index - len(self.special_tokens)]
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer configuration and vocabulary."""
        self.bpe.save(f"{save_directory}/vocab.json")
        
        # Save tokenizer config
        config = {
            "vocab_size": len(self.bpe.vocab) + len(self.special_tokens),
            "special_tokens": self.special_tokens,
        }
        with open(f"{save_directory}/config.json", "w") as f:
            json.dump(config, f)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load pretrained tokenizer."""
        tokenizer = cls(**kwargs)
        tokenizer.bpe.load(f"{pretrained_model_name_or_path}/vocab.json")
        return tokenizer 