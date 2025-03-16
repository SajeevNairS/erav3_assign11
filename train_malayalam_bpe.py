import glob
from malayalam_bpe import MalayalamBPE

def load_malayalam_texts(directory: str) -> list:
    """Load Malayalam text files from directory."""
    texts = []
    for filepath in glob.glob(f"{directory}/*.txt"):
        with open(filepath, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return texts

def calculate_compression_ratio(original_texts: list, encoded_texts: list) -> float:
    """Calculate compression ratio between original and encoded texts."""
    original_size = sum(len(text.encode('utf-8')) for text in original_texts)
    encoded_size = sum(len(''.join(tokens).encode('utf-8')) for tokens in encoded_texts)
    return original_size / encoded_size

def main():
    # Load Malayalam texts
    texts = load_malayalam_texts("malayalam_texts")
    
    # Initialize and train BPE
    bpe = MalayalamBPE(vocab_size=4500)  # Target slightly lower to have room for adjustment
    vocab_size = bpe.train(texts, min_frequency=3)
    
    # Encode texts and calculate compression ratio
    encoded_texts = [bpe.encode(text) for text in texts]
    compression_ratio = calculate_compression_ratio(texts, encoded_texts)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Compression ratio: {compression_ratio:.2f}")
    
    # Verify decoding
    decoded_text = bpe.decode(encoded_texts[0])
    print("\nVerification:")
    print(f"Original text sample: {texts[0][:100]}")
    print(f"Decoded text sample: {decoded_text[:100]}")
    
    # Save model if requirements are met
    if vocab_size < 5000 and compression_ratio >= 3.2:
        bpe.save("malayalam_bpe_model.json")
        print("\nModel saved successfully!")
    else:
        print("\nRequirements not met:")
        print(f"Vocabulary size {'OK' if vocab_size < 5000 else 'Too large'}")
        print(f"Compression ratio {'OK' if compression_ratio >= 3.2 else 'Too low'}")

if __name__ == "__main__":
    main() 