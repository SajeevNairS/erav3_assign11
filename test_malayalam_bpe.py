import os
from malayalam_bpe import MalayalamBPE

def create_test_data():
    """Create test Malayalam text files."""
    os.makedirs("malayalam_texts", exist_ok=True)
    
    # Sample Malayalam text (you should replace this with real Malayalam text)
    test_text = """
    മലയാളം ഒരു ദ്രാവിഡ ഭാഷയാണ്. ഇന്ത്യയിലെ കേരള സംസ്ഥാനത്തിന്റെ ഔദ്യോഗിക ഭാഷയാണ് മലയാളം.
    കേരളത്തിലെ ഭൂരിഭാഗം ജനങ്ങളും സംസാരിക്കുന്ന ഭാഷയാണിത്.
    ലക്ഷദ്വീപിലെയും ഔദ്യോഗിക ഭാഷയാണ് മലയാളം.
    """
    
    # Create multiple test files
    for i in range(3):
        with open(f"malayalam_texts/test_{i}.txt", "w", encoding="utf-8") as f:
            f.write(test_text * (i + 1))  # Varying lengths of text

def test_bpe():
    # Create test data
    create_test_data()
    
    # Train BPE
    from train_malayalam_bpe import main
    main()
    
    # Test loading and using the model
    if os.path.exists("malayalam_bpe_model.json"):
        bpe = MalayalamBPE()
        bpe.load("malayalam_bpe_model.json")
        
        # Test encoding and decoding
        test_text = "മലയാളം ഒരു ഭാഷയാണ്"
        encoded = bpe.encode(test_text)
        decoded = bpe.decode(encoded)
        
        print("\nTesting encoding and decoding:")
        print(f"Original: {test_text}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
        print(f"Match: {test_text == decoded}")
        
        # Print vocabulary statistics
        print(f"\nVocabulary size: {len(bpe.vocab)}")
        print(f"Sample tokens: {list(bpe.vocab)[:10]}")

if __name__ == "__main__":
    test_bpe() 