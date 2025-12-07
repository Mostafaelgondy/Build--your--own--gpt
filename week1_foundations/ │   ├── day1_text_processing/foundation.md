# Week 1: Foundations - Day 1: Text Processing

## 📚 Learning Objectives
By the end of today, you will be able to:
1. Understand the fundamentals of text processing for language models
2. Implement basic tokenization methods
3. Create vocabulary and numerical encoding systems
4. Build a simple text dataset loader

## 🎯 Today's Focus: Text Processing Pipeline
```
Raw Text → Tokenization → Vocabulary → Numerical Encoding → Batches
```

## 📁 Project Structure
```
week1_foundations/day1_text_processing/
│
├── README.md                          # This file
├── requirements.txt                   # Day-specific dependencies
│
├── notebooks/
│   ├── 01_text_processing_intro.ipynb # Interactive tutorial
│   └── exercises.ipynb                # Practice exercises
│
├── src/
│   ├── __init__.py
│   ├── tokenizer.py                   # Tokenizer implementations
│   ├── vocabulary.py                  # Vocabulary class
│   ├── dataset.py                     # Text dataset loader
│   └── utils.py                       # Helper functions
│
├── data/
│   ├── sample.txt                     # Sample text for practice
│   └── shakespeare.txt                # Shakespeare dataset
│
├── tests/
│   ├── test_tokenizer.py
│   └── test_vocabulary.py
│
└── examples/
    ├── basic_tokenization.py          # Basic usage examples
    └── custom_dataset.py              # Dataset creation example
```

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Navigate to day1 directory
cd week1_foundations/day1_text_processing

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Required Packages (`requirements.txt`)
```txt
numpy>=1.21.0
tqdm>=4.62.0
matplotlib>=3.5.0
pytest>=7.0.0
```

## 📖 Core Concepts

### 1. What is Tokenization?
Tokenization is the process of converting raw text into smaller units called tokens.

**Types of Tokenization:**
- **Character-level**: Each character is a token
- **Word-level**: Each word is a token
- **Subword-level**: Balance between character and word

### 2. Why Text Processing Matters
- Models understand numbers, not text
- Proper tokenization affects model performance
- Vocabulary size impacts memory and speed

## 🛠️ Implementation Guide

### 1. Basic Tokenizer (`src/tokenizer.py`)
```python
class BasicTokenizer:
    """A simple tokenizer for educational purposes"""
    
    def __init__(self, method='word'):
        self.method = method
        self.vocab = None
        
    def tokenize(self, text):
        """Convert text to tokens based on method"""
        if self.method == 'char':
            return list(text)
        elif self.method == 'word':
            return text.split()
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def train(self, corpus):
        """Build vocabulary from corpus"""
        tokens = []
        for text in corpus:
            tokens.extend(self.tokenize(text))
        
        # Create vocabulary
        unique_tokens = sorted(set(tokens))
        self.vocab = {
            token: idx for idx, token in enumerate(unique_tokens)
        }
        self.vocab_size = len(unique_tokens)
        
        return self.vocab
```

### 2. Vocabulary Class (`src/vocabulary.py`)
```python
class Vocabulary:
    """Manages token-to-id and id-to-token mappings"""
    
    def __init__(self):
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.add_token('[PAD]')  # Padding token
        self.add_token('[UNK]')  # Unknown token
        self.add_token('[BOS]')  # Beginning of sequence
        self.add_token('[EOS]')  # End of sequence
    
    def add_token(self, token):
        """Add a new token to vocabulary"""
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
    
    def encode(self, tokens):
        """Convert tokens to indices"""
        return [
            self.token_to_idx.get(token, self.token_to_idx['[UNK]'])
            for token in tokens
        ]
    
    def decode(self, indices):
        """Convert indices back to tokens"""
        return [
            self.idx_to_token.get(idx, '[UNK]')
            for idx in indices
        ]
```

### 3. Text Dataset (`src/dataset.py`)
```python
import numpy as np

class TextDataset:
    """Loads and processes text data for training"""
    
    def __init__(self, file_path, tokenizer, vocab, seq_length=50):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.seq_length = seq_length
        
        # Load and process data
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # Tokenize
        self.tokens = self.tokenizer.tokenize(self.text)
        self.indices = self.vocab.encode(self.tokens)
    
    def __len__(self):
        return len(self.indices) - self.seq_length
    
    def __getitem__(self, idx):
        # Get sequence of length seq_length
        x = self.indices[idx:idx + self.seq_length]
        y = self.indices[idx + 1:idx + self.seq_length + 1]
        
        return np.array(x), np.array(y)
    
    def create_batches(self, batch_size=32):
        """Create batches for training"""
        num_batches = len(self) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch_x = []
            batch_y = []
            
            for idx in range(start_idx, end_idx):
                x, y = self[idx]
                batch_x.append(x)
                batch_y.append(y)
            
            yield np.array(batch_x), np.array(batch_y)
```

## 📓 Jupyter Notebook Tutorial

### Notebook 1: `01_text_processing_intro.ipynb`

#### Section 1: Basic Text Processing
```python
# Cell 1: Import and setup
import sys
sys.path.append('..')

from src.tokenizer import BasicTokenizer
from src.vocabulary import Vocabulary

# Sample text
text = "Hello, world! This is a test."
```

#### Section 2: Tokenization Methods
```python
# Character-level tokenization
char_tokenizer = BasicTokenizer(method='char')
char_tokens = char_tokenizer.tokenize(text)
print(f"Character tokens: {char_tokens}")

# Word-level tokenization
word_tokenizer = BasicTokenizer(method='word')
word_tokens = word_tokenizer.tokenize(text)
print(f"Word tokens: {word_tokens}")
```

#### Section 3: Building Vocabulary
```python
# Create vocabulary from sample corpus
corpus = [
    "Hello world!",
    "This is a test.",
    "Another sentence for vocabulary."
]

vocab = Vocabulary()
word_tokenizer.train(corpus)

# Add tokens to vocabulary
for text in corpus:
    tokens = word_tokenizer.tokenize(text)
    for token in tokens:
        vocab.add_token(token)

print(f"Vocabulary size: {len(vocab.token_to_idx)}")
print(f"Sample mappings: {list(vocab.token_to_idx.items())[:10]}")
```

#### Section 4: Encoding and Decoding
```python
# Encode text
tokens = word_tokenizer.tokenize("Hello test world")
indices = vocab.encode(tokens)
print(f"Encoded: {indices}")

# Decode back to text
decoded_tokens = vocab.decode(indices)
decoded_text = ' '.join(decoded_tokens)
print(f"Decoded: {decoded_text}")
```

#### Section 5: Working with Real Data
```python
# Load Shakespeare dataset
with open('data/shakespeare.txt', 'r') as f:
    shakespeare_text = f.read()[:1000]  # First 1000 chars

# Process the text
tokens = word_tokenizer.tokenize(shakespeare_text)
print(f"Total tokens: {len(tokens)}")
print(f"Unique tokens: {len(set(tokens))}")
```

## 🏋️ Exercises (`notebooks/exercises.ipynb`)

### Exercise 1: Implement Byte Pair Encoding (BPE)
```python
# TODO: Implement BPE tokenizer
class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        
    def train(self, corpus):
        """Implement BPE training algorithm"""
        pass
    
    def tokenize(self, text):
        """Tokenize text using learned BPE merges"""
        pass
```

### Exercise 2: Handle Special Cases
```python
# TODO: Improve tokenizer to handle:
# 1. Punctuation
# 2. Case sensitivity
# 3. Numbers
# 4. URLs and emails

class ImprovedTokenizer(BasicTokenizer):
    def tokenize(self, text):
        # Your implementation here
        pass
```

### Exercise 3: Create Text Statistics
```python
# TODO: Analyze text data
def analyze_text(text):
    """
    Return statistics about the text:
    - Total characters
    - Total words
    - Vocabulary size
    - Average word length
    - Most common words
    """
    pass
```

## 🧪 Testing Your Code

### Run Tests
```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_tokenizer.py -v

# Run with coverage
pytest --cov=src tests/
```

### Example Test (`tests/test_tokenizer.py`)
```python
import pytest
from src.tokenizer import BasicTokenizer

def test_char_tokenizer():
    tokenizer = BasicTokenizer(method='char')
    result = tokenizer.tokenize("abc")
    assert result == ['a', 'b', 'c']

def test_word_tokenizer():
    tokenizer = BasicTokenizer(method='word')
    result = tokenizer.tokenize("hello world")
    assert result == ['hello', 'world']
```

## 📊 Visualization Examples

### 1. Token Distribution
```python
import matplotlib.pyplot as plt
from collections import Counter

# Analyze token frequencies
tokens = word_tokenizer.tokenize(shakespeare_text)
token_counts = Counter(tokens)

# Plot top 20 tokens
top_tokens = token_counts.most_common(20)
words, counts = zip(*top_tokens)

plt.figure(figsize=(10, 6))
plt.bar(range(len(words)), counts)
plt.xticks(range(len(words)), words, rotation=45)
plt.title('Top 20 Tokens in Shakespeare Text')
plt.tight_layout()
plt.show()
```

### 2. Vocabulary Growth
```python
# Simulate vocabulary growth with more text
vocab_sizes = []
text_lengths = []

for i in range(100, len(tokens), 100):
    sample_tokens = tokens[:i]
    vocab_size = len(set(sample_tokens))
    vocab_sizes.append(vocab_size)
    text_lengths.append(i)

plt.plot(text_lengths, vocab_sizes)
plt.xlabel('Number of tokens')
plt.ylabel('Vocabulary size')
plt.title('Vocabulary Growth')
plt.show()
```

## 🚀 Quick Start Script

Create `run_example.py`:
```python
"""Quick start example for Day 1"""
import sys
import os
sys.path.append('src')

from tokenizer import BasicTokenizer
from vocabulary import Vocabulary
from dataset import TextDataset

def main():
    # 1. Load sample data
    with open('data/sample.txt', 'r') as f:
        text = f.read()
    
    # 2. Initialize tokenizer and vocabulary
    tokenizer = BasicTokenizer(method='word')
    vocab = Vocabulary()
    
    # 3. Process text
    tokens = tokenizer.tokenize(text)
    
    # 4. Build vocabulary
    for token in set(tokens):
        vocab.add_token(token)
    
    # 5. Create dataset
    dataset = TextDataset('data/sample.txt', tokenizer, vocab, seq_length=20)
    
    # 6. Show statistics
    print(f"Text length: {len(text)} characters")
    print(f"Number of tokens: {len(tokens)}")
    print(f"Vocabulary size: {len(vocab.token_to_idx)}")
    print(f"Dataset size: {len(dataset)} sequences")
    
    # 7. Show example batch
    for i, (x, y) in enumerate(dataset.create_batches(batch_size=2)):
        if i == 0:  # Show first batch only
            print(f"\nFirst batch - X shape: {x.shape}, Y shape: {y.shape}")
            print(f"Sample input: {vocab.decode(x[0])}")
            print(f"Sample target: {vocab.decode(y[0])}")
            break

if __name__ == "__main__":
    main()
```

## 📚 Additional Resources

### Reading Material
1. [Byte Pair Encoding Paper](https://arxiv.org/abs/1508.07909)
2. [WordPiece Tokenization](https://arxiv.org/abs/1609.08144)
3. [SentencePiece: Unsupervised Text Tokenization](https://arxiv.org/abs/1808.06226)

### Tools to Explore
- Hugging Face Tokenizers library
- spaCy tokenizer
- NLTK tokenization

### Cheat Sheet
```
Key Concepts:
• Tokenization: Text → Tokens
• Vocabulary: Tokens ↔ Indices
• Encoding: Tokens → Numbers
• Decoding: Numbers → Tokens

Common Issues:
• OOV (Out-of-Vocabulary): Use [UNK] token
• Sequence length: Padding/truncation
• Case handling: Lowercasing vs preserving
```

## 🎯 Daily Challenge

**Challenge:** Build a tokenizer that can handle code (Python/JavaScript)

**Requirements:**
1. Preserve indentation
2. Handle variables and function names
3. Manage special characters (+, -, *, /, =, etc.)
4. Support comments and strings

**Starter Code:**
```python
class CodeTokenizer:
    def tokenize(self, code):
        # Your implementation
        pass
```

## 📝 Summary

Today you learned:
- ✓ Text processing fundamentals
- ✓ Tokenization methods (char, word)
- ✓ Vocabulary creation and management
- ✓ Dataset preparation for language modeling
- ✓ Basic text analysis and visualization

**Tomorrow:** Day 2 - Neural Network Basics for Language Models

---

## 🆘 Need Help?

### Common Issues:
1. **Import errors**: Make sure you're in the right directory
2. **Memory issues**: Process text in chunks for large files
3. **Vocabulary explosion**: Consider subword tokenization for large texts

### Get Support:
- Check the [GitHub Issues](https://github.com/yourusername/build-your-own-gpt/issues)
- Join our [Discord community](link-to-discord)
- Review the [FAQ](../../docs/FAQ.md)

---

**Happy Coding!** 🚀

Remember: Text processing is the foundation of all language models. Take your time to understand these concepts thoroughly!
