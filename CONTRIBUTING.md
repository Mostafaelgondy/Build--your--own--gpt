# CONTRIBUTING.md

# 🚀 Contributing to Build Your Own GPT

First off, thank you for considering contributing to **Build Your Own GPT**! 🙌 This is a community-driven project designed to help developers build their own AI assistants from scratch. Whether you're a beginner or an experienced AI engineer, your contributions are welcome and valued.

---

## 📋 Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Quick Start](#quick-start)
3. [How Can I Contribute?](#how-can-i-contribute)
   - [Reporting Bugs](#reporting-bugs)
   - [Suggesting Enhancements](#suggesting-enhancements)
   - [Your First Code Contribution](#your-first-code-contribution)
   - [Improving Documentation](#improving-documentation)
   - [Adding New Lessons](#adding-new-lessons)
4. [Development Workflow](#development-workflow)
5. [Commit Guidelines](#commit-guidelines)
6. [Pull Request Process](#pull-request-process)
7. [Project Structure](#project-structure)
8. [Style Guides](#style-guides)
   - [Python Style Guide](#python-style-guide)
   - [Documentation Style Guide](#documentation-style-guide)
9. [Testing](#testing)
10. [Community](#community)
11. [Recognition](#recognition)

---

## 📜 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [mostafaeljindy8@example.com].

## 🚦 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/build-your-own-gpt.git
   cd build-your-own-gpt
   ```
3. **Set up the development environment**:
   ```bash
   # Create a virtual environment
   python -m venv venv
   
   # Activate it (Linux/Mac)
   source venv/bin/activate
   # Activate it (Windows)
   venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements-dev.txt
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🤔 How Can I Contribute?

### 🐛 Reporting Bugs

Found a bug? Great! We want to fix it. 

**Before submitting a bug report:**
- Check if the bug hasn't already been reported in [Issues](https://github.com/yourusername/build-your-own-gpt/issues)
- Try to reproduce the bug with the latest version

**How to submit a good bug report:**
1. Use the **Bug Report** template when creating a new issue
2. Include a clear, descriptive title
3. Describe the exact steps to reproduce the bug
4. Include code snippets, error messages, and screenshots if applicable
5. Specify your environment (OS, Python version, dependencies)

**Example of a good bug report title:**  
`[Bug] Day 3 - Embeddings calculation returns NaN for empty strings`

### 💡 Suggesting Enhancements

Have an idea to make this project better? We'd love to hear it!

**Before suggesting an enhancement:**
- Check if a similar suggestion exists in [Issues](https://github.com/yourusername/build-your-own-gpt/issues)
- Think about whether it aligns with the project goals

**How to suggest an enhancement:**
1. Use the **Feature Request** template
2. Start the title with `[Enhancement]`
3. Describe the current behavior and what you'd like to see
4. Explain why this would be useful to other learners
5. Include examples if possible

**Example of a good enhancement request:**  
`[Enhancement] Add visualization for attention weights in Day 8`

### 👩‍💻 Your First Code Contribution

Unsure where to begin? Look for issues labeled:
- `good first issue` - Perfect for newcomers
- `help wanted` - Areas where we need assistance
- `beginner-friendly` - Minimal prior knowledge needed

**Good first issues:**
1. Fix typos in documentation
2. Add more examples to existing lessons
3. Improve error messages
4. Add test cases

### 📚 Improving Documentation

Great documentation is crucial for learning! You can help by:
- Fixing grammatical errors
- Clarifying confusing explanations
- Adding more examples
- Translating documentation (check with maintainers first)
- Adding inline comments to complex code

**Documentation files to consider:**
- `README.md`
- Individual lesson `theory.md` files
- Docstrings in Python files
- Comments in code examples

### 📝 Adding New Lessons

Want to contribute a new lesson? Fantastic!

**Guidelines for new lessons:**
1. **Discuss first**: Open an issue to discuss your lesson idea
2. **Follow structure**: Use existing lessons as templates
3. **Keep it practical**: Focus on hands-on, runnable code
4. **Difficulty level**: Label appropriately (Beginner/Intermediate/Advanced)
5. **Prerequisites**: List what learners should know before starting

**Required files for each lesson:**
```
dayX_lesson_name/
├── theory.md          # Concepts explained simply
├── practice.py        # Main code file
├── exercises/         # Challenge exercises
├── solutions/         # Exercise solutions
├── data/              # Sample data (if needed)
├── README.md          # Lesson overview
└── requirements.txt   # Lesson-specific dependencies
```

## 🔄 Development Workflow

1. **Sync your fork** with upstream:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b type/description
   # Types: feature/, fix/, docs/, test/, refactor/
   ```

3. **Make your changes**:
   - Write or modify code
   - Add/update tests
   - Update documentation
   - Commit with clear messages

4. **Test your changes**:
   ```bash
   # Run the specific lesson's tests
   python -m pytest lessons/dayX_lesson_name/tests/ -v
   
   # Run all tests
   python -m pytest
   ```

5. **Push to your fork**:
   ```bash
   git push origin type/description
   ```

6. **Create a Pull Request** (see below)

## 📝 Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

**Format:** `type(scope): description`

**Types:**
- `feat`: New feature or lesson
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `style`: Code style changes (no logic change)
- `chore`: Maintenance tasks

**Examples:**
```
feat(day15): add attention visualization
fix(day3): correct embedding dimension calculation
docs(readme): update installation instructions
test(day8): add test for multi-head attention
```

**Commit Message Tips:**
- Use the imperative mood ("Add feature" not "Added feature")
- Keep the first line under 50 characters
- Reference issues: `Closes #123` or `Fixes #456`

## 🎯 Pull Request Process

1. **Ensure your PR is ready**:
   - All tests pass
   - Documentation is updated
   - Code follows style guides
   - Commit messages are clear

2. **Create a Pull Request**:
   - Use the PR template
   - Link related issues (`Closes #123`)
   - Describe what and why, not just how

3. **PR Review**:
   - A maintainer will review within 48 hours
   - Address review comments promptly
   - Make requested changes and push updates
   - PRs require at least one approval before merging

4. **After approval**:
   - Maintainers will merge your PR
   - Your contribution will be credited in CHANGELOG.md

## 🏗️ Project Structure

```
build-your-own-gpt/
├── week1_foundations/          # Week 1 lessons
│   ├── day1_text_processing/
│   └── ...
├── week2_gpt_components/       # Week 2 lessons
├── week3_full_gpt/             # Week 3 lessons
├── week4_production/           # Week 4 lessons
├── templates/                  # Lesson templates
├── tests/                      # Test suite
├── scripts/                    # Utility scripts
├── docs/                       # Project documentation
├── community_projects/         # Community submissions
├── .github/                    # GitHub workflows
│   ├── workflows/
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE/
├── requirements-dev.txt        # Development dependencies
├── pyproject.toml              # Project configuration
├── LICENSE                     # MIT License
└── CODE_OF_CONDUCT.md          # Community guidelines
```

## 🎨 Style Guides

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with these specifics:

**General Rules:**
- Use 4 spaces per indentation level
- Maximum line length: 88 characters (Black formatter)
- Use double quotes for docstrings, single quotes for strings
- Import order: standard library → third-party → local

**Example of good Python style:**
```python
"""Example module demonstrating proper Python style."""

import os
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from .attention import MultiHeadAttention


def calculate_embeddings(
    text: str,
    model_name: str = "bert-base-uncased",
    max_length: int = 512,
) -> np.ndarray:
    """
    Calculate embeddings for a given text.
    
    Args:
        text: Input text to embed
        model_name: Name of the pretrained model
        max_length: Maximum sequence length
        
    Returns:
        Array of shape (embedding_dim,)
        
    Raises:
        ValueError: If text is empty
    """
    if not text.strip():
        raise ValueError("Text cannot be empty")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    
    # Process embeddings
    with torch.no_grad():
        embeddings = get_embeddings(inputs)
    
    return embeddings.numpy()


class GPTBlock:
    """A single block in the GPT architecture."""
    
    def __init__(self, hidden_size: int, num_heads: int):
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the block."""
        # Residual connection
        attn_output = self.attention(x)
        x = x + attn_output
        x = self.norm(x)
        return x
```

**Required tools:**
- **Black**: Code formatting
- **isort**: Import sorting
- **Flake8**: Linting
- **mypy**: Type checking

**Setup formatting/linting:**
```bash
# Install pre-commit hooks
pre-commit install

# Manual formatting
black .
isort .
flake8 .
mypy .
```

### Documentation Style Guide

**Markdown Files:**
- Use heading hierarchy properly
- Include code fences with language specification
- Use relative links for internal references
- Add alt text for images

**Example documentation structure:**
```markdown
# Lesson Title

## Learning Objectives
- What learners will achieve
- Specific skills they'll gain

## Prerequisites
- Required knowledge
- Previous lessons to complete

## Theory

### Concept 1
Explanation with examples...

```python
# Code example
def example():
    return "Hello"
```

### Concept 2
More explanation...

## Hands-On Practice

### Exercise 1: Basic Implementation
**Goal**: Implement X

**Steps**:
1. Step one
2. Step two

**Starter Code**:
```python
# Fill in the blanks
def exercise():
    _____
```

## Common Pitfalls
- What learners often get wrong
- How to avoid these mistakes

## Further Reading
- Links to relevant resources
- Research papers
- Blog posts
```

## 🧪 Testing

### Writing Tests

**Location:** `tests/dayX_lesson_name/test_practice.py`

**Example test structure:**
```python
"""Tests for Day 3 - Embeddings."""

import numpy as np
import pytest

from week1_foundations.day3_embeddings.practice import (
    calculate_similarity,
    TextEmbedder,
)


def test_calculate_similarity_positive():
    """Test that similar vectors have high similarity."""
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.9, 0.1, 0.0])
    
    similarity = calculate_similarity(vec1, vec2)
    assert similarity > 0.8, "Similar vectors should have high similarity"


def test_calculate_similarity_negative():
    """Test that orthogonal vectors have zero similarity."""
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.0, 1.0, 0.0])
    
    similarity = calculate_similarity(vec1, vec2)
    assert abs(similarity) < 1e-6, "Orthogonal vectors should have zero similarity"


def test_text_embedder_initialization():
    """Test TextEmbedder initializes correctly."""
    embedder = TextEmbedder(model_name="bert-base-uncased")
    assert embedder.model_name == "bert-base-uncased"
    assert embedder.tokenizer is not None


class TestTextEmbedder:
    """Test suite for TextEmbedder class."""
    
    @pytest.fixture
    def embedder(self):
        return TextEmbedder()
    
    def test_embed_single_text(self, embedder):
        """Test embedding a single text."""
        text = "Hello, world!"
        embedding = embedder.embed(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)  # BERT base dimension
    
    def test_embed_empty_text(self, embedder):
        """Test that empty text raises error."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            embedder.embed("")
```

**Running tests:**
```bash
# Run all tests
pytest

# Run specific lesson tests
pytest tests/week1_foundations/day3_embeddings/ -v

# Run with coverage
pytest --cov=.

# Run in parallel
pytest -n auto
```

## 👥 Community

### Where to Get Help
- **GitHub Discussions**: For questions and discussions
- **Issue Tracker**: For bugs and feature requests
- **Discord Server**: [Link to your Discord] - Real-time chat
- **Twitter**: [@YourHandle] - Updates and announcements

### Weekly Community Events
- **Mondays**: Office hours (video call)
- **Wednesdays**: Code review sessions
- **Fridays**: Show & tell (share what you built)

### How to Help Others
1. Answer questions in GitHub Discussions
2. Review Pull Requests
3. Help troubleshoot issues
4. Share your learning journey
5. Create tutorial videos/write-ups

## 🏆 Recognition

Your contributions will be recognized in several ways:

### Contributor Levels
- **🌱 Beginner**: 1-3 merged PRs
- **🌿 Contributor**: 4-10 merged PRs
- **🌳 Maintainer**: 11+ PRs with significant impact
- **🦉 Core Team**: Invitation to join core team

### Recognition Methods
1. **CONTRIBUTORS.md**: All contributors listed
2. **GitHub Contributor Graph**: Automatic
3. **Monthly Spotlight**: Featured contributor in README
4. **Digital Badges**: For completing contribution milestones
5. **Swag**: For top contributors (stickers, t-shirts)

### Hall of Fame
Top contributors will be featured in our Hall of Fame:
```
🏆 Hall of Fame
---------------
🥇 @username1 - 50+ PRs, Added Week 3 content
🥈 @username2 - 30+ PRs, Major bug fixes
🥉 @username3 - 20+ PRs, Documentation overhaul
```

---

## ❓ Still Have Questions?

Check our [FAQ](docs/FAQ.md) or:
- Open a [GitHub Discussion](https://github.com/yourusername/build-your-own-gpt/discussions)
- Join our [Discord community](your-discord-link)
- Email: [your-email@example.com]

---

## 🙏 Thank You!

Thank you for taking the time to contribute! Your efforts help make AI education more accessible to everyone around the world. Every contribution, no matter how small, makes a difference.

**Remember:** This project exists because of contributors like you. Let's build something amazing together! 🚀

