# Contributing to ZKML-Guard

Thank you for your interest in contributing to ZKML-Guard! This document provides guidelines for contributions.

## 🤝 Ways to Contribute

- **Report bugs** via GitHub Issues
- **Suggest features** or improvements
- **Submit pull requests** for bug fixes or enhancements
- **Improve documentation**
- **Share use cases** and deployment experiences
- **Contribute training data** (with appropriate privacy measures)

## 🔬 Areas of Interest

We're particularly interested in contributions in these areas:

1. **Performance Optimization**
   - Faster proof generation
   - More efficient feature extraction
   - Optimized model architectures for ZKML

2. **Security Enhancements**
   - New attack pattern detection
   - Adversarial robustness improvements
   - Novel feature engineering

3. **Integration Support**
   - Additional MPC custody platform integrations
   - API clients and SDKs
   - Deployment automation

4. **Research Extensions**
   - Federated learning implementations
   - On-chain verification
   - Multi-chain support

## 📋 Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/zkml-guard.git
   cd zkml-guard
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Run tests**
   ```bash
   pytest tests/
   ```

## 🏗️ Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Run quality checks**
   ```bash
   # Format code
   black src/ scripts/ tests/
   
   # Check style
   flake8 src/ scripts/ tests/
   
   # Type checking
   mypy src/
   
   # Run tests
   pytest tests/ --cov=src
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```
   
   Use conventional commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `test:` for tests
   - `refactor:` for code refactoring
   - `perf:` for performance improvements

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## 📝 Code Style

- **Python**: Follow PEP 8 guidelines
- **Line length**: 100 characters max
- **Docstrings**: Use Google style docstrings
- **Type hints**: Add type hints for all functions

Example:
```python
def extract_features(transaction: Dict[str, Any]) -> np.ndarray:
    """
    Extract features from a blockchain transaction.
    
    Args:
        transaction: Transaction dictionary with required fields
        
    Returns:
        Feature vector of shape (42,)
        
    Raises:
        ValueError: If transaction is missing required fields
    """
    pass
```

## 🧪 Testing Guidelines

- Write unit tests for all new functions
- Add integration tests for new features
- Maintain >80% code coverage
- Test edge cases and error conditions

Example test:
```python
def test_feature_extraction():
    """Test that feature extraction produces correct shape."""
    extractor = TransactionFeatureExtractor()
    tx = create_mock_transaction()
    features = extractor.extract_features(tx)
    
    assert features.shape == (42,)
    assert not np.any(np.isnan(features))
```

## 🔒 Security Considerations

- **Never commit** private keys, API keys, or sensitive data
- **Sanitize** all transaction data in examples
- **Review** security implications of changes
- **Report** security vulnerabilities privately to security@zkml-guard.org

## 📚 Documentation

- Update README.md for user-facing changes
- Add docstrings for all new functions/classes
- Update API documentation
- Include examples for new features

## 🐛 Bug Reports

Good bug reports include:

1. **Description**: Clear description of the bug
2. **Reproduction**: Step-by-step instructions to reproduce
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**: OS, Python version, dependencies
6. **Logs**: Relevant error messages or logs

Use this template:
```markdown
## Bug Description
[Clear description]

## Steps to Reproduce
1. Step one
2. Step two
3. ...

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happens]

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10.8]
- ZKML-Guard version: [e.g., 1.0.0]

## Additional Context
[Logs, screenshots, etc.]
```

## 💡 Feature Requests

Good feature requests include:

1. **Use case**: Why is this feature needed?
2. **Description**: What should the feature do?
3. **Alternatives**: What alternatives have you considered?
4. **Implementation ideas**: Any thoughts on implementation?

## 🎯 Code Review

All submissions require review. We look for:

- **Correctness**: Does it work as intended?
- **Tests**: Are there adequate tests?
- **Documentation**: Is it well documented?
- **Style**: Does it follow our guidelines?
- **Performance**: Are there performance concerns?
- **Security**: Are there security implications?

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in release notes
- Credited in academic citations (for significant contributions)

## ❓ Questions?

- **General questions**: GitHub Discussions
- **Bug reports**: GitHub Issues
- **Security**: nv262@cornell.edu
- **Community**: Discord server

Thank you for contributing to ZKML-Guard! 🚀
