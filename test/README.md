# Test Suite

This folder contains **test files** for the Research Paper Assistant project.

## Test Files:

- `test_beautification.py` - Tests for response beautification feature
- `test_citation_analyzer.py` - Tests for citation analysis (NLP vs Semantic comparison)
- `test_lazy_loading.py` - Tests for lazy model loading optimization
- `test_year_extraction.py` - Tests for year extraction from citations

## Running Tests:

### Individual Tests:
```bash
# Run a specific test
python test/test_beautification.py
python test/test_citation_analyzer.py
python test/test_lazy_loading.py
python test/test_year_extraction.py
```

### All Tests (if using pytest):
```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest test/

# Run with verbose output
pytest test/ -v

# Run specific test
pytest test/test_beautification.py
```

## Adding New Tests:

When adding new tests:
1. Create a file starting with `test_` (e.g., `test_new_feature.py`)
2. Follow the existing test structure
3. Include docstrings explaining what is being tested
4. Add assertions to verify expected behavior

## Test Coverage:

- ✅ Response beautification (metadata removal)
- ✅ Citation analysis (NLP vs Semantic)
- ✅ Lazy model loading
- ✅ Year extraction validation
- ⏳ RAG processor (todo)
- ⏳ PDF processing (todo)
- ⏳ Chat interface (todo)

**Last updated:** November 5, 2025
