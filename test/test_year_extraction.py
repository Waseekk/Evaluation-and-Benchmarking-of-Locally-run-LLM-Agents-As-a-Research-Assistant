"""
Test the improved year extraction from references section
"""

from citation_analyzer_hybrid import HybridSemanticCitationAnalyzer

# Sample text with a references section
sample_paper = """
Introduction
This paper presents a novel approach to machine learning.
Smith et al. (2020) demonstrated similar results.

Methods
We used the technique from Jones (2021) with modifications.

Results
Our results outperform the baseline from Brown (2022).

REFERENCES

[1] Smith, J., & Doe, A. (2020). Machine learning approaches. Journal of AI, 15(3), 245-260.

[2] Jones, M. (2021). Deep learning techniques. Neural Networks, 28(1), 112-125.

[3] Brown, T., et al. (2022). Baseline performance evaluation. ICML Proceedings, pp. 1950-1960.

[4] Miller, K. (2019). Theoretical foundations. Science, 364(6443), 1234-1238.

[5] Davis, L. (2023). Comparative analysis. Nature Machine Intelligence, 5(1), 45-52.

[6] Wilson, R., & Taylor, S. (2018). Data collection protocols. Methods in Research, 12(2), 89-102.

APPENDIX

Additional materials here.
"""

print("=" * 80)
print("Testing Improved Year Extraction from References Section")
print("=" * 80)

# Initialize analyzer
analyzer = HybridSemanticCitationAnalyzer(
    use_embeddings=False,  # Disable for faster testing
    enable_spacy=False,
    log_file='test_year_extraction.log'
)

# Extract citations (which includes year extraction)
results = analyzer.extract_citations(sample_paper)

# Display results
print("\nYear Distribution Results:")
print("-" * 80)

if results['year_distribution']:
    for year, count in sorted(results['year_distribution'].items()):
        print(f"  {year}: {count} citation(s)")

    print(f"\nTotal unique years found: {len(results['year_distribution'])}")
    print(f"Total citations counted: {sum(results['year_distribution'].values())}")
else:
    print("  No years found!")

print("\n" + "=" * 80)
print("Test complete! Check 'test_year_extraction.log' for detailed logs")
print("=" * 80)

# Expected output:
# 2018: 1
# 2019: 1
# 2020: 1
# 2021: 1
# 2022: 1
# 2023: 1
# Total: 6 unique years

print("\nExpected: 6 years (2018, 2019, 2020, 2021, 2022, 2023)")
print(f"Actual: {len(results['year_distribution'])} years")

if len(results['year_distribution']) == 6:
    print("SUCCESS: All years extracted correctly from references!")
else:
    print("WARNING: Year count doesn't match expected")
