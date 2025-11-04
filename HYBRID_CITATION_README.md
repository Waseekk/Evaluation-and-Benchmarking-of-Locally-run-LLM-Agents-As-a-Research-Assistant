# Research Paper Assistant with Hybrid Citation Analysis

## Overview

`research_paper_assistant_hybrid_citation.py` is an enhanced version of the Research Paper Assistant that integrates the **HybridSemanticCitationAnalyzer** for advanced citation analysis.

## What's New? 🚀

### Enhanced Citation Analysis Features

1. **Semantic Similarity Analysis**
   - Uses ML embeddings to detect semantically related citations
   - Identifies citations discussing similar topics even without direct co-occurrence
   - Provides similarity scores between citation pairs

2. **Stance Detection**
   - Automatically detects citation stance: **Supporting**, **Refuting**, **Contrasting**, **Neutral**
   - Shows confidence scores for each stance classification
   - Visualizes stance distribution with pie charts

3. **Purpose Classification**
   - Classifies citations into 6 categories:
     - **Background**: Foundational/historical work
     - **Methodology**: Methods and techniques adopted
     - **Comparison**: Comparative analysis and benchmarking
     - **Theory**: Theoretical frameworks
     - **Results**: Findings and evidence
     - **Data**: Datasets and resources
   - Displays purpose distribution with bar charts

4. **Better False Positive Filtering**
   - Advanced stopword filtering to remove non-citation text
   - Validates author names and citation formats
   - Filters out paper titles and journal names from relationship graphs

5. **Enhanced Network Analysis**
   - Citation-to-citation relationships (not just star topology)
   - Co-occurrence detection (citations in same sentence)
   - Community/cluster detection for topic grouping
   - Centrality metrics to identify influential citations

6. **Performance Tracking**
   - Detailed timing statistics for each analysis stage
   - Displays total processing time
   - Shows which features are active

## Key Differences from Original

| Feature | Original Version | Hybrid Version |
|---------|-----------------|----------------|
| **Citation Extraction** | Regex-based | Regex + spaCy NLP |
| **Relationships** | Sequential only | Co-occurrence + Semantic similarity |
| **False Positives** | Basic filtering | Advanced stopword + validation |
| **Stance Detection** | None | 4 categories with confidence |
| **Purpose Analysis** | None | 6 categories with confidence |
| **Network Analysis** | Basic | Communities + centrality + clusters |
| **Context Extraction** | Surrounding text | Sentence-level with spaCy |
| **Logging** | Minimal | Comprehensive to file |

## Installation & Requirements

### Required Dependencies (Core)
```bash
pip install streamlit pandas plotly networkx
```

### Optional Dependencies (Hybrid Features)
```bash
# For semantic similarity
pip install sentence-transformers

# For advanced NLP
pip install spacy
python -m spacy download en_core_web_sm

# Usually already installed
pip install scikit-learn
```

### Fallback Behavior
If optional dependencies are missing, the system automatically falls back to the basic citation analyzer. You'll see a warning in the sidebar with installation instructions.

## How to Use

### Starting the Application
```bash
streamlit run research_paper_assistant_hybrid_citation.py
```

### Sidebar Controls

#### Analyzer Status
- **Green**: ✓ Hybrid Semantic Analyzer Active
- **Yellow**: ⚠️ Using Basic Analyzer (Hybrid unavailable)

#### Citation Analysis Features
- **Semantic Similarity**: Enable ML-based relationship detection
- **Stance Detection**: Identify supporting/refuting citations
- **Purpose Classification**: Categorize citation purposes

#### Visualization Options
- **Show Citation Year Distribution**: Bar chart of citations by year
- **Show Citation Network Graph**: Interactive network visualization
- **Show Structure Radar Chart**: Document structure completeness

### Analysis Results

#### Citation Metrics Display
- Total Citations
- Unique Citations
- Numbered Citations
- Author-Year Citations
- Network Edges (relationships)

#### Stance Distribution (New!)
- Pie chart showing distribution of citation stances
- Supporting, Refuting, Contrasting, Neutral counts

#### Purpose Distribution (New!)
- Bar chart showing citation purposes
- Background, Methodology, Comparison, Theory, Results, Data counts

#### Network Analysis
- **Total Citations**: Unique citation nodes
- **Citation Links**: Edges/relationships between citations
- **Network Density**: How interconnected the citations are
- **Avg. Citations/Paper**: Average connections per citation
- **Topic Clusters**: Automatically detected research topic groups

#### Citation Relationships Table
Shows detailed relationship information:
- From/To citations
- Relationship type (co-occurrence, semantic_similar)
- Weight (strength of relationship)
- Similarity score (for semantic relationships)
- Source/Target stance
- Source/Target purpose

#### Performance Timing (New!)
Displays timing for each analysis stage:
- Citation extraction
- Year extraction
- Network building
- Total time

## Features Compatibility Matrix

| Feature | Requires | Works Without |
|---------|----------|---------------|
| Basic citation extraction | ✓ Always | - |
| Year distribution | ✓ Always | - |
| Co-occurrence detection | ✓ Always | - |
| Semantic similarity | sentence-transformers | ✓ Degrades gracefully |
| Stance detection | ✓ Always | - |
| Purpose classification | ✓ Always | - |
| Better sentence detection | spaCy | ✓ Falls back to regex |
| Network visualization | networkx | - |

## Output Files

### Log File
- **File**: `citation_analysis.log`
- **Content**: Detailed analysis logs with timestamps
- **Location**: Same directory as script

### Excel Export
When you export analysis results, the hybrid version includes:
- Citation details with stance and purpose
- Relationship table with semantic scores
- Network metrics and communities
- Performance timing statistics

## Troubleshooting

### "Using Basic Analyzer" Warning
**Cause**: Optional dependencies missing
**Solution**: Install dependencies listed in sidebar expander "Why Basic Mode?"

### spaCy Model Error
**Error**: `Can't find model 'en_core_web_sm'`
**Solution**:
```bash
python -m spacy download en_core_web_sm
```

### Sentence-transformers Download
**Issue**: First run downloads ~80MB model
**Expected**: One-time download, cached for future use

### Network Graph Not Showing
**Cause**: No citation relationships detected
**Check**:
- Paper has citations in same sentences
- Citations have semantic similarity > 0.5

## Performance Considerations

### Analysis Speed
- **Small papers** (<5 pages): 5-10 seconds
- **Medium papers** (10-20 pages): 15-30 seconds
- **Large papers** (>20 pages): 30-60 seconds

### With Semantic Analysis
Add ~10-20 seconds for embedding computation (first run only, then cached)

### Memory Usage
- Basic mode: ~200MB
- Hybrid mode: ~500MB (includes embedding model)

## Comparison with Other Versions

### vs. `research_paper_assistant.py`
- Original version with basic NLP citation analysis
- No semantic similarity
- No stance/purpose detection
- Use if you want lightweight, fast analysis

### vs. `research_paper_assistant_hybrid_citation.py` (This Version)
- Full semantic analysis with embeddings
- Stance and purpose classification
- Enhanced network analysis
- Use if you want comprehensive, research-grade analysis

## Tips for Best Results

1. **PDF Quality**: Use native PDFs (not scanned) for best text extraction
2. **Paper Length**: Works best with papers containing 10+ citations
3. **Citation Style**: Handles numbered, author-year, and Harvard styles
4. **Network Visualization**: Most useful for papers with 20+ citations
5. **Semantic Analysis**: Enable for detecting non-obvious citation relationships

## Known Limitations

1. **Mixed Citation Styles**: Papers with multiple citation styles may have some duplicate entries
2. **Informal Citations**: Blog posts, websites without proper formatting may be missed
3. **Language**: Currently optimized for English papers only
4. **Network Scalability**: Very large papers (>200 citations) may have cluttered network graphs

## Future Enhancements

- [ ] Multi-language support
- [ ] Citation recommendation based on topic
- [ ] Temporal analysis of citation trends
- [ ] Export to citation management tools (BibTeX, EndNote)
- [ ] Citation sentiment analysis
- [ ] Automatic literature gap identification

## Support & Feedback

- **Issues**: Check `citation_analysis.log` for detailed error messages
- **Performance**: Use basic mode for faster analysis if hybrid features not needed
- **Dependencies**: All optional - system will work without them

## Version History

### v1.0 (Current)
- Initial release with hybrid semantic citation analysis
- Stance and purpose detection
- Enhanced network analysis with communities
- Graceful fallback to basic analyzer
- Comprehensive error handling

---

**Happy Analyzing! 📚🔬**
