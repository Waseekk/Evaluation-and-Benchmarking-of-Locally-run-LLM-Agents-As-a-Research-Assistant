# Quick Start Guide: Hybrid Citation Analysis

## 1. Installation (One-Time Setup)

### Basic Installation (Required)
```bash
pip install streamlit pandas plotly networkx
```

### Full Hybrid Mode (Recommended)
```bash
# Install semantic analysis
pip install sentence-transformers

# Install advanced NLP
pip install spacy
python -m spacy download en_core_web_sm

# Install scikit-learn (usually already installed)
pip install scikit-learn
```

## 2. Run the Application

```bash
cd "E:\Olama\langchain_deepseek\v2\v4\Evaluation-and-Benchmarking-of-Locally-run-LLM-Agents-As-a-Research-Assistant"
streamlit run research_paper_assistant_hybrid_citation.py
```

## 3. Using the Interface

### Upload Your Paper
1. Click "Browse files" or drag & drop a PDF
2. OR paste text directly in the text area

### Check Analyzer Status
Look at the sidebar:
- ✓ Green = Hybrid mode active (all features)
- ⚠️ Yellow = Basic mode (missing dependencies)

### Configure Analysis
In sidebar, enable/disable:
- ✅ **Semantic Similarity** - Find related citations
- ✅ **Stance Detection** - Supporting vs. refuting
- ✅ **Purpose Classification** - Why citations are used
- ✅ **Show Citation Network Graph** - Visual relationships

### Run Analysis
Click **"Analyze Paper"** button

### View Results
Results appear in tabs:
- **Citation counts** - Total, unique, by type
- **Stance distribution** - Pie chart of citation stances
- **Purpose distribution** - Bar chart of purposes
- **Network graph** - Visual citation relationships
- **Relationship table** - Detailed edge information
- **Timing stats** - Performance breakdown

## 4. Export Results

1. Analyze a paper first
2. In sidebar, see "📥 Export Data"
3. Click "📊 Export All Data to Excel"
4. Click "⬇️ Download Excel File"

Excel includes all citation details, relationships, stance, and purpose data!

## 5. Common Issues

### Issue: "Using Basic Analyzer" Warning
**Fix**: Install optional dependencies
```bash
pip install sentence-transformers spacy
python -m spacy download en_core_web_sm
```

### Issue: Slow Analysis
**Tip**: First run downloads embedding model (~80MB), subsequent runs are faster
**Alternative**: Disable "Semantic Similarity" for faster analysis

### Issue: No Network Graph
**Reason**: Not enough citation relationships detected
**Fix**: Use papers with more citations in close proximity

### Issue: spaCy Error
**Fix**: Download the language model:
```bash
python -m spacy download en_core_web_sm
```

## 6. What to Expect

### Typical Analysis Results:
- **10-page paper**: ~20-50 citations detected
- **Network edges**: ~10-30 relationships
- **Stance distribution**: Mix of supporting/neutral
- **Purpose**: Mostly background + methodology
- **Analysis time**: 10-30 seconds

### Best Use Cases:
✅ Research paper literature review analysis
✅ Understanding citation patterns in a field
✅ Identifying influential works
✅ Finding related research through semantic links
✅ Analyzing how papers support/refute each other

## 7. Tips for Best Results

1. **Use native PDFs** (not scanned images)
2. **Papers with 10+ citations** work best
3. **Enable all features** for comprehensive analysis
4. **Check the log file** (`citation_analysis.log`) for details
5. **Export to Excel** to analyze relationships further

## 8. Comparison: Basic vs. Hybrid

| Feature | Basic Mode | Hybrid Mode |
|---------|-----------|-------------|
| Speed | ⚡ Faster | 🐢 Slightly slower |
| Citation extraction | ✅ Yes | ✅ Yes |
| Stance detection | ✅ Yes | ✅ Yes |
| Purpose detection | ✅ Yes | ✅ Yes |
| Semantic similarity | ❌ No | ✅ Yes |
| Advanced NLP | ❌ Basic | ✅ spaCy |
| False positive filtering | ⚠️ Good | ✅ Excellent |

**Recommendation**: Install dependencies for hybrid mode - it's worth it!

## 9. Example Workflow

```
1. Upload PDF → "MyPaper.pdf"
2. Wait for processing → ~5 seconds
3. Check analyzer status → "✓ Hybrid Semantic Analyzer Active"
4. Enable all features in sidebar
5. Click "Analyze Paper" → ~20 seconds
6. Review results:
   - 45 citations detected
   - 32 relationships found
   - Stance: 60% supporting, 30% neutral, 10% contrasting
   - Purpose: 40% background, 30% methodology, 20% comparison
7. Export to Excel
8. Download "paper_analysis_hybrid_20251030_143022.xlsx"
```

## 10. Next Steps

- Read full documentation: `HYBRID_CITATION_README.md`
- Check example papers in `examples/` (if available)
- Explore Excel export sheets
- Review log file for detailed analysis info

---

**Questions?** Check the log file: `citation_analysis.log`

**Need Help?** See full README: `HYBRID_CITATION_README.md`
