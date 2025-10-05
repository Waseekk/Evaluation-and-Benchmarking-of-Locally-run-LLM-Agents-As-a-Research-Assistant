# Research Paper Analysis Assistant

An advanced AI-powered tool for comprehensive research paper analysis using multiple LLM models (Deepseek, Mistral, LLaMA3) with RAG capabilities, citation network visualization, and interactive chat interface.

## Features

**PDF Processing**
- Extract text from PDFs with OCR support for scanned documents
- Metadata extraction (author, title, page count, creation date)
- Image and figure detection

**Multi-Model Analysis**
- Compare analyses from Deepseek (1.5B & 8B), Mistral, and LLaMA3-8B models
- Performance benchmarking with response time, token count, and throughput metrics
- Consistency scoring across multiple inference runs

**Citation Analysis**
- Extract citations in multiple formats (DOI, arXiv, numbered, author-year, Harvard)
- Build and visualize citation networks with contextual relationships
- Year distribution analysis
- Citation density and network metrics

**Structure Analysis**
- Automatically detect paper sections (Abstract, Introduction, Methods, Results, Discussion, Conclusion)
- Structure completeness scoring with radar chart visualization
- Figure and table detection with enhanced pattern matching
- Header extraction and organization analysis

**RAG-Powered Chat**
- Interactive Q&A about papers using retrieval-augmented generation
- Context-aware responses with FAISS vector similarity search
- Multiple chat modes (general, focused, technical)
- Chat history summarization and key point extraction

**Performance Metrics**
- Quality evaluation: BLEU, METEOR, ROUGE scores
- Language metrics: Perplexity, n-gram diversity, coherence
- Factual consistency checking
- Resource usage tracking (CPU, memory, GPU)

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- Tesseract OCR (for scanned PDF support)

### Required Ollama Models

```bash
ollama pull deepseek-r1:1.5b
ollama pull deepseek-r1:8b
ollama pull mistral
ollama pull llama3:8b
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd research-paper-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Usage

1. Ensure Ollama is running:
```bash
ollama serve
```

2. Launch the application:
```bash
streamlit run app.py
```

3. Open your browser at `http://localhost:8501`

4. **Analysis Tab**:
   - Upload a PDF or paste research paper text
   - Configure analysis options in the sidebar
   - Click "Analyze Paper" to run comprehensive analysis

5. **Chat Tab**:
   - Ask questions about the paper
   - Select preferred model from sidebar
   - Choose chat mode (general/focused/technical)
   - Generate summaries and extract key points

## Project Structure

```
.
├── app.py                          # Main entry point
├── research_paper_assistant.py     # Core application logic
├── pdf_processor.py                # PDF extraction and OCR
├── citation_analyzer.py            # Citation detection and network analysis
├── structure_analyzer.py           # Paper structure analysis
├── rag_processor.py                # RAG implementation with FAISS
├── enhanced_chat_interface.py      # Chat interface with model integration
├── model_comparison_analyzer.py    # Multi-model benchmarking
├── performance_analyzer.py         # Quality and performance metrics
└── requirements.txt                # Python dependencies
```

## Analysis Capabilities

**Citation Network Analysis**
- Network visualization with nodes and edges
- Metrics: node count, edge count, density, average degree
- Context extraction around citation pairs

**Structure Completeness**
- Section detection with scoring
- Visual radar chart representation
- Figure/table enumeration

**Model Performance Comparison**
- Response time distribution (box plots)
- Token count analysis
- Consistency scoring
- Throughput calculation (tokens/second)

**Quality Metrics**
- BLEU: Translation quality metric
- METEOR: Semantic similarity
- ROUGE-1, ROUGE-2, ROUGE-L: Summary evaluation
- Perplexity: Language model quality
- Coherence: Inter-sentence semantic similarity

## Technology Stack

- **Frontend**: Streamlit
- **LLM Framework**: LangChain with Ollama integration
- **Vector Store**: FAISS
- **Embeddings**: HuggingFace Sentence-Transformers
- **PDF Processing**: PyMuPDF (fitz), Tesseract OCR, Pillow
- **Visualization**: Plotly, NetworkX
- **NLP Libraries**: NLTK, Rouge-Score, Transformers
- **Metrics**: SentenceTransformers, GPT-2 (for perplexity)

## Configuration

Adjust analysis settings in the sidebar:
- **Analysis Depth**: Controls detail level (1-5)
- **Analysis Focus**: Toggle citation, structure, and model comparison
- **Visualization Options**: Enable/disable specific charts
- **Chat Model Selection**: Choose from 4 available models
- **Chat Mode**: General, focused, or technical responses

## Performance Notes

- First-time model loading may take 2-5 minutes
- Large PDFs (>100 pages) require additional processing time
- GPU acceleration recommended for CUDA-enabled systems
- RAG initialization occurs automatically on paper upload
- Memory usage scales with paper size and model selection

## Limitations

- Requires Ollama server running locally
- Models must be pre-downloaded
- OCR accuracy depends on scan quality
- Citation detection patterns may not cover all formats
- Performance metrics require reference text for some calculations

## Contributing

Contributions are welcome. Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request

## License

[Specify your license here]

## Troubleshooting

**Ollama Connection Error**: Ensure Ollama is running on `http://localhost:11434`

**Model Not Found**: Pull the required model using `ollama pull <model-name>`

**OCR Failures**: Install Tesseract OCR and verify it's in your system PATH

**Memory Issues**: Reduce analysis depth or use smaller models

**NLTK Data Missing**: Run the NLTK download commands in Installation step 4