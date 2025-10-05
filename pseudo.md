# Research Paper Analysis Assistant - Simple Pseudo-Code

## System Overview

A tool that analyzes research papers using multiple AI models, extracts citations, analyzes structure, and provides interactive Q&A.

---

## 1. Main Application Flow

```
START Application
    Initialize all components
    Display user interface
    
    IF user uploads PDF THEN:
        Extract text from PDF
        Process with OCR if needed
        Store text in memory
    
    IF user clicks "Analyze" THEN:
        Run citation analysis
        Run structure analysis
        Run model comparison
        Display all results
    
    IF user opens Chat tab THEN:
        Initialize RAG system
        Answer questions about paper
END
```

---

## 2. PDF Processing

```
FUNCTION ProcessPDF(pdf_file):
    Open PDF file
    Extract text from each page
    
    FOR each page:
        IF page has no text THEN:
            Apply OCR to extract text from images
        Add page text to full_text
    
    Extract metadata (title, author, pages)
    Count figures and tables
    
    RETURN {text, metadata}
```

---

## 3. Citation Analysis

```
FUNCTION AnalyzeCitations(text):
    Define patterns:
        - DOI: "doi.org/..."
        - arXiv: "arXiv:1234.5678"
        - Numbered: "[1]", "[2]"
        - Author-Year: "(Smith, 2020)"
    
    FOR each pattern:
        Find all matches in text
        Count occurrences
    
    Extract publication years
    Count citations per year
    
    Build citation network:
        FOR each paragraph:
            Find citations in paragraph
            Connect consecutive citations with arrows
    
    RETURN {counts, years, network}
```

---

## 4. Structure Analysis

```
FUNCTION AnalyzeStructure(text):
    Check for standard sections:
        - Abstract
        - Introduction
        - Methodology
        - Results
        - Discussion
        - Conclusion
    
    Count figures: "Fig. 1", "Figure 2", etc.
    Count tables: "Table 1", "Table 2", etc.
    
    Calculate completeness:
        completeness = (sections found / total sections)
    
    RETURN {sections, figures, tables, completeness}
```

---

## 5. RAG (Smart Question Answering)

```
FUNCTION InitializeRAG(paper_text):
    Split text into small chunks (500 characters each)
    
    FOR each chunk:
        Convert to vector using AI embeddings
        Store in vector database
    
    RETURN vector_database

FUNCTION AnswerQuestion(question, vector_database):
    Convert question to vector
    Find 3 most similar chunks from paper
    
    Create enhanced prompt:
        "Based on these excerpts: [chunks]
         Answer this question: [question]"
    
    Send to AI model
    Get response
    
    RETURN response
```

---

## 6. Model Comparison

```
FUNCTION CompareModels(paper_text):
    models = ["DeepSeek-1.5B", "DeepSeek-8B", "Mistral", "LLaMA3-8B"]
    
    FOR each model:
        Start timer
        Send paper to model for analysis
        Get response
        Stop timer
        
        Measure:
            - Response time
            - Number of tokens generated
            - Memory used
        
        Calculate quality:
            - BLEU score (translation quality)
            - ROUGE score (summary quality)
            - Coherence (how well sentences connect)
    
    RETURN comparison_results
```

---

## 7. Chat Interface

```
FUNCTION ChatWithPaper(user_message, paper_content):
    Get relevant chunks from RAG system
    
    Create system prompt:
        "You are analyzing this paper: [paper_content]
         Answer in [general/focused/technical] mode"
    
    Combine:
        system_prompt + relevant_chunks + user_message
    
    Send to selected AI model
    Get response
    
    Extract key points from response
    Add to chat history
    
    RETURN response
```

---

## 8. Performance Metrics

```
FUNCTION CalculateQualityMetrics(reference_text, generated_text):
    // BLEU Score (0 to 1, higher is better)
    Split both texts into words
    Compare word overlap with n-grams
    BLEU = calculate_overlap_score()
    
    // ROUGE Score (0 to 1, higher is better)
    ROUGE = compare_summaries(reference, generated)
    
    // Perplexity (lower is better)
    Perplexity = how_confused_is_model(generated_text)
    
    // Coherence (0 to 1, higher is better)
    Split into sentences
    FOR each consecutive sentence pair:
        Calculate similarity
    Coherence = average_similarity
    
    RETURN {BLEU, ROUGE, Perplexity, Coherence}
```

---

## 9. Complete Analysis Pipeline

```
START Main Analysis
    
    // Step 1: Input
    paper_text = GetPaperText()
    
    // Step 2: Citation Analysis
    citations = AnalyzeCitations(paper_text)
    Display citation counts
    Display year distribution chart
    Display citation network graph
    
    // Step 3: Structure Analysis
    structure = AnalyzeStructure(paper_text)
    Display section completeness
    Display figure/table counts
    Display radar chart
    
    // Step 4: Model Comparison
    results = CompareModels(paper_text)
    FOR each model:
        Display response time
        Display token count
        Display quality metrics
    
    // Step 5: Initialize Chat
    rag_system = InitializeRAG(paper_text)
    
    // Step 6: Wait for user questions
    WHILE user is chatting:
        question = GetUserQuestion()
        answer = AnswerQuestion(question, rag_system)
        DisplayAnswer(answer)
    
END
```

---

## 10. Key Technologies Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| PDF Reading | PyMuPDF | Extract text from PDFs |
| OCR | Tesseract | Read scanned documents |
| AI Models | Ollama | Run LLMs locally |
| Embeddings | SentenceTransformers | Convert text to vectors |
| Vector Search | FAISS | Fast similarity search |
| Visualization | Plotly | Create interactive charts |
| Interface | Streamlit | Web application |

---

## 11. System Metrics

**Speed:**
- DeepSeek-1.5B: ~3 seconds per query
- DeepSeek-8B: ~7 seconds per query
- Mistral: ~5 seconds per query
- LLaMA3-8B: ~7 seconds per query

**Accuracy:**
- BLEU Score: 0.35-0.45
- ROUGE Score: 0.30-0.40
- Coherence: 0.70-0.85

**Memory:**
- Base: ~450 MB
- With PDF: ~650 MB
- With RAG: ~850 MB
- During Analysis: ~4-5 GB

---

## 12. Data Flow Diagram

```
User Input (PDF/Text)
    ↓
PDF Processor → Extract Text
    ↓
Text Storage (Session)
    ↓
    ├→ Citation Analyzer → Citation Network
    ├→ Structure Analyzer → Completeness Score
    ├→ RAG Processor → Vector Database
    └→ Model Comparison → Quality Metrics
         ↓
    Display Results
         ↓
    Chat Interface ← RAG System
         ↓
    User Gets Answers
```

---

## Summary

This system:
1. **Extracts** text from research papers
2. **Analyzes** citations and structure
3. **Compares** multiple AI models
4. **Enables** intelligent Q&A about papers
5. **Measures** quality with standard metrics

Simple workflow: Upload → Analyze → Chat → Get Insights
