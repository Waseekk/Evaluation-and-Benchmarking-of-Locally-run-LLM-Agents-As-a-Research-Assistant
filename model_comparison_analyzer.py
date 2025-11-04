# model_comparison_analyzer.py

import time
import psutil
import numpy as np
import pandas as pd
import re
from typing import List, Dict
from langchain_ollama import ChatOllama
import plotly.graph_objects as go

class ModelComparisonAnalyzer:
    """Enhanced comparison of different language models."""
    
    def __init__(self):
        # Lazy loading: models loaded on-demand, not at startup
        self._loaded_models = {}

        # Model configurations (metadata only, no actual loading)
        self.model_configs = {
            "deepseek-1.5b": {
                "model": "deepseek-r1:1.5b",
                "temperature": 0.3,
                "base_url": "http://localhost:11434"
            },
            "deepseek-8b": {
                "model": "deepseek-r1:8b",
                "temperature": 0.3,
                "base_url": "http://localhost:11434"
            },
            "mistral": {
                "model": "mistral",
                "temperature": 0.3,
                "base_url": "http://localhost:11434"
            },
            "llama3-8b": {
                "model": "llama3:8b",
                "temperature": 0.3,
                "base_url": "http://localhost:11434"
            }
        }
        
        # Model-specific scaling factors
        self.scaling_factors = {
            'deepseek-1.5b': 1.0,
            'deepseek-8b': 1.0,
            'mistral': 1.00,
            'llama3-8b': 1.00
        }
        
        self.performance_metrics = {model: {
            'response_times': [],
            'token_counts': [],
            'consistency_scores': []
        } for model in self.model_configs.keys()}

    def get_model(self, model_name: str) -> ChatOllama:
        """
        Lazy load models on-demand.
        Only loads a model when it's actually needed, not at startup.

        Args:
            model_name: Name of the model to load (e.g., "deepseek-1.5b")

        Returns:
            ChatOllama instance for the requested model
        """
        if model_name not in self._loaded_models:
            if model_name not in self.model_configs:
                raise ValueError(f"Unknown model: {model_name}. Available: {list(self.model_configs.keys())}")

            print(f"Loading model: {model_name}...")
            config = self.model_configs[model_name]
            self._loaded_models[model_name] = ChatOllama(
                model=config["model"],
                temperature=config["temperature"],
                base_url=config["base_url"]
            )
            print(f"[OK] Model {model_name} loaded successfully")

        return self._loaded_models[model_name]

    # ⭐ NEW METHOD: Estimate token count for intelligent chunking
    def estimate_token_count(self, text: str) -> int:
        """Estimate the number of tokens in text."""
        # Rough estimate: 1 token ≈ 0.75 words (or 1.3 tokens per word)
        words = len(text.split())
        return int(words * 1.3)
    
    def _extract_chunk_summary(self, chunk: str, chunk_num: int, total_chunks: int, model) -> str:
        """
        Enhanced first pass: Extract structured information from a chunk.
        
        Improvements:
        - Structured JSON-like output for easier parsing
        - Section-aware extraction (detects if chunk is intro/methods/results)
        - Handles incomplete sentences at chunk boundaries
        - Extracts specific quantitative data
        """
        
        # Detect likely section type based on content
        section_hint = self._detect_chunk_section(chunk)
        
        extraction_prompt = f"""You are analyzing part {chunk_num} of {total_chunks} from a research paper.

    SECTION TYPE: This appears to be from the {section_hint} section.

    TASK: Extract key factual information in a structured format. Do NOT analyze or interpret - just extract facts.

    Extract the following if present:
    1. RESEARCH_TOPIC: Main subject or research question
    2. METHODOLOGY: Specific methods, algorithms, or approaches mentioned
    3. RESULTS: Quantitative findings, metrics, percentages, or outcomes
    4. KEY_TERMS: Important technical terms or concepts (3-5 terms)
    5. CITATIONS: Author names and years mentioned (e.g., "Smith 2020")
    6. DATA_POINTS: Any specific numbers, statistics, or measurements

    FORMAT your response as:
    RESEARCH_TOPIC: [extract or write "Not mentioned"]
    METHODOLOGY: [extract or write "Not mentioned"]
    RESULTS: [extract or write "Not mentioned"]
    KEY_TERMS: [list terms separated by semicolons]
    CITATIONS: [list citations separated by semicolons]
    DATA_POINTS: [list numbers/stats separated by semicolons]

    CHUNK POSITION: Section {chunk_num}/{total_chunks}
    {"[NOTE: This is the first section - may contain abstract/introduction]" if chunk_num == 1 else ""}
    {"[NOTE: This is the final section - may contain conclusions]" if chunk_num == total_chunks else ""}

    TEXT TO ANALYZE:
    {chunk}

    EXTRACTED INFORMATION:"""
        
        try:
            response = model.invoke(extraction_prompt)
            response_text = str(response) if not isinstance(response, str) else response
            
            # Add metadata for synthesis phase
            structured_output = f"""=== CHUNK {chunk_num}/{total_chunks} ({section_hint}) ===
    {response_text}
    """
            return structured_output
            
        except Exception as e:
            print(f"Error extracting from chunk {chunk_num}: {str(e)}")
            return f"=== CHUNK {chunk_num}/{total_chunks} ===\nERROR: {str(e)}\n"


    def _synthesize_analysis(self, chunk_summaries: list, analysis_type: str, model) -> str:
        """
        Enhanced second pass: Synthesize coherent analysis from structured extracts.
        
        Improvements:
        - Aware of chunk positions and section types
        - Explicit deduplication instructions
        - Maintains academic tone
        - Handles contradictions between chunks
        - Creates narrative flow
        """
        
        # Combine all chunk summaries with clear separation
        combined_context = "\n".join(chunk_summaries)
        
        # Count total information blocks
        num_chunks = len(chunk_summaries)
        
        # Analysis type specific instructions
        analysis_instructions = {
            "general": """Provide a comprehensive analysis with this structure:
    1. **Main Topic & Research Objective** (2-3 sentences)
    2. **Key Methodology** (3-4 sentences focusing on approach and techniques)
    3. **Major Findings** (3-4 sentences with specific results if available)
    4. **Strengths** (2-3 bullet points)
    5. **Areas for Improvement** (2-3 bullet points)

    Keep analysis concise, specific, and evidence-based.""",
            
            "methodology": """Focus on methodological analysis:
    1. **Approach Overview** (what methods were used)
    2. **Methodological Rigor** (assess completeness and validity)
    3. **Data Collection & Analysis** (evaluate techniques)
    4. **Reproducibility** (assess if methods are clearly described)
    5. **Limitations** (identify methodological gaps)""",
            
            "results": """Focus on results analysis:
    1. **Key Findings Summary** (what was discovered)
    2. **Data Presentation Quality** (clarity of tables/figures)
    3. **Statistical Validity** (assess rigor of analysis)
    4. **Results Interpretation** (how well results support claims)
    5. **Missing Elements** (what additional analysis would help)"""
        }
        
        analysis_instruction = analysis_instructions.get(analysis_type, analysis_instructions["general"])
        
        synthesis_prompt = f"""You are synthesizing a comprehensive research paper analysis from {num_chunks} extracted information blocks.

    EXTRACTED INFORMATION FROM ALL SECTIONS:
    {combined_context}

    ANALYSIS INSTRUCTIONS:
    {analysis_instruction}

    IMPORTANT SYNTHESIS RULES:
    1. **Deduplicate**: If the same information appears in multiple chunks, mention it only ONCE
    2. **Integrate**: Combine related information from different chunks into coherent paragraphs
    3. **Position-aware**: Information from early chunks (abstract/intro) vs late chunks (results/conclusion) should be weighted appropriately
    4. **Evidence-based**: Use specific data points, citations, and metrics from the extracts
    5. **Academic tone**: Maintain formal, objective language
    6. **Flow**: Create logical narrative connections between sections
    7. **Contradictions**: If chunks contain conflicting info, note this explicitly
    8. **Completeness**: If certain aspects are "Not mentioned" across all chunks, state this clearly

    AVOID:
    - Repeating the same point multiple times
    - Generic statements without evidence from the extracts
    - Analyzing information that wasn't in the chunks
    - Overly verbose or redundant language

    SYNTHESIZED ANALYSIS:"""
        
        try:
            response = model.invoke(synthesis_prompt)
            response_text = str(response) if not isinstance(response, str) else response
            
            # Post-process to ensure quality
            response_text = self._post_process_synthesis(response_text)
            
            return response_text
            
        except Exception as e:
            print(f"Error in synthesis: {str(e)}")
            return f"Error synthesizing analysis: {str(e)}"


    def _detect_chunk_section(self, chunk: str) -> str:
        """
        Detect which section of the paper this chunk likely belongs to.
        Helps provide better context to the extraction prompt.
        """
        chunk_lower = chunk.lower()
        
        # Section indicators
        if any(word in chunk_lower for word in ['abstract', 'summary', 'overview']):
            return "ABSTRACT/OVERVIEW"
        elif any(word in chunk_lower for word in ['introduction', 'background', 'motivation']):
            return "INTRODUCTION"
        elif any(word in chunk_lower for word in ['method', 'approach', 'algorithm', 'procedure', 'design']):
            return "METHODOLOGY"
        elif any(word in chunk_lower for word in ['result', 'finding', 'outcome', 'experiment', 'evaluation']):
            return "RESULTS"
        elif any(word in chunk_lower for word in ['discussion', 'analysis', 'interpretation']):
            return "DISCUSSION"
        elif any(word in chunk_lower for word in ['conclusion', 'summary', 'future work', 'limitation']):
            return "CONCLUSION"
        elif any(word in chunk_lower for word in ['reference', 'bibliography', 'citation']):
            return "REFERENCES"
        else:
            return "BODY/MIXED"


    def _post_process_synthesis(self, text: str) -> str:
        """
        Clean up synthesized analysis for better readability.
        """
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Ensure section headers are properly formatted
        text = re.sub(r'(\d+)\.\s*\*\*([^*]+)\*\*', r'\n\n**\1. \2**\n', text)
        
        # Remove any leftover chunk markers
        text = re.sub(r'=== CHUNK.*?===\n?', '', text)
        
        # Clean up bullet points
        text = re.sub(r'\n-\s+', '\n• ', text)
        
        return text.strip()

    def _chunk_text(self, text: str, max_length: int = 1000) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        for word in words:
            word_tokens = len(word.split()) * 1.3
            if current_length + word_tokens > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_tokens
            else:
                current_chunk.append(word)
                current_length += word_tokens
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks
    
    def calculate_token_count(self, text: str) -> int:
        words = text.split()
        return len(words) * 1.3
    
    def calculate_consistency_score(self, responses: List[str]) -> float:
        if not responses:
            return 0.0
        word_sets = [set(response.lower().split()) for response in responses]
        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                if union > 0:
                    similarities.append(intersection / union)
        return sum(similarities) / max(1, len(similarities))
    
    def analyze_paper(self, text: str, analysis_type: str = "general", num_trials: int = 1) -> Dict[str, any]:
        results = {}
        prompts = {
            "general": """Analyze this research paper with the following structure:
1. Main Topic & Objective
2. Key Methodology
3. Major Findings
4. Strengths
5. Areas for Improvement

Keep the analysis concise and focused.""",
            "methodology": "Focus on analyzing the methodology section of this paper. Evaluate its completeness and rigor.",
            "results": "Analyze the results section, focusing on data presentation and statistical validity."
        }
        system_prompt = """You are a research paper analysis expert. Analyze the given text and provide:
- Clear, structured feedback
- Specific examples from the text
- Constructive suggestions
Keep your response concise and well-organized."""
        prompt = prompts.get(analysis_type, prompts["general"])

        for model_name in self.model_configs.keys():
            model = self.get_model(model_name)  # Lazy load only when needed
            model_results = {
                'responses': [],
                'response_times': [],
                'token_counts': [],
                'memory_usage': [],
                'errors': [],
                'success_count': 0
            }
            
            for trial in range(num_trials):
                try:
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024
                    
                    # ⭐ INTELLIGENT CHUNKING: Only chunk if text is large
                    estimated_tokens = self.estimate_token_count(text)
                    
                    if estimated_tokens > 3000:
                        # Large paper: chunk it
                        chunks = self._chunk_text(text, max_length=1000)
                        print(f"📄 {model_name}: Large paper detected ({estimated_tokens} tokens) - splitting into {len(chunks)} chunks")
                    else:
                        # Small/medium paper: process as single chunk
                        chunks = [text]
                        print(f"📄 {model_name}: Processing paper as single chunk ({estimated_tokens} tokens)")
                    
                    combined_response = ""
                    start_time = time.time()
                    
                    for i, chunk in enumerate(chunks):
                        chunk_prompt = f"{prompt}\n\nAnalyzing part {i+1} of {len(chunks)}:\n\nText: {chunk}"
                        formatted_message = f"System: {system_prompt}\n\nHuman: {chunk_prompt}"
                        chunk_response = model.invoke(formatted_message)
                        if not isinstance(chunk_response, str):
                            chunk_response = str(chunk_response)
                        combined_response += chunk_response + "\n\n"
                    
                    response_time = time.time() - start_time
                    memory_after = process.memory_info().rss / 1024 / 1024
                    memory_usage = memory_after - memory_before
                    token_count = self.calculate_token_count(combined_response)
                    
                    model_results['responses'].append(combined_response)
                    model_results['response_times'].append(response_time)
                    model_results['token_counts'].append(token_count)
                    model_results['memory_usage'].append(memory_usage)
                    model_results['success_count'] += 1
                except Exception as e:
                    error_info = {
                        'trial': trial,
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    }
                    model_results['errors'].append(error_info)
                    print(f"Error in trial {trial} with {model_name}: {str(e)}")
            
            if model_results['responses']:
                response_times = np.array(model_results['response_times'])
                token_counts = np.array(model_results['token_counts'])
                memory_usage = np.array(model_results['memory_usage'])
                results[model_name] = {
                    'response': model_results['responses'][-1],
                    'avg_response_time': np.mean(response_times),
                    'std_response_time': np.std(response_times),
                    'avg_token_count': np.mean(token_counts),
                    'std_token_count': np.std(token_counts),
                    'avg_memory_usage': np.mean(memory_usage),
                    'std_memory_usage': np.std(memory_usage),
                    'consistency_score': self.calculate_consistency_score(model_results['responses']),
                    'error_rate': len(model_results['errors']) / num_trials,
                    'success_rate': model_results['success_count'] / num_trials,
                    'performance_history': {
                        'response_times': model_results['response_times'],
                        'token_counts': model_results['token_counts'],
                        'memory_usage': model_results['memory_usage']
                    },
                    'errors': model_results['errors']
                }
            else:
                results[model_name] = {
                    'response': f"Error analyzing with {model_name}. Please try again.",
                    'error_rate': 1.0,
                    'success_rate': 0.0,
                    'errors': model_results['errors']
                }
            self.performance_metrics[model_name]['response_times'].extend(model_results['response_times'])
            self.performance_metrics[model_name]['token_counts'].extend(model_results['token_counts'])
            if results[model_name].get('consistency_score'):
                self.performance_metrics[model_name]['consistency_scores'].append(
                    results[model_name]['consistency_score']
                )
        return results
    
    # ⭐ NEW METHOD: Analyze with single model for progress tracking
    def analyze_paper_single_model(self, text: str, model_name: str, model,
                               analysis_type: str = "general",
                               num_trials: int = 1,
                               progress_callback=None,
                               custom_prompts: Dict[str, str] = None) -> Dict[str, any]:
        """
        Analyze paper with a single model using two-pass approach for long papers.

        Args:
            text: Research paper text
            model_name: Name of the model
            model: Model instance
            analysis_type: Type of analysis ('general', 'methodology', 'results')
            num_trials: Number of analysis trials (for consistency)
            progress_callback: Optional callback(status_text, progress_value)
            custom_prompts: Optional dict of custom prompts {"general": str, "methodology": str, "results": str}

        Returns:
            Dictionary with analysis results and metrics
        """
        
        model_results = {
            'responses': [],
            'response_times': [],
            'token_counts': [],
            'memory_usage': [],
            'errors': [],
            'success_count': 0,
            'chunking_stats': {}
        }
        
        for trial in range(num_trials):
            try:
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024
                
                # Intelligent chunking decision
                estimated_tokens = self.estimate_token_count(text)
                
                if estimated_tokens > 3000:
                    # LONG PAPER: Use two-pass analysis
                    chunks = self._chunk_text(text, max_length=1000)
                    num_chunks = len(chunks)
                    
                    print(f"📄 {model_name}: Large paper ({estimated_tokens} tokens) - using two-pass analysis with {num_chunks} chunks")
                    
                    if progress_callback:
                        progress_callback(f"Pass 1/2: Extracting key points from {num_chunks} chunks...", 0.0)
                    
                    # === PASS 1: Extract key points from each chunk ===
                    chunk_summaries = []
                    start_time = time.time()
                    
                    for i, chunk in enumerate(chunks):
                        if progress_callback:
                            progress = (i + 1) / num_chunks * 0.5  # First pass is 0-50%
                            progress_callback(
                                f"Pass 1/2: Extracting from chunk {i+1}/{num_chunks}...", 
                                progress
                            )
                        
                        summary = self._extract_chunk_summary(chunk, i + 1, num_chunks, model)
                        chunk_summaries.append(summary)
                    
                    pass1_time = time.time() - start_time
                    
                    # === PASS 2: Synthesize holistic analysis ===
                    if progress_callback:
                        progress_callback(f"Pass 2/2: Synthesizing holistic analysis...", 0.6)
                    
                    start_time = time.time()
                    combined_response = self._synthesize_analysis(chunk_summaries, analysis_type, model)
                    pass2_time = time.time() - start_time
                    
                    response_time = pass1_time + pass2_time
                    
                    # Store chunking statistics
                    model_results['chunking_stats'] = {
                        'num_chunks': num_chunks,
                        'pass1_time': pass1_time,
                        'pass2_time': pass2_time,
                        'chunk_summaries': chunk_summaries  # For debugging if needed
                    }
                    
                    if progress_callback:
                        progress_callback(f"✓ Two-pass analysis complete!", 1.0)
                    
                else:
                    # SMALL PAPER: Direct analysis (no chunking)
                    print(f"📄 {model_name}: Processing paper as single chunk ({estimated_tokens} tokens)")
                    
                    if progress_callback:
                        progress_callback(f"Analyzing paper (single pass)...", 0.5)

                    # Default prompts
                    default_prompts = {
                        "general": """Analyze this research paper systematically:

1. **Main Topic & Research Question**: 
   - What specific problem/gap does this paper address?
   - What is the central research question or hypothesis?
   - Why is this important? (2-3 sentences)

2. **Key Methodology**: 
   - What approach/methods were used? (experimental, theoretical, computational, etc.)
   - Include specific techniques, tools, or frameworks
   - Sample size/dataset characteristics if applicable (3-4 sentences)

3. **Major Findings**: 
   - What were the main results? State clearly and quantitatively
   - Include specific metrics, percentages, or statistical measures (e.g., "accuracy improved by 15%", "p<0.05")
   - How do findings answer the research question? (3-4 sentences)

4. **Contribution & Significance**:
   - What is novel about this work?
   - How does it advance the field?
   - Compare to prior work if mentioned (2-3 points)

5. **Strengths**: 
   - What does the paper do well? Be specific
   - Provide evidence from the paper (e.g., "Figure 3 clearly shows...", "The authors validated using...")
   - Consider: rigor, clarity, reproducibility (2-3 points)

6. **Limitations & Areas for Improvement**: 
   - What could be enhanced or is missing?
   - Are there methodological gaps, limited scope, or unclear explanations?
   - Suggest specific improvements (2-3 points)

**Citation Format**: Reference specific sections, figures, or tables (e.g., "Section 3.2 describes...", "Table 1 shows...")""",

                            "methodology": """Analyze the methodology in detail:

1. **Research Design & Approach**: 
   - What type of study is this? (experimental, observational, simulation, theoretical, etc.)
   - Is the design appropriate for the research question?
   - What is the overall experimental/analytical framework?

2. **Data Collection**: 
   - What data sources were used? (sample size, dataset name, collection period)
   - How was data collected/generated?
   - Are there potential sampling biases or data quality issues?

3. **Methods & Techniques**: 
   - Describe specific methods, algorithms, or statistical tests used
   - Were these methods appropriate and state-of-the-art?
   - Were controls, baselines, or comparison groups established?

4. **Rigor & Validity**: 
   - Internal validity: Are results reliable within the study context?
   - External validity: Can results generalize beyond this study?
   - Were confounding variables addressed?
   - Was statistical power adequate?

5. **Reproducibility**: 
   - How clearly are methods described? Could someone replicate this?
   - Are code, data, or supplementary materials available?
   - Are parameters, hyperparameters, or settings specified?

6. **Methodological Limitations**: 
   - What are the main weaknesses or gaps?
   - What assumptions were made? Are they reasonable?
   - What alternative methods could strengthen the work?

**Provide evidence-based critique**: Reference specific sections, equations, or procedures (e.g., "Section 2.3 lacks detail on...", "The choice of α=0.05 in Equation 4...")""",

                            "results": """Analyze the results comprehensively:

1. **Key Findings Summary**: 
   - What are the primary results? State with specific metrics
   - Include quantitative measures: percentages, effect sizes, confidence intervals, p-values
   - Are results statistically AND practically significant?
   - Example format: "Method X achieved 87% accuracy (±2.3%), outperforming baseline by 12%"

2. **Presentation Quality**: 
   - Are tables and figures clear, well-labeled, and informative?
   - Do visualizations effectively communicate the data?
   - Are trends, patterns, or key differences easily visible?
   - Identify best/worst presented results

3. **Statistical Rigor & Interpretation**: 
   - Were appropriate statistical tests used?
   - Are error bars, confidence intervals, or significance tests reported?
   - Were multiple comparisons or false discovery rates addressed?
   - Is statistical significance confused with practical importance?

4. **Results vs. Claims Alignment**: 
   - Do the results fully support the conclusions drawn?
   - Are there overclaims or unsupported generalizations?
   - Are limitations of the results acknowledged?
   - Are alternative explanations considered?

5. **Completeness & Missing Elements**: 
   - What additional analyses would strengthen findings?
   - Are negative or null results reported, or only positive ones?
   - Are subgroup analyses or sensitivity analyses needed?
   - What follow-up experiments are suggested?

6. **Comparison to Prior Work**: 
   - How do results compare to existing literature?
   - Are comparisons fair (same datasets, metrics, conditions)?
   - Do results confirm, contradict, or extend prior findings?

**Focus on**: Specific data points, quantitative measures, and evidence from figures/tables (e.g., "Figure 2b shows...", "Table 3 indicates...")"""
                    }

                    # Use custom prompts if provided, otherwise use defaults
                    if custom_prompts:
                        prompts = {
                            "general": custom_prompts.get("general") or default_prompts["general"],
                            "methodology": custom_prompts.get("methodology") or default_prompts["methodology"],
                            "results": custom_prompts.get("results") or default_prompts["results"]
                        }
                    else:
                        prompts = default_prompts

                    # More directive system prompt
                    system_prompt = """You are an expert research paper analyst. Your analysis must be:
    - Structured following the exact format provided
    - Evidence-based with specific examples from the text
    - Objective and constructively critical
    - Concise but substantive

    Avoid generic statements - focus on this specific paper's content."""
                    
                    prompt = prompts.get(analysis_type, prompts["general"])
                    formatted_message = f"System: {system_prompt}\n\nHuman: {prompt}\n\nText: {text}"
                    
                    start_time = time.time()
                    combined_response = model.invoke(formatted_message)
                    response_time = time.time() - start_time
                    
                    if not isinstance(combined_response, str):
                        combined_response = str(combined_response)
                    
                    if progress_callback:
                        progress_callback(f"✓ Analysis complete!", 1.0)
                
                # Calculate metrics
                memory_after = process.memory_info().rss / 1024 / 1024
                memory_usage = memory_after - memory_before
                token_count = self.calculate_token_count(combined_response)
                
                # Store results
                model_results['responses'].append(combined_response)
                model_results['response_times'].append(response_time)
                model_results['token_counts'].append(token_count)
                model_results['memory_usage'].append(memory_usage)
                model_results['success_count'] += 1
                
            except Exception as e:
                error_info = {
                    'trial': trial,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
                model_results['errors'].append(error_info)
                print(f"Error in trial {trial} with {model_name}: {str(e)}")
        
        # Calculate final metrics
        if model_results['responses']:
            response_times = np.array(model_results['response_times'])
            token_counts = np.array(model_results['token_counts'])
            memory_usage = np.array(model_results['memory_usage'])
            
            result = {
                'response': model_results['responses'][-1],  # Last (best) response
                'avg_response_time': np.mean(response_times),
                'std_response_time': np.std(response_times),
                'avg_token_count': np.mean(token_counts),
                'std_token_count': np.std(token_counts),
                'avg_memory_usage': np.mean(memory_usage),
                'std_memory_usage': np.std(memory_usage),
                'consistency_score': self.calculate_consistency_score(model_results['responses']),
                'error_rate': len(model_results['errors']) / num_trials,
                'success_rate': model_results['success_count'] / num_trials,
                'performance_history': {
                    'response_times': model_results['response_times'],
                    'token_counts': model_results['token_counts'],
                    'memory_usage': model_results['memory_usage']
                },
                'errors': model_results['errors'],
                'chunking_stats': model_results.get('chunking_stats', {})
            }
            
            # Update global performance metrics
            self.performance_metrics[model_name]['response_times'].extend(model_results['response_times'])
            self.performance_metrics[model_name]['token_counts'].extend(model_results['token_counts'])
            if result.get('consistency_score'):
                self.performance_metrics[model_name]['consistency_scores'].append(
                    result['consistency_score']
                )
            
            return result
        else:
            return {
                'response': f"Error analyzing with {model_name}. Please try again.",
                'error_rate': 1.0,
                'success_rate': 0.0,
                'errors': model_results['errors']
            }
    
    def create_performance_visualizations(self, results: Dict) -> List[go.Figure]:
        figures = []
        response_times = go.Figure()
        for model_name, metrics in results.items():
            if 'performance_history' in metrics:
                response_times.add_trace(go.Box(
                    y=metrics['performance_history']['response_times'],
                    name=model_name,
                    boxpoints='all'
                ))
        response_times.update_layout(
            title="Response Time Distribution",
            yaxis_title="Time (seconds)",
            showlegend=True
        )
        figures.append(response_times)
        
        token_counts = go.Figure(data=[
            go.Bar(
                x=list(results.keys()),
                y=[m.get('avg_token_count', 0) for m in results.values()],
                error_y=dict(
                    type='data',
                    array=[np.std(m['performance_history']['token_counts']) 
                          if 'performance_history' in m else 0 
                          for m in results.values()]
                )
            )
        ])
        token_counts.update_layout(
            title="Average Token Count per Response",
            yaxis_title="Tokens",
            showlegend=False
        )
        figures.append(token_counts)
        
        consistency = go.Figure(data=[
            go.Bar(
                x=list(results.keys()),
                y=[m.get('consistency_score', 0) for m in results.values()],
                marker_color='rgb(55, 83, 109)'
            )
        ])
        consistency.update_layout(
            title="Model Consistency Scores",
            yaxis_title="Consistency Score",
            showlegend=False
        )
        figures.append(consistency)
        return figures
        
    def generate_performance_report(self, results: Dict) -> pd.DataFrame:
        """Generates performance report with model-specific scaling and throughput."""
        report_data = []
        for model_name, metrics in results.items():
            if 'avg_response_time' in metrics:
                response_times = metrics['performance_history']['response_times']
                token_counts = metrics['performance_history']['token_counts']
                mean_time = np.mean(response_times)
                std_time = np.std(response_times)
                mean_tokens = np.mean(token_counts)
                std_tokens = np.std(token_counts)
                
                # Apply model-specific scaling
                scale = self.scaling_factors.get(model_name, 1.0)
                
                # Calculate throughput
                throughput = mean_tokens / mean_time if mean_time > 0 else 0
                
                report_data.append({
                    "Model": model_name,
                    "Response Time": f"{mean_time * scale:.3f} ± {std_time:.3f}s",
                    "Token Count": f"{mean_tokens:.1f} ± {std_tokens:.1f}",
                 #   "Consistency": f"{metrics.get('consistency_score', 0):.2f}",
                  #  "Error Rate": f"{metrics.get('error_rate', 1.0):.2%}",
                    "Throughput": f"{throughput:.1f} tokens/s",
                    "Memory Usage": f"{metrics.get('avg_memory_usage', 0):.1f} MB"
                })
            else:
                report_data.append({
                    "Model": model_name,
                    "Error Rate": "100%",
                    "Status": "Failed to analyze"
                })
        return pd.DataFrame(report_data)