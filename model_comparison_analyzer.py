# model_comparison_analyzer.py

import time
import psutil
import numpy as np
import pandas as pd
from typing import List, Dict
from langchain_ollama import ChatOllama
import plotly.graph_objects as go

class ModelComparisonAnalyzer:
    """Enhanced comparison of different language models."""
    
    def __init__(self):
        self.models = {
            "deepseek-1.5b": ChatOllama(
                model="deepseek-r1:1.5b",
                temperature=0.3,
                base_url="http://localhost:11434"
            ),
            "deepseek-8b": ChatOllama(
                model="deepseek-r1:8b",
                temperature=0.3,
                base_url="http://localhost:11434"
            ),
            "mistral": ChatOllama(
                model="mistral",
                temperature=0.3,
                base_url="http://localhost:11434"
            ),
            "llama3-8b": ChatOllama(
                model="llama3:8b",
                temperature=0.3,
                base_url="http://localhost:11434"
            )
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
        } for model in self.models.keys()}

    # ⭐ NEW METHOD: Estimate token count for intelligent chunking
    def estimate_token_count(self, text: str) -> int:
        """Estimate the number of tokens in text."""
        # Rough estimate: 1 token ≈ 0.75 words (or 1.3 tokens per word)
        words = len(text.split())
        return int(words * 1.3)

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
        
        for model_name, model in self.models.items():
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
                                   num_trials: int = 1) -> Dict[str, any]:
        """Analyze paper with a single model - used for progress tracking."""
        
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
                
                # Intelligent chunking
                estimated_tokens = self.estimate_token_count(text)
                
                if estimated_tokens > 3000:
                    chunks = self._chunk_text(text, max_length=1000)
                    print(f"📄 {model_name}: Large paper ({estimated_tokens} tokens) - {len(chunks)} chunks")
                else:
                    chunks = [text]
                    print(f"📄 {model_name}: Single chunk ({estimated_tokens} tokens)")
                
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
        
        # Calculate metrics
        if model_results['responses']:
            response_times = np.array(model_results['response_times'])
            token_counts = np.array(model_results['token_counts'])
            memory_usage = np.array(model_results['memory_usage'])
            
            result = {
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
            
            # Update performance metrics
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
                    "Consistency": f"{metrics.get('consistency_score', 0):.2f}",
                    "Error Rate": f"{metrics.get('error_rate', 1.0):.2%}",
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