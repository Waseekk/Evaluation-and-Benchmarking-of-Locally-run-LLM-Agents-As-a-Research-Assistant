# performance_analyzer.py

import time
import numpy as np
import psutil
import torch
import pandas as pd
from typing import Tuple, Dict, List
import plotly.graph_objects as go
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
import nltk
from nltk.translate import bleu_score
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer

class PerformanceAnalyzer:
    def __init__(self):
        self.metrics = {
            "response_times": [],
            "memory_usage": [],
            "token_counts": [],
            "error_rates": [],
            "cpu_usage": [],
            "gpu_usage": [],
            "peak_memory": [],
            "memory_growth": [],
            "thread_usage": [],
            "perplexity_scores": [],
            "ngram_diversity": [],
            "response_coherence": [],
            "semantic_similarity": [],
            "consistency_scores": [],
            "bleu_scores": [],
            "meteor_scores": [],
            "rouge_scores": [],
            "factual_consistency": []
        }
        
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
        except Exception as e:
            print(f"Error initializing NLTK: {str(e)}")
            self.rouge_scorer = None
            
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Error loading SentenceTransformer: {str(e)}")
            self.encoder = None
            
        try:
            self.fact_check_model = AutoModelForSequenceClassification.from_pretrained(
                'microsoft/deberta-base-mnli', ignore_mismatched_sizes=True
            )
            self.fact_check_tokenizer = AutoTokenizer.from_pretrained(
                'microsoft/deberta-base-mnli', ignore_mismatched_sizes=True
            )
        except Exception as e:
            print(f"Error loading fact-checking model: {str(e)}")
            self.fact_check_model = None
            self.fact_check_tokenizer = None

    def calculate_perplexity(self, text) -> float:
        try:
            if hasattr(text, 'content'):
                text = text.content
            text_str = str(text)
            if not text_str.strip():
                return 0.0
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            inputs = tokenizer(text_str, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
            return torch.exp(outputs.loss).item()
        except Exception as e:
            print(f"Error calculating perplexity: {str(e)}")
            return 0.0

    def calculate_ngram_diversity(self, text, n: int = 3) -> float:
        try:
            if hasattr(text, 'content'):
                text = text.content
            text_str = str(text)
            from nltk import ngrams
            tokens = text_str.split()
            if len(tokens) < n:
                return 0.0
            ngram_list = list(ngrams(tokens, n))
            unique_ngrams = len(set(ngram_list))
            total_ngrams = len(ngram_list)
            return unique_ngrams / total_ngrams if total_ngrams > 0 else 0
        except Exception as e:
            print(f"Error calculating n-gram diversity: {str(e)}")
            return 0.0

    def calculate_response_coherence(self, text) -> float:
        try:
            if hasattr(text, 'content'):
                text = text.content
            text_str = str(text)
            import nltk
            nltk.download('punkt', quiet=True)
            sentences = nltk.sent_tokenize(text_str)
            if len(sentences) < 2:
                return 1.0
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(sentences)
            coherence_scores = []
            for i in range(len(embeddings) - 1):
                similarity = np.dot(embeddings[i], embeddings[i+1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
                )
                coherence_scores.append(similarity)
            return np.mean(coherence_scores)
        except Exception as e:
            print(f"Error calculating coherence: {str(e)}")
            return 0.0

    def calculate_bleu(self, reference: str, candidate: str) -> float:
        try:
            from nltk.tokenize import word_tokenize
            reference_tokens = word_tokenize(reference.lower())
            candidate_tokens = word_tokenize(candidate.lower())
            if not reference_tokens or not candidate_tokens:
                return 0.0
            weights = (0.25, 0.25, 0.25, 0.25)
            smoothing_function = bleu_score.SmoothingFunction().method1
            score = bleu_score.sentence_bleu(
                [reference_tokens], 
                candidate_tokens,
                weights=weights,
                smoothing_function=smoothing_function
            )
            return float(score)
        except Exception as e:
            print(f"Error calculating BLEU: {str(e)}")
            return 0.0

    def calculate_meteor(self, reference: str, candidate: str) -> float:
        try:
            if not reference.strip() or not candidate.strip():
                return 0.0
            reference_tokens = reference.lower().split()
            candidate_tokens = candidate.lower().split()
            score = single_meteor_score(reference_tokens, candidate_tokens)
            return float(score)
        except Exception as e:
            print(f"Error calculating METEOR: {str(e)}")
            return 0.0

    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        try:
            if self.rouge_scorer is None:
                return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
            scores = self.rouge_scorer.score(reference, candidate)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            print(f"Error calculating ROUGE: {str(e)}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    def check_factual_consistency(self, reference: str, candidate: str) -> float:
        try:
            if self.fact_check_model is None or self.fact_check_tokenizer is None:
                return 0.0
            inputs = self.fact_check_tokenizer(
                reference, candidate, truncation=True, padding=True, return_tensors='pt'
            )
            with torch.no_grad():
                outputs = self.fact_check_model(**inputs)
                prediction = torch.softmax(outputs.logits, dim=1)
                entailment_score = prediction[0][2].item()
            return entailment_score
        except Exception as e:
            print(f"Error checking factual consistency: {str(e)}")
            return 0.0

    def measure_response_time(self, model, text: str) -> Tuple[float, str]:
        start_time = time.time()
        try:
            response = model.invoke(text)
            elapsed_time = time.time() - start_time
            return elapsed_time, response
        except Exception as e:
            print(f"Error during response time measurement: {str(e)}")
            return -1, str(e)

    def track_resource_usage(self, model) -> Dict:
        cpu_percent = psutil.cpu_percent(interval=1)
        gpu_usage = None
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        process = psutil.Process()
        memory_info = process.memory_info()
        current_memory = memory_info.rss / 1024 / 1024
        thread_count = len(process.threads())
        return {
            "cpu_percent": cpu_percent,
            "gpu_usage": gpu_usage,
            "current_memory": current_memory,
            "thread_count": thread_count
        }

    def analyze_model_performance(self, models: Dict, sample_text: str, num_runs: int = 2) -> Dict:
        results = {}
        for model_name, model in models.items():
            responses = []
            resource_metrics = []
            quality_metrics = []
            response_metrics = []
            memory_growth = []
            
            for _ in range(num_runs):
                resource_usage = self.track_resource_usage(model)
                resource_metrics.append(resource_usage)
                memory_growth.append(resource_usage["current_memory"])
                response_time, response = self.measure_response_time(model, sample_text)
                responses.append(response)
                
                try:
                    if hasattr(response, 'content'):
                        response_text = str(response.content)
                    else:
                        response_text = str(response)
                    response_text = response_text.strip()
                    if response_text:
                        bleu = self.calculate_bleu(sample_text, response_text)
                        meteor = self.calculate_meteor(sample_text, response_text)
                        rouge_scores = self.calculate_rouge(sample_text, response_text)
                        factual_score = self.check_factual_consistency(sample_text, response_text)
                        response_metrics.append({
                            'bleu': bleu,
                            'meteor': meteor,
                            'rouge': rouge_scores,
                            'factual_consistency': factual_score
                        })
                except Exception as e:
                    print(f"Error calculating response metrics: {str(e)}")
                
                quality_metrics.append({
                    "perplexity": self.calculate_perplexity(response),
                    "ngram_diversity": self.calculate_ngram_diversity(response),
                    "coherence": self.calculate_response_coherence(response)
                })
            
            results[model_name] = {
                "resource_usage": {
                    "avg_cpu_percent": np.mean([m["cpu_percent"] for m in resource_metrics]),
                    "avg_gpu_usage": np.mean([m["gpu_usage"] for m in resource_metrics if m["gpu_usage"]]),
                    "peak_memory": max([m["current_memory"] for m in resource_metrics]),
                    "avg_thread_count": np.mean([m["thread_count"] for m in resource_metrics]),
                    "memory_growth": memory_growth
                },
                "quality_metrics": {
                    "avg_perplexity": np.mean([m["perplexity"] for m in quality_metrics]),
                    "avg_ngram_diversity": np.mean([m["ngram_diversity"] for m in quality_metrics]),
                    "avg_coherence": np.mean([m["coherence"] for m in quality_metrics])
                },
                "response_metrics": {
                    "avg_bleu": np.mean([m["bleu"] for m in response_metrics]),
                    "avg_meteor": np.mean([m["meteor"] for m in response_metrics]),
                    "avg_rouge1": np.mean([m["rouge"]["rouge1"] for m in response_metrics]),
                    "avg_rouge2": np.mean([m["rouge"]["rouge2"] for m in response_metrics]),
                    "avg_rougeL": np.mean([m["rouge"]["rougeL"] for m in response_metrics]),
                    "avg_factual_consistency": np.mean([m["factual_consistency"] for m in response_metrics])
                }
            }
        return results

    def create_performance_visualizations(self, results: Dict) -> list:
        figures = []
        resource_fig = go.Figure()
        models = list(results.keys())
        resource_fig.add_trace(go.Bar(
            name='CPU Usage (%)', x=models,
            y=[results[m]['resource_usage']['avg_cpu_percent'] for m in models]
        ))
        resource_fig.add_trace(go.Bar(
            name='Peak Memory (MB)', x=models,
            y=[results[m]['resource_usage']['peak_memory'] for m in models]
        ))
        resource_fig.update_layout(title='Resource Usage by Model', barmode='group', showlegend=True)
        figures.append(resource_fig)
        
        quality_fig = go.Figure()
        quality_metrics = ['avg_perplexity', 'avg_ngram_diversity', 'avg_coherence']
        for metric in quality_metrics:
            quality_fig.add_trace(go.Bar(
                name=metric.replace('avg_', '').replace('_', ' ').title(), x=models,
                y=[results[m]['quality_metrics'][metric] for m in models]
            ))
        quality_fig.update_layout(title='Quality Metrics by Model', barmode='group', showlegend=True)
        figures.append(quality_fig)
        
        response_fig = go.Figure()
        response_metrics = ['avg_bleu', 'avg_meteor', 'avg_rouge1', 'avg_rouge2', 'avg_rougeL', 'avg_factual_consistency']
        for metric in response_metrics:
            response_fig.add_trace(go.Bar(
                name=metric.replace('avg_', '').replace('_', ' ').title(), x=models,
                y=[results[m]['response_metrics'][metric] for m in models]
            ))
        response_fig.update_layout(title='Response Metrics by Model', barmode='group', showlegend=True)
        figures.append(response_fig)
        return figures

    def generate_performance_report(self, results: Dict) -> pd.DataFrame:
        report_data = []
        for model_name, metrics in results.items():
            data = {
                "Model": model_name,
                "CPU Usage (%)": f"{metrics['resource_usage']['avg_cpu_percent']:.1f}",
                "Peak Memory (MB)": f"{metrics['resource_usage']['peak_memory']:.1f}",
                "BLEU Score": f"{metrics['response_metrics']['avg_bleu']:.3f}",
                "METEOR Score": f"{metrics['response_metrics']['avg_meteor']:.3f}",
                "ROUGE-1": f"{metrics['response_metrics']['avg_rouge1']:.3f}",
                "ROUGE-2": f"{metrics['response_metrics']['avg_rouge2']:.3f}",
                "ROUGE-L": f"{metrics['response_metrics']['avg_rougeL']:.3f}",
                "Factual Consistency": f"{metrics['response_metrics']['avg_factual_consistency']:.3f}",
                "Perplexity": f"{metrics['quality_metrics']['avg_perplexity']:.2f}",
                "N-gram Diversity": f"{metrics['quality_metrics']['avg_ngram_diversity']:.2f}",
                "Coherence": f"{metrics['quality_metrics']['avg_coherence']:.2f}"
            }
            if 'gpu_usage' in metrics['resource_usage']:
                data["GPU Usage (%)"] = f"{metrics['resource_usage']['avg_gpu_usage']:.1f}"
            report_data.append(data)
        return pd.DataFrame(report_data)