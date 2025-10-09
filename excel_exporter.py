# excel_exporter.py

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import io

class ExcelExporter:
    """Handles comprehensive Excel export for all analysis data."""
    
    def __init__(self):
        self.excel_buffer = io.BytesIO()
        self.metadata = {
            'export_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'analysis_version': '1.0',
            'paper_title': 'Research Paper Analysis'
        }
    
    def create_comprehensive_export(
        self,
        citation_data: Optional[Dict] = None,
        structure_data: Optional[Dict] = None,
        model_results: Optional[Dict] = None,
        performance_results: Optional[Dict] = None,
        paper_metadata: Optional[Dict] = None
    ) -> io.BytesIO:
        """
        Creates a comprehensive Excel file with all analysis data.
        
        Args:
            citation_data: Citation analysis results
            structure_data: Structure analysis results
            model_results: Model comparison results
            performance_results: Performance analysis results
            paper_metadata: PDF metadata
            
        Returns:
            BytesIO buffer containing the Excel file
        """
        with pd.ExcelWriter(self.excel_buffer, engine='openpyxl') as writer:
            # Sheet 1: Metadata
            self._write_metadata_sheet(writer, paper_metadata)
            
            # Sheets 2-5: Citation Analysis
            if citation_data:
                self._write_citation_sheets(writer, citation_data)
            
            # Sheets 6-8: Structure Analysis
            if structure_data:
                self._write_structure_sheets(writer, structure_data)
            
            # Sheets 9-12: Model Comparison
            if model_results:
                self._write_model_sheets(writer, model_results)
            
            # Sheets 13-15: Performance Metrics
            if performance_results:
                self._write_performance_sheets(writer, performance_results)
        
        self.excel_buffer.seek(0)
        return self.excel_buffer
    
    def _write_metadata_sheet(self, writer: pd.ExcelWriter, paper_metadata: Optional[Dict]):
        """Write metadata information sheet."""
        metadata_rows = [
            ['Research Paper Analysis - Data Export'],
            [''],
            ['Export Information'],
            ['Export Date', self.metadata['export_timestamp']],
            ['Analysis Version', self.metadata['analysis_version']],
            [''],
            ['Paper Information']
        ]
        
        if paper_metadata:
            metadata_rows.extend([
                ['Title', paper_metadata.get('title', 'Unknown')],
                ['Author', paper_metadata.get('author', 'Unknown')],
                ['Total Pages', paper_metadata.get('total_pages', 'N/A')],
                ['Creation Date', paper_metadata.get('creation_date', 'N/A')],
                ['File Size (KB)', paper_metadata.get('file_size_kb', 'N/A')],
                ['Contains Images', 'Yes' if paper_metadata.get('has_images') else 'No'],
                ['OCR Applied', 'Yes' if paper_metadata.get('ocr_applied') else 'No']
            ])
        
        df_metadata = pd.DataFrame(metadata_rows)
        df_metadata.to_excel(writer, sheet_name='Metadata', index=False, header=False)
    
    def _write_citation_sheets(self, writer: pd.ExcelWriter, citation_data: Dict):
        """Write citation analysis sheets."""
        
        # Sheet 2: Citation Summary
        citation_summary = []
        for key, value in citation_data.items():
            if key.endswith('_count'):
                citation_type = key.replace('_count', '').replace('_', ' ').title()
                citation_summary.append({
                    'Citation Type': citation_type,
                    'Count': value
                })
        
        if citation_summary:
            df_summary = pd.DataFrame(citation_summary)
            df_summary.to_excel(writer, sheet_name='Citation_Summary', index=False)
        
        # Sheet 3: Citation Year Distribution
        if 'year_distribution' in citation_data and citation_data['year_distribution']:
            year_dist = citation_data['year_distribution']
            df_years = pd.DataFrame([
                {'Year': year, 'Citation Count': count}
                for year, count in sorted(year_dist.items())
            ])
            df_years.to_excel(writer, sheet_name='Citation_Years', index=False)
        
        # Sheet 4: Citation Network Nodes
        if 'network' in citation_data and citation_data['network'].get('citations_in_context'):
            contexts = citation_data['network']['citations_in_context']
            if contexts:
                df_contexts = pd.DataFrame(contexts)
                
                # ✅ FIXED: Handle both old and new formats
                if len(df_contexts.columns) == 4:
                    # NLP version with relationship metadata
                    df_contexts.columns = ['From Citation', 'To Citation', 'Context', 'Relationship']
                else:
                    # Regex-only version (backward compatible)
                    df_contexts.columns = ['From Citation', 'To Citation', 'Context']
                
                df_contexts.to_excel(writer, sheet_name='Citation_Contexts', index=False)
        
        # Sheet 5: Citation Network Edges
        if 'network' in citation_data and citation_data['network'].get('edges'):
            network = citation_data['network']
            df_edges = pd.DataFrame(network['edges'], columns=['From Citation', 'To Citation'])
            df_edges.to_excel(writer, sheet_name='Citation_Network_Edges', index=False)
        
        # Sheet 6: Citation Network Metrics
        if 'network' in citation_data and 'metrics' in citation_data['network']:
            metrics = citation_data['network']['metrics']
            df_metrics = pd.DataFrame([
                {'Metric': 'Total Citations (Nodes)', 'Value': metrics.get('node_count', 0)},
                {'Metric': 'Citation Links (Edges)', 'Value': metrics.get('edge_count', 0)},
                {'Metric': 'Network Density', 'Value': f"{metrics.get('density', 0):.4f}"},
                {'Metric': 'Average Degree', 'Value': f"{metrics.get('average_degree', 0):.4f}"}
            ])
            df_metrics.to_excel(writer, sheet_name='Citation_Network_Metrics', index=False)
        
        # Sheet 7: Citation Contexts
        if 'network' in citation_data and citation_data['network'].get('citations_in_context'):
            contexts = citation_data['network']['citations_in_context']
            if contexts:
                df_contexts = pd.DataFrame(contexts)
                df_contexts.columns = ['From Citation', 'To Citation', 'Context']
                df_contexts.to_excel(writer, sheet_name='Citation_Contexts', index=False)
    
    def _write_structure_sheets(self, writer: pd.ExcelWriter, structure_data: Dict):
        """Write structure analysis sheets."""
        
        # Sheet 8: Section Scores
        if 'section_scores' in structure_data:
            section_data = []
            for section, score in structure_data['section_scores'].items():
                section_name = section.replace('_score', '').replace('_', ' ').title()
                section_data.append({
                    'Section': section_name,
                    'Present': 'Yes' if score > 0 else 'No',
                    'Score': score
                })
            
            df_sections = pd.DataFrame(section_data)
            df_sections.to_excel(writer, sheet_name='Structure_Sections', index=False)
        
        # Sheet 9: Structure Summary
        summary_data = [
            {'Metric': 'Structure Completeness', 'Value': f"{structure_data.get('structure_completeness', 0):.2%}"},
            {'Metric': 'Number of Figures', 'Value': structure_data.get('num_figures', 0)},
            {'Metric': 'Number of Tables', 'Value': structure_data.get('num_tables', 0)},
            {'Metric': 'Detected Headers', 'Value': len(structure_data.get('detected_headers', []))}
        ]
        df_structure_summary = pd.DataFrame(summary_data)
        df_structure_summary.to_excel(writer, sheet_name='Structure_Summary', index=False)
        
        # Sheet 10: Detected Headers
        if structure_data.get('detected_headers'):
            df_headers = pd.DataFrame({
                'Header Number': range(1, len(structure_data['detected_headers']) + 1),
                'Header Text': structure_data['detected_headers']
            })
            df_headers.to_excel(writer, sheet_name='Detected_Headers', index=False)
        
        # Sheet 11: Figures and Tables List
        content_elements = []
        
        if structure_data.get('figures'):
            for fig in structure_data['figures']:
                content_elements.append({
                    'Type': 'Figure',
                    'Reference': fig
                })
        
        if structure_data.get('tables'):
            for table in structure_data['tables']:
                content_elements.append({
                    'Type': 'Table',
                    'Reference': f"Table {table}"
                })
        
        if content_elements:
            df_content = pd.DataFrame(content_elements)
            df_content.to_excel(writer, sheet_name='Figures_Tables_List', index=False)
    
    def _write_model_sheets(self, writer: pd.ExcelWriter, model_results: Dict):
        """Write model comparison sheets."""
        
        # Sheet 12: Model Performance Summary
        performance_data = []
        for model_name, metrics in model_results.items():
            if isinstance(metrics, dict) and 'avg_response_time' in metrics:
                performance_data.append({
                    'Model': model_name,
                    'Avg Response Time (s)': f"{metrics.get('avg_response_time', 0):.3f}",
                    'Std Response Time (s)': f"{metrics.get('std_response_time', 0):.3f}",
                    'Avg Token Count': f"{metrics.get('avg_token_count', 0):.1f}",
                    'Std Token Count': f"{metrics.get('std_token_count', 0):.1f}",
                    'Consistency Score': f"{metrics.get('consistency_score', 0):.4f}",
                    'Success Rate': f"{metrics.get('success_rate', 0):.2%}",
                    'Error Rate': f"{metrics.get('error_rate', 0):.2%}"
                })
        
        if performance_data:
            df_performance = pd.DataFrame(performance_data)
            df_performance.to_excel(writer, sheet_name='Model_Performance', index=False)
        
        # Sheet 13: Response Times (detailed)
        response_times_data = []
        for model_name, metrics in model_results.items():
            if isinstance(metrics, dict) and 'performance_history' in metrics:
                for i, time_val in enumerate(metrics['performance_history'].get('response_times', [])):
                    response_times_data.append({
                        'Model': model_name,
                        'Trial': i + 1,
                        'Response Time (s)': time_val
                    })
        
        if response_times_data:
            df_response_times = pd.DataFrame(response_times_data)
            df_response_times.to_excel(writer, sheet_name='Model_Response_Times', index=False)
        
        # Sheet 14: Token Counts (detailed)
        token_data = []
        for model_name, metrics in model_results.items():
            if isinstance(metrics, dict) and 'performance_history' in metrics:
                for i, token_count in enumerate(metrics['performance_history'].get('token_counts', [])):
                    token_data.append({
                        'Model': model_name,
                        'Trial': i + 1,
                        'Token Count': token_count
                    })
        
        if token_data:
            df_tokens = pd.DataFrame(token_data)
            df_tokens.to_excel(writer, sheet_name='Model_Token_Counts', index=False)
        
        # Sheet 15: Memory Usage
        memory_data = []
        for model_name, metrics in model_results.items():
            if isinstance(metrics, dict) and 'performance_history' in metrics:
                for i, mem_usage in enumerate(metrics['performance_history'].get('memory_usage', [])):
                    memory_data.append({
                        'Model': model_name,
                        'Trial': i + 1,
                        'Memory Usage (MB)': mem_usage
                    })
        
        if memory_data:
            df_memory = pd.DataFrame(memory_data)
            df_memory.to_excel(writer, sheet_name='Model_Memory_Usage', index=False)
    
    def _write_performance_sheets(self, writer: pd.ExcelWriter, performance_results: Dict):
        """Write performance analysis sheets."""
        
        # Sheet 16: Resource Usage
        resource_data = []
        for model_name, metrics in performance_results.items():
            if 'resource_usage' in metrics:
                resource = metrics['resource_usage']
                resource_data.append({
                    'Model': model_name,
                    'Avg CPU (%)': f"{resource.get('avg_cpu_percent', 0):.2f}",
                    'Peak Memory (MB)': f"{resource.get('peak_memory', 0):.2f}",
                    'Avg Thread Count': f"{resource.get('avg_thread_count', 0):.1f}",
                    'Avg GPU Usage (%)': f"{resource.get('avg_gpu_usage', 0):.2f}" if resource.get('avg_gpu_usage') else 'N/A'
                })
        
        if resource_data:
            df_resource = pd.DataFrame(resource_data)
            df_resource.to_excel(writer, sheet_name='Resource_Usage', index=False)
        
        # Sheet 17: Quality Metrics
        quality_data = []
        for model_name, metrics in performance_results.items():
            if 'quality_metrics' in metrics:
                quality = metrics['quality_metrics']
                quality_data.append({
                    'Model': model_name,
                    'Avg Perplexity': f"{quality.get('avg_perplexity', 0):.2f}",
                    'Avg N-gram Diversity': f"{quality.get('avg_ngram_diversity', 0):.4f}",
                    'Avg Coherence': f"{quality.get('avg_coherence', 0):.4f}"
                })
        
        if quality_data:
            df_quality = pd.DataFrame(quality_data)
            df_quality.to_excel(writer, sheet_name='Quality_Metrics', index=False)
        
        # Sheet 18: Response Metrics (BLEU, ROUGE, etc.)
        response_metrics_data = []
        for model_name, metrics in performance_results.items():
            if 'response_metrics' in metrics:
                response = metrics['response_metrics']
                response_metrics_data.append({
                    'Model': model_name,
                    'BLEU Score': f"{response.get('avg_bleu', 0):.4f}",
                    'METEOR Score': f"{response.get('avg_meteor', 0):.4f}",
                    'ROUGE-1': f"{response.get('avg_rouge1', 0):.4f}",
                    'ROUGE-2': f"{response.get('avg_rouge2', 0):.4f}",
                    'ROUGE-L': f"{response.get('avg_rougeL', 0):.4f}",
                    'Factual Consistency': f"{response.get('avg_factual_consistency', 0):.4f}"
                })
        
        if response_metrics_data:
            df_response_metrics = pd.DataFrame(response_metrics_data)
            df_response_metrics.to_excel(writer, sheet_name='Response_Metrics', index=False)
    
    def get_sheet_summary(self) -> Dict[str, str]:
        """Returns a summary of all sheets in the Excel file."""
        return {
            'Metadata': 'Export information and paper metadata',
            'Citation_Summary': 'Summary of citation counts by type',
            'Citation_Years': 'Citation distribution by year',
            'Citation_Network_Nodes': 'All citations found (nodes)',
            'Citation_Network_Edges': 'Citation relationships (edges)',
            'Citation_Network_Metrics': 'Network statistics',
            'Citation_Contexts': 'Context around citation pairs',
            'Structure_Sections': 'Paper sections detected',
            'Structure_Summary': 'Overall structure metrics',
            'Detected_Headers': 'All headers found in paper',
            'Figures_Tables_List': 'List of figures and tables',
            'Model_Performance': 'Model comparison summary',
            'Model_Response_Times': 'Detailed response times per trial',
            'Model_Token_Counts': 'Token counts per trial',
            'Model_Memory_Usage': 'Memory usage per trial',
            'Resource_Usage': 'CPU, GPU, memory statistics',
            'Quality_Metrics': 'Perplexity, diversity, coherence',
            'Response_Metrics': 'BLEU, ROUGE, factual consistency'
        }