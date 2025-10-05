# structure_analyzer.py

import re
import plotly.graph_objects as go
from typing import List, Dict, Any

class StructureAnalyzer:
    """Analyzes the structure and organization of research papers."""
    
    def __init__(self):
        self.sections = [
            "abstract", "introduction", "background", "methodology", 
            "methods", "results", "discussion", "conclusion", 
            "references", "acknowledgments", "appendix"
        ]
        
        # Enhanced figure patterns (12+ patterns)
        self.figure_patterns = [
            r'Fig(?:ure)?\.?\s*(\d+[\.\:]?\s*[A-Za-z])',  # Fig. 1: or Figure 1. with text
            r'Fig(?:ure)?\.?\s*(\d+)\s+[A-Z]',  # Fig 1 Description
            r'Fig(?:ure)?\.?\s*(\d+)(?=[^.]*)',  # Any Fig 1 reference
            r'Fig(?:ure)?\.?\s*(\d+)\s*[A-Za-z]',  # Figure 1 text
            r'(?:Figure|Fig\.?)[s\s]*(\d+)[-–](\d+)',  # Figure/Fig 1-2
            r'(?:Figure|Fig\.?)[s\s]*(\d+)\s*(?:and|&)\s*(\d+)',  # Figure/Fig 1 and 2
            r'(?:Figure|Fig\.?)[s\s]*(\d+)\s*,\s*(\d+)',  # Figure/Fig 1, 2
            r'Fig(?:ure)?\.?\s*(\w+)',  # Handles Roman numerals (e.g., Fig. I)
            r'Fig(?:ure)?\.?\s*(\d+[A-Za-z]?)',  # Handles mixed (e.g., Fig. 1A)
            r'Fig(?:ure)?\.?\s*(\d+)\s*[\^]?\d*',  # Handles superscripts
            r'Fig(?:ure)?\.?\s*(\d+)\s*[–-]\s*(\d+)',  # Handles ranges
            r'Fig(?:ure)?\.?\s*(\d+)\s*[\s]*\(?[\dA-Za-z]+\)?'  # Parentheses
        ]
        
        # Enhanced table patterns
        self.section_patterns = {
            "header_pattern": r'^(?:[0-9.]+\s+)?([A-Z][A-Za-z\s]+)$',
            "table_pattern": [
                r'Table\s*(\d+)[:.]?\s*[A-Z]',  # Table 1: Description
                r'Table\s*(\d+)\s+[A-Z]',  # Table 1 Description
                r'Table\s*(\d+)(?=[^.])',  # Any Table 1 reference
                r'Tables?\s*(\d+)[-–](\d+)',  # Table 1-2
                r'Tables?\s*(\d+)\s*(?:and|&)\s*(\d+)',  # Table 1 and 2
                r'Tables?\s*(\d+)\s*,\s*(\d+)',  # Table 1, 2
                r'Tab(?:le)?\.?\s*(\w+)',  # Roman numerals (Tab. I)
                r'Tab(?:le)?\.?\s*(\d+[A-Za-z]?)',  # Mixed (Tab. 1A)
                r'Tab(?:le)?\.?\s*(\d+)\s*[\^]?\d*',  # Superscripts
                r'Tab(?:le)?\.?\s*(\d+)\s*[–-]\s*(\d+)',  # Ranges
                r'Tab(?:le)?\.?\s*(\d+)\s*[\s]*\(?[\dA-Za-z]+\)?'  # Parentheses
            ]
        }
    
    def find_figures(self, text: str) -> List[str]:
        """Find all figure references using comprehensive patterns."""
        figures = []
        for pattern in self.figure_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if match.groups():
                    figures.append(match.group(1).strip())
                else:
                    figures.append(match.group(0).strip())
        return list(set(figures))
    
    def count_figure_references(self, text: str) -> int:
        """Count all unique figure references."""
        all_refs = []
        for pattern in self.figure_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            all_refs.extend(matches)
        
        figure_numbers = set()
        for ref in all_refs:
            numbers = re.findall(r'\d+', str(ref))
            figure_numbers.update(numbers)
        return len(figure_numbers)
    
    def find_tables(self, text: str) -> List[int]:
        """Find all table references using comprehensive patterns."""
        table_numbers = set()
        for pattern in self.section_patterns["table_pattern"]:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                groups = match.groups()
                if groups:
                    for group in groups:
                        if group and str(group).isdigit():
                            table_numbers.add(int(group))
        return sorted(list(table_numbers))
    
    def count_table_references(self, text: str) -> int:
        """Count unique table numbers."""
        return len(self.find_tables(text))
    
    def analyze_structure(self, text: str) -> Dict[str, Any]:
        """Enhanced structure analysis."""
        text_lower = text.lower()
        lines = text.split('\n')
        
        section_scores = {
            f"{section}_score": 1.0 if section in text_lower else 0.0
            for section in self.sections
        }
        
        headers = []
        for line in lines:
            match = re.match(self.section_patterns["header_pattern"], line.strip())
            if match:
                headers.append(match.group(1))
        
        figures = self.find_figures(text)
        num_figures = self.count_figure_references(text)
        tables = self.find_tables(text)
        num_tables = len(tables)
        
        analysis = {
            "section_scores": section_scores,
            "structure_completeness": sum(section_scores.values()) / len(self.sections),
            "detected_headers": headers,
            "num_figures": num_figures,
            "num_tables": num_tables,
            "figures": figures[:5],
            "tables": tables[:5]
        }
        
        return analysis
    
    def create_radar_chart(self, structure_metrics: Dict) -> go.Figure:
        """Creates a radar chart visualization with enhanced styling."""
        scores = [structure_metrics["section_scores"][f"{section}_score"] for section in self.sections]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=scores,
            theta=self.sections,
            fill='toself',
            name='Structure Completeness',
            fillcolor='rgba(135, 206, 250, 0.5)',
            line=dict(color='royalblue')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            title="Paper Structure Radar Chart",
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig