# citation_analyzer.py

import re
import networkx as nx

class EnhancedCitationAnalyzer:
    """Enhanced analyzer for citations and references."""
    
    def __init__(self):
        self.citation_patterns = {
            "doi": r'(?:https?://)?(?:dx\.)?doi\.org/([^\s]+)',
            "arxiv": r'arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)',
            "url": r'https?://[^\s<>"]+|www\.[^\s<>"]+',
            "numbered": r'\[\d+\]',
            "author_year": r'\([A-Za-z]+(?: and [A-Za-z]+)?(?: et al\.)?,?\s+\d{4}\)',
            "harvard": r'[A-Z][a-z]+\s+\(\d{4}\)',
            "inline": r'[A-Z][a-z]+ \d{4}'
        }
    
    def extract_citations(self, text: str) -> dict:
        citations = {}
        for citation_type, pattern in self.citation_patterns.items():
            matches = re.findall(pattern, text)
            citations[f"{citation_type}_count"] = len(matches)
            citations[f"{citation_type}_examples"] = matches[:3]
        citations["total_count"] = sum(
            citations[k] for k in citations.keys() if k.endswith('_count')
        )
        years = re.findall(r'\d{4}', text)
        years = [int(y) for y in years if 1900 <= int(y) <= 2024]
        citations["year_distribution"] = {year: years.count(year) for year in set(years)}
        citations["network"] = self.build_citation_network(text)
        return citations
    
    def build_citation_network(self, text: str) -> dict:
        """Builds a citation network with context extraction."""
        G = nx.DiGraph()
        paragraphs = text.split('\n\n')
        citations_in_context = []
        
        for para in paragraphs:
            all_matches = []
            for pattern in self.citation_patterns.values():
                matches = re.finditer(pattern, para)
                all_matches.extend((m.group(), m.start(), m.end()) for m in matches)
            all_matches.sort(key=lambda x: x[1])
            
            # Extract context around citation pairs
            for i in range(len(all_matches) - 1):
                cite1, cite2 = all_matches[i][0], all_matches[i+1][0]
                G.add_edge(cite1, cite2)
                
                # Extract 50 characters before and after for context
                context_start = max(0, all_matches[i][1] - 50)
                context_end = min(len(para), all_matches[i+1][2] + 50)
                
                citations_in_context.append({
                    'from': cite1,
                    'to': cite2,
                    'context': para[context_start:context_end]
                })
        
        network_data = {
            'nodes': list(G.nodes()),
            'edges': list(G.edges()),
            'metrics': {
                'node_count': G.number_of_nodes(),
                'edge_count': G.number_of_edges(),
                'density': nx.density(G) if G.number_of_nodes() > 0 else 0,
                'average_degree': sum(dict(G.degree()).values()) / max(1, G.number_of_nodes())
            },
            'citations_in_context': citations_in_context
        }
        
        return network_data