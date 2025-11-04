# citation_analyzer_semantic_enhanced.py

import re
import networkx as nx
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np

class SemanticCitationAnalyzer:
    """
    Advanced citation analyzer that finds relationships between research papers
    based on semantic similarity and co-citation patterns.

    Enhanced Features:
    - Extracts citation context (surrounding text)
    - Uses sentence embeddings to find semantically similar citations
    - Builds network showing which papers discuss similar topics
    - Co-citation analysis (papers cited together = related)
    - **NEW: Citation Sentiment/Stance Detection** (supporting, refuting, neutral, contrasting)
    - **NEW: Citation Purpose Classification** (background, methodology, comparison, etc.)
    - **NEW: Citations in context output** (similar to NLP version)
    """

    def __init__(self, use_embeddings: bool = True):
        """
        Initialize the semantic citation analyzer.

        Args:
            use_embeddings: If True, downloads and uses sentence-transformers
                          for semantic similarity. If False, uses co-citation only.
        """
        self.citation_patterns = {
            "doi": r'(?:https?://)?(?:dx\.)?doi\.org/([^\s]+)',
            "arxiv": r'arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)',
            "url": r'https?://[^\s<>"]+|www\.[^\s<>"]+',
            "numbered": r'\[(\d+)\]',
            "author_year": r'\(([A-Z][a-z]+(?: (?:and |& )?[A-Z][a-z]+)?(?: et al\.)?),?\s+(\d{4})\)',
            "harvard": r'([A-Z][a-z]+)\s+\((\d{4})\)',
        }

        # Sentiment/Stance Detection Signals
        self.stance_signals = {
            'supporting': [
                'confirms', 'supports', 'validates', 'verifies', 'corroborates',
                'agrees with', 'consistent with', 'aligns with', 'demonstrates',
                'proves', 'shows', 'establishes', 'reinforces', 'substantiates'
            ],
            'refuting': [
                'contradicts', 'refutes', 'challenges', 'disputes', 'opposes',
                'questions', 'undermines', 'disproves', 'conflicts with',
                'contrary to', 'disagrees', 'fails to', 'unable to'
            ],
            'contrasting': [
                'however', 'although', 'whereas', 'while', 'but', 'nevertheless',
                'in contrast', 'on the other hand', 'conversely', 'unlike',
                'different from', 'differs from', 'alternatively'
            ],
            'neutral': [
                'reports', 'describes', 'discusses', 'presents', 'reviews',
                'examines', 'investigates', 'analyzes', 'studies', 'explores'
            ]
        }

        # Citation Purpose Classification Signals
        self.purpose_signals = {
            'background': [
                'background', 'previously', 'earlier work', 'established',
                'well-known', 'traditional', 'historical', 'foundational',
                'seminal', 'pioneering', 'literature review', 'prior research'
            ],
            'methodology': [
                'method', 'approach', 'technique', 'algorithm', 'procedure',
                'protocol', 'framework', 'model', 'implementation', 'using',
                'adopted', 'applied', 'following', 'based on', 'adapted from'
            ],
            'comparison': [
                'compared to', 'comparison with', 'versus', 'vs', 'relative to',
                'in contrast to', 'unlike', 'similar to', 'outperforms',
                'better than', 'worse than', 'benchmark', 'baseline'
            ],
            'theory': [
                'theory', 'framework', 'hypothesis', 'principle', 'concept',
                'paradigm', 'model', 'theorem', 'proposition', 'assumption',
                'postulates', 'argues', 'suggests', 'proposes'
            ],
            'results': [
                'results', 'findings', 'outcome', 'showed', 'demonstrated',
                'found', 'observed', 'reported', 'indicated', 'revealed',
                'discovered', 'achieved', 'obtained'
            ],
            'data': [
                'dataset', 'data', 'corpus', 'collected', 'gathered',
                'obtained from', 'sourced from', 'available at', 'downloaded',
                'provided by', 'repository'
            ]
        }

        self.use_embeddings = use_embeddings
        self.embedding_model = None

        if use_embeddings:
            self._load_embedding_model()

    def _load_embedding_model(self):
        """Lazy load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading sentence embedding model (one-time download ~80MB)...")
            # Using a lightweight but effective model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Embedding model loaded successfully")
        except ImportError:
            print("⚠ sentence-transformers not installed. Run: pip install sentence-transformers")
            print("  Falling back to co-citation analysis only.")
            self.use_embeddings = False
        except Exception as e:
            print(f"⚠ Error loading embedding model: {e}")
            print("  Falling back to co-citation analysis only.")
            self.use_embeddings = False

    def _clean_context_text(self, text: str, max_length: int = 600) -> str:
        """
        Clean and format context text for better readability.
        
        Args:
            text: Raw context text
            max_length: Maximum length of context to return
            
        Returns:
            Cleaned, readable text
        """
        if not text:
            return "No context available"
        
        # Remove excessive whitespace and normalize spacing
        text = ' '.join(text.split())
        
        # Fix common PDF extraction issues
        text = text.replace('- ', '')   # Remove hyphenation
        text = text.replace('\ufeff', '')  # Remove BOM
        text = text.replace('\u200b', '')  # Remove zero-width space
        
        # Limit length
        if len(text) > max_length:
            text = text[:max_length]
            
            # Try to end at a complete sentence
            last_sentence_end = max(
                text.rfind('.'),
                text.rfind('?'),
                text.rfind('!')
            )
            
            # If we found a sentence ending and it's not too short, cut there
            if last_sentence_end > len(text) * 0.6:  # At least 60% through
                text = text[:last_sentence_end + 1]
            else:
                # Try to end at a word boundary
                last_space = text.rfind(' ')
                if last_space > len(text) * 0.8:  # At least 80% through
                    text = text[:last_space] + '...'
                else:
                    text = text + '...'
        
        return text.strip()

    def detect_citation_stance(self, context: str) -> Dict[str, any]:
        """
        Detect the stance/sentiment of how a citation is used.

        Args:
            context: Text surrounding the citation

        Returns:
            Dict with 'stance' (supporting/refuting/contrasting/neutral) and 'confidence'
        """
        context_lower = context.lower()
        
        # Count signals for each stance
        stance_scores = {}
        for stance, signals in self.stance_signals.items():
            score = sum(1 for signal in signals if signal in context_lower)
            stance_scores[stance] = score
        
        # Get the dominant stance
        max_score = max(stance_scores.values())
        
        if max_score == 0:
            return {'stance': 'neutral', 'confidence': 'low', 'score': 0}
        
        dominant_stance = max(stance_scores, key=stance_scores.get)
        
        # Calculate confidence based on score
        if max_score >= 3:
            confidence = 'high'
        elif max_score >= 2:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'stance': dominant_stance,
            'confidence': confidence,
            'score': max_score,
            'all_scores': stance_scores
        }

    def classify_citation_purpose(self, context: str) -> Dict[str, any]:
        """
        Classify the purpose of a citation.

        Args:
            context: Text surrounding the citation

        Returns:
            Dict with 'purpose' (background/methodology/comparison/etc) and 'confidence'
        """
        context_lower = context.lower()
        
        # Count signals for each purpose
        purpose_scores = {}
        for purpose, signals in self.purpose_signals.items():
            score = sum(1 for signal in signals if signal in context_lower)
            purpose_scores[purpose] = score
        
        # Get the dominant purpose
        max_score = max(purpose_scores.values())
        
        if max_score == 0:
            return {'purpose': 'general', 'confidence': 'low', 'score': 0}
        
        dominant_purpose = max(purpose_scores, key=purpose_scores.get)
        
        # Calculate confidence
        if max_score >= 3:
            confidence = 'high'
        elif max_score >= 2:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'purpose': dominant_purpose,
            'confidence': confidence,
            'score': max_score,
            'all_scores': purpose_scores
        }

    def extract_citations_with_context(self, text: str, context_window: int = 600) -> List[Dict]:
        """
        Extract all citations with their surrounding context, stance, and purpose.

        Args:
            text: Full paper text
            context_window: Characters before/after citation to capture

        Returns:
            List of dicts with keys: citation_id, citation_text, context, stance, purpose, position
        """
        citations = []
        citation_id = 0

        # Split into paragraphs for better context
        paragraphs = text.split('\n\n')
        char_offset = 0

        for para_idx, paragraph in enumerate(paragraphs):
            # Find all citation matches in this paragraph
            for pattern_name, pattern in self.citation_patterns.items():
                for match in re.finditer(pattern, paragraph):
                    citation_text = match.group(0)
                    match_start = match.start()
                    match_end = match.end()

                    # Extract context around citation
                    context_start = max(0, match_start - context_window)
                    context_end = min(len(paragraph), match_end + context_window)
                    raw_context = paragraph[context_start:context_end].strip()
                    
                    # Clean context for better readability
                    context = self._clean_context_text(raw_context, max_length=600)

                    # Normalize citation for matching
                    if pattern_name == "numbered":
                        normalized_id = f"[{match.group(1)}]"
                    elif pattern_name in ["author_year", "harvard"]:
                        normalized_id = f"{match.group(1)} ({match.group(2)})"
                    else:
                        normalized_id = citation_text

                    # Detect stance and purpose
                    stance_info = self.detect_citation_stance(context)
                    purpose_info = self.classify_citation_purpose(context)

                    citations.append({
                        'id': citation_id,
                        'citation_text': citation_text,
                        'normalized_id': normalized_id,
                        'context': context,
                        'paragraph_idx': para_idx,
                        'position': char_offset + match_start,
                        'pattern_type': pattern_name,
                        'stance': stance_info['stance'],
                        'stance_confidence': stance_info['confidence'],
                        'purpose': purpose_info['purpose'],
                        'purpose_confidence': purpose_info['confidence']
                    })
                    citation_id += 1

            char_offset += len(paragraph) + 2  # +2 for \n\n

        return citations

    def build_cocitation_network(self, citations: List[Dict]) -> nx.Graph:
        """
        Build network based on co-citation: citations appearing in same paragraph = related.

        Args:
            citations: List of citation dicts from extract_citations_with_context

        Returns:
            NetworkX graph where edges represent co-citation relationships
        """
        G = nx.Graph()

        # Group citations by paragraph
        citations_by_paragraph = defaultdict(list)
        for cite in citations:
            citations_by_paragraph[cite['paragraph_idx']].append(cite)

        # Create edges for citations in same paragraph
        for para_idx, para_citations in citations_by_paragraph.items():
            # Get unique normalized IDs in this paragraph
            unique_cites = {}
            for cite in para_citations:
                norm_id = cite['normalized_id']
                if norm_id not in unique_cites:
                    unique_cites[norm_id] = cite
                else:
                    # Keep the citation with more context
                    if len(cite['context']) > len(unique_cites[norm_id]['context']):
                        unique_cites[norm_id] = cite

            # Add nodes with enhanced attributes
            for norm_id, cite in unique_cites.items():
                if not G.has_node(norm_id):
                    G.add_node(norm_id,
                              citation_text=cite['citation_text'],
                              pattern_type=cite['pattern_type'],
                              contexts=[cite['context']],
                              stances=[cite['stance']],
                              purposes=[cite['purpose']])
                else:
                    # Accumulate contexts, stances, and purposes
                    G.nodes[norm_id]['contexts'].append(cite['context'])
                    G.nodes[norm_id]['stances'].append(cite['stance'])
                    G.nodes[norm_id]['purposes'].append(cite['purpose'])

            # Create edges between co-cited papers
            cite_list = list(unique_cites.keys())
            for i in range(len(cite_list)):
                for j in range(i + 1, len(cite_list)):
                    cite_a = cite_list[i]
                    cite_b = cite_list[j]

                    if G.has_edge(cite_a, cite_b):
                        # Increment weight for repeated co-citation
                        G[cite_a][cite_b]['weight'] += 1
                        G[cite_a][cite_b]['shared_paragraphs'].append(para_idx)
                    else:
                        G.add_edge(cite_a, cite_b,
                                 weight=1,
                                 shared_paragraphs=[para_idx],
                                 relationship='co-cited')

        return G

    def add_semantic_similarity_edges(self, G: nx.Graph, similarity_threshold: float = 0.5):
        """
        Add edges between citations with semantically similar contexts.

        Args:
            G: NetworkX graph with nodes containing 'contexts' attribute
            similarity_threshold: Minimum cosine similarity to create edge (0-1)
        """
        if not self.use_embeddings or self.embedding_model is None:
            print("⚠ Semantic similarity not available (embeddings disabled)")
            return G

        nodes_list = list(G.nodes())

        # Compute embeddings for each citation's contexts
        node_embeddings = {}
        for node in nodes_list:
            contexts = G.nodes[node].get('contexts', [])
            if contexts:
                # Average embedding of all contexts for this citation
                context_texts = [ctx[:500] for ctx in contexts]  # Limit length
                embeddings = self.embedding_model.encode(context_texts)
                node_embeddings[node] = np.mean(embeddings, axis=0)

        # Compute pairwise similarities
        added_edges = 0
        for i, node_a in enumerate(nodes_list):
            if node_a not in node_embeddings:
                continue

            for node_b in nodes_list[i+1:]:
                if node_b not in node_embeddings:
                    continue

                # Skip if already co-cited
                if G.has_edge(node_a, node_b):
                    continue

                # Compute cosine similarity
                emb_a = node_embeddings[node_a]
                emb_b = node_embeddings[node_b]
                similarity = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))

                if similarity >= similarity_threshold:
                    G.add_edge(node_a, node_b,
                             weight=similarity,
                             relationship='semantic_similarity',
                             similarity_score=float(similarity))
                    added_edges += 1

        print(f"✓ Added {added_edges} semantic similarity edges")
        return G

    def build_citations_in_context(self, citations: List[Dict]) -> List[Dict]:
        """
        Build citations_in_context list similar to NLP version.

        Args:
            citations: List of citation dicts

        Returns:
            List of citation relationships with context
        """
        citations_in_context = []
        
        # Group by paragraph to find co-citations
        citations_by_paragraph = defaultdict(list)
        for cite in citations:
            citations_by_paragraph[cite['paragraph_idx']].append(cite)
        
        # Build relationships
        for para_idx, para_citations in citations_by_paragraph.items():
            # Get unique citations in this paragraph
            unique_cites = {}
            for cite in para_citations:
                norm_id = cite['normalized_id']
                if norm_id not in unique_cites:
                    unique_cites[norm_id] = cite
            
            # Create pairwise relationships
            cite_list = list(unique_cites.values())
            for i in range(len(cite_list)):
                for j in range(i + 1, len(cite_list)):
                    cite_a = cite_list[i]
                    cite_b = cite_list[j]
                    
                    # Determine relationship based on stances
                    if cite_a['stance'] == cite_b['stance']:
                        if cite_a['stance'] == 'supporting':
                            relationship = 'supports'
                        elif cite_a['stance'] == 'contrasting':
                            relationship = 'contrasts'
                        else:
                            relationship = 'parallel'
                    elif 'refuting' in [cite_a['stance'], cite_b['stance']]:
                        relationship = 'conflicts'
                    else:
                        relationship = 'sequential'
                    
                    # Use the longer context
                    context = cite_a['context'] if len(cite_a['context']) > len(cite_b['context']) else cite_b['context']
                    
                    citations_in_context.append({
                        'from': cite_a['normalized_id'],
                        'to': cite_b['normalized_id'],
                        'relationship': relationship,
                        'context': context,
                        'paragraph_idx': para_idx,
                        'from_stance': cite_a['stance'],
                        'to_stance': cite_b['stance'],
                        'from_purpose': cite_a['purpose'],
                        'to_purpose': cite_b['purpose']
                    })
        
        return citations_in_context

    def extract_citations(self, text: str, semantic_threshold: float = 0.5) -> Dict:
        """
        Main method: Extract citations and build semantic network with enhanced analysis.

        Args:
            text: Full research paper text
            semantic_threshold: Minimum similarity to connect citations (0-1)

        Returns:
            Dict with citation counts, network data, stance analysis, purpose analysis, and insights
        """
        # Extract all citations with context, stance, and purpose
        print("Extracting citations with context, stance, and purpose...")
        citations = self.extract_citations_with_context(text)

        # Count citation types
        citation_counts = defaultdict(int)
        for cite in citations:
            citation_counts[f"{cite['pattern_type']}_count"] += 1

        # Count stances and purposes
        stance_distribution = defaultdict(int)
        purpose_distribution = defaultdict(int)
        for cite in citations:
            stance_distribution[cite['stance']] += 1
            purpose_distribution[cite['purpose']] += 1

        # Get unique citations
        unique_citations = {}
        for cite in citations:
            norm_id = cite['normalized_id']
            if norm_id not in unique_citations:
                unique_citations[norm_id] = cite

        total_unique = len(unique_citations)
        citation_counts['total_count'] = len(citations)
        citation_counts['unique_count'] = total_unique

        # Build co-citation network
        print(f"Building citation network from {total_unique} unique citations...")
        G = self.build_cocitation_network(citations)

        # Add semantic similarity edges
        if self.use_embeddings:
            G = self.add_semantic_similarity_edges(G, semantic_threshold)

        # Build citations_in_context
        print("Building citation relationships...")
        citations_in_context = self.build_citations_in_context(citations)

        # Calculate network metrics
        network_metrics = {
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges(),
            'density': nx.density(G) if G.number_of_nodes() > 0 else 0,
            'average_degree': sum(dict(G.degree()).values()) / max(1, G.number_of_nodes())
        }

        # Find citation clusters (communities)
        communities = []
        if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
            try:
                from networkx.algorithms import community
                communities_generator = community.greedy_modularity_communities(G)
                communities = [list(comm) for comm in communities_generator]
                network_metrics['num_communities'] = len(communities)
            except:
                network_metrics['num_communities'] = 0

        # Find most influential citations (highest degree)
        if G.number_of_nodes() > 0:
            degree_dict = dict(G.degree())
            top_citations = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            network_metrics['top_citations'] = [
                {'citation': cite, 'connections': deg} for cite, deg in top_citations
            ]

        # Extract year distribution
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        years = [int(y) for y in years if 1900 <= int(y) <= 2025]
        year_distribution = {year: years.count(year) for year in set(years)}

        # Build a context and relationship lookup for edges
        edge_details_map = {}
        for ctx in citations_in_context:
            key = (ctx['from'], ctx['to'])
            if key not in edge_details_map:
                edge_details_map[key] = {
                    'context': ctx['context'],
                    'from_stance': ctx['from_stance'],
                    'to_stance': ctx['to_stance'],
                    'from_purpose': ctx['from_purpose'],
                    'to_purpose': ctx['to_purpose']
                }
            # Also add reverse direction
            reverse_key = (ctx['to'], ctx['from'])
            if reverse_key not in edge_details_map:
                edge_details_map[reverse_key] = {
                    'context': ctx['context'],
                    'from_stance': ctx['to_stance'],  # Swapped
                    'to_stance': ctx['from_stance'],  # Swapped
                    'from_purpose': ctx['to_purpose'],  # Swapped
                    'to_purpose': ctx['from_purpose']  # Swapped
                }

        # Prepare network data for visualization
        network_data = {
            'nodes': [
                {
                    'id': node,
                    'label': node,
                    'degree': G.degree(node),
                    'pattern_type': G.nodes[node].get('pattern_type', 'unknown'),
                    'contexts': G.nodes[node].get('contexts', [])[:3],  # First 3 contexts
                    'num_contexts': len(G.nodes[node].get('contexts', [])),
                    'dominant_stance': max(set(G.nodes[node].get('stances', ['neutral'])), 
                                          key=G.nodes[node].get('stances', ['neutral']).count),
                    'dominant_purpose': max(set(G.nodes[node].get('purposes', ['general'])), 
                                           key=G.nodes[node].get('purposes', ['general']).count)
                }
                for node in G.nodes()
            ],
            'edges': [
                {
                    'source': u,
                    'target': v,
                    'weight': G[u][v].get('weight', 1),
                    'relationship': G[u][v].get('relationship', 'unknown'),
                    'similarity': G[u][v].get('similarity_score', None),
                    'context': edge_details_map.get((u, v), {}).get('context', 'No context available'),  # Already cleaned in extraction
                    'source_stance': edge_details_map.get((u, v), {}).get('from_stance', 'unknown'),
                    'target_stance': edge_details_map.get((u, v), {}).get('to_stance', 'unknown'),
                    'source_purpose': edge_details_map.get((u, v), {}).get('from_purpose', 'unknown'),
                    'target_purpose': edge_details_map.get((u, v), {}).get('to_purpose', 'unknown')
                }
                for u, v in G.edges()
            ],
            'metrics': network_metrics,
            'communities': communities,
            'citations_in_context': citations_in_context
        }

        # Create detailed citation table
        citation_details = []
        for cite in citations:
            citation_details.append({
                'citation': cite['normalized_id'],
                'type': cite['pattern_type'],
                'stance': cite['stance'],
                'stance_confidence': cite['stance_confidence'],
                'purpose': cite['purpose'],
                'purpose_confidence': cite['purpose_confidence'],
                'context': cite['context'],  # Already cleaned during extraction
                'position': cite['position'],
                'paragraph_idx': cite['paragraph_idx']
            })

        result = {
            **citation_counts,
            'year_distribution': year_distribution,
            'stance_distribution': dict(stance_distribution),
            'purpose_distribution': dict(purpose_distribution),
            'network': network_data,
            'citation_details': citation_details,
            'method': 'semantic_similarity' if self.use_embeddings else 'co-citation_only'
        }

        return result

    def get_citation_insights(self, network_data: Dict) -> str:
        """
        Generate human-readable insights from the citation network.

        Args:
            network_data: The 'network' dict from extract_citations()

        Returns:
            String with formatted insights
        """
        insights = []
        metrics = network_data['metrics']

        insights.append(f"📊 Network Overview:")
        insights.append(f"  • {metrics['node_count']} unique citations found")
        insights.append(f"  • {metrics['edge_count']} relationships identified")
        insights.append(f"  • Network density: {metrics['density']:.2%}")
        insights.append(f"  • Average connections per citation: {metrics['average_degree']:.1f}")

        if metrics.get('num_communities', 0) > 0:
            insights.append(f"\n🔍 Topic Clusters:")
            insights.append(f"  • Found {metrics['num_communities']} research topic clusters")
            insights.append(f"  • Citations grouped by common themes/co-occurrence")

        if 'top_citations' in metrics and metrics['top_citations']:
            insights.append(f"\n⭐ Most Influential Citations:")
            for cite_info in metrics['top_citations'][:3]:
                insights.append(f"  • {cite_info['citation']}: {cite_info['connections']} connections")

        # Add citation relationships preview
        if 'citations_in_context' in network_data and network_data['citations_in_context']:
            insights.append(f"\n📑 Sample Citation Relationships:")
            for ctx in network_data['citations_in_context'][:3]:
                insights.append(f"  • {ctx['from']} → {ctx['to']}")
                insights.append(f"    Relationship: {ctx['relationship']}")
                insights.append(f"    Stances: {ctx['from_stance']} ↔ {ctx['to_stance']}")

        return "\n".join(insights)

    def get_stance_purpose_summary(self, result: Dict) -> str:
        """
        Generate summary of stance and purpose distributions.

        Args:
            result: The full result dict from extract_citations()

        Returns:
            String with formatted summary
        """
        summary = []
        
        summary.append("📈 Citation Stance Distribution:")
        stance_dist = result.get('stance_distribution', {})
        total_stances = sum(stance_dist.values())
        for stance, count in sorted(stance_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_stances * 100) if total_stances > 0 else 0
            summary.append(f"  • {stance.capitalize()}: {count} ({percentage:.1f}%)")
        
        summary.append("\n🎯 Citation Purpose Distribution:")
        purpose_dist = result.get('purpose_distribution', {})
        total_purposes = sum(purpose_dist.values())
        for purpose, count in sorted(purpose_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_purposes * 100) if total_purposes > 0 else 0
            summary.append(f"  • {purpose.capitalize()}: {count} ({percentage:.1f}%)")
        
        return "\n".join(summary)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    sample_text = """
    Recent work by Smith et al. (2020) has shown promising results in machine learning.
    Building on this methodology, Jones (2021) extended the approach with deep learning.
    However, Brown et al. (2022) contradicts these findings and questions the validity.
    Similar methods were used in previous research [1] and [2].
    The background literature demonstrates that Miller (2019) established the theoretical framework.
    For comparison, our results outperform the baseline presented in Davis (2023).
    We adopted the data collection protocol from Wilson et al. (2018).
    """
    
    print("=" * 70)
    print("🚀 Enhanced Semantic Citation Analyzer")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = SemanticCitationAnalyzer(use_embeddings=True)
    
    # Extract citations
    print("\n📊 Analyzing citations...")
    results = analyzer.extract_citations(sample_text)
    
    # Display results
    print(f"\n✓ Total citations found: {results['total_count']}")
    print(f"✓ Unique citations: {results['unique_count']}")
    print(f"✓ Analysis method: {results['method']}")
    
    # Display stance and purpose summary
    print(f"\n{analyzer.get_stance_purpose_summary(results)}")
    
    # Display network insights
    print(f"\n{analyzer.get_citation_insights(results['network'])}")
    
    # Display sample citation details
    print("\n📋 Sample Citation Details:")
    for cite in results['citation_details'][:3]:
        print(f"\n  Citation: {cite['citation']}")
        print(f"  Stance: {cite['stance']} (confidence: {cite['stance_confidence']})")
        print(f"  Purpose: {cite['purpose']} (confidence: {cite['purpose_confidence']})")
        print(f"  Context: {cite['context'][:80]}...")
    
    print("\n" + "=" * 70)
    print("✅ Analysis complete!")
    print("=" * 70)