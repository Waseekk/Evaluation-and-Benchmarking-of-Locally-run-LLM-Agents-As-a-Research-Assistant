#!/usr/bin/env python3
"""
Citation Analyzer Comparison Tool - Interactive Streamlit App

Upload a PDF and compare NLP vs Semantic vs Hybrid citation analysis approaches.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import time
from typing import Dict, Any

from pdf_processor import PDFProcessor
from citation_analyzer_nlp import EnhancedCitationAnalyzer as NLPAnalyzer
from citation_analyzer_semantic import SemanticCitationAnalyzer
from citation_analyzer_hybrid import HybridSemanticCitationAnalyzer

# Page configuration
st.set_page_config(
    page_title="Citation Analyzer Comparison",
    page_icon="🔬",
    layout="wide"
)

# Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #F0F2F6;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 5px;
    }
    .comparison-table {
        background: white;
        padding: 20px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)


# ⭐ CRITICAL FIX: Helper function to validate citation references
def is_valid_citation_reference(text: str) -> bool:
    """
    Check if text is a valid citation reference (not a paper title/journal name).
    Filters out entries like "Computers 2025", "Access 2022"
    """
    import re
    text = str(text).strip()
    
    # Valid patterns
    # Pattern 1: Numbered references [1], [2], [1,2], etc.
    if re.match(r'^\[\d+(?:\s*[,;-]\s*\d+)*\]$', text):
        return True
    
    # Pattern 2: DOI
    if 'doi.org' in text.lower():
        return True
    
    # Pattern 3: arXiv
    if text.startswith('arXiv:'):
        return True
    
    # Pattern 4: Author-year (short form only - not long titles)
    if re.match(r'^[A-Z][a-z]+.*\d{4}[a-z]?$', text) and len(text) < 50:
        return True
    
    # Exclude anything that looks like a journal/paper title
    exclude_words = [
        'computers', 'journal', 'proceedings', 'conference', 
        'access', 'review', 'transactions', 'international',
        'science', 'nature', 'ieee', 'acm', 'springer',
        'elsevier', 'wiley', 'communications', 'letters'
    ]
    
    text_lower = text.lower()
    if any(word in text_lower for word in exclude_words):
        return False
    
    return False  # If none of the patterns match, it's not valid



def create_network_visualization(network_data: Dict[str, Any], title: str) -> go.Figure:
    """
    Create a network graph visualization using Plotly.

    Args:
        network_data: Network dictionary with nodes, edges, and metrics
        title: Title for the graph

    Returns:
        Plotly figure object
    """
    # Handle different network formats
    if 'nodes' in network_data and isinstance(network_data['nodes'], list):
        # Semantic analyzer format (list of dicts)
        if network_data['nodes'] and isinstance(network_data['nodes'][0], dict):
            nodes = [node['id'] for node in network_data['nodes']]
            edges = [(edge['source'], edge['target']) for edge in network_data.get('edges', [])]
        else:
            nodes = network_data['nodes']
            edges = network_data.get('edges', [])
    else:
        # NLP analyzer format (simple list)
        nodes = network_data.get('nodes', [])
        edges = network_data.get('edges', [])

    if not nodes:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No citation network data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(title=title)
        return fig

    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(nodes)

    if edges:
        # Handle both tuple and dict formats
        if isinstance(edges[0], tuple):
            G.add_edges_from(edges)
        elif isinstance(edges[0], dict):
            G.add_edges_from([(e['source'], e['target']) for e in edges])

    # Create layout
    try:
        if len(nodes) > 50:
            pos = nx.spring_layout(G, k=0.5, iterations=30)
        else:
            pos = nx.spring_layout(G, k=1, iterations=50)
    except:
        # Fallback if spring layout fails
        pos = {node: (i % 10, i // 10) for i, node in enumerate(nodes)}

    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='Relationships'
    )

    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        # Truncate long citation text
        text = str(node)[:50] + "..." if len(str(node)) > 50 else str(node)
        node_text.append(text)
        # Size based on degree (number of connections)
        degree = G.degree(node)
        node_size.append(10 + degree * 2)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(
            size=node_size,
            color='#1f77b4',
            line_width=2,
            line_color='white'
        ),
        name='Citations'
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title=title,
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )

    return fig


def create_year_distribution_chart(year_dist: Dict[int, int], title: str) -> go.Figure:
    """Create a bar chart of citation year distribution."""
    if not year_dist:
        fig = go.Figure()
        fig.add_annotation(
            text="No year data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title=title)
        return fig

    years = sorted(year_dist.keys())
    counts = [year_dist[year] for year in years]

    fig = go.Figure(data=[
        go.Bar(x=years, y=counts, marker_color='#3B82F6')
    ])

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Number of Citations",
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=400
    )

    return fig


def create_stance_distribution_chart(stance_dist: Dict[str, int], title: str) -> go.Figure:
    """Create a pie chart of citation stance distribution."""
    if not stance_dist:
        fig = go.Figure()
        fig.add_annotation(
            text="No stance data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title=title)
        return fig

    labels = list(stance_dist.keys())
    values = list(stance_dist.values())
    
    colors = {
        'supporting': '#10B981',
        'refuting': '#EF4444',
        'contrasting': '#F59E0B',
        'neutral': '#6B7280'
    }
    
    color_list = [colors.get(label, '#3B82F6') for label in labels]

    fig = go.Figure(data=[
        go.Pie(labels=labels, values=values, marker=dict(colors=color_list))
    ])

    fig.update_layout(
        title=title,
        paper_bgcolor='white',
        height=400
    )

    return fig


def create_purpose_distribution_chart(purpose_dist: Dict[str, int], title: str) -> go.Figure:
    """Create a horizontal bar chart of citation purpose distribution."""
    if not purpose_dist:
        fig = go.Figure()
        fig.add_annotation(
            text="No purpose data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title=title)
        return fig

    purposes = list(purpose_dist.keys())
    counts = list(purpose_dist.values())

    fig = go.Figure(data=[
        go.Bar(y=purposes, x=counts, orientation='h', marker_color='#8B5CF6')
    ])

    fig.update_layout(
        title=title,
        xaxis_title="Number of Citations",
        yaxis_title="Purpose",
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=400
    )

    return fig


def display_hybrid_results(results: Dict[str, Any], paper_text: str):
    """Display Hybrid Semantic analyzer results."""
    st.header("🚀 Hybrid Semantic Citation Analyzer Results")
    
    st.success("✨ **Best of Both Worlds**: Combines semantic similarity with co-occurrence detection!")

    # Main Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Citations", results['total_count'])
    with col2:
        st.metric("Unique Citations", results['unique_count'])
    with col3:
        st.metric("Network Nodes", results['network']['metrics']['node_count'])
    with col4:
        st.metric("Network Edges", results['network']['metrics']['edge_count'])

    # Citation Types
    st.subheader("📚 Citation Types Breakdown")
    type_cols = st.columns(4)
    
    with type_cols[0]:
        st.metric("Numbered", results.get('numbered_count', 0))
    with type_cols[1]:
        st.metric("Author-Year", results.get('author_year_count', 0))
    with type_cols[2]:
        st.metric("Harvard", results.get('harvard_count', 0))
    with type_cols[3]:
        st.metric("DOI/arXiv", results.get('doi_count', 0) + results.get('arxiv_count', 0))

    # Features
    st.subheader("✨ Hybrid Features Enabled")
    feature_cols = st.columns(4)
    
    with feature_cols[0]:
        if results['features']['embeddings']:
            st.success("✅ **Semantic Similarity**\nEmbedding-based connections")
        else:
            st.warning("⚠️ **Semantic Similarity**\nDisabled")
    
    with feature_cols[1]:
        if results['features']['spacy']:
            st.success("✅ **spaCy NLP**\nAccurate sentence detection")
        else:
            st.warning("⚠️ **spaCy NLP**\nFallback to regex")
    
    with feature_cols[2]:
        st.success("✅ **Co-occurrence**\nSame sentence/paragraph")
    
    with feature_cols[3]:
        st.success("✅ **Year Distribution**\nWith validation")

    # Performance
    st.subheader("⏱️ Performance Breakdown")
    timing_cols = st.columns(len(results['timing']))
    for idx, (stage, duration) in enumerate(results['timing'].items()):
        with timing_cols[idx]:
            st.metric(stage.replace('_', ' ').title(), f"{duration:.3f}s")

    # Year Distribution
    if results['year_distribution']:
        st.subheader("📅 Year Distribution")
        year_fig = create_year_distribution_chart(
            results['year_distribution'],
            "Citation Years with Context Validation"
        )
        st.plotly_chart(year_fig, use_container_width=True)

    # Stance and Purpose
    col1, col2 = st.columns(2)
    
    with col1:
        if results.get('stance_distribution'):
            st.subheader("📈 Citation Stance Distribution")
            stance_fig = create_stance_distribution_chart(
                results['stance_distribution'],
                "How Citations Are Used"
            )
            st.plotly_chart(stance_fig, use_container_width=True)
    
    with col2:
        if results.get('purpose_distribution'):
            st.subheader("🎯 Citation Purpose Distribution")
            purpose_fig = create_purpose_distribution_chart(
                results['purpose_distribution'],
                "Why Citations Are Made"
            )
            st.plotly_chart(purpose_fig, use_container_width=True)

    # Network Visualization
    st.subheader("🔄 Citation Network (Hybrid: Co-occurrence + Semantic)")
    network = results['network']
    
    metrics_cols = st.columns(4)
    with metrics_cols[0]:
        st.metric("Nodes", network['metrics']['node_count'])
    with metrics_cols[1]:
        st.metric("Edges", network['metrics']['edge_count'])
    with metrics_cols[2]:
        st.metric("Density", f"{network['metrics']['density']:.3f}")
    with metrics_cols[3]:
        st.metric("Avg Degree", f"{network['metrics']['average_degree']:.1f}")
    
    if network['metrics'].get('num_communities', 0) > 0:
        st.info(f"🔍 Found **{network['metrics']['num_communities']} topic clusters** in the citation network")

    network_fig = create_network_visualization(network, "Hybrid Citation Network")
    st.plotly_chart(network_fig, use_container_width=True)

    # Top Citations
    if 'top_citations' in network['metrics']:
        st.subheader("⭐ Most Influential Citations")
        top_citations_data = []
        for cite_info in network['metrics']['top_citations'][:10]:
            top_citations_data.append({
                'Citation': cite_info['citation'],
                'Connections': cite_info['connections'],
                'Centrality': f"{cite_info.get('centrality', 0):.3f}"
            })
        
        if top_citations_data:
            st.dataframe(pd.DataFrame(top_citations_data), use_container_width=True, hide_index=True)

    # Citation Relationships
    if network.get('citations_in_context'):
        st.subheader("📑 Citation Relationships in Context")
        
        relationships_data = []
        for ctx in network['citations_in_context'][:20]:  # Show first 20
            rel_type = ctx['relationship']
            if ctx.get('similarity'):
                rel_type += f" (sim: {ctx['similarity']:.2f})"
            
            relationships_data.append({
                'From': ctx['from'][:40] + '...' if len(ctx['from']) > 40 else ctx['from'],
                'To': ctx['to'][:40] + '...' if len(ctx['to']) > 40 else ctx['to'],
                'Relationship': rel_type,
                'Weight': ctx.get('weight', 1),
                'From Stance': ctx.get('from_stance', 'N/A'),
                'To Stance': ctx.get('to_stance', 'N/A'),
                'From Purpose': ctx.get('from_purpose', 'N/A'),
                'To Purpose': ctx.get('to_purpose', 'N/A')
                #!
            })
        
        if relationships_data:
            st.dataframe(pd.DataFrame(relationships_data), use_container_width=True, hide_index=True)
            
            # ⭐ CSV Export Button (with validated data)
            st.divider()
            st.subheader("📥 Export Citation Relationships")
            
            # Create export DataFrame from network edges (already filtered by hybrid analyzer)
            export_df = pd.DataFrame(network['edges'])
            
            # Additional filtering to be extra sure (double-check)
            if not export_df.empty:
                export_df = export_df[
                    export_df['source'].apply(is_valid_citation_reference) & 
                    export_df['target'].apply(is_valid_citation_reference)
                ]
            
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                st.metric("Edges to Export", len(export_df))
                st.caption("Only valid citation-to-citation relationships")
            
            with col_exp2:
                if not export_df.empty:
                    csv_data = export_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv_data,
                        file_name="citation_relationships_hybrid.csv",
                        mime="text/csv",
                        help="Download filtered citation relationships (no paper titles or journal names)"
                    )
                else:
                    st.warning("No valid relationships to export")

    # Citation Details
    with st.expander("📋 Detailed Citation Analysis"):
        if results.get('citation_details'):
            details_data = []
            for cite in results['citation_details'][:50]:  # Show first 50
                details_data.append({
                    'Citation': cite['citation'][:50] + '...' if len(cite['citation']) > 50 else cite['citation'],
                    'Type': cite['type'],
                    'Stance': cite['stance'],
                    'Stance Conf.': cite['stance_confidence'],
                    'Purpose': cite['purpose'],
                    'Purpose Conf.': cite['purpose_confidence'],
                    'Context': cite['context'][:100] + '...' if len(cite['context']) > 100 else cite['context']
                })
            
            if details_data:
                st.dataframe(pd.DataFrame(details_data), use_container_width=True, hide_index=True)


def main():
    """Main application."""
    st.title("🔬 Citation Analyzer Comparison Tool")
    st.caption("Compare different approaches for citation analysis: NLP, Semantic, and Hybrid")

    # Sidebar
    with st.sidebar:
        st.header("📊 Settings")

        analyzer_type = st.radio(
            "Select Analyzer",
            ["Hybrid (Recommended ⭐)", "NLP (Fast)", "Semantic (Topic Clustering)", "Compare All"],
            help="""
            **Hybrid**: Best of both worlds - semantic + co-occurrence + spaCy
            **NLP**: Uses spaCy for relationship detection (fast, ~200-400ms)
            **Semantic**: Uses embeddings for similarity (finds hidden connections)
            **Compare All**: Run all three and compare side-by-side
            """
        )

        st.divider()

        st.markdown("### 🎯 Analyzer Features")

        if "Hybrid" in analyzer_type or "Compare" in analyzer_type:
            with st.expander("🚀 Hybrid Analyzer (NEW!)"):
                st.markdown("""
                - ✅ Semantic similarity (embeddings)
                - ✅ Co-occurrence detection
                - ✅ spaCy sentence segmentation
                - ✅ Year distribution validation
                - ✅ Stance detection (4 types)
                - ✅ Purpose classification (6 types)
                - ✅ Topic clustering
                - ✅ Combined relationship types
                """)

        if "NLP" in analyzer_type or "Compare" in analyzer_type:
            with st.expander("🤖 NLP Analyzer"):
                st.markdown("""
                - ✅ Fuzzy deduplication
                - ✅ 5 relationship types
                - ✅ False positive filtering
                - ✅ Fast processing (~200-400ms)
                - ❌ No topic clustering
                - ❌ No stance detection
                """)

        if "Semantic" in analyzer_type or "Compare" in analyzer_type:
            with st.expander("🧠 Semantic Analyzer"):
                st.markdown("""
                - ✅ Semantic similarity
                - ✅ Topic clustering
                - ✅ Stance detection
                - ✅ Purpose classification
                - ❌ No co-occurrence
                - ❌ No spaCy
                - ❌ Slower processing
                """)

        st.divider()

        # Hybrid/Semantic settings
        if "Hybrid" in analyzer_type or "Semantic" in analyzer_type or "Compare" in analyzer_type:
            st.markdown("### ⚙️ Advanced Settings")
            
            use_embeddings = st.checkbox(
                "Use Embeddings",
                value=True,
                help="Enable sentence-transformers (~80MB download on first use)"
            )
            
            if "Hybrid" in analyzer_type or "Compare" in analyzer_type:
                enable_spacy = st.checkbox(
                    "Use spaCy",
                    value=True,
                    help="Better sentence detection (requires: pip install spacy)"
                )
            else:
                enable_spacy = False

            if use_embeddings:
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Minimum similarity score to connect citations"
                )
            else:
                similarity_threshold = 0.5
        else:
            use_embeddings = False
            enable_spacy = False
            similarity_threshold = 0.5

    # Main content
    st.header("📄 Upload Research Paper")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a research paper in PDF format"
    )

    if uploaded_file is not None:
        # Process PDF
        with st.spinner("📖 Extracting text from PDF..."):
            pdf_processor = PDFProcessor()
            pdf_data = pdf_processor.extract_text_from_pdf(uploaded_file)
            paper_text = pdf_data.get("text", "")
            metadata = pdf_data.get("metadata", {})

        if not paper_text:
            st.error("❌ Failed to extract text from PDF. Please try another file.")
            return

        st.success(f"✅ Extracted {len(paper_text)} characters from PDF")

        # Show PDF metadata
        with st.expander("📋 PDF Metadata"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pages", metadata.get("total_pages", "N/A"))
            with col2:
                st.metric("Author", metadata.get("author", "Unknown"))
            with col3:
                st.metric("Document Type", metadata.get("document_type", "Unknown"))

        # Run analysis
        if st.button("🚀 Run Analysis", type="primary"):
            
            # Store results for comparison
            all_results = {}

            # Hybrid Analysis
            if "Hybrid" in analyzer_type or "Compare" in analyzer_type:
                st.divider()
                
                if use_embeddings:
                    st.info("📥 First run will download sentence-transformers model (~80MB)")
                
                with st.spinner("🚀 Running Hybrid analysis..."):
                    hybrid_analyzer = HybridSemanticCitationAnalyzer(
                        use_embeddings=use_embeddings,
                        enable_spacy=enable_spacy
                    )
                    hybrid_start = time.time()
                    hybrid_results = hybrid_analyzer.extract_citations(paper_text)
                    hybrid_time = time.time() - hybrid_start

                st.success(f"✅ Hybrid analysis completed in {hybrid_time:.3f}s")
                display_hybrid_results(hybrid_results, paper_text)
                all_results['hybrid'] = (hybrid_results, hybrid_time)

            # NLP Analysis
            if "NLP" in analyzer_type or "Compare" in analyzer_type:
                st.divider()
                with st.spinner("🤖 Running NLP analysis..."):
                    from citation_analyzer_comparison_app import display_nlp_results
                    nlp_analyzer = NLPAnalyzer()
                    nlp_start = time.time()
                    nlp_results = nlp_analyzer.extract_citations(paper_text)
                    nlp_time = time.time() - nlp_start

                st.success(f"✅ NLP analysis completed in {nlp_time:.3f}s")
                display_nlp_results(nlp_results, paper_text)
                all_results['nlp'] = (nlp_results, nlp_time)

            # Semantic Analysis
            if "Semantic" in analyzer_type or "Compare" in analyzer_type:
                st.divider()

                if use_embeddings:
                    st.info("📥 First run will download sentence-transformers model (~80MB)")

                with st.spinner("🧠 Running Semantic analysis..."):
                    from citation_analyzer_comparison_app import display_semantic_results
                    semantic_analyzer = SemanticCitationAnalyzer(use_embeddings=use_embeddings)
                    semantic_start = time.time()
                    semantic_results = semantic_analyzer.extract_citations(
                        paper_text,
                        semantic_threshold=similarity_threshold
                    )
                    semantic_time = time.time() - semantic_start

                st.success(f"✅ Semantic analysis completed in {semantic_time:.3f}s")
                display_semantic_results(semantic_results, paper_text)
                all_results['semantic'] = (semantic_results, semantic_time)

            # Three-way Comparison
            if "Compare" in analyzer_type and len(all_results) >= 2:
                st.divider()
                st.header("📊 Three-Way Comparison")

                comparison_data = {
                    'Metric': [
                        'Processing Time',
                        'Total Citations',
                        'Unique Citations',
                        'Network Nodes',
                        'Network Edges',
                        'Network Density',
                        'Year Distribution',
                        'Stance Detection',
                        'Purpose Classification',
                        'Co-occurrence',
                        'Semantic Similarity',
                        'spaCy Integration',
                        'Topic Clustering'
                    ]
                }

                if 'hybrid' in all_results:
                    hybrid_results, hybrid_time = all_results['hybrid']
                    comparison_data['Hybrid Analyzer'] = [
                        f"{hybrid_time:.3f}s",
                        hybrid_results['total_count'],
                        hybrid_results['unique_count'],
                        hybrid_results['network']['metrics']['node_count'],
                        hybrid_results['network']['metrics']['edge_count'],
                        f"{hybrid_results['network']['metrics']['density']:.3f}",
                        "✅ Yes (validated)" if hybrid_results.get('year_distribution') else "❌ No",
                        "✅ Yes (4 types)" if hybrid_results.get('stance_distribution') else "❌ No",
                        "✅ Yes (6 types)" if hybrid_results.get('purpose_distribution') else "❌ No",
                        "✅ Yes",
                        "✅ Yes" if hybrid_results['features']['embeddings'] else "❌ No",
                        "✅ Yes" if hybrid_results['features']['spacy'] else "❌ No",
                        f"✅ Yes ({hybrid_results['network']['metrics'].get('num_communities', 0)} clusters)"
                    ]

                if 'nlp' in all_results:
                    nlp_results, nlp_time = all_results['nlp']
                    comparison_data['NLP Analyzer'] = [
                        f"{nlp_time:.3f}s",
                        nlp_results['total_count'],
                        nlp_results['unique_citations'],
                        nlp_results['network']['metrics']['node_count'],
                        nlp_results['network']['metrics']['edge_count'],
                        f"{nlp_results['network']['metrics']['density']:.3f}",
                        "✅ Yes (validated)" if nlp_results.get('year_distribution') else "❌ No",
                        "❌ No",
                        "❌ No",
                        "✅ Yes",
                        "❌ No",
                        "✅ Yes",
                        "❌ No"
                    ]

                if 'semantic' in all_results:
                    semantic_results, semantic_time = all_results['semantic']
                    comparison_data['Semantic Analyzer'] = [
                        f"{semantic_time:.3f}s",
                        semantic_results['total_count'],
                        semantic_results['unique_count'],
                        semantic_results['network']['metrics']['node_count'],
                        semantic_results['network']['metrics']['edge_count'],
                        f"{semantic_results['network']['metrics']['density']:.3f}",
                        "❌ No",
                        "✅ Yes (4 types)" if semantic_results.get('stance_distribution') else "❌ No",
                        "✅ Yes (6 types)" if semantic_results.get('purpose_distribution') else "❌ No",
                        "❌ No",
                        "✅ Yes" if semantic_results.get('method') == 'semantic_similarity' else "❌ No",
                        "❌ No",
                        f"✅ Yes ({semantic_results['network']['metrics'].get('num_communities', 0)} clusters)"
                    ]

                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                # Recommendations
                st.subheader("💡 Recommendations")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.success("""
                    **✨ Hybrid Analyzer (BEST)**
                    - ⭐ Combines all strengths
                    - 🎯 Most accurate relationships
                    - 📊 Complete analysis
                    - 🚀 Best for research papers
                    - ✅ Year distribution validated
                    - ✅ Stance + Purpose detection
                    """)

                with col2:
                    st.info("""
                    **🤖 NLP Analyzer**
                    - ⚡ Fastest processing
                    - ✅ Accurate citation counts
                    - ✅ Good for production
                    - ❌ No stance detection
                    - ❌ No semantic similarity
                    """)

                with col3:
                    st.info("""
                    **🧠 Semantic Analyzer**
                    - ✅ Topic clustering
                    - ✅ Stance detection
                    - ❌ No co-occurrence
                    - ❌ No year distribution
                    - ❌ Missing spaCy integration
                    """)

    else:
        # Instructions
        st.info("👆 Upload a PDF file to begin analysis")

        st.markdown("""
        ### 📖 How to Use

        1. **Upload** a research paper PDF using the file uploader above
        2. **Select** analyzer type from the sidebar
        3. **Configure** settings (embeddings, spaCy, etc.)
        4. **Click** "Run Analysis" to see results

        ### 🚀 NEW: Hybrid Analyzer

        The **Hybrid Analyzer** combines the best features from both approaches:
        
        - ✅ **Semantic Similarity**: Finds related citations based on meaning
        - ✅ **Co-occurrence**: Detects citations in same sentence/paragraph
        - ✅ **spaCy Integration**: Accurate sentence boundary detection
        - ✅ **Year Distribution**: With context validation
        - ✅ **Stance Detection**: Supporting, refuting, contrasting, neutral
        - ✅ **Purpose Classification**: Background, methodology, comparison, etc.
        - ✅ **Combined Relationships**: Multiple relationship types

        ### 📊 Analyzer Comparison

        | Feature | NLP | Semantic | Hybrid |
        |---------|-----|----------|--------|
        | Speed | ⚡ Fast | 🐢 Slower | ⚡ Medium |
        | Co-occurrence | ✅ Yes | ❌ No | ✅ Yes |
        | Semantic | ❌ No | ✅ Yes | ✅ Yes |
        | spaCy | ✅ Yes | ❌ No | ✅ Yes |
        | Year Dist. | ✅ Yes | ❌ No | ✅ Yes |
        | Stance | ❌ No | ✅ Yes | ✅ Yes |
        | Purpose | ❌ No | ✅ Yes | ✅ Yes |
        | Best For | Production | Exploration | Research |
        """)


if __name__ == "__main__":
    main()