# citation_analyzer_app_FULL.py
"""
COMPLETE Citation Network Analyzer - ALL FEATURES INCLUDED
~730 lines - Nothing removed
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from pathlib import Path
import json
from datetime import datetime
import sys

# Setup comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('citation_analyzer_app_full.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import analyzers
try:
    from citation_analyzer_nlp import EnhancedCitationAnalyzer as NLPAnalyzer
    logger.info("✓ NLP Analyzer imported")
except ImportError as e:
    logger.error(f"NLP Analyzer import failed: {e}")
    NLPAnalyzer = None

try:
    from citation_analyzer_semantic_enhanced import SemanticCitationAnalyzer
    logger.info("✓ Semantic Analyzer imported")
except ImportError as e:
    logger.error(f"Semantic Analyzer import failed: {e}")
    SemanticCitationAnalyzer = None

# Page config
st.set_page_config(
    page_title="Citation Network Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stDataFrame {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def create_network_graph(results, analyzer_name):
    """
    Create interactive network graph using Plotly
    """
    
    network_data = results.get('network', {})
    nodes = network_data.get('nodes', [])
    edges = network_data.get('edges', [])
    
    if not nodes or not edges:
        st.warning("No network data to visualize")
        return None
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for node in nodes:
        G.add_node(node['id'], **node)
    
    # Add edges
    for edge in edges:
        G.add_edge(edge['source'], edge['target'], **edge)
    
    # Calculate layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Edge info for hover
        relationship = edge[2].get('relationship', 'unknown')
        weight = edge[2].get('weight', 1)
        edge_text.append(f"{edge[0]} → {edge[1]}<br>Type: {relationship}<br>Weight: {weight}")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        text=edge_text,
        mode='lines'
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Node info
        node_data = G.nodes[node]
        degree = node_data.get('degree', 0)
        stance = node_data.get('dominant_stance', 'neutral')
        purpose = node_data.get('dominant_purpose', 'general')
        
        node_text.append(
            f"{node}<br>"
            f"Connections: {degree}<br>"
            f"Stance: {stance}<br>"
            f"Purpose: {purpose}"
        )
        
        node_size.append(10 + degree * 2)
        
        # Color by stance
        stance_colors = {
            'supporting': '#2ecc71',
            'refuting': '#e74c3c',
            'contrasting': '#f39c12',
            'neutral': '#95a5a6'
        }
        node_color.append(stance_colors.get(stance, '#95a5a6'))
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white')
        ),
        textposition="top center",
        textfont=dict(size=8)
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f'{analyzer_name} Citation Network',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
    )
    
    return fig


def display_relationships_table(results, analyzer_name):
    """
    Display Citation Relationships table with ALL columns including context
    """
    
    st.subheader(f"📊 Citation Relationships - {analyzer_name}")
    
    # Get edges
    edges = results.get('network', {}).get('edges', [])
    
    if not edges:
        st.warning(f"No citation relationships found in {analyzer_name}")
        logger.warning(f"{analyzer_name}: No edges to display")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"{analyzer_name} - RELATIONSHIP TABLE")
    logger.info(f"{'='*60}")
    logger.info(f"Number of edges: {len(edges)}")
    
    # Log first edge structure
    first_edge = edges[0]
    logger.info(f"First edge keys: {list(first_edge.keys())}")
    
    # Check for context
    has_context = 'context' in first_edge
    logger.info(f"Has 'context' field: {has_context}")
    
    if has_context:
        context_value = first_edge['context']
        logger.info(f"Context type: {type(context_value)}")
        logger.info(f"Context length: {len(str(context_value))}")
        logger.info(f"Context preview: {str(context_value)[:150]}...")
    
    # Create DataFrame
    df = pd.DataFrame(edges)
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"DataFrame columns: {list(df.columns)}")
    
    # Verify context column
    if 'context' not in df.columns:
        st.error(f"⚠️ ERROR: Context column is MISSING from {analyzer_name} edges!")
        st.write("**Available columns:**", list(df.columns))
        logger.error(f"❌ Context column NOT in DataFrame!")
        logger.error(f"Columns: {list(df.columns)}")
        st.dataframe(df, use_container_width=True)
        return
    
    logger.info(f"✅ Context column found in DataFrame")
    
    # Context statistics
    null_count = df['context'].isnull().sum()
    empty_count = (df['context'] == '').sum()
    valid_count = len(df) - null_count - empty_count
    
    logger.info(f"Context stats: {valid_count} valid, {null_count} null, {empty_count} empty")
    
    # Filter options
    st.markdown("#### Filter Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        relationship_types = ['All'] + df['relationship'].unique().tolist()
        selected_relationship = st.selectbox(
            "Relationship Type",
            relationship_types,
            key=f"rel_{analyzer_name}"
        )
    
    with col2:
        if 'source_stance' in df.columns:
            stance_types = ['All'] + df['source_stance'].unique().tolist()
            selected_stance = st.selectbox(
                "Source Stance",
                stance_types,
                key=f"stance_{analyzer_name}"
            )
        else:
            selected_stance = 'All'
    
    with col3:
        if 'source_purpose' in df.columns:
            purpose_types = ['All'] + df['source_purpose'].unique().tolist()
            selected_purpose = st.selectbox(
                "Source Purpose",
                purpose_types,
                key=f"purpose_{analyzer_name}"
            )
        else:
            selected_purpose = 'All'
    
    # Apply filters
    df_filtered = df.copy()
    
    if selected_relationship != 'All':
        df_filtered = df_filtered[df_filtered['relationship'] == selected_relationship]
    
    if selected_stance != 'All' and 'source_stance' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['source_stance'] == selected_stance]
    
    if selected_purpose != 'All' and 'source_purpose' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['source_purpose'] == selected_purpose]
    
    # Select columns to display
    display_columns = ['source', 'target', 'relationship', 'weight', 'similarity', 'context']
    
    # Add optional columns
    optional_columns = ['source_stance', 'target_stance', 'source_purpose', 'target_purpose']
    for col in optional_columns:
        if col in df_filtered.columns:
            display_columns.append(col)
            logger.info(f"Including optional column: {col}")
    
    # Filter to available columns
    display_columns = [col for col in display_columns if col in df_filtered.columns]
    logger.info(f"Displaying columns: {display_columns}")
    
    # Create display DataFrame
    df_display = df_filtered[display_columns].copy()
    
    # Rename columns
    column_mapping = {
        'source': 'Source',
        'target': 'Target',
        'relationship': 'Relationship',
        'weight': 'Weight',
        'similarity': 'Similarity',
        'context': 'Context',
        'source_stance': 'Source Stance',
        'target_stance': 'Target Stance',
        'source_purpose': 'Source Purpose',
        'target_purpose': 'Target Purpose'
    }
    
    df_display = df_display.rename(columns=column_mapping)
    
    # Log sample contexts
    logger.info(f"\nSample contexts from {analyzer_name}:")
    for idx in range(min(3, len(df_display))):
        context = df_display.iloc[idx]['Context']
        logger.info(f"  Row {idx}: {str(context)[:100]}...")
    
    # Display info
    st.info(f"📊 Showing {len(df_display)} of {len(df)} total relationships")
    
    # Display table
    st.dataframe(
        df_display,
        use_container_width=True,
        height=500,
        column_config={
            "Context": st.column_config.TextColumn(
                "Context",
                width="large",
                help="Surrounding text for this citation relationship"
            ),
            "Source": st.column_config.TextColumn("Source", width="medium"),
            "Target": st.column_config.TextColumn("Target", width="medium"),
            "Relationship": st.column_config.TextColumn("Relationship", width="small"),
            "Similarity": st.column_config.NumberColumn("Similarity", format="%.3f"),
        }
    )
    
    # Statistics row
    st.markdown("#### Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Relationships", len(df))
    
    with col2:
        cocited = len(df[df['relationship'] == 'co-cited'])
        st.metric("Co-cited", cocited)
    
    with col3:
        semantic = len(df[df['relationship'] == 'semantic_similarity'])
        st.metric("Semantic Similarity", semantic)
    
    with col4:
        st.metric("With Valid Context", valid_count)
    
    # Download buttons
    st.markdown("#### Export Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"{analyzer_name.lower().replace(' ', '_')}_relationships.csv",
            mime='text/csv',
        )
    
    with col2:
        # Excel download
        try:
            from io import BytesIO
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_display.to_excel(writer, index=False, sheet_name='Relationships')
            
            st.download_button(
                label="📥 Download as Excel",
                data=buffer.getvalue(),
                file_name=f"{analyzer_name.lower().replace(' ', '_')}_relationships.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            )
        except ImportError:
            st.info("Install openpyxl for Excel export")
    
    with col3:
        # JSON download
        json_data = df_display.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name=f"{analyzer_name.lower().replace(' ', '_')}_relationships.json",
            mime='application/json',
        )
    
    logger.info(f"✅ {analyzer_name} - Display completed successfully\n")


def display_citation_details(results, analyzer_name):
    """Display detailed citation information table"""
    
    st.subheader(f"📋 Citation Details - {analyzer_name}")
    
    citation_details = results.get('citation_details', [])
    
    if not citation_details:
        st.warning("No citation details available")
        return
    
    df = pd.DataFrame(citation_details)
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        if 'stance' in df.columns:
            stance_filter = st.multiselect(
                "Filter by Stance",
                df['stance'].unique().tolist(),
                key=f"details_stance_{analyzer_name}"
            )
            if stance_filter:
                df = df[df['stance'].isin(stance_filter)]
    
    with col2:
        if 'purpose' in df.columns:
            purpose_filter = st.multiselect(
                "Filter by Purpose",
                df['purpose'].unique().tolist(),
                key=f"details_purpose_{analyzer_name}"
            )
            if purpose_filter:
                df = df[df['purpose'].isin(purpose_filter)]
    
    st.dataframe(
        df,
        use_container_width=True,
        height=400,
        column_config={
            "context": st.column_config.TextColumn("Context", width="large"),
            "citation": st.column_config.TextColumn("Citation", width="medium"),
        }
    )
    
    # Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Citation Details CSV",
        data=csv,
        file_name=f"{analyzer_name.lower().replace(' ', '_')}_citation_details.csv",
        mime='text/csv',
    )


def display_visualizations(results, analyzer_name):
    """Display comprehensive visualizations"""
    
    st.subheader(f"📈 Visualizations - {analyzer_name}")
    
    # Stance distribution
    if 'stance_distribution' in results:
        stance_dist = results['stance_distribution']
        if stance_dist:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    x=list(stance_dist.keys()),
                    y=list(stance_dist.values()),
                    title="Citation Stance Distribution",
                    labels={'x': 'Stance', 'y': 'Count'},
                    color=list(stance_dist.keys()),
                    color_discrete_map={
                        'supporting': '#2ecc71',
                        'refuting': '#e74c3c',
                        'contrasting': '#f39c12',
                        'neutral': '#95a5a6'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Stance Counts:**")
                for stance, count in sorted(stance_dist.items(), key=lambda x: x[1], reverse=True):
                    st.metric(stance.capitalize(), count)
    
    # Purpose distribution
    if 'purpose_distribution' in results:
        purpose_dist = results['purpose_distribution']
        if purpose_dist:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.pie(
                    values=list(purpose_dist.values()),
                    names=list(purpose_dist.keys()),
                    title="Citation Purpose Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Purpose Counts:**")
                for purpose, count in sorted(purpose_dist.items(), key=lambda x: x[1], reverse=True):
                    st.metric(purpose.capitalize(), count)
    
    # Year distribution
    if 'year_distribution' in results:
        year_dist = results['year_distribution']
        if year_dist:
            fig = px.line(
                x=sorted(year_dist.keys()),
                y=[year_dist[year] for year in sorted(year_dist.keys())],
                title="Citations by Year",
                labels={'x': 'Year', 'y': 'Citation Count'}
            )
            st.plotly_chart(fig, use_container_width=True)


def display_network_metrics(results, analyzer_name):
    """Display network analysis metrics"""
    
    st.subheader(f"🔗 Network Metrics - {analyzer_name}")
    
    metrics = results.get('network', {}).get('metrics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Network Nodes",
            metrics.get('node_count', 0),
            help="Total number of unique citations"
        )
    
    with col2:
        st.metric(
            "Network Edges",
            metrics.get('edge_count', 0),
            help="Total number of relationships"
        )
    
    with col3:
        density = metrics.get('density', 0)
        st.metric(
            "Network Density",
            f"{density:.2%}",
            help="How connected the network is"
        )
    
    with col4:
        avg_degree = metrics.get('average_degree', 0)
        st.metric(
            "Avg Connections",
            f"{avg_degree:.1f}",
            help="Average connections per citation"
        )
    
    # Communities
    if metrics.get('num_communities', 0) > 0:
        st.markdown(f"**Research Topic Clusters:** {metrics['num_communities']}")
        
        communities = results.get('network', {}).get('communities', [])
        if communities:
            with st.expander("View Communities"):
                for i, community in enumerate(communities, 1):
                    st.write(f"**Community {i}:** {', '.join(list(community)[:5])}")
                    if len(community) > 5:
                        st.write(f"   ... and {len(community) - 5} more")
    
    # Top citations
    if 'top_citations' in metrics:
        st.markdown("**Most Influential Citations:**")
        for cite_info in metrics['top_citations'][:5]:
            st.write(f"• **{cite_info['citation']}**: {cite_info['connections']} connections")


def run_nlp_analysis(paper_text):
    """Run NLP citation analysis"""
    
    logger.info("\n" + "="*60)
    logger.info("STARTING NLP ANALYSIS")
    logger.info("="*60)
    
    if NLPAnalyzer is None:
        st.error("❌ NLP Analyzer not available")
        logger.error("NLP Analyzer is None")
        return None
    
    try:
        with st.spinner("🔄 Running NLP analysis..."):
            analyzer = NLPAnalyzer()
            results = analyzer.extract_citations(paper_text)
        
        logger.info("✅ NLP analysis completed")
        logger.info(f"  Total citations: {results.get('total_count', 0)}")
        logger.info(f"  Unique citations: {results.get('unique_citations', 0)}")
        
        st.success("✅ NLP Analysis Complete!")
        return results
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        logger.error(f"NLP analysis failed: {e}", exc_info=True)
        return None


def run_semantic_analysis(paper_text):
    """Run Semantic citation analysis"""
    
    logger.info("\n" + "="*60)
    logger.info("STARTING SEMANTIC ANALYSIS")
    logger.info("="*60)
    
    if SemanticCitationAnalyzer is None:
        st.error("❌ Semantic Analyzer not available")
        logger.error("Semantic Analyzer is None")
        return None
    
    try:
        with st.spinner("🔄 Running semantic analysis..."):
            analyzer = SemanticCitationAnalyzer(use_embeddings=True)
            results = analyzer.extract_citations(paper_text)
        
        logger.info("✅ Semantic analysis completed")
        logger.info(f"  Total citations: {results.get('total_count', 0)}")
        logger.info(f"  Unique citations: {results.get('unique_count', 0)}")
        
        st.success("✅ Semantic Analysis Complete!")
        return results
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        logger.error(f"Semantic analysis failed: {e}", exc_info=True)
        return None


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Citation Network Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("**Compare NLP vs Semantic citation analysis with advanced network visualization**")
    
    logger.info("\n" + "="*80)
    logger.info(f"NEW SESSION STARTED - {datetime.now()}")
    logger.info("="*80)
    
    # Sidebar
    with st.sidebar:
        st.header("📄 Upload Paper")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['txt'],
            help="Upload a text file containing your research paper"
        )
        
        st.markdown("---")
        
        st.header("⚙️ Settings")
        
        show_network_graph = st.checkbox("Show Network Graph", value=True)
        show_visualizations = st.checkbox("Show Visualizations", value=True)
        show_metrics = st.checkbox("Show Network Metrics", value=True)
        show_citation_details = st.checkbox("Show Citation Details", value=False)
        
        st.markdown("---")
        
        st.header("ℹ️ About")
        st.markdown("""
        This tool analyzes citations in research papers using:
        
        - **NLP Analysis**: Natural language processing
        - **Semantic Analysis**: Semantic similarity & co-citation
        
        Features:
        - Context extraction
        - Stance detection
        - Purpose classification
        - Network visualization
        """)
        
        st.markdown("---")
        st.caption(f"Log file: citation_analyzer_app_full.log")
    
    # Main content
    if uploaded_file is None:
        st.info("👈 **Upload a paper to begin analysis**")
        
        # Show example/features
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 NLP Analyzer")
            st.markdown("""
            - Position-based deduplication
            - Context-aware validation
            - Relationship detection
            - Fast processing
            """)
        
        with col2:
            st.markdown("### 🔬 Semantic Analyzer")
            st.markdown("""
            - Semantic similarity
            - Stance detection
            - Purpose classification
            - Network analysis
            """)
        
        return
    
    # Read paper
    try:
        paper_text = uploaded_file.read().decode('utf-8')
        logger.info(f"✓ File uploaded: {uploaded_file.name}")
        logger.info(f"  Size: {len(paper_text)} characters")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        logger.error(f"File read error: {e}")
        return
    
    st.success(f"✅ Loaded: **{uploaded_file.name}**")
    st.text(f"Paper length: {len(paper_text):,} characters")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 NLP Analyzer",
        "🔬 Semantic Analyzer",
        "⚖️ Comparison",
        "📖 Documentation"
    ])
    
    # ==================== NLP ANALYZER TAB ====================
    with tab1:
        st.header("📊 NLP Citation Analyzer")
        st.markdown("Advanced natural language processing for citation detection")
        
        if st.button("🚀 Run NLP Analysis", key="nlp_btn", type="primary"):
            results = run_nlp_analysis(paper_text)
            if results:
                st.session_state['nlp_results'] = results
        
        if 'nlp_results' in st.session_state:
            results = st.session_state['nlp_results']
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Citations", results.get('total_count', 0))
            with col2:
                st.metric("Unique Citations", results.get('unique_citations', 0))
            with col3:
                edges = results.get('network', {}).get('metrics', {}).get('edge_count', 0)
                st.metric("Relationships", edges)
            with col4:
                timing = results.get('timing', {}).get('total_time', 0)
                st.metric("Processing Time", f"{timing:.2f}s")
            
            # Network metrics
            if show_metrics:
                display_network_metrics(results, "NLP Analyzer")
            
            # Network graph
            if show_network_graph:
                st.markdown("---")
                graph_fig = create_network_graph(results, "NLP Analyzer")
                if graph_fig:
                    st.plotly_chart(graph_fig, use_container_width=True)
            
            # Relationships table
            st.markdown("---")
            display_relationships_table(results, "NLP Analyzer")
            
            # Citation details
            if show_citation_details:
                st.markdown("---")
                display_citation_details(results, "NLP Analyzer")
    
    # ==================== SEMANTIC ANALYZER TAB ====================
    with tab2:
        st.header("🔬 Semantic Citation Analyzer")
        st.markdown("Semantic similarity and co-citation network analysis")
        
        if st.button("🚀 Run Semantic Analysis", key="semantic_btn", type="primary"):
            results = run_semantic_analysis(paper_text)
            if results:
                st.session_state['semantic_results'] = results
        
        if 'semantic_results' in st.session_state:
            results = st.session_state['semantic_results']
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Citations", results.get('total_count', 0))
            with col2:
                st.metric("Unique Citations", results.get('unique_count', 0))
            with col3:
                edges = results.get('network', {}).get('metrics', {}).get('edge_count', 0)
                st.metric("Relationships", edges)
            with col4:
                supporting = results.get('stance_distribution', {}).get('supporting', 0)
                st.metric("Supporting", supporting)
            
            # Network metrics
            if show_metrics:
                display_network_metrics(results, "Semantic Analyzer")
            
            # Visualizations
            if show_visualizations:
                st.markdown("---")
                display_visualizations(results, "Semantic Analyzer")
            
            # Network graph
            if show_network_graph:
                st.markdown("---")
                graph_fig = create_network_graph(results, "Semantic Analyzer")
                if graph_fig:
                    st.plotly_chart(graph_fig, use_container_width=True)
            
            # Relationships table
            st.markdown("---")
            display_relationships_table(results, "Semantic Analyzer")
            
            # Citation details
            if show_citation_details:
                st.markdown("---")
                display_citation_details(results, "Semantic Analyzer")
    
    # ==================== COMPARISON TAB ====================
    with tab3:
        st.header("⚖️ Analysis Comparison")
        
        if 'nlp_results' not in st.session_state or 'semantic_results' not in st.session_state:
            st.info("ℹ️ Run both analyses to see comparison")
            return
        
        nlp_results = st.session_state['nlp_results']
        semantic_results = st.session_state['semantic_results']
        
        # Side-by-side metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 NLP Analyzer")
            st.metric("Total Citations", nlp_results.get('total_count', 0))
            st.metric("Unique Citations", nlp_results.get('unique_citations', 0))
            nlp_edges = nlp_results.get('network', {}).get('metrics', {}).get('edge_count', 0)
            st.metric("Relationships", nlp_edges)
            nlp_time = nlp_results.get('timing', {}).get('total_time', 0)
            st.metric("Processing Time", f"{nlp_time:.2f}s")
        
        with col2:
            st.subheader("🔬 Semantic Analyzer")
            st.metric("Total Citations", semantic_results.get('total_count', 0))
            st.metric("Unique Citations", semantic_results.get('unique_count', 0))
            semantic_edges = semantic_results.get('network', {}).get('metrics', {}).get('edge_count', 0)
            st.metric("Relationships", semantic_edges)
            st.metric("Method", semantic_results.get('method', 'N/A'))
        
        # Comparison chart
        st.markdown("---")
        st.subheader("📊 Comparison Chart")
        
        comparison_data = {
            'Metric': ['Total Citations', 'Unique Citations', 'Relationships'],
            'NLP': [
                nlp_results.get('total_count', 0),
                nlp_results.get('unique_citations', 0),
                nlp_edges
            ],
            'Semantic': [
                semantic_results.get('total_count', 0),
                semantic_results.get('unique_count', 0),
                semantic_edges
            ]
        }
        
        fig = go.Figure(data=[
            go.Bar(name='NLP', x=comparison_data['Metric'], y=comparison_data['NLP']),
            go.Bar(name='Semantic', x=comparison_data['Metric'], y=comparison_data['Semantic'])
        ])
        fig.update_layout(barmode='group', title='NLP vs Semantic Comparison')
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison
        st.markdown("---")
        st.subheader("🔍 Detailed Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**NLP Strengths:**")
            st.markdown("""
            - ✅ Faster processing
            - ✅ Better deduplication
            - ✅ Context-aware validation
            - ✅ Relationship detection
            """)
        
        with col2:
            st.markdown("**Semantic Strengths:**")
            st.markdown("""
            - ✅ Semantic similarity
            - ✅ Stance detection
            - ✅ Purpose classification
            - ✅ Richer context
            """)
    
    # ==================== DOCUMENTATION TAB ====================
    with tab4:
        st.header("📖 Documentation")
        
        st.markdown("""
        ## How to Use
        
        1. **Upload Paper**: Upload a .txt file containing your research paper
        2. **Run Analysis**: Choose NLP or Semantic analyzer (or both)
        3. **Explore Results**: View relationships, network graph, and metrics
        4. **Export Data**: Download results as CSV, Excel, or JSON
        
        ## Features
        
        ### Citation Detection
        - DOI patterns
        - arXiv identifiers
        - Numbered citations [1]
        - Author-year (Smith, 2020)
        - Harvard style
        
        ### Context Analysis
        - Extracts surrounding text
        - Detects citation stance
        - Classifies purpose
        - Shows relationships
        
        ### Network Analysis
        - Co-citation detection
        - Semantic similarity
        - Community detection
        - Network metrics
        
        ### Stance Detection
        - **Supporting**: Confirms, validates
        - **Refuting**: Contradicts, challenges
        - **Contrasting**: Alternative views
        - **Neutral**: Descriptive
        
        ### Purpose Classification
        - **Background**: Literature review
        - **Methodology**: Methods used
        - **Comparison**: Benchmarking
        - **Theory**: Frameworks
        - **Results**: Findings
        - **Data**: Datasets
        
        ## Export Options
        
        - **CSV**: Universal format
        - **Excel**: Formatted spreadsheet
        - **JSON**: For developers
        
        ## Logging
        
        All analysis is logged to `citation_analyzer_app_full.log`
        
        Check this file for debugging and detailed information.
        """)


if __name__ == "__main__":
    main()
    logger.info("\n" + "="*80)
    logger.info(f"SESSION ENDED - {datetime.now()}")
    logger.info("="*80 + "\n")