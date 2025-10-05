# research_paper_assistant.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from typing import Dict
from pdf_processor import PDFProcessor
from citation_analyzer import EnhancedCitationAnalyzer
from structure_analyzer import StructureAnalyzer
from model_comparison_analyzer import ModelComparisonAnalyzer
from enhanced_chat_interface_1 import EnhancedChatInterface
from performance_analyzer import PerformanceAnalyzer

class ResearchPaperAssistant:
    """Enhanced main class for the Research Paper Analysis Assistant."""
    
    def __init__(self):
        self.setup_page()
        self.initialize_components()
        self.setup_session_state()
    
    def setup_page(self):
        """Configures the Streamlit page layout with enhanced styling."""
        st.set_page_config(
            page_title="Research Paper Analysis Assistant",
            page_icon="📚",
            layout="wide"
        )
        
        st.markdown("""
            <style>
            .stApp {
                background-color: #F0F2F6;
                color: #000000 !important;
            }
            .stSidebar {
                background-color: #f4f4f4;
                color: #333;
            }
            div[data-baseweb="select"] > div {
                background-color: #f9f9f9 !important;
                color: #000000 !important;
            }
            div[data-baseweb="select"] div[role="option"] {
                background-color: #ffffff !important;
                color: #000000 !important;
            }
            .stFileUploader > div {
                background-color: #FFFFFF !important;
                color: #000000 !important;
                border: 1px solid #CCC;
            }
            .stTextArea textarea {
                color: #000000 !important;
                background-color: #FFFFFF !important;
                border: 1px solid #CCC !important;
            }
            .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6 {
                color: #000000 !important;
            }
            div {
                color: #000000 !important;
            }
            .stMetric {
                background-color: #FFFFFF !important;
            }
            .stMetric label, .stMetric .value {
                color: #000000 !important;
            }
            .stProgress > div {
                background-color: #E0E0E0;
            }
            .stProgress > div > div {
                background-color: #3B82F6;
            }
            .stButton button {
                background-color: #3B82F6;
                color: white;
            }
            .dataframe {
                color: #000000 !important;
            }
            .metric-container {
                padding: 10px;
                margin: 10px 0;
            }
            .metric-card {
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 5px;
            }
            </style>
        """, unsafe_allow_html=True)

    def initialize_components(self):
        """Initializes enhanced analysis components."""
        self.pdf_processor = PDFProcessor()
        self.citation_analyzer = EnhancedCitationAnalyzer()
        self.structure_analyzer = StructureAnalyzer()
        self.model_analyzer = ModelComparisonAnalyzer()
        self.chat_interface = EnhancedChatInterface()
        self.performance_analyzer = PerformanceAnalyzer()

    def setup_session_state(self):
        """Initializes enhanced session state variables."""
        if "paper_content" not in st.session_state:
            st.session_state.paper_content = ""
        if "model_responses" not in st.session_state:
            st.session_state.model_responses = {}
        if "pdf_metadata" not in st.session_state:
            st.session_state.pdf_metadata = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
    
    def create_sidebar(self):
        """Creates enhanced configuration sidebar."""
        with st.sidebar:
            st.header("📊 Analysis Settings")
            analysis_depth = st.slider(
                "Analysis Depth",
                min_value=1,
                max_value=5,
                value=3,
                help="Controls how detailed the analysis should be"
            )
            st.divider()
            st.markdown("### 🎯 Analysis Focus")
            analyze_citations = st.checkbox("Citation Analysis", value=True)
            analyze_structure = st.checkbox("Structure Analysis", value=True)
            compare_models = st.checkbox("Model Comparison", value=True)
            st.divider()
            st.markdown("### 📈 Visualization Options")
            show_citation_graph = st.checkbox("Show Citation Year Distribution", value=True)
            show_structure_radar = st.checkbox("Show Structure Radar Chart", value=True)
            return {
                "depth": analysis_depth,
                "analyze_citations": analyze_citations,
                "analyze_structure": analyze_structure,
                "compare_models": compare_models,
                "show_citation_graph": show_citation_graph,
                "show_structure_radar": show_structure_radar
            }

    def display_pdf_metadata(self, metadata: Dict):
        """Displays PDF metadata with improved styling."""
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.subheader("📄 Document Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Pages", metadata.get("total_pages", "N/A"))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Author", metadata.get("author", "Unknown"))
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Contains Images", "Yes" if metadata.get("has_images") else "No")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    def display_performance_analysis(self, text: str):
        st.subheader("🚀 Performance Analysis")
        with st.spinner("Analyzing model performance..."):
            results = self.performance_analyzer.analyze_model_performance(
                self.model_analyzer.models,
                text[:1000]
            )
            figures = self.performance_analyzer.create_performance_visualizations(results)
            tabs = st.tabs(["Resource Usage", "Quality Metrics", "Response Metrics"])
            for tab, fig in zip(tabs, figures):
                with tab:
                    st.plotly_chart(fig, use_container_width=True)
            st.subheader("📊 Detailed Performance Metrics")
            report_df = self.performance_analyzer.generate_performance_report(results)
            st.dataframe(report_df, hide_index=True)
            csv = report_df.to_csv(index=False)
            st.download_button(
                "Download Performance Report",
                csv,
                "performance_report.csv",
                "text/csv",
                key='download-performance-csv'
            )
    # Continuation of ResearchPaperAssistant class
    
    def display_analysis_results(self, text: str, settings: Dict):
        """Displays enhanced analysis results with all new features."""
        
        # Citation Analysis with Network Visualization
        if settings["analyze_citations"]:
            with st.spinner("Analyzing citations..."):
                citation_metrics = self.citation_analyzer.extract_citations(text)
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.subheader("📚 Citation Analysis")
                
                # Display citation count metrics
                cols = st.columns(4)
                metrics = [
                    ("Total Citations", citation_metrics["total_count"]),
                    ("Numbered Citations", citation_metrics["numbered_count"]),
                    ("Author-Year Citations", citation_metrics["author_year_count"]),
                    ("Harvard Style", citation_metrics["harvard_count"])
                ]
                for col, (label, value) in zip(cols, metrics):
                    with col:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(label, value)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Citation Year Distribution
                if settings["show_citation_graph"] and citation_metrics["year_distribution"]:
                    years = list(citation_metrics["year_distribution"].keys())
                    counts = list(citation_metrics["year_distribution"].values())
                    fig = go.Figure(data=[go.Bar(x=years, y=counts, name="Citations per Year")])
                    fig.update_layout(
                        title="Citation Year Distribution",
                        xaxis_title="Year",
                        yaxis_title="Number of Citations",
                        paper_bgcolor='white',
                        plot_bgcolor='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Citation Network Analysis
                if "network" in citation_metrics:
                    network_data = citation_metrics["network"]
                    st.markdown("### 🔄 Citation Network Analysis")
                    
                    # Display network metrics
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Total Citations", network_data["metrics"]["node_count"])
                    with metric_cols[1]:
                        st.metric("Citation Links", network_data["metrics"]["edge_count"])
                    with metric_cols[2]:
                        st.metric("Network Density", f"{network_data['metrics']['density']:.2f}")
                    with metric_cols[3]:
                        st.metric("Avg. Citations/Paper", f"{network_data['metrics']['average_degree']:.2f}")
                    
                    # Create network visualization
                    if network_data["nodes"] and network_data["edges"]:
                        G = nx.DiGraph()
                        G.add_edges_from(network_data["edges"])
                        pos = nx.spring_layout(G)
                        
                        network_fig = go.Figure()
                        
                        # Add edges
                        edge_x, edge_y = [], []
                        for edge in network_data["edges"]:
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                        
                        network_fig.add_trace(go.Scatter(
                            x=edge_x, y=edge_y,
                            line=dict(width=0.5, color='#888'),
                            hoverinfo='none',
                            mode='lines',
                            name='Citation Links'
                        ))
                        
                        # Add nodes
                        node_x = [pos[node][0] for node in G.nodes()]
                        node_y = [pos[node][1] for node in G.nodes()]
                        
                        network_fig.add_trace(go.Scatter(
                            x=node_x, y=node_y,
                            mode='markers+text',
                            hoverinfo='text',
                            marker=dict(size=10, color='#1f77b4', line_width=2),
                            text=[str(node) for node in G.nodes()],
                            textposition="top center",
                            name='Citations'
                        ))
                        
                        network_fig.update_layout(
                            title="Citation Network Visualization",
                            showlegend=True,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            paper_bgcolor="white",
                            plot_bgcolor="white",
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        )
                        st.plotly_chart(network_fig, use_container_width=True)
                        
                        # Display citation contexts
                        st.markdown("### 📑 Citation Contexts")
                        contexts_df = pd.DataFrame(network_data["citations_in_context"])
                        if not contexts_df.empty:
                            contexts_df.columns = ["From Citation", "To Citation", "Context"]
                            st.dataframe(contexts_df, hide_index=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Structure Analysis with Progress Bar and Sections
        if settings["analyze_structure"]:
            with st.spinner("Analyzing structure..."):
                structure_metrics = self.structure_analyzer.analyze_structure(text)
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.subheader("🏗️ Structure Analysis")
                
                # Structure completeness progress bar
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                completeness = structure_metrics["structure_completeness"]
                st.progress(completeness, text=f"Structure Completeness: {completeness:.0%}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display section information
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 📋 Sections Found")
                    sections_df = pd.DataFrame([
                        {"Section": k.replace("_score", "").title(), "Present": "✓" if v else "✗"}
                        for k, v in structure_metrics["section_scores"].items()
                    ])
                    st.dataframe(sections_df, hide_index=True)
                
                with col2:
                    st.markdown("### 📊 Content Elements")
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Number of Figures", structure_metrics["num_figures"])
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Number of Tables", structure_metrics["num_tables"])
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Add radar chart
                if settings["show_structure_radar"]:
                    radar_fig = self.structure_analyzer.create_radar_chart(structure_metrics)
                    st.plotly_chart(radar_fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Model Comparison with Enhanced Metrics
        if settings["compare_models"]:
            with st.spinner("Analyzing models..."):
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.subheader("🤖 Enhanced Model Analysis")
                
                model_results = self.model_analyzer.analyze_paper(text)
                st.session_state.model_responses = model_results
                
                # Display performance metrics with throughput
                st.markdown("### 📊 Model Performance Metrics")
                metrics_df = self.model_analyzer.generate_performance_report(model_results)
                st.dataframe(metrics_df, hide_index=True)
                
                # Display performance visualizations
                visualizations = self.model_analyzer.create_performance_visualizations(model_results)
                viz_tabs = st.tabs(["Response Times", "Token Counts", "Consistency Scores"])
                for tab, fig in zip(viz_tabs, visualizations):
                    with tab:
                        st.plotly_chart(fig, use_container_width=True)
                
                # Display model analyses
                st.markdown("### 📝 Model Analyses")
                row1_col1, row1_col2 = st.columns(2)
                row2_col1, row2_col2 = st.columns(2)
                
                models_grid = [
                    (row1_col1, "deepseek-1.5b", "Deepseek 1.5B Analysis"),
                    (row1_col2, "deepseek-8b", "Deepseek 8B Analysis"),
                    (row2_col1, "mistral", "Mistral Analysis"),
                    (row2_col2, "llama3-8b", "LLaMA3 8B Analysis")
                ]
                
                for col, model_name, title in models_grid:
                    with col:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown(f"### {title}")
                        response = model_results.get(model_name, {}).get('response', f"Error analyzing with {model_name}")
                        st.markdown(response)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance Analysis (BLEU, METEOR, ROUGE)
        if settings["compare_models"]:
            self.display_performance_analysis(text)
    
    def run(self):
        """Enhanced main application loop with improved UI."""
        st.title("📚 Research Paper Analysis Assistant")
        st.caption("Advanced PDF Analysis with Deepseek, Llama and Mistral models")
        
        settings = self.create_sidebar()
        selected_model = st.sidebar.selectbox(
            "Select Chat Model",
            ["deepseek-1.5b", "deepseek-8b", "mistral", "llama3-8b"],
            help="Choose which model to use for chat interactions"
        )
        
        tab1, tab2 = st.tabs(["📊 Analysis", "💬 Chat"])
        
        with tab1:
            uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=['pdf'])
            text_input = st.text_area(
                "Or paste your research paper text here:",
                height=200,
                placeholder="Enter or paste your research paper text for analysis..."
            )
            
            if uploaded_file is not None:
                with st.spinner("Processing PDF..."):
                    pdf_data = self.pdf_processor.extract_text_from_pdf(uploaded_file)
                    if pdf_data.get("metadata"):
                        self.display_pdf_metadata(pdf_data["metadata"])
                    text_input = pdf_data.get("text", "")
                    if text_input:
                        st.session_state.paper_content = text_input
                        self.chat_interface.initialize_rag(text_input)
            
            if st.button("Analyze Paper"):
                if text_input:
                    with st.spinner("Analyzing paper..."):
                        self.display_analysis_results(text_input, settings)
                        st.session_state.paper_content = text_input
                else:
                    st.warning("Please upload a PDF or enter text to analyze.")
        
        with tab2:
            if st.session_state.paper_content:
                self.chat_interface.display_chat_interface(selected_model)
            else:
                st.info("Please upload a paper or enter text in the Analysis tab first.")