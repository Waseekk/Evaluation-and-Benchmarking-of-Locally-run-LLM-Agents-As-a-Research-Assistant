# research_paper_assistant.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from typing import Dict
from datetime import datetime
from pdf_processor import PDFProcessor
#from citation_analyzer import EnhancedCitationAnalyzer
from citation_analyzer_nlp import EnhancedCitationAnalyzer


from structure_analyzer import StructureAnalyzer
from model_comparison_analyzer import ModelComparisonAnalyzer
from enhanced_chat_interface_1 import EnhancedChatInterface
from performance_analyzer import PerformanceAnalyzer
from excel_exporter import ExcelExporter

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
        self.excel_exporter = ExcelExporter()

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
        
        # Add session state for Excel export
        if "citation_results" not in st.session_state:
            st.session_state.citation_results = None
        if "structure_results" not in st.session_state:
            st.session_state.structure_results = None
        if "performance_results" not in st.session_state:
            st.session_state.performance_results = None
    
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
            st.divider()
            st.markdown("### 🔬 Advanced Metrics")
            enable_quality_metrics = st.checkbox(
                "📊 Detailed Quality Metrics", 
                value=False,
                help="Enable BLEU, ROUGE, Perplexity analysis (adds ~20 seconds)"
            )
            return {
                "depth": analysis_depth,
                "analyze_citations": analyze_citations,
                "analyze_structure": analyze_structure,
                "compare_models": compare_models,
                "show_citation_graph": show_citation_graph,
                "show_structure_radar": show_structure_radar,
                "enable_quality_metrics": enable_quality_metrics
            }

    def create_export_section(self):
        """Creates a section for exporting all analysis data to Excel."""
        st.sidebar.divider()
        st.sidebar.markdown("### 📥 Export Data")
        
        # ✅ FIXED: Check if ANY useful analysis data exists (performance is optional)
        has_citation = st.session_state.citation_results is not None
        has_structure = st.session_state.structure_results is not None
        has_models = bool(st.session_state.model_responses)
        
        # Export should be available if we have core analysis data
        has_data = any([has_citation, has_structure, has_models])
        
        # Check if performance metrics exist (optional bonus)
        has_performance = bool(st.session_state.performance_results)
        
        if has_data:
            st.sidebar.success("✓ Analysis data available")
            
            # Show what data is available
            with st.sidebar.expander("📋 Available Data"):
                st.write("✅ Citations" if has_citation else "❌ Citations")
                st.write("✅ Structure" if has_structure else "❌ Structure")
                st.write("✅ Model Comparisons" if has_models else "❌ Model Comparisons")
                
                # Show performance status
                if has_performance:
                    # Check if it has quality metrics or just basic metrics
                    sample_model = list(st.session_state.performance_results.keys())[0]
                    sample_data = st.session_state.performance_results[sample_model]
                    has_quality = sample_data['response_metrics']['avg_bleu'] > 0
                    
                    if has_quality:
                        st.write("✅ Performance Metrics (with BLEU/ROUGE/Perplexity)")
                    else:
                        st.write("✅ Performance Metrics (basic only)")
                else:
                    st.write("⚠️ Performance Metrics (not calculated)")
            
            # Show sheet summary in an expander
            with st.sidebar.expander("📑 Excel Sheets Preview"):
                sheet_summary = self.excel_exporter.get_sheet_summary()
                for sheet_name, description in list(sheet_summary.items())[:5]:
                    st.caption(f"**{sheet_name}**: {description}")
                st.caption(f"*...and {len(sheet_summary) - 5} more sheets*")
            
            # ✅ EXPORT BUTTON - Now works with or without performance metrics
            if st.sidebar.button("📊 Export All Data to Excel", type="primary"):
                with st.spinner("Generating Excel file..."):
                    try:
                        # Create comprehensive export (includes whatever data exists)
                        excel_buffer = self.excel_exporter.create_comprehensive_export(
                            citation_data=st.session_state.citation_results,
                            structure_data=st.session_state.structure_results,
                            model_results=st.session_state.model_responses,
                            performance_results=st.session_state.performance_results,  # Can be None or basic or full
                            paper_metadata=st.session_state.pdf_metadata
                        )
                        
                        # Generate filename with timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"paper_analysis_{timestamp}.xlsx"
                        
                        # Create download button
                        st.sidebar.download_button(
                            label="⬇️ Download Excel File",
                            data=excel_buffer,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_excel"
                        )
                        
                        # Show success message with details
                        sheet_count = len(self.excel_exporter.get_sheet_summary())
                        if has_performance:
                            st.sidebar.success(f"✅ Excel ready! ({sheet_count} sheets, includes performance metrics)")
                        else:
                            st.sidebar.success(f"✅ Excel ready! ({sheet_count} sheets)")
                            st.sidebar.info("💡 Enable 'Detailed Quality Metrics' for BLEU/ROUGE analysis")
                        
                    except Exception as e:
                        st.sidebar.error(f"Error generating Excel: {str(e)}")
                        st.sidebar.exception(e)  # Show full traceback for debugging
        else:
            st.sidebar.info("Run an analysis first to enable export")
            st.sidebar.markdown("**Steps:**")
            st.sidebar.markdown("1. Upload a PDF or paste text")
            st.sidebar.markdown("2. Enable analysis options")
            st.sidebar.markdown("3. Click 'Analyze Paper'")
            st.sidebar.markdown("4. Export button will appear here")

    def display_pdf_metadata(self, metadata: Dict):
        """
        Displays enhanced PDF metadata with OCR info, sections, and content lists.
        """
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        
        # === DOCUMENT INFORMATION ===
        st.subheader("📄 Document Information")
        
        col1, col2, col3, col4 = st.columns(4)
        
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
            page_format = metadata.get("page_format", "Unknown")
            st.metric("Page Format", page_format)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            doc_type = metadata.get("document_type", "unknown")
            doc_type_display = {
                "native": "Native PDF",
                "scanned": "Scanned PDF",
                "mixed": "Mixed PDF"
            }.get(doc_type, "Unknown")
            st.metric("Document Type", doc_type_display)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # === OCR INFORMATION (if OCR was used) ===
        if metadata.get("ocr_applied", False):
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.subheader("🔍 OCR Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                ocr_count = metadata.get("ocr_page_count", 0)
                total_pages = metadata.get("total_pages", 1)
                st.metric(
                    "Pages Processed with OCR", 
                    f"{ocr_count}/{total_pages}",
                    help="Number of pages where OCR was applied"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                ocr_conf = metadata.get("ocr_confidence_avg", 0.0)
                confidence_pct = f"{ocr_conf * 100:.1f}%"
                
                # Color coding based on confidence
                if ocr_conf >= 0.8:
                    color = "🟢"
                elif ocr_conf >= 0.6:
                    color = "🟡"
                else:
                    color = "🔴"
                
                st.metric(
                    "OCR Confidence", 
                    f"{color} {confidence_pct}",
                    help="Average confidence score for OCR text extraction"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                ocr_pages = metadata.get("ocr_pages", [])
                if ocr_pages:
                    # Show first few page numbers
                    if len(ocr_pages) <= 5:
                        pages_str = ", ".join(map(str, ocr_pages))
                    else:
                        pages_str = f"{', '.join(map(str, ocr_pages[:5]))}..."
                    
                    st.metric(
                        "OCR Pages", 
                        pages_str,
                        help=f"Pages where OCR was applied: {', '.join(map(str, ocr_pages))}"
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            
            # OCR Quality Warning
            if ocr_conf < 0.7:
                st.warning(
                    "⚠️ OCR confidence is below 70%. Text extraction quality may be lower than expected. "
                    "Consider using a higher quality scan for better results."
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # === CONTENT STRUCTURE ===
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.subheader("📝 Document Structure")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**Sections Detected:**")
            
            sections = {
                "Abstract": metadata.get("has_abstract", False),
                "Introduction": metadata.get("has_introduction", False),
                "Methodology": metadata.get("has_methodology", False),
                "Results": metadata.get("has_results", False),
                "Discussion": metadata.get("has_discussion", False),
                "Conclusion": metadata.get("has_conclusion", False),
                "References": metadata.get("has_references", False)
            }
            
            for section_name, found in sections.items():
                icon = "✅" if found else "❌"
                st.markdown(f"{icon} {section_name}")
            
            # Calculate completeness
            completeness = sum(sections.values()) / len(sections)
            st.progress(completeness, text=f"Structure Completeness: {completeness:.0%}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            
            # Abstract information
            if metadata.get("has_abstract", False):
                st.markdown("**Abstract:**")
                word_count = metadata.get("abstract_word_count", 0)
                st.markdown(f"• Word Count: {word_count}")
                st.markdown(f"• Status: ✅ Found")
            else:
                st.markdown("**Abstract:**")
                st.markdown("• Status: ❌ Not found")
            
            st.markdown("---")
            
            # Content counts
            st.markdown("**Content Elements:**")
            st.markdown(f"• Figures: {metadata.get('figures_count', 0)}")
            st.markdown(f"• Tables: {metadata.get('tables_count', 0)}")
            st.markdown(f"• Images: {metadata.get('images_count', 0)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # === FIGURES LIST ===
        if metadata.get("figure_list"):
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.subheader("📊 Figures")
            
            figure_list = metadata.get("figure_list", [])
            
            # Create DataFrame for better display
            fig_df = pd.DataFrame(figure_list)
            
            if not fig_df.empty:
                # Rename columns for display
                fig_df = fig_df.rename(columns={
                    'number': 'Figure #',
                    'caption': 'Caption',
                    'page': 'Page'
                })
                
                st.dataframe(
                    fig_df,
                    hide_index=True,
                    use_container_width=True
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # === TABLES LIST ===
        if metadata.get("table_list"):
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.subheader("📋 Tables")
            
            table_list = metadata.get("table_list", [])
            
            # Create DataFrame for better display
            table_df = pd.DataFrame(table_list)
            
            if not table_df.empty:
                # Rename columns for display
                table_df = table_df.rename(columns={
                    'number': 'Table #',
                    'caption': 'Caption',
                    'page': 'Page'
                })
                
                st.dataframe(
                    table_df,
                    hide_index=True,
                    use_container_width=True
                )
            
            st.markdown('</div>', unsafe_allow_html=True)

    def display_performance_analysis(self, text: str) -> Dict:
        """Display performance analysis and return results."""
        # st.subheader("🚀 Performance Analysis")
        # with st.spinner("Analyzing model performance..."):
        #     results = self.performance_analyzer.analyze_model_performance(
        #         self.model_analyzer.models,
        #         text[:1000]
        #     )
        #     figures = self.performance_analyzer.create_performance_visualizations(results)
        #     tabs = st.tabs(["Resource Usage", "Quality Metrics", "Response Metrics"])
        #     for tab, fig in zip(tabs, figures):
        #         with tab:
        #             st.plotly_chart(fig, use_container_width=True)
        #     st.subheader("📊 Detailed Performance Metrics")
        #     report_df = self.performance_analyzer.generate_performance_report(results)
        #     st.dataframe(report_df, hide_index=True)
        #     csv = report_df.to_csv(index=False)
        #     st.download_button(
        #         "Download Performance Report",
        #         csv,
        #         "performance_report.csv",
        #         "text/csv",
        #         key='download-performance-csv'
        #     )
        
        # return results
        st.info("ℹ️ Performance metrics are now integrated into model comparison for faster processing.")
        return {}
    
    def display_analysis_results(self, text: str, settings: Dict):
        """Displays enhanced analysis results with all new features."""
        
        # Citation Analysis with Network Visualization
        if settings["analyze_citations"]:
            with st.spinner("Analyzing citations..."):
                citation_metrics = self.citation_analyzer.extract_citations(text)
                
                # Store in session state for export
                st.session_state.citation_results = citation_metrics
                
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
                            # ✅ FIXED: Handle both old (3 cols) and new (4 cols) formats
                            if len(contexts_df.columns) == 4:
                                # NLP version with relationship detection
                                contexts_df.columns = ["From Citation", "To Citation", "Context", "Relationship"]
                                
                                # Add color coding for relationships
                                def highlight_relationship(row):
                                    colors = {
                                        'builds_on': '🔵',
                                        'conflicts': '🔴', 
                                        'parallel': '🟢',
                                        'supports': '🟡',
                                        'sequential': '⚪'
                                    }
                                    return colors.get(row['Relationship'], '⚪')
                                
                                # Add emoji indicators
                                contexts_df['Type'] = contexts_df['Relationship'].apply(
                                    lambda x: highlight_relationship({'Relationship': x})
                                )
                                
                                # Reorder columns: Type, From, To, Context, Relationship
                                contexts_df = contexts_df[['Type', 'From Citation', 'To Citation', 'Context', 'Relationship']]
                                
                                st.dataframe(contexts_df, hide_index=True, use_container_width=True)
                                
                                # Add relationship legend
                                with st.expander("📖 Relationship Types"):
                                    st.markdown("""
                                    - 🔵 **Builds On**: Second citation extends first
                                    - 🔴 **Conflicts**: Citations contradict each other  
                                    - 🟢 **Parallel**: Citations are comparable/similar
                                    - 🟡 **Supports**: Citations support same conclusion
                                    - ⚪ **Sequential**: No special relationship detected
                                    """)
                            else:
                                # Fallback to old 3-column format (regex-only mode)
                                contexts_df.columns = ["From Citation", "To Citation", "Context"]
                                st.dataframe(contexts_df, hide_index=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Structure Analysis with Progress Bar and Sections
        if settings["analyze_structure"]:
            with st.spinner("Analyzing structure..."):
                structure_metrics = self.structure_analyzer.analyze_structure(text)
                
                # Store in session state for export
                st.session_state.structure_results = structure_metrics
                
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
        # Performance Analysis (BLEU, METEOR, ROUGE)
        if settings["compare_models"]:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.subheader("🤖 Enhanced Model Analysis")
            
            # ⭐ STEP 1: RUN MODEL ANALYSIS (Always runs)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            estimated_tokens = self.model_analyzer.estimate_token_count(text)
            will_chunk = estimated_tokens > 3000
            
            if will_chunk:
                chunks_count = len(self.model_analyzer._chunk_text(text, max_length=1000))
                status_text.text(f"📄 Paper size: {estimated_tokens} tokens → splitting into {chunks_count} chunks")
            else:
                status_text.text(f"📄 Paper size: {estimated_tokens} tokens → processing as single chunk")
            
            import time
            time.sleep(1)
            
            model_names = list(self.model_analyzer.models.keys())
            total_models = len(model_names)
            
            # Process all models
            model_results = {}
            for idx, model_name in enumerate(model_names):
                progress_percentage = (idx) / total_models
                progress_bar.progress(progress_percentage)
                status_text.text(f"🤖 Analyzing with {model_name}... ({idx + 1}/{total_models})")
                
                result = self.model_analyzer.analyze_paper_single_model(
                    text, 
                    model_name, 
                    self.model_analyzer.models[model_name],
                    analysis_type="general", 
                    num_trials=1
                )
                model_results[model_name] = result
            
            progress_bar.progress(1.0)
            status_text.text("✅ Analysis complete!")
            time.sleep(0.5)
            
            progress_bar.empty()
            status_text.empty()
            
            # ✅ ALWAYS store model responses for export
            st.session_state.model_responses = model_results
            
            # Display basic performance metrics (always shown)
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
            
            # ⭐ STEP 2: OPTIONAL QUALITY METRICS (User choice)
            if settings["enable_quality_metrics"]:
                st.markdown("### 🔬 Detailed Quality Analysis")
                st.info("📊 Running comprehensive quality metrics (BLEU, ROUGE, Perplexity)...")
                
                with st.spinner("Loading quality analysis models..."):
                    self.performance_analyzer.initialize_quality_models()
                
                with st.spinner("Calculating quality metrics..."):
                    # Run FULL performance analysis with all quality metrics
                    performance_results = self.performance_analyzer.analyze_model_performance(
                        self.model_analyzer.models,
                        text[:1000],
                        num_runs=1
                    )
                    
                    # ✅ Store detailed performance results
                    st.session_state.performance_results = performance_results
                    
                    # Display detailed visualizations
                    figures = self.performance_analyzer.create_performance_visualizations(performance_results)
                    viz_tabs = st.tabs(["Resource Usage", "Quality Metrics", "Response Metrics"])
                    for tab, fig in zip(viz_tabs, figures):
                        with tab:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Display detailed report
                    st.markdown("### 📊 Detailed Performance Report")
                    report_df = self.performance_analyzer.generate_performance_report(performance_results)
                    st.dataframe(report_df, hide_index=True, use_container_width=True)
                    
                    csv = report_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Performance Report (CSV)",
                        csv,
                        "performance_report.csv",
                        "text/csv",
                        key='download-performance-csv'
                    )
            else:
                # ✅ QUALITY METRICS DISABLED: Create basic performance data for export
                # This ensures Excel export always has data to work with
                
                st.info("💡 Enable 'Detailed Quality Metrics' in the sidebar for BLEU, ROUGE, and Perplexity analysis")
                
                # Create simplified but complete performance structure
                simplified_results = {}
                for model_name, result in model_results.items():
                    # Extract what we have from basic analysis
                    simplified_results[model_name] = {
                        'resource_usage': {
                            'avg_cpu_percent': result.get('avg_memory_usage', 0) / 10 if isinstance(result, dict) else 0,
                            'peak_memory': result.get('avg_memory_usage', 0) if isinstance(result, dict) else 0,
                            'avg_thread_count': 1.0,
                            'avg_gpu_usage': 0.0
                        },
                        'quality_metrics': {
                            'avg_perplexity': 0.0,  # Not calculated without quality metrics
                            'avg_ngram_diversity': 0.0,  # Not calculated
                            'avg_coherence': result.get('consistency_score', 0) if isinstance(result, dict) else 0
                        },
                        'response_metrics': {
                            'avg_bleu': 0.0,  # Not calculated without quality metrics
                            'avg_meteor': 0.0,  # Not calculated
                            'avg_rouge1': 0.0,  # Not calculated
                            'avg_rouge2': 0.0,  # Not calculated
                            'avg_rougeL': 0.0,  # Not calculated
                            'avg_factual_consistency': 0.0  # Not calculated
                        }
                    }
                
                # ✅ Store simplified results (Excel export will work with this)
                st.session_state.performance_results = simplified_results
                
                # Show what's available
                st.markdown("**Available Metrics (without detailed quality analysis):**")
                st.markdown("- ✅ Response times")
                st.markdown("- ✅ Token counts") 
                st.markdown("- ✅ Memory usage")
                st.markdown("- ✅ Consistency scores")
                st.markdown("- ❌ BLEU, METEOR, ROUGE scores (enable quality metrics)")
                st.markdown("- ❌ Perplexity analysis (enable quality metrics)")
                    
    
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
        
        # Add Excel export section
        self.create_export_section()
        
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
                        # Store metadata for export
                        st.session_state.pdf_metadata = pdf_data["metadata"]
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