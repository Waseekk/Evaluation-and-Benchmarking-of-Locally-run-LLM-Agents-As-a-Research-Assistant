# research_paper_assistant_hybrid_citation.py
# Enhanced version using HybridSemanticCitationAnalyzer
"""
Enhanced Research Paper Analysis Assistant with Hybrid Semantic Citation Analysis

This version integrates the HybridSemanticCitationAnalyzer which provides:
- Semantic similarity analysis using sentence embeddings
- Citation stance detection (supporting, refuting, contrasting, neutral)
- Citation purpose classification (methodology, background, comparison, etc.)
- Enhanced co-occurrence and semantic relationship detection
- Better false positive filtering

Required dependencies for full hybrid features:
- sentence-transformers (pip install sentence-transformers)
- spacy (pip install spacy && python -m spacy download en_core_web_sm)
- sklearn (usually included with scikit-learn)
If dependencies are missing, the system automatically falls back to basic citation analysis.

Usage:
    streamlit run research_paper_assistant_hybrid_citation.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from typing import Dict
from datetime import datetime
from pdf_processor import PDFProcessor
# ✨ NEW: Using Hybrid Semantic Citation Analyzer
from citation_analyzer_hybrid import HybridSemanticCitationAnalyzer

#from structure_analyzer import StructureAnalyzer
from model_comparison_analyzer import ModelComparisonAnalyzer
from enhanced_chat_interface import EnhancedChatInterface
from performance_analyzer import PerformanceAnalyzer
#from excel_exporter import ExcelExporter

class ResearchPaperAssistantHybrid:
    """Enhanced main class for the Research Paper Analysis Assistant with Hybrid Citation Analysis."""

    def __init__(self):
        self.setup_page()
        self.initialize_components()
        self.setup_session_state()

    @staticmethod
    def apply_dark_mode_styles():
        """Apply dark mode CSS styles."""
        st.markdown("""
        <style>
            /* Dark Mode Styles */
            .stApp {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }

            /* Cards and containers */
            .metric-container, .metric-card {
                background-color: #2d2d2d !important;
                border-color: #404040 !important;
            }

            /* Text elements */
            h1, h2, h3, h4, h5, h6, p, span, div {
                color: #e0e0e0 !important;
            }

            /* Headers with specific styling */
            .model-response h1, .model-response h2, .model-response h3 {
                color: #58a6ff !important;
            }

            /* Links */
            a {
                color: #58a6ff !important;
            }

            /* Code blocks */
            code {
                background-color: #2d2d2d !important;
                color: #79c0ff !important;
            }

            /* Dataframes and tables */
            .dataframe {
                background-color: #2d2d2d !important;
                color: #e0e0e0 !important;
            }

            /* Input fields */
            .stTextInput input, .stTextArea textarea, .stSelectbox select {
                background-color: #2d2d2d !important;
                color: #e0e0e0 !important;
                border-color: #404040 !important;
            }

            /* Buttons */
            .stButton button {
                background-color: #3a3a3a !important;
                color: #e0e0e0 !important;
                border-color: #404040 !important;
            }

            .stButton button:hover {
                background-color: #4a4a4a !important;
                border-color: #58a6ff !important;
            }

            /* Sidebar */
            [data-testid="stSidebar"] {
                background-color: #252525 !important;
            }

            /* Expanders */
            .streamlit-expanderHeader {
                background-color: #2d2d2d !important;
                color: #e0e0e0 !important;
            }

            /* Metrics */
            [data-testid="stMetricValue"] {
                color: #e0e0e0 !important;
            }

            /* Dividers */
            hr {
                border-color: #404040 !important;
            }

            /* Chat messages */
            .stChatMessage {
                background-color: #2d2d2d !important;
            }

            /* Progress bars */
            .stProgress > div > div {
                background-color: #58a6ff !important;
            }

            /* Tabs */
            .stTabs [data-baseweb="tab"] {
                background-color: #2d2d2d !important;
                color: #e0e0e0 !important;
            }

            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                background-color: #3a3a3a !important;
                color: #58a6ff !important;
            }
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def beautify_model_response(response: str) -> str:
        """
        Clean and beautify model responses by removing metadata and formatting tags.

        Args:
            response: Raw model response string

        Returns:
            Cleaned and formatted response
        """
        import re

        if not isinstance(response, str):
            response = str(response)

        # Remove content=' or content=" wrapper and metadata (from LangChain/Ollama responses)
        # Try with single quotes first
        if "content='" in response:
            match = re.search(r"content='(.*?)' additional_kwargs=", response, re.DOTALL)
            if match:
                response = match.group(1)
        # Try with double quotes
        elif 'content="' in response:
            match = re.search(r'content="(.*?)" additional_kwargs=', response, re.DOTALL)
            if match:
                response = match.group(1)
            else:
                # Handle case where content=" is at the start without closing metadata
                match = re.search(r'content="(.*?)"\s*$', response, re.DOTALL)
                if match:
                    response = match.group(1)
                else:
                    # Just remove the content=" prefix if no pattern matches
                    response = re.sub(r'^content="', '', response)
                    response = re.sub(r'"\s*$', '', response)

        # Remove <think> tags and their content (from reasoning models)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)

        # Remove metadata patterns
        response = re.sub(r"additional_kwargs=\{.*?\}", '', response, flags=re.DOTALL)
        response = re.sub(r"response_metadata=\{.*?\}", '', response, flags=re.DOTALL)
        response = re.sub(r"usage_metadata=\{.*?\}", '', response, flags=re.DOTALL)
        response = re.sub(r"id='run-[^']*'", '', response)

        # Clean up excessive separators
        response = re.sub(r'-{3,}', '\n\n---\n\n', response)  # Convert to markdown HR
        response = re.sub(r'={3,}', '\n\n---\n\n', response)

        # Clean up excessive newlines
        response = re.sub(r'\n{4,}', '\n\n', response)

        # Remove escape characters
        response = response.replace('\\n', '\n')
        response = response.replace('\\t', '  ')

        # Fix common formatting issues
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)  # Max 2 newlines

        # Improve list formatting
        response = re.sub(r'\n-\s+', '\n- ', response)  # Standardize list spacing
        response = re.sub(r'\n\*\s+', '\n* ', response)

        # Strip leading/trailing whitespace
        response = response.strip()

        # If response is still messy (contains Python dict/object syntax), try to extract just text
        if "Message(" in response or "role=" in response:
            # Try to extract just the main content
            lines = []
            for line in response.split('\n'):
                # Skip lines with metadata
                if any(skip in line for skip in ['additional_kwargs', 'response_metadata', 'usage_metadata', 'Message(', 'role=']):
                    continue
                lines.append(line)
            response = '\n'.join(lines).strip()

        return response

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
            .stance-badge {
                display: inline-block;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 0.8em;
                font-weight: bold;
            }
            .stance-supporting { background-color: #d4edda; color: #155724; }
            .stance-refuting { background-color: #f8d7da; color: #721c24; }
            .stance-contrasting { background-color: #fff3cd; color: #856404; }
            .stance-neutral { background-color: #e2e3e5; color: #383d41; }

            /* Model response styling */
            .model-response {
                line-height: 1.6;
                font-size: 0.95rem;
            }
            .model-response h1, .model-response h2, .model-response h3 {
                color: #1f77b4;
                margin-top: 1em;
                margin-bottom: 0.5em;
            }
            .model-response p {
                margin-bottom: 0.8em;
            }
            .model-response ul, .model-response ol {
                margin-left: 1.5em;
                margin-bottom: 0.8em;
            }
            .model-response li {
                margin-bottom: 0.3em;
            }
            .model-response strong {
                color: #2c3e50;
                font-weight: 600;
            }
            .model-response hr {
                border: none;
                border-top: 1px solid #ddd;
                margin: 1em 0;
            }
            </style>
        """, unsafe_allow_html=True)

    def initialize_components(self):
        """Initializes enhanced analysis components with Hybrid Citation Analyzer."""
        self.pdf_processor = PDFProcessor()

        # ✨ NEW: Initialize Hybrid Semantic Citation Analyzer with error handling
        self.hybrid_mode = False
        try:
            self.citation_analyzer = HybridSemanticCitationAnalyzer(
                use_embeddings=True,  # Enable semantic similarity
                enable_spacy=True,     # Enable spaCy for better sentence detection
                log_file='citation_analysis.log'
            )
            self.hybrid_mode = True
        except Exception as e:
            # Fallback to basic analyzer if hybrid fails
            try:
                from citation_analyzer_nlp import EnhancedCitationAnalyzer
                self.citation_analyzer = EnhancedCitationAnalyzer()
            except:
                # Ultimate fallback
                from citation_analyzer import EnhancedCitationAnalyzer
                self.citation_analyzer = EnhancedCitationAnalyzer()

       # self.structure_analyzer = StructureAnalyzer()
        self.model_analyzer = ModelComparisonAnalyzer()
        self.chat_interface = EnhancedChatInterface()
        self.performance_analyzer = PerformanceAnalyzer()
        #self.excel_exporter = ExcelExporter()

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

        # ✨ NEW: Dark mode state
        if "dark_mode" not in st.session_state:
            st.session_state.dark_mode = False
        if "performance_results" not in st.session_state:
            st.session_state.performance_results = None

    def _check_dependencies(self) -> Dict[str, bool]:
        """Check availability of optional dependencies for hybrid mode."""
        dependencies = {}

        # Check sentence-transformers
        try:
            import sentence_transformers
            dependencies["sentence-transformers"] = True
        except ImportError:
            dependencies["sentence-transformers"] = False

        # Check spaCy
        try:
            import spacy
            # Try to load the model
            try:
                nlp = spacy.load("en_core_web_sm")
                dependencies["spacy (with en_core_web_sm)"] = True
            except:
                dependencies["spacy (with en_core_web_sm)"] = False
        except ImportError:
            dependencies["spacy (with en_core_web_sm)"] = False

        # Check sklearn (usually available)
        try:
            import sklearn
            dependencies["scikit-learn"] = True
        except ImportError:
            dependencies["scikit-learn"] = False

        return dependencies

    def create_sidebar(self):
        """Creates enhanced configuration sidebar."""
        with st.sidebar:
            st.header("📊 Analysis Settings")

            # # ✨ NEW: Dark Mode Toggle
            # st.markdown("### 🎨 Appearance")
            # dark_mode = st.toggle(
            #     "🌙 Dark Mode",
            #     value=st.session_state.dark_mode,
            #     help="Switch between light and dark theme"
            # )
            # if dark_mode != st.session_state.dark_mode:
            #     st.session_state.dark_mode = dark_mode
            #     st.rerun()

            # # Apply dark mode styles
            # if st.session_state.dark_mode:
            #     self.apply_dark_mode_styles()

            # st.divider()

            # # Display analyzer mode
            # if self.hybrid_mode:
            #     st.success("✓ Hybrid Semantic Analyzer Active")

            #     # Show dependency status in expander
            #     with st.expander("📦 Dependencies Status"):
            #         deps_status = self._check_dependencies()
            #         for dep_name, is_available in deps_status.items():
            #             if is_available:
            #                 st.markdown(f"✅ {dep_name}")
            #             else:
            #                 st.markdown(f"❌ {dep_name} (feature disabled)")
            # else:
            #     st.warning("⚠️ Using Basic Analyzer (Hybrid unavailable)")
            #     with st.expander("❓ Why Basic Mode?"):
            #         st.markdown("**Missing dependencies for hybrid mode:**")
            #         deps_status = self._check_dependencies()
            #         for dep_name, is_available in deps_status.items():
            #             if not is_available:
            #                 st.markdown(f"❌ {dep_name}")

            #         st.markdown("\n**To enable hybrid mode, install:**")
            #         st.code("pip install sentence-transformers\npip install spacy\npython -m spacy download en_core_web_sm")

            # # ✨ NEW: Hybrid Citation Analysis Settings
            # st.markdown("### 🔬 Citation Analysis Features")
            # enable_semantic = st.checkbox("Semantic Similarity", value=True,
            #     help="Use ML embeddings to detect semantically related citations")
            # enable_stance = st.checkbox("Stance Detection", value=True,
            #     help="Detect whether citations are supporting, refuting, or contrasting")
            # enable_purpose = st.checkbox("Purpose Classification", value=True,
            #     help="Classify citation purposes (methodology, background, comparison, etc.)")

            # Set default values for commented-out options
            enable_semantic = True
            enable_stance = True
            enable_purpose = True

            analysis_depth = st.slider(
                "Analysis Depth",
                min_value=1,
                max_value=5,
                value=3,
                help="Controls how detailed the analysis should be"
            )
            st.divider()
            # st.markdown("### 🎯 Analysis Focus")
            # analyze_citations = st.checkbox("Citation Analysis", value=True)
            # compare_models = st.checkbox("Model Comparison", value=True)

            # Set default values for Analysis Focus (always enabled)
            analyze_citations = True
            compare_models = True

            # ✨ NEW: Analysis Type Selector
            if compare_models:
                st.markdown("**Model Analysis Type:**")
                analysis_type = st.radio(
                    "Choose which prompt to use:",
                    options=["general", "methodology", "results"],
                    format_func=lambda x: {
                        "general": "📊 General Analysis",
                        "methodology": "🔬 Methodology Focus",
                        "results": "📈 Results Focus"
                    }[x],
                    help="Select which analysis prompt the models should use",
                    horizontal=False
                )
            else:
                analysis_type = "general"

            st.divider()
            # st.markdown("### 📈 Visualization Options")
            # show_citation_graph = st.checkbox("Show Citation Year Distribution", value=True)
            # show_citation_network = st.checkbox("Show Citation Network Graph", value=True)
            # st.divider()

            # Set default values for Visualization Options (always enabled)
            show_citation_graph = True
            show_citation_network = True

            # # ✨ NEW: Model comparison view option
            # if compare_models:
            #     st.markdown("### 🔍 Model Comparison View")
            #     comparison_view = st.radio(
            #         "Display mode:",
            #         options=["grid", "side-by-side"],
            #         format_func=lambda x: "📊 Grid View (2x2)" if x == "grid" else "⚖️ Side-by-Side",
            #         help="Choose how to display model comparisons"
            #     )
            # else:
            #     comparison_view = "grid"

            # Set default value for comparison view
            comparison_view = "grid"

            # st.divider()
            st.markdown("### 🔬 Advanced Metrics")
            enable_quality_metrics = st.checkbox(
                "📊 Detailed Quality Metrics",
                value=False,
                help="Enable BLEU, ROUGE, Perplexity analysis (adds ~20 seconds)"
            )
            return {
                "depth": analysis_depth,
                "analyze_citations": analyze_citations,
                "compare_models": compare_models,
                "analysis_type": analysis_type,
                "comparison_view": comparison_view,
                "show_citation_graph": show_citation_graph,
                "show_citation_network": show_citation_network,
                "enable_quality_metrics": enable_quality_metrics,
                "enable_semantic": enable_semantic,
                "enable_stance": enable_stance,
                "enable_purpose": enable_purpose
            }

    def create_export_section(self):
        """Creates a section for exporting all analysis data to Excel."""
        # COMMENTED OUT - Export section moved from sidebar
        pass
        # st.sidebar.divider()
        # st.sidebar.markdown("### 📥 Export Data")

        # has_citation = st.session_state.citation_results is not None
        # has_structure = st.session_state.structure_results is not None
        # has_models = bool(st.session_state.model_responses)

        # has_data = any([has_citation, has_structure, has_models])
        # has_performance = bool(st.session_state.performance_results)

        # if has_data:
        #     st.sidebar.success("✓ Analysis data available")

        #     with st.sidebar.expander("📋 Available Data"):
        #         st.write("✅ Citations" if has_citation else "❌ Citations")
        #         st.write("✅ Structure" if has_structure else "❌ Structure")
        #         st.write("✅ Model Comparisons" if has_models else "❌ Model Comparisons")

        #         if has_performance:
        #             sample_model = list(st.session_state.performance_results.keys())[0]
        #             sample_data = st.session_state.performance_results[sample_model]
        #             has_quality = sample_data['response_metrics']['avg_bleu'] > 0

        #             if has_quality:
        #                 st.write("✅ Performance Metrics (with BLEU/ROUGE/Perplexity)")
        #             else:
        #                 st.write("✅ Performance Metrics (basic only)")
        #         else:
        #             st.write("⚠️ Performance Metrics (not calculated)")

        #     with st.sidebar.expander("📑 Excel Sheets Preview"):
        #         sheet_summary = self.excel_exporter.get_sheet_summary()
        #         for sheet_name, description in list(sheet_summary.items())[:5]:
        #             st.caption(f"**{sheet_name}**: {description}")
        #         st.caption(f"*...and {len(sheet_summary) - 5} more sheets*")

        #     if st.sidebar.button("📊 Export All Data to Excel", type="primary"):
        #         with st.spinner("Generating Excel file..."):
        #             try:
        #                 excel_buffer = self.excel_exporter.create_comprehensive_export(
        #                     citation_data=st.session_state.citation_results,
        #                     structure_data=st.session_state.structure_results,
        #                     model_results=st.session_state.model_responses,
        #                     performance_results=st.session_state.performance_results,
        #                     paper_metadata=st.session_state.pdf_metadata
        #                 )

        #                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #                 filename = f"paper_analysis_hybrid_{timestamp}.xlsx"

        #                 st.sidebar.download_button(
        #                     label="⬇️ Download Excel File",
        #                     data=excel_buffer,
        #                     file_name=filename,
        #                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        #                     key="download_excel"
        #                 )

        #                 sheet_count = len(self.excel_exporter.get_sheet_summary())
        #                 if has_performance:
        #                     st.sidebar.success(f"✅ Excel ready! ({sheet_count} sheets, includes performance metrics)")
        #                 else:
        #                     st.sidebar.success(f"✅ Excel ready! ({sheet_count} sheets)")
        #                     st.sidebar.info("💡 Enable 'Detailed Quality Metrics' for BLEU/ROUGE analysis")

        #             except Exception as e:
        #                 st.sidebar.error(f"Error generating Excel: {str(e)}")
        #                 st.sidebar.exception(e)
        # else:
        #     st.sidebar.info("Run an analysis first to enable export")
        #     st.sidebar.markdown("**Steps:**")
        #     st.sidebar.markdown("1. Upload a PDF or paste text")
        #     st.sidebar.markdown("2. Enable analysis options")
        #     st.sidebar.markdown("3. Click 'Analyze Paper'")
        #     st.sidebar.markdown("4. Export button will appear here")

    def display_pdf_metadata(self, metadata: Dict):
        """Displays enhanced PDF metadata with OCR info, sections, and content lists."""
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

        # Rest of the PDF metadata display (OCR info, structure, etc.)
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

            if ocr_conf < 0.7:
                st.warning(
                    "⚠️ OCR confidence is below 70%. Text extraction quality may be lower than expected. "
                    "Consider using a higher quality scan for better results."
                )

            st.markdown('</div>', unsafe_allow_html=True)

    def display_citation_stance_purpose(self, citation_metrics: Dict, settings: Dict):
        """✨ NEW: Display stance and purpose distributions."""

        if not settings.get("enable_stance") and not settings.get("enable_purpose"):
            return

        st.markdown("### 🎯 Citation Analysis: Stance & Purpose")

        col1, col2 = st.columns(2)

        # Stance Distribution
        if settings.get("enable_stance") and "stance_distribution" in citation_metrics:
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("**Stance Distribution**")

                stance_dist = citation_metrics["stance_distribution"]
                if stance_dist:
                    # Create pie chart
                    fig = go.Figure(data=[go.Pie(
                        labels=list(stance_dist.keys()),
                        values=list(stance_dist.values()),
                        marker=dict(colors=['#28a745', '#dc3545', '#ffc107', '#6c757d'])
                    )])
                    fig.update_layout(
                        showlegend=True,
                        height=300,
                        margin=dict(t=0, b=0, l=0, r=0),
                        paper_bgcolor='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Show counts
                    for stance, count in sorted(stance_dist.items(), key=lambda x: x[1], reverse=True):
                        st.markdown(f"- **{stance.capitalize()}**: {count}")
                else:
                    st.info("No stance data available")

                st.markdown('</div>', unsafe_allow_html=True)

        # Purpose Distribution
        if settings.get("enable_purpose") and "purpose_distribution" in citation_metrics:
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("**Purpose Distribution**")

                purpose_dist = citation_metrics["purpose_distribution"]
                if purpose_dist:
                    # Create bar chart
                    purposes = list(purpose_dist.keys())
                    counts = list(purpose_dist.values())

                    fig = go.Figure(data=[go.Bar(
                        x=purposes,
                        y=counts,
                        marker_color='#3B82F6'
                    )])
                    fig.update_layout(
                        showlegend=False,
                        height=300,
                        xaxis_title="Purpose",
                        yaxis_title="Count",
                        margin=dict(t=20, b=40, l=40, r=20),
                        paper_bgcolor='white',
                        plot_bgcolor='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Show counts
                    for purpose, count in sorted(purpose_dist.items(), key=lambda x: x[1], reverse=True):
                        st.markdown(f"- **{purpose.capitalize()}**: {count}")
                else:
                    st.info("No purpose data available")

                st.markdown('</div>', unsafe_allow_html=True)

    def display_analysis_results(self, text: str, settings: Dict):
        """Displays enhanced analysis results with hybrid citation features."""

        # ✨ ENHANCED Citation Analysis with Hybrid Analyzer
        if settings["analyze_citations"]:
            with st.spinner("Analyzing citations with hybrid semantic analysis..."):
                citation_metrics = self.citation_analyzer.extract_citations(text)

                # Store in session state for export
                st.session_state.citation_results = citation_metrics

                st.markdown('<div class="metric-container">', unsafe_allow_html=True)

                # Display title based on analyzer mode
                if self.hybrid_mode:
                    st.subheader("📚 Hybrid Citation Analysis")
                else:
                    st.subheader("📚 Citation Analysis")

                # Display enhanced info banner (if hybrid features available)
                if 'features' in citation_metrics:
                    features_enabled = []
                    if citation_metrics.get('features', {}).get('embeddings'):
                        features_enabled.append("Semantic Similarity")
                    if citation_metrics.get('features', {}).get('spacy'):
                        features_enabled.append("Advanced NLP")
                    if citation_metrics.get('features', {}).get('false_positive_filtering'):
                        features_enabled.append("False Positive Filtering")

                    method = citation_metrics.get('method', 'standard')
                    if features_enabled:
                        st.info(f"🔬 **Method**: {method} | **Features**: {', '.join(features_enabled)}")
                    else:
                        st.info(f"🔬 **Method**: {method}")

                # Display citation count metrics
                cols = st.columns(5)

                # Build metrics list with safe access
                network_edges = 0
                if "network" in citation_metrics and "metrics" in citation_metrics["network"]:
                    network_edges = citation_metrics["network"]["metrics"].get("edge_count", 0)

                metrics = [
                    ("Total Citations", citation_metrics.get("total_count", 0)),
                    ("Unique Citations", citation_metrics.get("unique_count", 0)),
                    ("Numbered", citation_metrics.get("numbered_count", 0)),
                    ("Author-Year", citation_metrics.get("author_year_count", 0)),
                    ("Network Edges", network_edges)
                ]
                for col, (label, value) in zip(cols, metrics):
                    with col:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(label, value)
                        st.markdown('</div>', unsafe_allow_html=True)

                # ✨ NEW: Display Stance and Purpose
                self.display_citation_stance_purpose(citation_metrics, settings)

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

                # ✨ ENHANCED Citation Network Analysis
                if "network" in citation_metrics:
                    network_data = citation_metrics["network"]
                    st.markdown("### 🔄 Citation Network Analysis")

                    # Display network metrics
                    metric_cols = st.columns(5)
                    with metric_cols[0]:
                        st.metric("Total Citations", network_data["metrics"]["node_count"])
                    with metric_cols[1]:
                        st.metric("Citation Links", network_data["metrics"]["edge_count"])
                    with metric_cols[2]:
                        st.metric("Network Density", f"{network_data['metrics']['density']:.3f}")
                    with metric_cols[3]:
                        st.metric("Avg. Citations/Paper", f"{network_data['metrics']['average_degree']:.2f}")
                    with metric_cols[4]:
                        communities = network_data["metrics"].get("num_communities", 0)
                        st.metric("Topic Clusters", communities)

                    # Display network insights
                    if network_data["metrics"].get("top_citations"):
                        st.markdown("#### ⭐ Most Connected Citations")
                        top_citations_df = pd.DataFrame(network_data["metrics"]["top_citations"])
                        st.dataframe(top_citations_df, hide_index=True, use_container_width=True)

                    # Create network visualization
                    if settings["show_citation_network"] and network_data["nodes"] and network_data["edges"]:
                        st.markdown("#### 🌐 Citation Network Graph")

                        G = nx.Graph()
                        for edge in network_data["edges"]:
                            G.add_edge(edge["source"], edge["target"],
                                      weight=edge.get("weight", 1),
                                      relationship=edge.get("relationship", "unknown"))

                        if len(G.nodes()) > 0:
                            pos = nx.spring_layout(G, k=1, iterations=50)

                            network_fig = go.Figure()

                            # Add edges
                            edge_x, edge_y = [], []
                            for edge in network_data["edges"]:
                                if edge["source"] in pos and edge["target"] in pos:
                                    x0, y0 = pos[edge["source"]]
                                    x1, y1 = pos[edge["target"]]
                                    edge_x.extend([x0, x1, None])
                                    edge_y.extend([y0, y1, None])

                            network_fig.add_trace(go.Scatter(
                                x=edge_x, y=edge_y,
                                line=dict(width=1, color='#888'),
                                hoverinfo='none',
                                mode='lines',
                                name='Citation Links'
                            ))

                            # Add nodes
                            node_x = [pos[node][0] for node in G.nodes() if node in pos]
                            node_y = [pos[node][1] for node in G.nodes() if node in pos]
                            node_text = [str(node) for node in G.nodes() if node in pos]

                            network_fig.add_trace(go.Scatter(
                                x=node_x, y=node_y,
                                mode='markers+text',
                                hoverinfo='text',
                                marker=dict(size=15, color='#1f77b4', line_width=2),
                                text=node_text,
                                textposition="top center",
                                textfont=dict(size=8),
                                name='Citations'
                            ))

                            network_fig.update_layout(
                                title="Citation Network Visualization",
                                showlegend=True,
                                hovermode='closest',
                                height=600,
                                margin=dict(b=20, l=5, r=5, t=40),
                                paper_bgcolor="white",
                                plot_bgcolor="white",
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                            )
                            st.plotly_chart(network_fig, use_container_width=True)
                        else:
                            st.info("Not enough citation relationships to visualize network")

                    # ✨ NEW: Display Citation Relationship Details
                    if network_data.get("edges"):
                        st.markdown("#### 📊 Citation Relationships")

                        edges_df = pd.DataFrame([
                            {
                                "From": edge["source"],
                                "To": edge["target"],
                                "Relationship": edge.get("relationship", "unknown"),
                                "Weight": edge.get("weight", 1),
                                "Similarity": f"{edge['similarity']:.2f}" if edge.get("similarity") else "N/A",
                                "Source Stance": edge.get("source_stance", "unknown"),
                                "Target Stance": edge.get("target_stance", "unknown"),
                                "Source Purpose": edge.get("source_purpose", "unknown"),
                                "Target Purpose": edge.get("target_purpose", "unknown")
                            }
                            for edge in network_data["edges"][:50]  # Limit to first 50
                        ])

                        if not edges_df.empty:
                            st.dataframe(edges_df, hide_index=True, use_container_width=True)

                            if len(network_data["edges"]) > 50:
                                st.caption(f"Showing first 50 of {len(network_data['edges'])} relationships")

                # Display detailed citation information
                if citation_metrics.get("citation_details"):
                    with st.expander("📑 Detailed Citation Information"):
                        details_df = pd.DataFrame(citation_metrics["citation_details"])

                        # Select relevant columns for display
                        display_cols = ['citation', 'type', 'stance', 'purpose', 'context']
                        available_cols = [col for col in display_cols if col in details_df.columns]

                        if available_cols:
                            st.dataframe(details_df[available_cols], hide_index=True, use_container_width=True)

                # Display performance timing (if available from hybrid analyzer)
                if citation_metrics.get("timing"):
                    with st.expander("⏱️ Analysis Performance"):
                        timing_df = pd.DataFrame([
                            {"Stage": k.replace("_", " ").title(), "Duration (s)": f"{v:.3f}"}
                            for k, v in citation_metrics.get("timing", {}).items()
                        ])
                        if not timing_df.empty:
                            st.dataframe(timing_df, hide_index=True)
                            total_time = citation_metrics["timing"].get("total_time", 0)
                            st.caption(f"Total analysis time: {total_time:.3f}s")

                st.markdown('</div>', unsafe_allow_html=True)

        # Model Comparison with Custom Prompt
        if settings["compare_models"]:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.subheader("🤖 Enhanced Model Analysis")

            # Display the prompt being used
            analysis_type = settings.get("analysis_type", "general")
            type_labels = {
                "general": "📊 General Analysis",
                "methodology": "🔬 Methodology Focus",
                "results": "📈 Results Focus"
            }

            with st.expander(f"📝 Using Prompt: {type_labels[analysis_type]}", expanded=False):
                st.info(f"**Active prompt type:** {type_labels[analysis_type]}")

                if analysis_type == "general":
                    st.markdown("**General Analysis Prompt:**")
                    st.code(settings.get("prompt_general", "Default prompt"), language="markdown")
                elif analysis_type == "methodology":
                    st.markdown("**Methodology Focus Prompt:**")
                    st.code(settings.get("prompt_methodology", "Default prompt"), language="markdown")
                elif analysis_type == "results":
                    st.markdown("**Results Focus Prompt:**")
                    st.code(settings.get("prompt_results", "Default prompt"), language="markdown")

            # Show all 3 prompts in a separate expander (not nested)
            with st.expander("📋 View All Available Prompts", expanded=False):
                st.markdown("**📊 General Analysis:**")
                st.code(settings.get("prompt_general", "Default"), language="markdown")
                st.divider()
                st.markdown("**🔬 Methodology Focus:**")
                st.code(settings.get("prompt_methodology", "Default"), language="markdown")
                st.divider()
                st.markdown("**📈 Results Focus:**")
                st.code(settings.get("prompt_results", "Default"), language="markdown")

            progress_bar = st.progress(0)
            status_text = st.empty()

            estimated_tokens = self.model_analyzer.estimate_token_count(text)
            will_use_two_pass = estimated_tokens > 3000

            if will_use_two_pass:
                chunks_count = len(self.model_analyzer._chunk_text(text, max_length=1000))
                status_text.text(f"📄 Paper size: {estimated_tokens} tokens → using two-pass analysis ({chunks_count} chunks)")
            else:
                status_text.text(f"📄 Paper size: {estimated_tokens} tokens → single-pass analysis")

            import time
            time.sleep(1)

            model_names = list(self.model_analyzer.model_configs.keys())
            total_models = len(model_names)

            def update_progress(status, progress):
                status_text.text(status)
                progress_bar.progress(min(progress, 1.0))

            model_results = {}
            for idx, model_name in enumerate(model_names):
                base_progress = idx / total_models
                progress_range = 1.0 / total_models

                def model_progress_callback(status, sub_progress):
                    overall_progress = base_progress + (sub_progress * progress_range)
                    status_text.text(f"🤖 {model_name}: {status}")
                    progress_bar.progress(overall_progress)

                model = self.model_analyzer.get_model(model_name)

                # Create custom prompts dictionary
                custom_prompts = {
                    "general": settings.get("prompt_general"),
                    "methodology": settings.get("prompt_methodology"),
                    "results": settings.get("prompt_results")
                }

                result = self.model_analyzer.analyze_paper_single_model(
                    text,
                    model_name,
                    model,
                    analysis_type=settings.get("analysis_type", "general"),  # ✨ Use selected type
                    num_trials=1,
                    progress_callback=model_progress_callback,
                    custom_prompts=custom_prompts  # ✨ Pass all 3 custom prompts
                )
                model_results[model_name] = result

            progress_bar.progress(1.0)
            status_text.text("✅ All models analyzed!")
            time.sleep(0.5)

            progress_bar.empty()
            status_text.empty()

            st.session_state.model_responses = model_results

            st.markdown("### 📊 Model Performance Metrics")
            metrics_df = self.model_analyzer.generate_performance_report(model_results)
            st.dataframe(metrics_df, hide_index=True)

            visualizations = self.model_analyzer.create_performance_visualizations(model_results)
            viz_tabs = st.tabs(["Response Times", "Token Counts", "Consistency Scores"])
            for tab, fig in zip(viz_tabs, visualizations):
                with tab:
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 📝 Model Analyses")

            # ✨ Display models based on selected view
            comparison_view = settings.get("comparison_view", "grid")

            if comparison_view == "side-by-side":
                # Side-by-side comparison view
                st.caption("💡 Tip: Scroll horizontally to compare all models")

                # Model selector for side-by-side
                col_left, col_right = st.columns(2)

                with col_left:
                    model_left = st.selectbox(
                        "Left model:",
                        options=["deepseek-1.5b", "deepseek-8b", "mistral", "llama3-8b"],
                        format_func=lambda x: {
                            "deepseek-1.5b": "Deepseek 1.5B",
                            "deepseek-8b": "Deepseek 8B",
                            "mistral": "Mistral",
                            "llama3-8b": "LLaMA3 8B"
                        }[x],
                        key="model_left"
                    )

                with col_right:
                    model_right = st.selectbox(
                        "Right model:",
                        options=["deepseek-1.5b", "deepseek-8b", "mistral", "llama3-8b"],
                        format_func=lambda x: {
                            "deepseek-1.5b": "Deepseek 1.5B",
                            "deepseek-8b": "Deepseek 8B",
                            "mistral": "Mistral",
                            "llama3-8b": "LLaMA3 8B"
                        }[x],
                        index=1,
                        key="model_right"
                    )

                # Display selected models side-by-side
                col1, col2 = st.columns(2)

                models_to_compare = [
                    (col1, model_left),
                    (col2, model_right)
                ]

                for col, model_name in models_to_compare:
                    with col:
                        model_titles = {
                            "deepseek-1.5b": "Deepseek 1.5B",
                            "deepseek-8b": "Deepseek 8B",
                            "mistral": "Mistral",
                            "llama3-8b": "LLaMA3 8B"
                        }

                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown(f"### {model_titles[model_name]}")

                        raw_response = model_results.get(model_name, {}).get('response', f"Error analyzing with {model_name}")
                        cleaned_response = self.beautify_model_response(raw_response)

                        # Add scrollable container for long responses
                        st.markdown(f'<div class="model-response" style="max-height: 600px; overflow-y: auto;">{cleaned_response}</div>', unsafe_allow_html=True)

                        # Show response metrics
                        response_time = model_results.get(model_name, {}).get('response_times', [0])[0]
                        st.caption(f"⏱️ Response time: {response_time:.2f}s")

                        st.markdown('</div>', unsafe_allow_html=True)

                # Add "Compare All" expander
                with st.expander("🔍 View All 4 Models"):
                    all_models = [
                        ("deepseek-1.5b", "Deepseek 1.5B"),
                        ("deepseek-8b", "Deepseek 8B"),
                        ("mistral", "Mistral"),
                        ("llama3-8b", "LLaMA3 8B")
                    ]

                    for model_name, title in all_models:
                        st.markdown(f"**{title}**")
                        raw_response = model_results.get(model_name, {}).get('response', f"Error analyzing")
                        cleaned_response = self.beautify_model_response(raw_response)
                        with st.container():
                            st.markdown(cleaned_response)
                        st.divider()

            else:
                # Grid view (original 2x2 layout)
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
                        raw_response = model_results.get(model_name, {}).get('response', f"Error analyzing with {model_name}")

                        # ✨ Beautify the response
                        cleaned_response = self.beautify_model_response(raw_response)

                        # Display with better formatting and styling
                        st.markdown(f'<div class="model-response">{cleaned_response}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            if settings["enable_quality_metrics"]:
                st.markdown("### 🔬 Detailed Quality Analysis")
                st.info("📊 Running comprehensive quality metrics (BLEU, ROUGE, Perplexity)...")

                with st.spinner("Loading quality analysis models..."):
                    self.performance_analyzer.initialize_quality_models()

                with st.spinner("Calculating quality metrics..."):
                    models_dict = {name: self.model_analyzer.get_model(name)
                                   for name in self.model_analyzer.model_configs.keys()}

                    performance_results = self.performance_analyzer.analyze_model_performance(
                        models_dict,
                        text[:1000],
                        num_runs=1
                    )

                    st.session_state.performance_results = performance_results

                    figures = self.performance_analyzer.create_performance_visualizations(performance_results)
                    viz_tabs = st.tabs(["Resource Usage", "Quality Metrics", "Response Metrics"])
                    for tab, fig in zip(viz_tabs, figures):
                        with tab:
                            st.plotly_chart(fig, use_container_width=True)

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
                st.info("💡 Enable 'Detailed Quality Metrics' in the sidebar for BLEU, ROUGE, and Perplexity analysis")

                simplified_results = {}
                for model_name, result in model_results.items():
                    simplified_results[model_name] = {
                        'resource_usage': {
                            'avg_cpu_percent': result.get('avg_memory_usage', 0) / 10 if isinstance(result, dict) else 0,
                            'peak_memory': result.get('avg_memory_usage', 0) if isinstance(result, dict) else 0,
                            'avg_thread_count': 1.0,
                            'avg_gpu_usage': 0.0
                        },
                        'quality_metrics': {
                            'avg_perplexity': 0.0,
                            'avg_ngram_diversity': 0.0,
                            'avg_coherence': result.get('consistency_score', 0) if isinstance(result, dict) else 0
                        },
                        'response_metrics': {
                            'avg_bleu': 0.0,
                            'avg_meteor': 0.0,
                            'avg_rouge1': 0.0,
                            'avg_rouge2': 0.0,
                            'avg_rougeL': 0.0,
                            'avg_factual_consistency': 0.0
                        }
                    }

                st.session_state.performance_results = simplified_results

                st.markdown("**Available Metrics (without detailed quality analysis):**")
                st.markdown("- ✅ Response times")
                st.markdown("- ✅ Token counts")
                st.markdown("- ✅ Memory usage")
                st.markdown("- ✅ Consistency scores")
                st.markdown("- ❌ BLEU, METEOR, ROUGE scores (enable quality metrics)")
                st.markdown("- ❌ Perplexity analysis (enable quality metrics)")

    def run(self):
        """Enhanced main application loop with improved UI."""
        st.title("📚 Research Paper Analysis Assistant (Hybrid Citation)")
        st.caption("Advanced PDF Analysis with Hybrid Semantic Citation Analysis, Deepseek, Llama and Mistral models")

        settings = self.create_sidebar()
        selected_model = st.sidebar.selectbox(
            "Select Chat Model",
            ["deepseek-1.5b", "deepseek-8b", "mistral", "llama3-8b"],
            help="Choose which model to use for chat interactions"
        )

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
                        st.session_state.pdf_metadata = pdf_data["metadata"]
                    text_input = pdf_data.get("text", "")
                    if text_input:
                        st.session_state.paper_content = text_input
                        self.chat_interface.initialize_rag(text_input)

            # ✨ Model Analysis Prompts Editor (on main page) - All 3 prompt types
            st.divider()
            st.markdown("### 📝 Model Analysis Prompts")

            # Show which prompt is selected
            selected_type = settings.get("analysis_type", "general")
            type_labels = {
                "general": "📊 General Analysis",
                "methodology": "🔬 Methodology Focus",
                "results": "📈 Results Focus"
            }
            if settings.get("compare_models", False):
                st.info(f"✓ **Currently selected:** {type_labels[selected_type]} (change in sidebar)")

            st.caption("Edit any of the 3 prompts below. The selected prompt type (from sidebar) will be used for model analysis.")

            # Initialize default prompts in session state if not exists
            if "prompt_general" not in st.session_state:
                st.session_state.prompt_general = """Analyze this research paper systematically:

1. **Main Topic & Objective**: What research question does this address? 
2. **Key Methodology**: What approach/methods were used? Include specific techniques. 
3. **Major Findings**: What were the main results? Include metrics or data if present. 
4. **Strengths**: What does the paper do well? ( specific points with evidence)
5. **Areas for Improvement**: What could be enhanced? (specific points)

Cite specific evidence from the paper."""

            if "prompt_methodology" not in st.session_state:
                st.session_state.prompt_methodology = """Analyze the methodology in detail:

1. **Approach Overview**: Summarize main methods and experimental design
2. **Rigor & Validity**: Evaluate methodological soundness
3. **Data & Analysis**: Assess data collection and analytical techniques
4. **Reproducibility**: How clearly are methods described?
5. **Limitations**: Identify gaps or weaknesses

Provide evidence-based critique with examples."""

            if "prompt_results" not in st.session_state:
                st.session_state.prompt_results = """Analyze the results comprehensively:

1. **Key Findings**: Summarize main results with specific metrics
2. **Presentation Quality**: Evaluate tables, figures, data visualization
3. **Statistical Rigor**: Assess validity of statistical methods
4. **Results vs Claims**: Do results support conclusions?
5. **Completeness**: What additional analysis would help?

Focus on specific data points and quantitative measures."""

            # Create tabs for different prompt types
            prompt_tabs = st.tabs(["📊 General Analysis", "🔬 Methodology Focus", "📈 Results Focus"])

            with prompt_tabs[0]:
                st.caption("This is the default prompt used for overall paper analysis.")
                col1, col2 = st.columns([4, 1])
                with col1:
                    prompt_general = st.text_area(
                        "General Analysis Prompt:",
                        value=st.session_state.prompt_general,
                        height=220,
                        help="Prompt for overall paper analysis",
                        key="prompt_general_editor"
                    )
                    st.session_state.prompt_general = prompt_general
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🔄 Reset", help="Reset general prompt to default", key="reset_general"):
                        st.session_state.prompt_general = """Analyze this research paper systematically:

1. **Main Topic & Objective**: What research question does this address? (2-3 sentences)
2. **Key Methodology**: What approach/methods were used? Include specific techniques. (3-4 sentences)
3. **Major Findings**: What were the main results? Include metrics or data if present. (3-4 sentences)
4. **Strengths**: What does the paper do well? (2-3 specific points with evidence)
5. **Areas for Improvement**: What could be enhanced? (2-3 specific points)

Cite specific evidence from the paper."""
                        st.rerun()

            with prompt_tabs[1]:
                st.caption("Focused on evaluating research methods and experimental design.")
                col1, col2 = st.columns([4, 1])
                with col1:
                    prompt_methodology = st.text_area(
                        "Methodology Focus Prompt:",
                        value=st.session_state.prompt_methodology,
                        height=220,
                        help="Prompt for methodology-focused analysis",
                        key="prompt_methodology_editor"
                    )
                    st.session_state.prompt_methodology = prompt_methodology
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🔄 Reset", help="Reset methodology prompt to default", key="reset_methodology"):
                        st.session_state.prompt_methodology = """Analyze the methodology in detail:

1. **Approach Overview**: Summarize main methods and experimental design
2. **Rigor & Validity**: Evaluate methodological soundness
3. **Data & Analysis**: Assess data collection and analytical techniques
4. **Reproducibility**: How clearly are methods described?
5. **Limitations**: Identify gaps or weaknesses

Provide evidence-based critique with examples."""
                        st.rerun()

            with prompt_tabs[2]:
                st.caption("Focused on research findings, data, and statistical analysis.")
                col1, col2 = st.columns([4, 1])
                with col1:
                    prompt_results = st.text_area(
                        "Results Focus Prompt:",
                        value=st.session_state.prompt_results,
                        height=220,
                        help="Prompt for results-focused analysis",
                        key="prompt_results_editor"
                    )
                    st.session_state.prompt_results = prompt_results
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🔄 Reset", help="Reset results prompt to default", key="reset_results"):
                        st.session_state.prompt_results = """Analyze the results comprehensively:

1. **Key Findings**: Summarize main results with specific metrics
2. **Presentation Quality**: Evaluate tables, figures, data visualization
3. **Statistical Rigor**: Assess validity of statistical methods
4. **Results vs Claims**: Do results support conclusions?
5. **Completeness**: What additional analysis would help?

Focus on specific data points and quantitative measures."""
                        st.rerun()

            st.divider()

            if st.button("Analyze Paper", type="primary"):
                if text_input:
                    # Add all 3 custom prompts to settings
                    settings["prompt_general"] = st.session_state.prompt_general
                    settings["prompt_methodology"] = st.session_state.prompt_methodology
                    settings["prompt_results"] = st.session_state.prompt_results
                    with st.spinner("Analyzing paper with hybrid semantic methods..."):
                        self.display_analysis_results(text_input, settings)
                        st.session_state.paper_content = text_input
                else:
                    st.warning("Please upload a PDF or enter text to analyze.")

        with tab2:
            if st.session_state.paper_content:
                self.chat_interface.display_chat_interface(selected_model)
            else:
                st.info("Please upload a paper or enter text in the Analysis tab first.")

if __name__ == "__main__":
    app = ResearchPaperAssistantHybrid()
    app.run()
