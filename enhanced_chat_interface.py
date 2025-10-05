import streamlit as st
import logging
from typing import Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from rag_processor import RAGProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedChatInterface:
    """Enhanced chat interface with RAG support."""
    
    def __init__(self, config=None):
        # Use config or defaults
        self.config = config
        self._loaded_models = {}
        self.rag_processor = RAGProcessor()
        self.rag_initialized = False
        
        # Model configurations
        self.model_configs = {
            "deepseek-1.5b": {"model": "deepseek-r1:1.5b", "temperature": 0.3},
            "deepseek-8b": {"model": "deepseek-r1:8b", "temperature": 0.3},
            "mistral": {"model": "mistral", "temperature": 0.3},
            "llama3-8b": {"model": "llama3:8b", "temperature": 0.3}
        }

    def get_model(self, model_name: str) -> ChatOllama:
        """Lazy load models on demand."""
        if model_name not in self._loaded_models:
            if model_name not in self.model_configs:
                raise ValueError(f"Unknown model: {model_name}")
            
            config = self.model_configs[model_name]
            self._loaded_models[model_name] = ChatOllama(
                model=config["model"],
                temperature=config["temperature"],
                base_url="http://localhost:11434"
            )
            logger.info(f"Loaded model: {model_name}")
        
        return self._loaded_models[model_name]

    def initialize_chat_state(self):
        """Initialize session state variables."""
        defaults = {
            "chat_history": [],
            "paper_content": None,
            "chat_summary": None,
            "key_points": [],
            "chat_mode": "general"
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def initialize_rag(self, paper_content: str) -> bool:
        """Initialize RAG processor with paper content."""
        try:
            if not self.rag_initialized and paper_content:
                self.rag_processor.process_paper(paper_content)
                self.rag_initialized = True
                logger.info("RAG initialized successfully")
                return True
        except Exception as e:
            logger.error(f"Error initializing RAG: {str(e)}", exc_info=True)
            return False
        return True
    
    def create_system_prompt(self, paper_content: str, chat_mode: str = "general") -> str:
        """Creates a context-aware system prompt based on chat mode."""
        base_prompt = f"""You are a research paper analysis assistant. Paper content:

{paper_content[:3000]}

"""
        mode_prompts = {
            "general": "Provide clear, accessible explanations suitable for general understanding.",
            "focused": "Focus on specific details and methodology, suitable for domain experts.",
            "technical": "Emphasize technical aspects, equations, and implementation details."
        }
        
        return base_prompt + mode_prompts.get(chat_mode, mode_prompts["general"])

    def generate_chat_summary(self, selected_model: str) -> str:
        """Generate summary of chat history."""
        if not st.session_state.chat_history:
            return "No chat history to summarize."
        
        try:
            # Convert messages to simple format
            messages = []
            for msg in st.session_state.chat_history[-10:]:
                messages.append({
                    "role": str(msg.get('role', 'unknown')),
                    "content": str(msg.get('content', ''))
                })
            
            conversation_text = "\n\n".join([
                f"{msg['role'].upper()}: {msg['content']}" 
                for msg in messages
            ])

            prompt = f"""Please analyze and summarize this conversation about a research paper:

{conversation_text}

Provide your summary in the following format:

MAIN TOPICS DISCUSSED:
- List the key topics and themes covered

KEY INSIGHTS:
- List important findings and conclusions

OPEN QUESTIONS:
- Note any unresolved or follow-up questions

Keep each section focused and clear."""

            model = self.get_model(selected_model)
            response = model.invoke(prompt)
            
            return str(response).strip() if response else "Error generating summary"
            
        except Exception as e:
            logger.error(f"Error in generate_chat_summary: {str(e)}", exc_info=True)
            return f"Error generating summary: {str(e)}"
    
    def extract_key_points(self, response: str) -> List[str]:
        """Extract key points from assistant response."""
        try:
            points = []
            lines = response.split('\n')
            current_point = ""
            
            key_terms = [
                'key', 'important', 'significant', 'finding', 'conclusion',
                'result', 'shows', 'demonstrates', 'reveals', 'highlights'
            ]
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if (line.startswith(('-', '*', '1.', '2.', '3.')) or 
                    any(term in line.lower() for term in key_terms)):
                    if current_point:
                        points.append(current_point.strip())
                        current_point = ""
                    current_point = line
                elif line.startswith('  ') and current_point:
                    current_point += " " + line.strip()
                elif any(term in line.lower() for term in key_terms):
                    points.append(line)
            
            if current_point:
                points.append(current_point.strip())
            
            # Remove duplicates while preserving order
            seen = set()
            unique_points = []
            for point in points:
                if point not in seen:
                    seen.add(point)
                    unique_points.append(point)
            
            return unique_points
            
        except Exception as e:
            logger.error(f"Error extracting key points: {str(e)}", exc_info=True)
            return []
    
    def display_chat_interface(self, selected_model: str):
        """Main chat interface display."""
        self.initialize_chat_state()
        self.display_chat_controls()
        
        # Initialize RAG if needed
        if st.session_state.paper_content and not self.rag_initialized:
            with st.spinner("Initializing paper analysis..."):
                success = self.initialize_rag(st.session_state.paper_content)
                if not success:
                    st.error("Failed to initialize paper analysis")
                    return
        
        st.subheader("💬 Chat with the Paper")
        
        chat_tab, summary_tab = st.tabs(["Chat", "Summary & Key Points"])
        
        with chat_tab:
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a question about the paper..."):
                st.session_state.chat_history.append({
                    "role": "user", 
                    "content": prompt
                })
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                try:
                    # Enhance prompt with RAG if available
                    if self.rag_initialized:
                        relevant_chunks = self.rag_processor.get_relevant_chunks(prompt)
                        enhanced_prompt = self.rag_processor.create_enhanced_prompt(
                            prompt, relevant_chunks
                        )
                    else:
                        enhanced_prompt = prompt
                    
                    # Create chat messages
                    messages = [
                        ("system", self.create_system_prompt(
                            st.session_state.paper_content,
                            st.session_state.chat_mode
                        )),
                        ("human", enhanced_prompt)
                    ]
                    
                    # Get model and generate response
                    model = self.get_model(selected_model)
                    chain = ChatPromptTemplate.from_messages(messages) | model
                    
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = chain.invoke({})
                            # if hasattr(response, 'content'):
                            #     response_text = response.content
                            # elif isinstance(response, dict) and 'content' in response:
                            #     response_text = response['content']
                            # else:
                            #     response_text = str(response)

                            # # Remove <think> tags if present (DeepSeek reasoning)
                            # import re
                            # response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
                            # response_text = response_text.strip()

                            # st.markdown(response_text)
                            response_text = str(response)
                            st.markdown(response_text)
                            
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response_text
                            })
                            
                            # Extract key points
                            new_points = self.extract_key_points(response_text)
                            if new_points:
                                st.session_state.key_points.extend(new_points)
                                
                except Exception as e:
                    logger.error(f"Error in chat processing: {str(e)}", exc_info=True)
                    st.error(f"Error: {str(e)}")
                    if not self.rag_initialized:
                        st.info("Note: Enhanced paper analysis is not available.")
        
        with summary_tab:
            self.display_summary_tab(selected_model)
    
    def display_summary_tab(self, selected_model: str):
        """Display summary and key points tab."""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📝 Chat Summary")
            if st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    summary = self.generate_chat_summary(selected_model)
                    st.session_state.chat_summary = summary
            
            if st.session_state.chat_summary:
                st.markdown(st.session_state.chat_summary)
        
        with col2:
            st.markdown("### 🎯 Key Points")
            if st.session_state.key_points:
                for point in st.session_state.key_points[-5:]:
                    st.markdown(f"• {point}")
            else:
                st.info("Key points will appear here as you chat.")
    
    def display_chat_controls(self):
        """Display chat control sidebar."""
        st.sidebar.markdown("### 💬 Chat Controls")
        
        # RAG status
        if self.rag_initialized:
            st.sidebar.success("✓ Enhanced paper analysis active")
        else:
            st.sidebar.warning("Basic chat mode (paper analysis not initialized)")
        
        # Chat mode selection
        chat_mode = st.sidebar.radio(
            "Chat Mode",
            ["general", "focused", "technical"],
            help="Select the level of technical detail in responses"
        )
        st.session_state.chat_mode = chat_mode
        
        # Clear chat button
        if st.sidebar.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.chat_summary = None
            st.session_state.key_points = []
            st.rerun()
        
        # Download chat history
        if st.session_state.chat_history:
            chat_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in st.session_state.chat_history
            ])
            st.sidebar.download_button(
                "Download Chat History",
                chat_text,
                file_name="chat_history.txt",
                mime="text/plain"
            )