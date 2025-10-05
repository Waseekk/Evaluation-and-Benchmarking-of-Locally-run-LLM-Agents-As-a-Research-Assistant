# enhanced_chat_interface.py

import streamlit as st
import time
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from rag_processor import RAGProcessor  # Adjust the import based on your project structure

class EnhancedChatInterface:
    """Enhanced chat interface with RAG support."""
    
    def __init__(self):
        self.models = {
            "deepseek-1.5b": ChatOllama(
                model="deepseek-r1:1.5b",
                temperature=0.3,
                base_url="http://localhost:11434"
            ),
            "deepseek-8b": ChatOllama(
                model="deepseek-r1:8b",
                temperature=0.3,
                base_url="http://localhost:11434"
            ),
            "mistral": ChatOllama(
                model="mistral",
                temperature=0.3,
                base_url="http://localhost:11434"
            ),
            "llama3-8b": ChatOllama(
                model="llama3:8b",
                temperature=0.3,
                base_url="http://localhost:11434"
            )
        }
        self.rag_processor = RAGProcessor()
        self.rag_initialized = False

    def initialize_chat_state(self):
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "paper_content" not in st.session_state:
            st.session_state.paper_content = None
        if "chat_summary" not in st.session_state:
            st.session_state.chat_summary = None
        if "key_points" not in st.session_state:
            st.session_state.key_points = []
        if "chat_mode" not in st.session_state:
            st.session_state.chat_mode = "general"
    
    def initialize_rag(self, paper_content: str) -> bool:
        try:
            if not self.rag_initialized and paper_content:
                self.rag_processor.process_paper(paper_content)
                self.rag_initialized = True
                return True
        except Exception as e:
            print(f"Error initializing RAG: {str(e)}")
            return False
        return True
    
    def create_system_prompt(self, paper_content: str, chat_mode: str = "general") -> str:
        try:
            if not self.rag_initialized:
                if not self.initialize_rag(paper_content):
                    return self.create_fallback_prompt(chat_mode)
            relevant_chunks = self.rag_processor.get_relevant_chunks(
                "What is the main topic and methodology of this paper?",
                k=2
            )
            context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
            base_prompt = f"""You are a research paper analysis assistant. Key context from the paper:

{context}

"""
            mode_prompts = {
                "general": "Provide clear, accessible explanations suitable for general understanding.",
                "focused": "Focus on specific details and methodology, suitable for domain experts.",
                "technical": "Emphasize technical aspects, equations, and implementation details."
            }
            return base_prompt + mode_prompts.get(chat_mode, mode_prompts["general"])
        except Exception as e:
            print(f"Error creating system prompt: {str(e)}")
            return self.create_fallback_prompt(chat_mode)
    
    def create_fallback_prompt(self, chat_mode: str) -> str:
        mode_prompts = {
            "general": "Provide clear, accessible explanations about research papers.",
            "focused": "Focus on specific details and methodology in research papers.",
            "technical": "Emphasize technical aspects and implementation details."
        }
        return f"You are a research paper analysis assistant. {mode_prompts.get(chat_mode, mode_prompts['general'])}"
    
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
    

    

    def generate_chat_summary(self, selected_model: str):
        """
        Generates a summary of the chat history with proper message type handling.
        
        Args:
            selected_model (str): Name of the model to use for summarization
            
        Returns:
            str: Formatted summary of the chat or error message
        """
        if not st.session_state.chat_history:
            return "No chat history to summarize."
        
        if selected_model not in self.models:
            return f"Error: Selected model '{selected_model}' not found."
        
        try:
            # Convert messages to simple dict format
            messages = []
            for msg in st.session_state.chat_history[-10:]:
                # Extract just the role and content, ignoring other attributes
                messages.append({
                    "role": str(msg['role']),
                    "content": str(msg['content'])
                })
            
            # Create conversation text directly without JSON serialization
            # Create conversation text directly without JSON serialization
            conversation_text = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in messages])

            # Print conversation_text to check for any curly braces
            print(conversation_text)

            # Create simple prompt without message structure
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

            # Use direct model invocation
            model = self.models[selected_model]
            response = model.invoke(prompt)
            
            # Ensure response is a string
            if not isinstance(response, str):
                response = str(response)
            
            return response.strip()
            
        except Exception as e:
            error_msg = str(e)
            print(f"Debug - Error in generate_chat_summary: {error_msg}")
            
            if "not JSON serializable" in error_msg:
                return "⚠️ Error: Message format conversion failed. Please try again."
            elif "torch.classes" in error_msg:
                return "⚠️ Model initialization error. Please try a different model."
            else:
                return f"⚠️ Error generating summary: {error_msg}"
    
    def extract_key_points(self, response: str) -> list:
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
                if (line.startswith(('•', '-', '*', '1.', '2.', '3.')) or 
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
            seen = set()
            unique_points = []
            for point in points:
                if point not in seen:
                    seen.add(point)
                    unique_points.append(point)
            return unique_points
        except Exception as e:
            print(f"Error extracting key points: {str(e)}")
            return []
    
    def display_chat_interface(self, selected_model: str):
        self.initialize_chat_state()
        self.display_chat_controls()
        if st.session_state.paper_content and not self.rag_initialized:
            with st.spinner("Initializing paper analysis..."):
                success = self.initialize_rag(st.session_state.paper_content)
                if not success:
                    st.error("Failed to initialize paper analysis")
                    return
        st.subheader("💬 Chat with the Paper")
        chat_tab, summary_tab = st.tabs(["Chat", "Summary & Key Points"])
        with chat_tab:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            if prompt := st.chat_input("Ask a question about the paper..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                try:
                    if self.rag_initialized:
                        relevant_chunks = self.rag_processor.get_relevant_chunks(prompt)
                        enhanced_prompt = self.rag_processor.create_enhanced_prompt(prompt, relevant_chunks)
                    else:
                        enhanced_prompt = prompt
                    messages = [
                        ("system", self.create_system_prompt(
                            st.session_state.paper_content,
                            st.session_state.chat_mode
                        )),
                        ("human", enhanced_prompt)
                    ]
                    chain = ChatPromptTemplate.from_messages(messages) | self.models[selected_model]
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = chain.invoke({})
                            st.markdown(response)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response
                            })
                            new_points = self.extract_key_points(response)
                            if new_points:
                                st.session_state.key_points.extend(new_points)
                except Exception as e:
                    st.error(f"Error in chat processing: {str(e)}")
                    if not self.rag_initialized:
                        st.info("Note: Enhanced paper analysis is not available. Using basic chat mode.")
        with summary_tab:
            self.display_summary_tab(selected_model)
    
    def display_summary_tab(self, selected_model: str):
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
        st.sidebar.markdown("### 💬 Chat Controls")
        if self.rag_initialized:
            st.sidebar.success("✓ Enhanced paper analysis active")
        else:
            st.sidebar.warning("Basic chat mode (paper analysis not initialized)")
        st.sidebar.markdown("### 💬 Chat Controls")
        chat_mode = st.sidebar.radio(
            "Chat Mode",
            ["general", "focused", "technical"],
            help="Select the level of technical detail in responses"
        )
        st.session_state.chat_mode = chat_mode
        if self.rag_initialized:
            st.sidebar.success("✓ Enhanced paper analysis active")
        else:
            st.sidebar.warning("Basic chat mode (paper analysis not initialized)")
        if st.sidebar.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.chat_summary = None
            st.session_state.key_points = []
            st.experimental_rerun()
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
