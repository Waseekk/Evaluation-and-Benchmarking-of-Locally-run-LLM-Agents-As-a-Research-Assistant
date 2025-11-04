# enhanced_chat_interface_2.py
# Enhanced chat interface with beautified responses and better UX

import streamlit as st
import time
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from rag_processor import RAGProcessor

class EnhancedChatInterface:
    """Enhanced chat interface with RAG support, beautified responses, and better UX."""

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

    @staticmethod
    def beautify_response(response: str) -> str:
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
        response = re.sub(r'-{3,}', '\n\n---\n\n', response)
        response = re.sub(r'={3,}', '\n\n---\n\n', response)

        # Clean up excessive newlines
        response = re.sub(r'\n{4,}', '\n\n', response)

        # Remove escape characters
        response = response.replace('\\n', '\n')
        response = response.replace('\\t', '  ')

        # Fix common formatting issues
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)

        # Improve list formatting
        response = re.sub(r'\n-\s+', '\n- ', response)
        response = re.sub(r'\n\*\s+', '\n* ', response)

        # Strip leading/trailing whitespace
        response = response.strip()

        # If response contains Python object syntax, extract content
        if "Message(" in response or "role=" in response:
            lines = []
            for line in response.split('\n'):
                if any(skip in line for skip in ['additional_kwargs', 'response_metadata', 'usage_metadata', 'Message(', 'role=']):
                    continue
                lines.append(line)
            response = '\n'.join(lines).strip()

        return response

    def initialize_chat_state(self):
        """Initialize chat-related session state variables."""
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
        """Initialize RAG processor with paper content."""
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
        """Creates a context-aware system prompt based on chat mode."""
        base_prompt = f"""You are a research paper analysis assistant. Paper content:

{paper_content[:3000]}

"""
        mode_prompts = {
    "general": (
        "Provide clear, accessible explanations suitable for general understanding. "
        "Use plain language, avoid jargon, and include analogies or real-world examples when helpful. "
        "Break down complex concepts into digestible steps. Aim for clarity over technical precision."
    ),
    
    "focused": (
        "Focus on specific details and methodology, suitable for domain experts. "
        "Use appropriate domain-specific terminology and frameworks. "
        "Provide concise, targeted responses that balance depth with relevance. "
        "Include key methodologies, assumptions, and limitations where applicable."
    ),
    
    "technical": (
        "Emphasize technical aspects, equations, and implementation details. "
        "Include mathematical formulations, algorithms, and pseudocode where relevant. "
        "Discuss computational complexity, performance considerations, and edge cases. "
        "Reference standards, best practices, and provide implementation-ready insights. "
        "Use precise technical terminology and formal notation."
    )
}

        return base_prompt + mode_prompts.get(chat_mode, mode_prompts["general"])

    def generate_chat_summary(self, selected_model: str):
        """Generate a summary of the chat history."""
        if not st.session_state.chat_history:
            return "No chat history to summarize."

        if selected_model not in self.models:
            return f"Error: Selected model '{selected_model}' not found."

        try:
            messages = []
            for msg in st.session_state.chat_history[-10:]:
                messages.append({
                    "role": str(msg['role']),
                    "content": str(msg['content'])
                })

            conversation_text = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in messages])

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

            model = self.models[selected_model]
            response = model.invoke(prompt)

            if not isinstance(response, str):
                response = str(response)

            # ✨ Beautify the summary response
            response = self.beautify_response(response)

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
        """Extract key points from a response."""
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
                    current_point = line.lstrip('•-*123456789. ')
                elif current_point:
                    current_point += " " + line
            if current_point:
                points.append(current_point.strip())
            return points[:5]
        except Exception as e:
            print(f"Error extracting key points: {e}")
            return []

    def display_chat_interface(self, selected_model: str):
        """Display the main chat interface with enhanced UX."""
        self.initialize_chat_state()
        self.display_chat_controls()

        # Initialize RAG with better progress indicator
        if st.session_state.paper_content and not self.rag_initialized:
            with st.spinner("🔄 Initializing paper analysis..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

                success = self.initialize_rag(st.session_state.paper_content)
                progress_bar.empty()

                if not success:
                    st.error("❌ Failed to initialize paper analysis")
                    return
                else:
                    st.success("✅ Paper analysis ready!")
                    time.sleep(0.5)
                    st.rerun()

        st.subheader("💬 Chat with the Paper")

        # Add helpful tips
        with st.expander("💡 Chat Tips", expanded=False):
            st.markdown("""
            **Effective questions to ask:**
            - "What is the main contribution of this paper?"
            - "Explain the methodology used"
            - "What are the key findings?"
            - "How does this compare to prior work?"
            - "What are the limitations?"

            **Chat modes:**
            - **General**: Easy-to-understand explanations
            - **Focused**: Detailed, expert-level analysis
            - **Technical**: Deep dive into technical details
            """)

        chat_tab, summary_tab = st.tabs(["💬 Chat", "📝 Summary & Key Points"])

        with chat_tab:
            # Display chat history with beautified responses
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    # ✨ Beautify assistant responses
                    if message["role"] == "assistant":
                        cleaned_content = self.beautify_response(message["content"])
                        st.markdown(cleaned_content)
                    else:
                        st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("Ask a question about the paper..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})

                with st.chat_message("user"):
                    st.markdown(prompt)

                try:
                    # Prepare prompt with RAG if available
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
                        # Enhanced loading indicator
                        with st.spinner(f"🤔 {selected_model} is thinking..."):
                            start_time = time.time()
                            response = chain.invoke({})
                            response_time = time.time() - start_time

                            # ✨ Beautify the response before displaying
                            cleaned_response = self.beautify_response(str(response))

                            st.markdown(cleaned_response)

                            # Show response time
                            st.caption(f"⏱️ Response time: {response_time:.2f}s")

                            # Save to history
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": str(response)  # Store original for summary generation
                            })

                            # Extract key points
                            new_points = self.extract_key_points(cleaned_response)
                            if new_points:
                                st.session_state.key_points.extend(new_points)

                except Exception as e:
                    st.error(f"❌ Error in chat processing: {str(e)}")
                    if not self.rag_initialized:
                        st.info("ℹ️ Note: Enhanced paper analysis is not available. Using basic chat mode.")

        with summary_tab:
            self.display_summary_tab(selected_model)

    def display_summary_tab(self, selected_model: str):
        """Display the summary tab with key points and chat summary."""
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 📝 Chat Summary")

            if st.button("🔄 Generate Summary", type="primary"):
                with st.spinner("Generating summary..."):
                    # Show progress
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)

                    summary = self.generate_chat_summary(selected_model)
                    st.session_state.chat_summary = summary
                    progress_bar.empty()

            if st.session_state.chat_summary:
                # Display beautified summary
                st.markdown(st.session_state.chat_summary)
            else:
                st.info("Click 'Generate Summary' to create a summary of your conversation.")

        with col2:
            st.markdown("### 🎯 Key Points")

            if st.session_state.key_points:
                for i, point in enumerate(st.session_state.key_points[-5:], 1):
                    st.markdown(f"**{i}.** {point}")

                if len(st.session_state.key_points) > 5:
                    with st.expander(f"📋 View all {len(st.session_state.key_points)} key points"):
                        for i, point in enumerate(st.session_state.key_points, 1):
                            st.markdown(f"**{i}.** {point}")
            else:
                st.info("💡 Key points will appear here as you chat.")

            # Add clear history button
            st.divider()
            if st.button("🗑️ Clear Key Points"):
                st.session_state.key_points = []
                st.rerun()

    def display_chat_controls(self):
        """Display chat control options in sidebar."""
        st.sidebar.markdown("### 💬 Chat Controls")

        # # RAG status indicator
        # if self.rag_initialized:
        #     st.sidebar.success("✅ Enhanced paper analysis active")
        # else:
        #     st.sidebar.warning("⚠️ Basic chat mode")

        # Chat mode selector
        chat_mode = st.sidebar.radio(
            "Chat Mode",
            ["general", "focused", "technical"],
            help="Select the level of technical detail in responses",
            index=["general", "focused", "technical"].index(st.session_state.chat_mode)
        )
        st.session_state.chat_mode = chat_mode

        # Show mode description
        mode_descriptions = {
            "general": "📖 Easy-to-understand explanations",
            "focused": "🎯 Detailed, expert-level analysis",
            "technical": "⚙️ Deep technical details"
        }
        st.sidebar.caption(mode_descriptions[chat_mode])

        st.sidebar.divider()

        # Clear chat button
        if st.sidebar.button("🗑️ Clear Chat History", help="Clear all chat messages"):
            st.session_state.chat_history = []
            st.session_state.chat_summary = None
            st.session_state.key_points = []
            st.rerun()

        # # Chat statistics
        # if st.session_state.chat_history:
        #     st.sidebar.divider()
        #     st.sidebar.markdown("### 📊 Chat Statistics")
        #     st.sidebar.metric("Messages", len(st.session_state.chat_history))
        #     st.sidebar.metric("Key Points", len(st.session_state.key_points))
