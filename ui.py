import streamlit as st
import os
import json
from dotenv import load_dotenv
from src.graphs.graph_builder import GraphBuilder
from src.llms.groqllm import GroqLLM

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="BloggerAI — Agentic Blog Assistant",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design & Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=Space+Grotesk:wght@400;500;700&display=swap');
    
    /* Global styles */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header/Sidebar customizations */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    
    .sidebar-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 24px;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    
    /* Premium Cards (Glassmorphism) */
    .glass-card {
        background: rgba(31, 41, 55, 0.4);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .output-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.05);
    }
    
    /* Titles & Headings */
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .main-title {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #a78bfa, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .section-title {
        font-size: 20px;
        font-weight: 600;
        color: #e2e8f0;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    
    /* Custom buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 16px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.6);
        transform: translateY(-2px);
    }
    
    div.stButton > button:first-child:active {
        transform: translateY(0);
    }
</style>
""", unsafe_allow_html=True)

# Render Banner Image
if os.path.exists("blogger_ai_banner.png"):
    st.image("blogger_ai_banner.png", use_container_width=True)

# Header Title
st.markdown("<h1 class='main-title'>BloggerAI</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:18px; color:#94a3b8; margin-top:-10px; margin-bottom: 30px;'>Advanced Agentic Blog Assistant powered by LangGraph and Llama 3</p>", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("<div class='sidebar-title'>⚙️ Configuration</div>", unsafe_allow_html=True)
    
    # API key check
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        api_input = st.text_input("Enter GROQ API Key", type="password")
        if api_input:
            os.environ["GROQ_API_KEY"] = api_input
            st.success("API Key set temporarily!")
    else:
        st.success("🔑 Groq API Key loaded from environment")
        
    st.markdown("---")
    
    # Model configuration
    model_option = st.selectbox(
        "Select LLM Model",
        options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0,
        help="Llama 3.3 70B is highly recommended for structured output and translation tasks."
    )
    
    st.markdown("---")
    st.markdown("### Agent Workflow")
    st.markdown(
        """
        This assistant implements a StateGraph with:
        - **Title Creation Node**: Generates an SEO-optimized, creative title.
        - **Content Generation Node**: Writes the blog body using markdown layout.
        - **Routing Node**: Analyzes the desired target language.
        - **Conditional Edges**: Dynamically routes to French or Hindi translation nodes, or finishes directly.
        """
    )

# Main Application Layout: Input and Output
col_input, col_output = st.columns([1, 2], gap="large")

with col_input:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📝 Topic & Parameters")
    
    topic = st.text_area(
        "Blog Topic",
        placeholder="e.g. Agentic AI and the Future of Software Development",
        height=100
    )
    
    usecase = st.radio(
        "Agent Workflow Mode",
        options=["Topic Only", "Topic + Translation"],
        index=0
    )
    
    language = "none"
    if usecase == "Topic + Translation":
        language = st.selectbox(
            "Target Translation Language",
            options=["French", "Hindi"],
            index=0,
            help="The graph will route to specific French or Hindi translation nodes using conditional edges."
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("🚀 Generate Blog Post")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Graph Visualization
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🗺️ Graph Visualization")
    try:
        # Initialize Groq LLM and Builder to draw graph
        groqllm = GroqLLM()
        llm = groqllm.get_llm(model=model_option)
            
        graph_builder = GraphBuilder(llm)
        if usecase == "Topic + Translation":
            graph = graph_builder.setup_graph(usecase="language")
        else:
            graph = graph_builder.setup_graph(usecase="topic")
            
        # Draw mermaid
        mermaid_png = graph.get_graph().draw_mermaid_png()
        st.image(mermaid_png, use_container_width=True)
    except Exception as e:
        st.warning("Could not render interactive graph layout: " + str(e))
    st.markdown("</div>", unsafe_allow_html=True)

with col_output:
    if generate_btn:
        if not topic.strip():
            st.error("Please enter a topic to start.")
        else:
            with st.spinner("🤖 Agentic AI workflow is running..."):
                try:
                    # Initialize LLM and build Graph
                    groqllm = GroqLLM()
                    llm = groqllm.get_llm(model=model_option)
                    
                    graph_builder = GraphBuilder(llm)
                    
                    # Run graph based on selected mode
                    if usecase == "Topic + Translation" and language:
                        st.info(f"Invoking Translation Graph (Target: {language})...")
                        graph = graph_builder.setup_graph(usecase="language")
                        inputs = {"topic": topic, "current_language": language.lower()}
                    else:
                        st.info("Invoking Topic Generation Graph...")
                        graph = graph_builder.setup_graph(usecase="topic")
                        inputs = {"topic": topic}
                        
                    # Stream or run graph
                    state = graph.invoke(inputs)
                    
                    # Show results in card
                    st.success("✨ Generation Complete!")
                    
                    # Extract title and content
                    blog_data = state.get("blog", {})
                    
                    # Handle both dict and object structures
                    if isinstance(blog_data, dict):
                        title = blog_data.get("title", "Untitled Blog")
                        content = blog_data.get("content", "")
                    else:
                        title = getattr(blog_data, "title", "Untitled Blog")
                        content = getattr(blog_data, "content", "")
                    
                    # Output Tabs
                    tab_preview, tab_raw, tab_state = st.tabs(["📖 Preview", "📄 Markdown Code", "⛓️ Graph State"])
                    
                    with tab_preview:
                        st.markdown("<div class='output-card'>", unsafe_allow_html=True)
                        st.markdown(f"# {title}")
                        st.markdown("---")
                        st.markdown(content)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    with tab_raw:
                        full_markdown = f"# {title}\n\n{content}"
                        st.code(full_markdown, language="markdown")
                        st.download_button(
                            label="📥 Download Markdown File",
                            data=full_markdown,
                            file_name=f"blog_{topic.lower().replace(' ', '_')[:30]}.md",
                            mime="text/markdown"
                        )
                        
                    with tab_state:
                        st.json(state)
                        
                except Exception as err:
                    st.error(f"An error occurred during graph execution: {err}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
    else:
        # Placeholder view before generation
        st.markdown("<div class='output-card' style='text-align: center; padding: 60px 20px;'>", unsafe_allow_html=True)
        st.markdown("<h2 style='color: #6366f1;'>Ready to Generate</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8;'>Enter a topic on the left and click 'Generate Blog Post' to invoke the AI agent workflow.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
