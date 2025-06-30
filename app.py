import os
import streamlit as st
import pickle
import time
import validators
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SOPBasedSR - AI News Research Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1e3a8a;
    text-align: center;
    margin-bottom: 1rem;
}
.subtitle {
    font-size: 1.1rem;
    color: #64748b;
    text-align: center;
    margin-bottom: 2rem;
}
.expert-prompt {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.success-msg {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: white;
    padding: 0.8rem;
    border-radius: 8px;
    text-align: center;
    margin: 0.5rem 0;
}
.error-msg {
    background: linear-gradient(135deg, #ff416c, #ff4757);
    color: white;
    padding: 0.8rem;
    border-radius: 8px;
    text-align: center;
    margin: 0.5rem 0;
}
.answer-box {
    background: linear-gradient(135deg, #a8edea, #fed6e3);
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 4px solid #667eea;
}
.sources-box {
    background: linear-gradient(135deg, #ffecd2, #fcb69f);
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 4px solid #ff7675;
}
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">🤖 SOP-Based-SR - AI Research Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your expert AI companion for analyzing articles and extracting insights</p>', unsafe_allow_html=True)

# Expert prompt information
with st.expander("🎯 About SOP-Based-SR - Your AI Expert", expanded=False):
    st.markdown("""
    <div class="expert-prompt">
    <h4>🧠 RockyBot is your specialized AI research expert that:</h4>
    <ul>
        <li>📰 <strong>Analyzes News Articles:</strong> Deep analysis of content from multiple sources</li>
        <li>🔍 <strong>Extracts Key Insights:</strong> Identifies trends, implications, and important findings</li>
        <li>💡 <strong>Provides Expert Analysis:</strong> Acts as a domain expert across various topics</li>
        <li>📊 <strong>Cross-References Information:</strong> Connects information across different articles</li>
        <li>🎯 <strong>Answers Complex Questions:</strong> Provides comprehensive, expert-level responses</li>
    </ul>
    <p><strong>Simply provide URLs of news articles, and I'll become your expert analyst in that domain!</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar for URLs
st.sidebar.title("📰 SOP Article URLs")
st.sidebar.markdown("**Enter 2-5 SOP article URLs for analysis:**")

urls = []
for i in range(5):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}", placeholder=f"Enter SOP article URL {i+1}...")
    if url.strip():
        urls.append(url.strip())

# Display URL validation status
if urls:
    valid_urls = [url for url in urls if validators.url(url)]
    if valid_urls:
        st.sidebar.success(f"✅ {len(valid_urls)} valid URLs detected")
    else:
        st.sidebar.error("❌ No valid URLs detected")

process_url_clicked = st.sidebar.button("🚀 Process URLs", help="Click to analyze the provided URLs")

# File path for storing embeddings
file_path = "faiss_store_embeddings.pkl"

# Main placeholder for status messages
main_placeholder = st.empty()

# Initialize LLM with expert prompting
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    max_tokens=2048
)

# URL processing
if process_url_clicked:
    if not urls:
        st.error("❌ Please enter at least one URL to process.")
    else:
        # Validate URLs
        valid_urls = [url for url in urls if validators.url(url)]
        
        if not valid_urls:
            st.error("❌ No valid URLs provided. Please check your URLs and try again.")
        else:
            try:
                # Progress tracking
                progress_bar = st.progress(0)
                
                # Load data
                main_placeholder.markdown('<div class="success-msg">📥 Loading articles from URLs...</div>', unsafe_allow_html=True)
                progress_bar.progress(20)
                
                loader = UnstructuredURLLoader(urls=valid_urls)
                data = loader.load()
                
                if not data:
                    st.error("❌ Failed to load content from the provided URLs. Please check if the URLs are accessible.")
                else:
                    # Split documents
                    main_placeholder.markdown('<div class="success-msg">✂️ Processing and splitting text content...</div>', unsafe_allow_html=True)
                    progress_bar.progress(40)
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        separators=['\n\n', '\n', '.', ',', ' '],
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    docs = text_splitter.split_documents(data)
                    
                    # Create embeddings
                    main_placeholder.markdown('<div class="success-msg">🧠 Creating knowledge base with embeddings...</div>', unsafe_allow_html=True)
                    progress_bar.progress(60)
                    
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    vectorstore_openai = FAISS.from_documents(docs, embeddings)
                    
                    # Save to file
                    main_placeholder.markdown('<div class="success-msg">💾 Saving knowledge base...</div>', unsafe_allow_html=True)
                    progress_bar.progress(80)
                    
                    with open(file_path, "wb") as f:
                        pickle.dump(vectorstore_openai, f)
                    
                    progress_bar.progress(100)
                    main_placeholder.markdown('<div class="success-msg">🎉 Knowledge base created successfully!</div>', unsafe_allow_html=True)
                    
                    # Success summary
                    st.success(f"✅ Successfully processed {len(valid_urls)} articles with {len(docs)} text chunks!")
                    
                    # Display processed URLs
                    with st.expander("📋 Processed URLs", expanded=False):
                        for i, url in enumerate(valid_urls, 1):
                            st.write(f"{i}. {url}")
                    
            except Exception as e:
                st.error(f"❌ Error processing URLs: {str(e)}")
                st.info("💡 Try checking your internet connection or using different URLs.")

# Query section
st.markdown("### 💬 Ask Questions to Your Assistant")
st.markdown("**Enter your question about the analyzed articles:**")

query = st.text_area(
    "Your Question:",
    placeholder="Examples:\n• What are the main trends discussed in these articles?\n• What are the key implications for the market?\n• Summarize the most important findings\n• What expert insights can you provide?",
    height=120
)

if st.button("🔍 Get Expert Analysis", help="Click to get AI expert analysis"):
    if query.strip():
        if os.path.exists(file_path):
            try:
                with st.spinner("🤖 ChatBot is analyzing as your expert assistant..."):
                    # Load vectorstore
                    with open(file_path, "rb") as f:
                        vectorstore = pickle.load(f)
                    
                    # Enhanced expert prompt
                    expert_prompt = f"""
                    You are RockyBot, an expert AI research assistant and domain specialist. You have analyzed multiple news articles and now act as a knowledgeable expert in the subject matter covered by these articles.

                    Your expertise includes:
                    - Deep analysis of current events and emerging trends
                    - Cross-referencing information from multiple reliable sources
                    - Providing contextual understanding and implications
                    - Identifying key insights that others might miss
                    - Offering expert-level commentary and analysis
                    - Maintaining objectivity while providing comprehensive insights

                    As an expert in this domain, please provide a comprehensive, insightful analysis for the following question:

                    Question: {query}

                    Please respond as a knowledgeable expert would, providing:
                    1. Direct answer to the question
                    2. Key insights and implications
                    3. Relevant context from the analyzed articles
                    4. Expert perspective on the significance
                    5. Any important trends or patterns identified

                    Provide your expert analysis:
                    """
                    
                    # Create retrieval chain
                    chain = RetrievalQAWithSourcesChain.from_llm(
                        llm=llm, 
                        retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
                    )
                    
                    # Get response
                    result = chain({"question": expert_prompt}, return_only_outputs=True)
                    
                    # Display results
                    if result and "answer" in result:
                        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                        st.markdown("### 🤖 RockyBot's Expert Analysis")
                        st.write(result["answer"])
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display sources
                        sources = result.get("sources", "")
                        if sources:
                            st.markdown('<div class="sources-box">', unsafe_allow_html=True)
                            st.markdown("### 📚 Sources Referenced")
                            sources_list = [s.strip() for s in sources.split("\n") if s.strip()]
                            for i, source in enumerate(sources_list, 1):
                                st.write(f"**{i}.** {source}")
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error("❌ Failed to generate analysis. Please try again.")
                        
            except Exception as e:
                st.error(f"❌ Error generating analysis: {str(e)}")
        else:
            st.warning("⚠️ Please process some URLs first to create the knowledge base.")
    else:
        st.warning("⚠️ Please enter a question to get analysis.")

# Footer with usage stats
if os.path.exists(file_path):
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        st.metric("📊 Knowledge Base Size", f"{file_size:.2f} MB")
    
    with col2:
        st.metric("🤖 AI Model", "Google Gemini 2.0 Flash")
    
    with col3:
        if st.button("🗑️ Clear Knowledge Base"):
            os.remove(file_path)
            st.success("Knowledge base cleared successfully!")
            st.rerun()

# Quick tips
with st.sidebar.expander("💡 Quick Tips", expanded=False):
    st.markdown("""
    **For best results:**
    - Use SOP URLs of sharepoint
    - Ask specific, detailed questions
    - Try questions like:
      - "What are the key findings?"
      - "What are the implications?"
      - "How does this relate to...?"
      - "What trends are emerging?"
      - "What's your expert opinion on...?"
    """)

# API key reminder
if not os.getenv("GOOGLE_API_KEY"):
    st.sidebar.error("⚠️ Please set your GOOGLE_API_KEY in .env file")
    st.sidebar.info("Get your API key from: https://aistudio.google.com/")