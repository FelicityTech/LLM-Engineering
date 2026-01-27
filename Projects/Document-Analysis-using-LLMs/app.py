import streamlit as st
import pdfplumber
import nltk
from transformers import pipeline
import json
import time
from typing import List, Dict
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="PDF Question & Answer System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .question-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .answer-card {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 6px;
        margin-top: 0.5rem;
    }
    .stats-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# CACHING & MODEL LOADING FUNCTIONS
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_nltk_data():
    """Download required NLTK data packages"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        return True
    except Exception as e:
        st.error(f"Error loading NLTK data: {e}")
        return False

@st.cache_resource(show_spinner=False)
def load_qg_model():
    """Load the question generation model"""
    try:
        qg_pipeline = pipeline(
            "text2text-generation", 
            model="valhalla/t5-base-qg-hl",
            device=-1  # Use CPU
        )
        return qg_pipeline
    except Exception as e:
        st.error(f"Error loading Question Generation model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_qa_model():
    """Load the question answering model"""
    try:
        qa_pipeline = pipeline(
            "question-answering", 
            model="deepset/roberta-base-squad2",
            device=-1  # Use CPU
        )
        return qa_pipeline
    except Exception as e:
        st.error(f"Error loading Question Answering model: {e}")
        return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            document_text = ""
            total_pages = len(pdf.pages)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    document_text += text + "\n"
                
                # Update progress
                progress = (idx + 1) / total_pages
                progress_bar.progress(progress)
                status_text.text(f"Processing page {idx + 1} of {total_pages}")
            
            progress_bar.empty()
            status_text.empty()
            
            return document_text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def split_into_passages(text: str, max_words: int = 100) -> List[str]:
    """Split text into manageable passages"""
    try:
        sentences = nltk.tokenize.sent_tokenize(text)
        passages = []
        current_passage = ""
        
        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            current_word_count = len(current_passage.split())
            
            if current_word_count + sentence_word_count <= max_words and current_passage:
                current_passage += " " + sentence
            else:
                if current_passage:
                    passages.append(current_passage.strip())
                current_passage = sentence
        
        if current_passage:
            passages.append(current_passage.strip())
        
        return passages
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return []

def generate_questions_from_passage(passage: str, qg_pipeline, min_questions: int = 3) -> List[str]:
    """Generate questions from a text passage"""
    try:
        input_text = f"generate questions: {passage}"
        results = qg_pipeline(input_text, max_length=256, num_return_sequences=1)
        questions = results[0]['generated_text'].split('<sep>')
        questions = [q.strip() for q in questions if q.strip()]
        
        # If we don't have enough questions, try with smaller chunks
        if len(questions) < min_questions:
            sentences = nltk.tokenize.sent_tokenize(passage)
            for i in range(0, len(sentences) - 1, 2):
                if len(questions) >= min_questions:
                    break
                chunk = ' '.join(sentences[i:i+2])
                if len(chunk.split()) > 5:
                    try:
                        additional_results = qg_pipeline(
                            f"generate questions: {chunk}", 
                            max_length=128
                        )
                        additional_questions = additional_results[0]['generated_text'].split('<sep>')
                        questions.extend([q.strip() for q in additional_questions if q.strip()])
                    except:
                        continue
        
        return list(dict.fromkeys(questions[:min_questions]))  # Remove duplicates, keep order
    except Exception as e:
        st.warning(f"Could not generate questions from passage: {e}")
        return []

def answer_question(question: str, context: str, qa_pipeline) -> Dict[str, any]:
    """Answer a question based on the context"""
    try:
        # Limit context length to avoid model limitations
        max_context_length = 3000  # Adjust based on model capacity
        if len(context.split()) > max_context_length:
            # Take beginning and end of context
            words = context.split()
            context = ' '.join(words[:max_context_length//2] + words[-max_context_length//2:])
        
        result = qa_pipeline({
            'question': question, 
            'context': context
        })
        
        return {
            'answer': result['answer'],
            'score': round(result['score'], 3),
            'start': result['start'],
            'end': result['end']
        }
    except Exception as e:
        return {
            'answer': f"Could not find an answer: {str(e)}",
            'score': 0.0,
            'start': -1,
            'end': -1
        }

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'document_text' not in st.session_state:
    st.session_state.document_text = ""
if 'generated_questions' not in st.session_state:
    st.session_state.generated_questions = []
if 'qa_pairs' not in st.session_state:
    st.session_state.qa_pairs = []
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = ""
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("📚 PDF Question & Answer System")
    st.markdown("Upload your PDF document to automatically generate questions or ask custom queries!")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model loading section
        st.subheader("Model Status")
        if not st.session_state.models_loaded:
            with st.spinner("🔄 Loading AI models... This may take a minute."):
                load_nltk_data()
                qg_model = load_qg_model()
                qa_model = load_qa_model()
                
                if qg_model and qa_model:
                    st.session_state.qg_model = qg_model
                    st.session_state.qa_model = qa_model
                    st.session_state.models_loaded = True
                    st.success("✅ Models loaded successfully!")
                else:
                    st.error("❌ Failed to load models. Please refresh the page.")
                    st.stop()
        else:
            st.success("✅ Models ready!")
        
        st.markdown("---")
        
        # Question generation settings
        st.subheader("Question Generation")
        num_questions = st.slider(
            "Questions per passage", 
            min_value=1, 
            max_value=5, 
            value=3,
            help="Number of questions to generate from each passage"
        )
        
        passage_length = st.slider(
            "Passage length (words)", 
            min_value=50, 
            max_value=200, 
            value=100,
            help="Maximum words per passage for question generation"
        )
        
        st.markdown("---")
        
        # Document info
        if st.session_state.document_text:
            st.subheader("📄 Document Info")
            word_count = len(st.session_state.document_text.split())
            char_count = len(st.session_state.document_text)
            
            st.metric("Words", f"{word_count:,}")
            st.metric("Characters", f"{char_count:,}")
            
            if st.button("🗑️ Clear Document"):
                st.session_state.document_text = ""
                st.session_state.generated_questions = []
                st.session_state.qa_pairs = []
                st.session_state.uploaded_file_name = ""
                st.session_state.processing_complete = False
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload section
        st.subheader("📤 Upload PDF Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type=["pdf"],
            help="Upload a PDF document to extract text and generate questions"
        )
        
        if uploaded_file is not None:
            # Check if new file or text needs extraction
            if (not st.session_state.document_text or 
                uploaded_file.name != st.session_state.uploaded_file_name):
                
                with st.spinner("📖 Extracting text from PDF..."):
                    try:
                        document_text = extract_text_from_pdf(uploaded_file)
                        
                        if document_text:
                            st.session_state.document_text = document_text
                            st.session_state.uploaded_file_name = uploaded_file.name
                            st.session_state.generated_questions = []
                            st.session_state.qa_pairs = []
                            st.session_state.processing_complete = False
                            st.success(f"✅ Successfully extracted text from '{uploaded_file.name}'!")
                        else:
                            st.warning("⚠️ No text could be extracted from the PDF. The file might be empty or contain only images.")
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
    
    with col2:
        if st.session_state.document_text:
            st.info(f"**Current Document:** {st.session_state.uploaded_file_name}")
    
    # Display extracted text preview
    if st.session_state.document_text:
        with st.expander("👁️ Preview Extracted Text (first 1000 characters)", expanded=False):
            st.text_area(
                "", 
                st.session_state.document_text[:1000] + "..." if len(st.session_state.document_text) > 1000 else st.session_state.document_text,
                height=200, 
                disabled=True
            )
        
        st.markdown("---")
        
        # Question Generation Section
        st.subheader("🤖 Auto-Generate Questions")
        
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        
        with col_btn1:
            generate_btn = st.button(
                "🎯 Generate Questions from Document", 
                type="primary",
                use_container_width=True
            )
        
        if generate_btn and st.session_state.models_loaded:
            with st.spinner('🔮 Generating questions and answers...'):
                try:
                    # Split document into passages
                    passages = split_into_passages(
                        st.session_state.document_text, 
                        max_words=passage_length
                    )
                    
                    if not passages:
                        st.error("Could not split document into passages.")
                        st.stop()
                    
                    st.info(f"📝 Processing {len(passages)} passages...")
                    
                    # Generate questions and answers
                    all_qa_pairs = []
                    progress_bar = st.progress(0)
                    
                    for idx, passage in enumerate(passages):
                        questions = generate_questions_from_passage(
                            passage, 
                            st.session_state.qg_model, 
                            min_questions=num_questions
                        )
                        
                        for question in questions:
                            answer_result = answer_question(
                                question, 
                                st.session_state.document_text, 
                                st.session_state.qa_model
                            )
                            all_qa_pairs.append({
                                'question': question,
                                'answer': answer_result['answer'],
                                'confidence': answer_result['score']
                            })
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / len(passages))
                    
                    progress_bar.empty()
                    
                    # Remove duplicates while preserving order
                    seen_questions = set()
                    unique_qa_pairs = []
                    for qa in all_qa_pairs:
                        if qa['question'] not in seen_questions:
                            seen_questions.add(qa['question'])
                            unique_qa_pairs.append(qa)
                    
                    st.session_state.qa_pairs = unique_qa_pairs
                    st.session_state.generated_questions = [qa['question'] for qa in unique_qa_pairs]
                    st.session_state.processing_complete = True
                    
                    st.success(f'✅ Generated {len(unique_qa_pairs)} unique questions and answers!')
                    time.sleep(0.5)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error during generation: {str(e)}")
        
        # Display generated Q&A
        if st.session_state.qa_pairs:
            st.markdown("---")
            st.subheader(f"📋 Generated Questions & Answers ({len(st.session_state.qa_pairs)})")
            
            # Download button
            col_download1, col_download2 = st.columns(2)
            
            with col_download1:
                json_data = json.dumps(st.session_state.qa_pairs, indent=4)
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_data,
                    file_name=f"{st.session_state.uploaded_file_name.replace('.pdf', '')}_qa.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col_download2:
                # Format as readable text
                text_output = ""
                for i, qa in enumerate(st.session_state.qa_pairs, 1):
                    text_output += f"Q{i}: {qa['question']}\n"
                    text_output += f"A{i}: {qa['answer']}\n"
                    text_output += f"Confidence: {qa['confidence']}\n\n"
                
                st.download_button(
                    label="📥 Download as Text",
                    data=text_output,
                    file_name=f"{st.session_state.uploaded_file_name.replace('.pdf', '')}_qa.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # Display Q&A with filtering
            show_confidence = st.checkbox("Show confidence scores", value=True)
            min_confidence = st.slider(
                "Minimum confidence threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help="Filter answers by confidence score"
            ) if show_confidence else 0.0
            
            filtered_qa = [qa for qa in st.session_state.qa_pairs if qa['confidence'] >= min_confidence]
            
            st.write(f"Showing {len(filtered_qa)} of {len(st.session_state.qa_pairs)} questions")
            
            for i, qa in enumerate(filtered_qa, 1):
                with st.container():
                    st.markdown(f"**Question {i}:** {qa['question']}")
                    
                    confidence_color = (
                        "green" if qa['confidence'] > 0.7 
                        else "orange" if qa['confidence'] > 0.4 
                        else "red"
                    )
                    
                    if show_confidence:
                        st.markdown(
                            f"**Answer:** {qa['answer']} "
                            f"<span style='color:{confidence_color}'>●</span> "
                            f"*Confidence: {qa['confidence']:.2%}*",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(f"**Answer:** {qa['answer']}")
                    
                    st.markdown("---")
        
        # Custom Question Section
        st.markdown("---")
        st.subheader("❓ Ask Your Own Question")
        
        user_question = st.text_input(
            "Type your question here:",
            placeholder="e.g., What is the main topic of this document?",
            key="user_question_input"
        )
        
        col_ask1, col_ask2 = st.columns([3, 1])
        
        with col_ask1:
            ask_btn = st.button("🔍 Get Answer", type="primary", use_container_width=True)
        
        if ask_btn and user_question:
            if st.session_state.models_loaded:
                with st.spinner('🔎 Finding answer...'):
                    try:
                        answer_result = answer_question(
                            user_question, 
                            st.session_state.document_text, 
                            st.session_state.qa_model
                        )
                        
                        st.markdown("### 💡 Answer")
                        
                        col_ans1, col_ans2 = st.columns([4, 1])
                        
                        with col_ans1:
                            st.success(answer_result['answer'])
                        
                        with col_ans2:
                            confidence_pct = answer_result['score'] * 100
                            st.metric("Confidence", f"{confidence_pct:.1f}%")
                        
                        # Show context if confidence is low
                        if answer_result['score'] < 0.5:
                            st.warning("⚠️ Low confidence answer. The model might not have found a clear answer in the document.")
                    
                    except Exception as e:
                        st.error(f"❌ Error answering question: {str(e)}")
            else:
                st.error("Models not loaded. Please refresh the page.")
    
    else:
        # Initial state - no document uploaded
        st.info("👆 Please upload a PDF document to get started!")
        
        st.markdown("### 🌟 Features")
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            st.markdown("""
            **Auto-Generate Questions**
            - Extract key questions from your document
            - Powered by AI language models
            - Customizable question count
            """)
        
        with col_f2:
            st.markdown("""
            **Ask Custom Questions**
            - Query your document naturally
            - Get instant AI-powered answers
            - Confidence scores included
            """)
        
        with col_f3:
            st.markdown("""
            **Export Results**
            - Download Q&A as JSON
            - Download Q&A as text
            - Easy sharing and storage
            """)

if __name__ == "__main__":
    main()