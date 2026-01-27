import streamlit as st
import pdfplumber
import nltk
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import json
from typing import List, Dict
import warnings
import torch

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
def load_models():
    """Load both models at once for faster initialization"""
    models = {}
    
    # Load Question Generation Model
    try:
        qg_model_name = "google/flan-t5-small"  # Using smaller, faster model
        qg_tokenizer = T5Tokenizer.from_pretrained(qg_model_name)
        qg_model = T5ForConditionalGeneration.from_pretrained(qg_model_name)
        models['qg'] = {
            'model': qg_model, 
            'tokenizer': qg_tokenizer, 
            'name': qg_model_name
        }
    except Exception as e:
        st.error(f"Error loading QG model: {e}")
        models['qg'] = None
    
    # Load Question Answering Model
    try:
        qa_model_name = "distilbert-base-uncased-distilled-squad"  # Smaller, faster model
        qa_pipeline = pipeline(
            "question-answering",
            model=qa_model_name,
            device=-1
        )
        models['qa'] = qa_pipeline
    except Exception as e:
        st.error(f"Error loading QA model: {e}")
        models['qa'] = None
    
    return models

# ============================================================================
# OPTIMIZED HELPER FUNCTIONS
# ============================================================================

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from uploaded PDF file - optimized"""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            document_text = ""
            total_pages = len(pdf.pages)
            
            # Only show progress for large PDFs
            if total_pages > 10:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            for idx, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    document_text += text + " "
                
                # Update progress only for large PDFs
                if total_pages > 10:
                    progress = (idx + 1) / total_pages
                    progress_bar.progress(progress)
                    status_text.text(f"Processing page {idx + 1}/{total_pages}")
            
            if total_pages > 10:
                progress_bar.empty()
                status_text.empty()
            
            return document_text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text: {str(e)}")

def split_into_passages(text: str, max_words: int = 150) -> List[str]:
    """Split text into passages - optimized"""
    sentences = nltk.tokenize.sent_tokenize(text)
    passages = []
    current_passage = ""
    
    for sentence in sentences:
        if len(current_passage.split()) + len(sentence.split()) <= max_words:
            current_passage += " " + sentence if current_passage else sentence
        else:
            if current_passage:
                passages.append(current_passage.strip())
            current_passage = sentence
    
    if current_passage:
        passages.append(current_passage.strip())
    
    return passages

def generate_questions_batch(passages: List[str], qg_model_dict, questions_per_passage: int = 2) -> List[str]:
    """Generate questions from multiple passages - optimized batch processing"""
    model = qg_model_dict['model']
    tokenizer = qg_model_dict['tokenizer']
    all_questions = []
    
    for passage in passages:
        try:
            # Prepare input with shorter max length for speed
            input_text = f"generate questions: {passage}"
            inputs = tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=256,  # Reduced for speed
                truncation=True
            )
            
            # Generate with optimized settings
            outputs = model.generate(
                **inputs,
                max_length=128,  # Shorter output for speed
                num_beams=2,  # Reduced beams for speed
                early_stopping=True,
                do_sample=False  # Deterministic for consistency
            )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=False)
            result = result.replace('</s>', '').replace('<pad>', '').strip()
            
            # Parse questions
            questions = [q.strip() for q in result.split('<sep>') if len(q.strip()) > 10]
            
            # Take only requested number
            all_questions.extend(questions[:questions_per_passage])
            
        except Exception as e:
            continue
    
    # Remove duplicates while preserving order
    seen = set()
    unique_questions = []
    for q in all_questions:
        q_lower = q.lower()
        if q_lower not in seen and len(q) > 10:
            seen.add(q_lower)
            unique_questions.append(q)
    
    return unique_questions

def answer_question_fast(question: str, context: str, qa_pipeline) -> Dict:
    """Answer question with optimized context handling"""
    try:
        # Limit context for speed
        max_words = 500
        words = context.split()
        if len(words) > max_words:
            context = ' '.join(words[:max_words])
        
        result = qa_pipeline({
            'question': question,
            'context': context
        })
        
        return {
            'answer': result['answer'],
            'confidence': round(result['score'], 3)
        }
    except Exception as e:
        return {
            'answer': f"Could not find answer: {str(e)}",
            'confidence': 0.0
        }

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'document_text' not in st.session_state:
    st.session_state.document_text = ""
if 'qa_pairs' not in st.session_state:
    st.session_state.qa_pairs = []
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = ""
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'extraction_done' not in st.session_state:
    st.session_state.extraction_done = False

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
            with st.spinner("🔄 Loading AI models..."):
                load_nltk_data()
                models = load_models()
                
                if models['qg'] and models['qa']:
                    st.session_state.qg_model = models['qg']
                    st.session_state.qa_model = models['qa']
                    st.session_state.models_loaded = True
                    st.success("✅ Models loaded!")
                else:
                    st.error("❌ Failed to load models.")
                    st.stop()
        else:
            st.success("✅ Models ready!")
        
        st.markdown("---")
        
        # Settings
        st.subheader("Question Generation")
        num_questions_per_passage = st.slider(
            "Questions per passage", 
            min_value=1, 
            max_value=3, 
            value=2,
            help="Fewer questions = faster processing"
        )
        
        max_passages = st.slider(
            "Max passages to process", 
            min_value=5, 
            max_value=30, 
            value=15,
            help="Fewer passages = faster processing"
        )
        
        st.markdown("---")
        
        # Document info
        if st.session_state.document_text:
            st.subheader("📄 Document Info")
            word_count = len(st.session_state.document_text.split())
            st.metric("Words", f"{word_count:,}")
            
            if st.button("🗑️ Clear Document"):
                st.session_state.document_text = ""
                st.session_state.qa_pairs = []
                st.session_state.uploaded_file_name = ""
                st.session_state.extraction_done = False
                st.rerun()
    
    # Main content area
    st.subheader("📤 Upload PDF Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type=["pdf"],
        help="Upload a PDF document"
    )
    
    # Handle file upload
    if uploaded_file is not None:
        # Only extract if it's a new file
        if uploaded_file.name != st.session_state.uploaded_file_name:
            with st.spinner("📖 Extracting text..."):
                try:
                    document_text = extract_text_from_pdf(uploaded_file)
                    
                    if document_text:
                        st.session_state.document_text = document_text
                        st.session_state.uploaded_file_name = uploaded_file.name
                        st.session_state.qa_pairs = []
                        st.session_state.extraction_done = True
                        st.success(f"✅ Extracted text from '{uploaded_file.name}'!")
                    else:
                        st.warning("⚠️ No text extracted. PDF might be image-based.")
                except Exception as e:
                    st.error(f"❌ {str(e)}")
        
        # Display document info
        if st.session_state.document_text:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📝 Words", f"{len(st.session_state.document_text.split()):,}")
            with col2:
                st.metric("📊 Characters", f"{len(st.session_state.document_text):,}")
            with col3:
                sentences = nltk.tokenize.sent_tokenize(st.session_state.document_text)
                st.metric("📑 Sentences", f"{len(sentences):,}")
            
            with st.expander("👁️ Preview Text", expanded=False):
                preview = st.session_state.document_text[:1000]
                if len(st.session_state.document_text) > 1000:
                    preview += "..."
                st.text_area("", preview, height=200, disabled=True, label_visibility="collapsed")
            
            st.markdown("---")
            
            # Question Generation Section
            st.subheader("🤖 Generate Questions")
            
            if st.button("🎯 Generate Questions & Answers", type="primary", use_container_width=True):
                if not st.session_state.models_loaded:
                    st.error("Models not loaded. Please refresh the page.")
                    st.stop()
                
                # Clear previous results
                st.session_state.qa_pairs = []
                
                with st.spinner('🔮 Generating questions and answers...'):
                    try:
                        # Split into passages
                        passages = split_into_passages(st.session_state.document_text, max_words=150)
                        
                        # Limit passages for speed
                        passages = passages[:max_passages]
                        
                        st.info(f"📝 Processing {len(passages)} passages...")
                        
                        # Generate questions (batch processing)
                        questions = generate_questions_batch(
                            passages,
                            st.session_state.qg_model,
                            questions_per_passage=num_questions_per_passage
                        )
                        
                        if not questions:
                            st.warning("No questions generated. Try adjusting settings.")
                            st.stop()
                        
                        st.info(f"💬 Generated {len(questions)} questions. Answering...")
                        
                        # Answer questions with progress bar
                        progress_bar = st.progress(0)
                        qa_pairs = []
                        
                        for idx, question in enumerate(questions):
                            answer_result = answer_question_fast(
                                question,
                                st.session_state.document_text,
                                st.session_state.qa_model
                            )
                            
                            qa_pairs.append({
                                'question': question,
                                'answer': answer_result['answer'],
                                'confidence': answer_result['confidence']
                            })
                            
                            progress_bar.progress((idx + 1) / len(questions))
                        
                        progress_bar.empty()
                        
                        # Store results
                        st.session_state.qa_pairs = qa_pairs
                        
                        st.success(f'✅ Generated {len(qa_pairs)} Q&A pairs!')
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        import traceback
                        with st.expander("Error details"):
                            st.code(traceback.format_exc())
            
            # Display Q&A
            if st.session_state.qa_pairs:
                st.markdown("---")
                st.subheader(f"📋 Questions & Answers ({len(st.session_state.qa_pairs)})")
                
                # Download buttons
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    json_data = json.dumps(st.session_state.qa_pairs, indent=4)
                    filename_base = st.session_state.uploaded_file_name.replace('.pdf', '')
                    st.download_button(
                        "📥 Download JSON",
                        json_data,
                        f"{filename_base}_qa.json",
                        "application/json",
                        use_container_width=True
                    )
                
                with col_d2:
                    text_output = "\n\n".join([
                        f"Q{i+1}: {qa['question']}\nA{i+1}: {qa['answer']}\nConfidence: {qa['confidence']:.0%}"
                        for i, qa in enumerate(st.session_state.qa_pairs)
                    ])
                    st.download_button(
                        "📥 Download Text",
                        text_output,
                        f"{filename_base}_qa.txt",
                        "text/plain",
                        use_container_width=True
                    )
                
                st.markdown("---")
                
                # Display options
                col_o1, col_o2 = st.columns(2)
                with col_o1:
                    show_confidence = st.checkbox("Show confidence scores", value=True)
                with col_o2:
                    view_mode = st.radio(
                        "View:",
                        ["Expandable", "Show All"],
                        horizontal=True
                    )
                
                min_confidence = st.slider(
                    "Min confidence",
                    0.0, 1.0, 0.0, 0.1
                )
                
                filtered_qa = [qa for qa in st.session_state.qa_pairs if qa['confidence'] >= min_confidence]
                st.write(f"Showing {len(filtered_qa)} of {len(st.session_state.qa_pairs)} questions")
                
                # Display Q&A
                if view_mode == "Expandable":
                    for i, qa in enumerate(filtered_qa, 1):
                        conf_icon = "🟢" if qa['confidence'] > 0.7 else "🟠" if qa['confidence'] > 0.4 else "🔴"
                        conf_text = f" {conf_icon} {qa['confidence']:.0%}" if show_confidence else ""
                        
                        with st.expander(f"**Q{i}:** {qa['question']}{conf_text}"):
                            st.markdown("### Answer")
                            st.info(qa['answer'])
                            if show_confidence:
                                st.metric("Confidence", f"{qa['confidence']:.1%}")
                else:
                    for i, qa in enumerate(filtered_qa, 1):
                        st.markdown(f"### Question {i}")
                        st.markdown(f"**Q:** {qa['question']}")
                        if show_confidence:
                            conf_color = "green" if qa['confidence'] > 0.7 else "orange" if qa['confidence'] > 0.4 else "red"
                            st.markdown(
                                f"**A:** {qa['answer']} "
                                f"<span style='color:{conf_color}'>●</span> "
                                f"*{qa['confidence']:.0%}*",
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(f"**A:** {qa['answer']}")
                        st.markdown("---")
            
            # Custom question section
            st.markdown("---")
            st.subheader("❓ Ask Your Own Question")
            
            user_question = st.text_input(
                "Type your question:",
                placeholder="What is the main topic?",
                key="user_q"
            )
            
            if st.button("🔍 Get Answer", disabled=not user_question):
                with st.spinner('Finding answer...'):
                    answer_result = answer_question_fast(
                        user_question,
                        st.session_state.document_text,
                        st.session_state.qa_model
                    )
                    
                    col_a1, col_a2 = st.columns([3, 1])
                    with col_a1:
                        st.success(answer_result['answer'])
                    with col_a2:
                        st.metric("Confidence", f"{answer_result['confidence']:.1%}")
    
    else:
        st.info("👆 Please upload a PDF to get started!")
        
        st.markdown("### 🌟 Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Auto-Generate Questions**
            - AI-powered question generation
            - Customizable settings
            - Fast processing
            """)
        
        with col2:
            st.markdown("""
            **Ask Custom Questions**
            - Natural language queries
            - Instant answers
            - Confidence scores
            """)
        
        with col3:
            st.markdown("""
            **Export Results**
            - Download as JSON
            - Download as text
            - Easy sharing
            """)

if __name__ == "__main__":
    main()