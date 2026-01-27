# PDF Question & Answer System 📚

An intelligent PDF document analysis tool that automatically generates questions and provides AI-powered answers using Hugging Face transformers.

## Features ✨

- **PDF Text Extraction**: Upload and extract text from PDF documents
- **Auto-Generate Questions**: AI-powered question generation from document content
- **Custom Q&A**: Ask your own questions about the document
- **Confidence Scores**: See how confident the AI is in its answers
- **Export Results**: Download questions and answers in JSON or text format
- **Beautiful UI**: Clean, modern interface built with Streamlit
- **Progress Tracking**: Visual feedback during processing

## Models Used 🤖

- **Question Generation**: `valhalla/t5-base-qg-hl` - Generates contextual questions from text
- **Question Answering**: `deepset/roberta-base-squad2` - Provides accurate answers with confidence scores

## Installation 🚀

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- 4GB+ RAM recommended (for model loading)

### Local Setup

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the app**
   - The app will automatically open in your browser
   - If not, navigate to `http://localhost:8501`

## Usage Guide 📖

### 1. Upload a PDF Document
- Click the "Choose a PDF file" button
- Select your PDF document
- Wait for text extraction to complete

### 2. Generate Questions
- Adjust settings in the sidebar:
  - Number of questions per passage (1-5)
  - Passage length (50-200 words)
- Click "Generate Questions from Document"
- Wait for AI to analyze and generate Q&A pairs

### 3. Review Generated Q&A
- View all generated questions and answers
- Filter by confidence score
- Toggle confidence score display
- Download results as JSON or text

### 4. Ask Custom Questions
- Type your question in the input field
- Click "Get Answer"
- View the answer with confidence score

## Project Structure 📁

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .streamlit/           # Streamlit config (optional)
    └── config.toml
```

## Deployment to Streamlit Cloud ☁️

### Step 1: Prepare Your Repository

1. **Create a GitHub repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git push -u origin main
   ```

2. **Ensure these files are in your repository:**
   - `app.py`
   - `requirements.txt`
   - `README.md` (optional but recommended)

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)

2. Sign in with GitHub

3. Click "New app"

4. Fill in the details:
   - **Repository**: Select your GitHub repository
   - **Branch**: `main`
   - **Main file path**: `app.py`

5. Click "Deploy"

6. Wait for deployment (usually 5-10 minutes for first deployment)

### Step 3: Configure Settings (Optional)

Create `.streamlit/config.toml` for custom settings:

```toml
[theme]
primaryColor = "#4CAF50"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true
enableCORS = false
```

## Performance Optimization Tips ⚡

### For Local Development

1. **Use CPU-optimized models**
   - The app is configured to use CPU by default
   - Models load once and are cached

2. **Limit document size**
   - Large PDFs may take longer to process
   - Consider splitting very large documents

3. **Adjust passage length**
   - Smaller passages = faster processing
   - Larger passages = better context

### For Production Deployment

1. **Streamlit Cloud Resources**
   - Free tier: 1 GB RAM, 1 CPU core
   - Upgrade if processing large documents frequently

2. **Model Caching**
   - Models are cached using `@st.cache_resource`
   - First load takes ~2-3 minutes
   - Subsequent loads are instant

3. **Memory Management**
   - Clear document when done
   - Limit simultaneous users if self-hosting

## Troubleshooting 🔧

### Common Issues

**Issue**: Models taking too long to load
- **Solution**: First load is slow (2-3 minutes). Models are cached after that.

**Issue**: Out of memory error
- **Solution**: Process smaller documents or upgrade your hosting plan.

**Issue**: No text extracted from PDF
- **Solution**: PDF might be image-based. Use OCR preprocessing or a different PDF.

**Issue**: Low confidence scores
- **Solution**: The answer might not be clearly stated in the document. Try rephrasing your question.

### Error Messages

- **"NLTK data not found"**: The app will automatically download required NLTK data
- **"Model not loaded"**: Refresh the page to retry model loading
- **"Could not find answer"**: The question might not be answerable from the document context

## Customization 🎨

### Modify Models

To use different Hugging Face models, edit these lines in `app.py`:

```python
# Question Generation Model
qg_pipeline = pipeline("text2text-generation", model="your-model-here")

# Question Answering Model
qa_pipeline = pipeline("question-answering", model="your-model-here")
```

### Adjust UI Theme

Modify the CSS in the `st.markdown()` section of `app.py` to customize colors, fonts, and layout.

### Change Question Generation Settings

Edit default values in the sidebar section:
```python
num_questions = st.slider("Questions per passage", min_value=1, max_value=10, value=3)
passage_length = st.slider("Passage length (words)", min_value=50, max_value=300, value=100)
```

## API Rate Limits & Considerations ⚠️

- **Hugging Face Models**: Run locally, no API limits
- **Processing Time**: Depends on document size and hardware
- **Concurrent Users**: Limited by hosting resources

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is open source and available under the MIT License.

## Support 💬

If you encounter any issues or have questions:
1. Check the Troubleshooting section above
2. Review Streamlit documentation: https://docs.streamlit.io
3. Check Hugging Face model documentation

## Acknowledgments 🙏

- [Streamlit](https://streamlit.io/) - Web framework
- [Hugging Face](https://huggingface.co/) - AI models
- [pdfplumber](https://github.com/jsvine/pdfplumber) - PDF extraction
- [NLTK](https://www.nltk.org/) - Natural language processing

## Future Enhancements 🔮

- [ ] Support for multiple PDF uploads
- [ ] Document comparison feature
- [ ] Advanced filtering and search
- [ ] Export to more formats (CSV, Excel)
- [ ] Multi-language support
- [ ] Chat history persistence
- [ ] Batch processing
- [ ] API endpoint creation

---

**Built with ❤️ using Streamlit and Hugging Face Transformers**
