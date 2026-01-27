# Testing Guide for PDF Q&A System 🧪

## Quick Test Checklist

Use this guide to test the application before deploying to production.

## 1. Installation Testing

### Test 1.1: Dependencies Installation
```bash
# Create fresh virtual environment
python3 -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Check for errors
```

**Expected Result**: All packages install successfully without errors

**Common Issues**:
- Torch installation may take time
- Some systems may need additional build tools

---

## 2. Application Launch Testing

### Test 2.1: Basic Launch
```bash
streamlit run app.py
```

**Expected Result**: 
- App launches without errors
- Opens in browser at `http://localhost:8501`
- Models begin loading

**Check**:
- [ ] No Python errors in terminal
- [ ] Browser page loads
- [ ] Sidebar shows "Loading AI models..."

---

## 3. Model Loading Testing

### Test 3.1: Model Initialization
**Steps**:
1. Wait for models to load (2-3 minutes first time)
2. Check sidebar status

**Expected Result**:
- ✅ "Models ready!" appears in sidebar
- No error messages

**If Failed**:
- Check internet connection (models download from Hugging Face)
- Check available RAM (need ~2GB free)
- Try refreshing the page

---

## 4. PDF Upload Testing

### Test 4.1: Valid PDF Upload
**Sample PDFs to test**:
- Small PDF (1-2 pages)
- Medium PDF (5-10 pages)
- Text-based PDF (not scanned images)

**Steps**:
1. Click "Choose a PDF file"
2. Select a test PDF
3. Wait for extraction

**Expected Result**:
- Progress bar shows extraction progress
- Success message appears
- Preview shows extracted text
- Word/character counts appear in sidebar

**Check**:
- [ ] Text extracted correctly
- [ ] No formatting issues
- [ ] Stats appear in sidebar

### Test 4.2: Invalid File Handling
**Steps**:
1. Try uploading non-PDF file (txt, docx, etc.)

**Expected Result**:
- File rejected by uploader
- Only PDF files accepted

### Test 4.3: Empty/Image-based PDF
**Steps**:
1. Upload PDF with no text (scanned image)

**Expected Result**:
- Warning message about no text extracted
- App doesn't crash

---

## 5. Question Generation Testing

### Test 5.1: Auto-Generate Questions
**Steps**:
1. Upload valid PDF with text
2. Click "Generate Questions from Document"
3. Wait for processing

**Expected Result**:
- Progress bar shows generation progress
- Success message with question count
- Questions appear with answers
- Confidence scores shown

**Check**:
- [ ] Questions are relevant to document
- [ ] Answers make sense
- [ ] Confidence scores between 0-1
- [ ] No duplicate questions

### Test 5.2: Settings Adjustment
**Steps**:
1. Adjust "Questions per passage" slider
2. Adjust "Passage length" slider
3. Generate questions again

**Expected Result**:
- Different number/quality of questions
- Generation time may vary

---

## 6. Custom Question Testing

### Test 6.1: Ask Valid Question
**Steps**:
1. Type question about document content
2. Click "Get Answer"

**Expected Result**:
- Answer appears quickly (< 5 seconds)
- Confidence score shown
- Answer is relevant

**Test Questions to Try**:
- "What is the main topic?"
- "Who are the key people mentioned?"
- "When did this event occur?"
- "What is the conclusion?"

### Test 6.2: Ask Unanswerable Question
**Steps**:
1. Ask question not related to document
2. Click "Get Answer"

**Expected Result**:
- Answer returned (may be low confidence)
- Warning about low confidence
- No error/crash

---

## 7. Export Functionality Testing

### Test 7.1: JSON Download
**Steps**:
1. Generate some questions
2. Click "Download as JSON"

**Expected Result**:
- File downloads successfully
- JSON is valid format
- Contains all Q&A pairs
- Includes confidence scores

**Validation**:
```bash
# Check JSON validity
python -m json.tool downloaded_file.json
```

### Test 7.2: Text Download
**Steps**:
1. Click "Download as Text"

**Expected Result**:
- Text file downloads
- Readable format
- All Q&A included

---

## 8. UI/UX Testing

### Test 8.1: Responsive Design
**Steps**:
1. Resize browser window
2. Test on different screen sizes

**Expected Result**:
- Layout adjusts properly
- No elements cut off
- Buttons remain clickable

### Test 8.2: Confidence Filtering
**Steps**:
1. Generate questions
2. Adjust confidence threshold slider
3. Toggle confidence display

**Expected Result**:
- Questions filter correctly
- Count updates
- Display toggles properly

### Test 8.3: Clear Document
**Steps**:
1. Upload document
2. Click "Clear Document" in sidebar

**Expected Result**:
- Document cleared
- Questions cleared
- Upload prompt returns
- No errors

---

## 9. Performance Testing

### Test 9.1: Document Size Limits
**Test with**:
- 1 page PDF
- 10 page PDF
- 50 page PDF
- 100 page PDF

**Monitor**:
- Processing time
- Memory usage
- Any slowdowns/crashes

**Expected Results**:
- 1-10 pages: Quick (<30 sec)
- 10-50 pages: Moderate (1-2 min)
- 50+ pages: Longer (may need optimization)

### Test 9.2: Concurrent Operations
**Steps**:
1. Generate questions
2. Immediately ask custom question
3. Try downloading while generating

**Expected Result**:
- Operations queue properly
- No crashes
- Graceful handling

---

## 10. Error Handling Testing

### Test 10.1: Network Interruption
**Steps**:
1. Disable internet during model loading
2. Try operations

**Expected Result**:
- Clear error messages
- Suggestion to check connection
- No cryptic errors

### Test 10.2: Memory Limits
**Steps**:
1. Upload very large PDF
2. Monitor memory

**Expected Result**:
- Graceful handling if out of memory
- Clear error message
- App doesn't crash completely

---

## 11. Browser Compatibility Testing

**Test in**:
- [ ] Chrome
- [ ] Firefox
- [ ] Safari
- [ ] Edge

**Check**:
- All features work
- UI displays correctly
- File upload works
- Downloads work

---

## 12. Streamlit Cloud Deployment Testing

### Pre-deployment Checklist
- [ ] requirements.txt is complete
- [ ] No hardcoded paths
- [ ] No local file dependencies
- [ ] Secrets properly handled
- [ ] .gitignore configured

### Post-deployment Testing
1. Deploy to Streamlit Cloud
2. Wait for deployment
3. Run through tests 1-11 above on deployed version

**Monitor**:
- Cold start time (first load)
- Warm start time (subsequent loads)
- Overall performance vs local

---

## Test Results Template

```
Test Date: _______________
Tester: __________________
Environment: Local / Cloud

| Test # | Description | Pass/Fail | Notes |
|--------|-------------|-----------|-------|
| 1.1    | Dependencies|           |       |
| 2.1    | Launch      |           |       |
| 3.1    | Models      |           |       |
| 4.1    | PDF Upload  |           |       |
| 5.1    | Questions   |           |       |
| 6.1    | Custom Q    |           |       |
| 7.1    | Export      |           |       |
| 8.1    | UI/UX       |           |       |
| 9.1    | Performance |           |       |
| 10.1   | Errors      |           |       |

Overall Status: ___________
Issues Found: _____________
```

---

## Quick Smoke Test (5 minutes)

For rapid testing, run these minimal tests:

1. ✅ App launches
2. ✅ Models load
3. ✅ Upload PDF works
4. ✅ Text extracted
5. ✅ Generate 3 questions
6. ✅ Ask custom question
7. ✅ Download JSON
8. ✅ Clear document

If all pass → Ready for deployment!

---

## Troubleshooting Common Test Failures

### Models Won't Load
- Check internet connection
- Clear cache: `~/.cache/huggingface/`
- Restart app

### PDF Won't Upload
- Check file size
- Verify PDF format
- Try different PDF

### Questions Not Relevant
- Adjust passage length
- Try different document
- Check if document has clear content

### Slow Performance
- Reduce question count
- Use smaller documents
- Check system resources

---

**Remember**: Test thoroughly before production deployment!
