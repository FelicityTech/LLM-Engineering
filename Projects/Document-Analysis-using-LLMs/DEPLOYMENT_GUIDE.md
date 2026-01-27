# Streamlit Cloud Deployment Guide 🚀

Complete step-by-step guide to deploy your PDF Q&A System to Streamlit Cloud.

## Prerequisites ✅

Before you begin, ensure you have:
- [ ] GitHub account
- [ ] Your app tested locally and working
- [ ] All files ready (app.py, requirements.txt, README.md, etc.)

## Step 1: Prepare Your Repository 📦

### 1.1 Create a GitHub Repository

**Option A: Via GitHub Website**
1. Go to [github.com](https://github.com)
2. Click the "+" icon in top right
3. Select "New repository"
4. Fill in details:
   - Repository name: `pdf-qa-system` (or your choice)
   - Description: "AI-powered PDF Question & Answer System"
   - Public or Private: Your choice (Public is fine)
   - Don't initialize with README (we have one)
5. Click "Create repository"

**Option B: Via Command Line**
```bash
# Initialize git in your project folder
cd /path/to/your/project
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: PDF Q&A System"

# Add remote (replace with your GitHub URL)
git remote add origin https://github.com/yourusername/pdf-qa-system.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 1.2 Verify Repository Contents

Make sure your repository contains:
```
pdf-qa-system/
├── app.py                    # Main application
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── .gitignore               # Git ignore rules
├── .streamlit/
│   └── config.toml          # Streamlit config
├── run.sh                   # Linux/Mac startup script
└── run.bat                  # Windows startup script
```

## Step 2: Configure for Streamlit Cloud ⚙️

### 2.1 Optimize requirements.txt

Your `requirements.txt` should be:
```txt
streamlit==1.31.0
pdfplumber==0.10.3
nltk==3.8.1
transformers==4.36.2
torch==2.1.2
sentencepiece==0.1.99
protobuf==4.25.2
```

**Important Notes**:
- Pin versions to ensure reproducibility
- Torch is large (~700MB) - deployment will take longer
- First deployment may take 10-15 minutes

### 2.2 Verify .gitignore

Ensure `.gitignore` excludes:
```
__pycache__/
*.pyc
venv/
.venv/
.streamlit/secrets.toml
.cache/
*.log
```

### 2.3 Create Optional packages.txt (if needed)

If you need system packages, create `packages.txt`:
```
poppler-utils
```

## Step 3: Deploy to Streamlit Cloud 🌐

### 3.1 Access Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "Sign in with GitHub"
3. Authorize Streamlit to access your GitHub account

### 3.2 Create New App

1. Click "New app" button (top right)
2. You'll see a form with three sections:

**Repository Section**:
- Repository: Select `yourusername/pdf-qa-system`
- Branch: `main` (or your default branch)
- Main file path: `app.py`

**Advanced Settings** (click to expand):
- Python version: 3.10 (recommended)
- Secrets: Leave empty (none needed for this app)
- Environment variables: Leave empty

3. Click "Deploy!"

### 3.3 Monitor Deployment

You'll see a deployment log showing:
```
Building...
Installing requirements...
Starting app...
```

**Timeline**:
- Initial setup: 2-3 minutes
- Installing requirements: 5-10 minutes (torch is large!)
- Starting app: 1-2 minutes
- Total: ~10-15 minutes for first deployment

**What's happening**:
1. Streamlit creates a container
2. Installs Python and dependencies
3. Downloads AI models (first run only)
4. Starts your app

### 3.4 First Launch

Once deployed, your app will:
1. Load in the browser
2. Download NLTK data (automatic)
3. Load AI models (2-3 minutes)
4. Show "Models ready!" in sidebar

**Important**: The first user to access the app each day may experience a "cold start" (~2-3 minutes). Subsequent users will have a faster experience.

## Step 4: Configure Your App 🔧

### 4.1 Custom Domain (Optional)

**Free subdomain** (included):
- Your app gets: `your-app-name.streamlit.app`
- Can be changed in settings

**Custom domain** (Community Cloud only):
1. Go to app settings
2. Add your domain
3. Configure DNS records

### 4.2 App Settings

Access via the ⚙️ icon on your deployed app:

**General**:
- App name
- App URL
- Description

**Sharing**:
- Public (anyone can access)
- Private (requires invitation)

**Resources**:
- View CPU/Memory usage
- Monitor performance

**Secrets** (if needed later):
- Add API keys
- Add credentials
- Format: TOML

### 4.3 Monitoring

Check your app's health:
1. View app logs (bottom right of settings)
2. Monitor resource usage
3. Check error reports

## Step 5: Update Your App 🔄

### 5.1 Making Changes

When you update your code:

```bash
# Make changes to your files
git add .
git commit -m "Description of changes"
git push
```

Streamlit Cloud will automatically:
1. Detect the push
2. Rebuild the app
3. Redeploy (usually 2-5 minutes)

### 5.2 Force Reboot

If something seems stuck:
1. Go to app settings (⚙️)
2. Click "Reboot app"
3. Wait for restart

### 5.3 Clear Cache

If models seem outdated:
1. Settings → Advanced
2. Click "Clear cache"
3. Reboot app

## Step 6: Troubleshooting 🔧

### Common Deployment Issues

#### Issue 1: Deployment Fails
**Symptoms**: Build fails, red error messages

**Solutions**:
- Check `requirements.txt` format
- Ensure all dependencies are available
- Check Python version compatibility
- Review deployment logs for specific errors

#### Issue 2: App Crashes on Start
**Symptoms**: App loads but immediately crashes

**Solutions**:
- Check for hardcoded file paths
- Verify all imports are in requirements.txt
- Look for local-only dependencies
- Check memory limits (1GB free tier)

#### Issue 3: Models Won't Load
**Symptoms**: "Model not loaded" error

**Solutions**:
- Check internet connectivity in logs
- Verify Hugging Face models are accessible
- Try rebootin app
- Check model names are correct

#### Issue 4: PDF Upload Fails
**Symptoms**: Upload button doesn't work

**Solutions**:
- Check file size limits (200MB default)
- Verify pdfplumber is installed
- Check for CORS issues in logs

#### Issue 5: Slow Performance
**Symptoms**: App is very slow

**Solutions**:
- Optimize passage length
- Reduce question count
- Consider upgrading plan
- Cache more aggressively

### Viewing Logs

Access logs:
1. Click ⚙️ (settings)
2. Scroll to "Logs" section
3. View real-time logs
4. Download logs for analysis

Common log patterns:
```
✅ Good: "Streamlit app started"
✅ Good: "Models loaded successfully"
⚠️  Warning: "Memory usage high"
❌ Error: "ModuleNotFoundError"
❌ Error: "Out of memory"
```

## Step 7: Best Practices 🌟

### 7.1 Performance Optimization

**Cache everything possible**:
```python
@st.cache_resource
@st.cache_data
```

**Minimize model reloads**:
- Load once at startup
- Keep in session state
- Don't reload unnecessarily

**Optimize UI**:
- Use spinners for long operations
- Show progress bars
- Provide user feedback

### 7.2 User Experience

**Add helpful messages**:
- Loading indicators
- Error messages with solutions
- Success confirmations
- Usage instructions

**Handle errors gracefully**:
```python
try:
    # Operation
except Exception as e:
    st.error(f"Something went wrong: {e}")
    st.info("Try: [suggestion]")
```

### 7.3 Monitoring & Analytics

**Track usage**:
- Monitor deployment logs
- Check resource usage
- Review error patterns
- Gather user feedback

**Set up alerts** (if available):
- High memory usage
- Frequent errors
- Deployment failures

## Step 8: Scaling Considerations 📈

### Free Tier Limits
- **RAM**: 1 GB
- **CPU**: 1 vCPU
- **Bandwidth**: Generous
- **Storage**: Ephemeral (no persistent storage)
- **Concurrent users**: Limited

### When to Upgrade

Consider upgrading if:
- Multiple concurrent users
- Large documents (50+ pages)
- High traffic
- Need more resources

### Upgrade Options
1. **Streamlit Community Cloud**: Free tier
2. **Streamlit Cloud (Paid)**: More resources, custom domains
3. **Self-hosting**: Full control, unlimited scaling

## Step 9: Sharing Your App 🔗

### 9.1 Get Your URL

Your app URL: `https://your-app-name.streamlit.app`

### 9.2 Share Options

**Public sharing**:
- Copy URL
- Share on social media
- Add to GitHub README
- Include in portfolio

**Private sharing**:
- Settings → Sharing
- Add email addresses
- Send invitations

### 9.3 Embed in Website

Add to your site:
```html
<iframe src="https://your-app.streamlit.app" 
        width="100%" 
        height="800px">
</iframe>
```

## Step 10: Maintenance 🛠️

### Regular Tasks

**Weekly**:
- Check app status
- Review error logs
- Monitor performance

**Monthly**:
- Update dependencies
- Check for security updates
- Review user feedback

**As Needed**:
- Fix bugs
- Add features
- Improve performance

### Keeping Dependencies Updated

```bash
# Check for updates
pip list --outdated

# Update requirements.txt
pip freeze > requirements.txt

# Test locally first!
# Then push to GitHub
```

## Quick Reference 📋

### Useful Commands

```bash
# Local testing
streamlit run app.py

# Check Python version
python --version

# List installed packages
pip list

# Check package versions
pip show streamlit

# Git status
git status

# Push changes
git add . && git commit -m "Update" && git push
```

### Useful URLs

- Streamlit Cloud: https://share.streamlit.io
- Streamlit Docs: https://docs.streamlit.io
- Hugging Face: https://huggingface.co
- Your App: https://your-app.streamlit.app
- GitHub Repo: https://github.com/yourusername/pdf-qa-system

### Support Resources

- Streamlit Forum: https://discuss.streamlit.io
- Streamlit Discord: https://discord.gg/streamlit
- GitHub Issues: https://github.com/streamlit/streamlit/issues
- Hugging Face Forums: https://discuss.huggingface.co

## Success Checklist ✅

Before considering deployment complete:

- [ ] App deploys without errors
- [ ] Models load successfully
- [ ] PDF upload works
- [ ] Question generation works
- [ ] Custom questions work
- [ ] Downloads work
- [ ] UI looks good
- [ ] No console errors
- [ ] Performance acceptable
- [ ] Error handling works
- [ ] URL is accessible
- [ ] Shared with users
- [ ] Documentation updated

## Next Steps 🎯

After successful deployment:

1. **Monitor**: Watch for issues in first few days
2. **Gather Feedback**: Ask users for input
3. **Iterate**: Make improvements based on feedback
4. **Promote**: Share your app!
5. **Maintain**: Keep dependencies updated

---

**Congratulations! Your PDF Q&A System is now live! 🎉**

Share it with the world: `https://your-app.streamlit.app`

Need help? Check the Troubleshooting section or reach out to the Streamlit community!
