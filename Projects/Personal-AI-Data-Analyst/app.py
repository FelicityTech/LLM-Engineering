# app.py
import streamlit as st
import os

# Load .env locally only (Streamlit Cloud ignores safely)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from analyst import (
    load_data,
    suggest_prompts,
    prompt_to_code,
    run_code,
    ask_llm,
    get_df_summary,
    validate_dataframe,
    extract_python_code,
    get_api_status,
    get_available_providers,
    AnalystConfig,
    FreeAPIConfig,
    _is_code_incomplete,
    _continuation_prompt
)
import pandas as pd
from datetime import datetime
import traceback
import time

# ────────────────────────────────────────────────
# Page config & styling
# ────────────────────────────────────────────────
# At the top of the file
MAX_PREVIEW_ROWS_DEFAULT = 150
HISTORY_DISPLAY_LIMIT = 10
PROMPT_TRUNCATE_LENGTH = 90
TOKEN_DISPLAY_FREQUENCY = 5


ERROR_NO_API_KEY = "❌ This analysis requires AI, but no API keys are configured!"
ERROR_NO_CODE = "❌ Could not extract valid Python code from AI response" 


st.set_page_config(
    page_title="FelicityTech AI Data Analyst",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.6rem;
        font-weight: bold;
        color: #1e88e5;
        margin-bottom: 0.4rem;
    }
    .creator {
        font-size: 0.95rem;
        color: #555;
        margin-bottom: 1.2rem;
    }
    .linkedin-btn {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.55rem 1.1rem;
        background: #0077b5;
        color: white !important;
        text-decoration: none;
        border-radius: 6px;
        font-weight: 500;
    }
    .linkedin-btn:hover {
        background: #005f8d;
    }
    .stSpinner > div > div {
        border-top-color: #1e88e5 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🧠 FelicityTech AI Data Analyst</p>', unsafe_allow_html=True)
st.markdown(
    '<div class="creator">'
    'Built by <strong>FelicityTech</strong> • '
    '<a href="https://www.linkedin.com/in/solomon-eniola-adegoke/" target="_blank" class="linkedin-btn">'
    '🔗 Connect on LinkedIn</a></div>',
    unsafe_allow_html=True
)
st.caption("Upload data → get insights!")

# ────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    # API Status Check
    api_status = get_api_status()
    available_providers = get_available_providers()
    
    with st.expander("🔑 API Keys Status", expanded=not bool(available_providers)):
        st.text(api_status)
        
        if not available_providers:
            st.warning("⚠️ No API keys found - Custom AI prompts disabled")
            st.markdown("""
            **Add to your `.env` file for Custom AI:**
            ```
            # Choose at least ONE (all are 100% FREE):
            
            GEMINI_API_KEY=your_key_here
            GROQ_API_KEY=your_key_here
            HF_API_KEY=your_key_here
            ```
            
            **Get FREE keys:**
            - Gemini: [aistudio.google.com](https://aistudio.google.com)
            - Groq: [console.groq.com](https://console.groq.com)
            - Hugging Face: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
            
            **Note:** Suggested prompts work without any API keys!
            """)
        else:
            st.success("✅ AI available for custom prompts")

    with st.expander("🤖 AI Settings", expanded=bool(available_providers)):
        if available_providers:
            # Provider selection
            provider_options = {k: v["name"] for k, v in available_providers.items()}
            selected_provider = st.selectbox(
                "API Provider",
                options=list(provider_options.keys()),
                format_func=lambda x: provider_options[x],
                help="All providers are completely FREE (no credit card needed)"
            )
            
            # Model selection for chosen provider
            model_options = available_providers[selected_provider]["models"]
            selected_model = st.selectbox(
                "Model",
                options=model_options,
                help=f"All {provider_options[selected_provider]} models are FREE"
            )
            
            llm_timeout = st.slider("Max wait time (seconds)", 30, 300, 120, 30)
            
            # Tips for each provider
            if selected_provider == "gemini":
                st.info("💡 **Gemini**: Best for data analysis, large context (1M tokens)")
            elif selected_provider == "groq":
                st.info("💡 **Groq**: Super fast inference (300+ tokens/sec)")
        else:
            st.info("⚡ **Suggested prompts** work without AI - just upload data and select!")
            selected_provider = None
            selected_model = None
            llm_timeout = 120
    
    with st.expander("📊 Display"):
        show_code = st.checkbox("Show generated code", value=True)
        max_preview_rows = st.slider("Max preview rows", 10, 800, 150, 50)

    st.divider()
    st.caption(f"Session • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────
if 'df' not in st.session_state:
    st.session_state.df = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

# ────────────────────────────────────────────────
# Upload & load data
# ────────────────────────────────────────────────
st.subheader("1. Upload your dataset")
uploaded_file = st.file_uploader(
    label="CSV, Excel, JSON, Parquet, Feather, HDF5, XML, TXT",
    type=["csv","xls","xlsx","json","parquet","feather","txt","xml","h5","hdf5"],
    help="Up to ~1 million rows recommended"
)

if uploaded_file is None:
    st.info("👆 Upload a file to start exploring")
    
    with st.expander("🎓 Example Analyses"):
        st.markdown("""
        **⚡ Suggested Prompts (NO AI NEEDED - Work Instantly):**
        - Dataset summary & statistics
        - Missing value analysis
        - Data type information
        - Column distribution analysis
        - Histograms & scatter plots
        - Correlation heatmaps
        - Top/bottom N rows
        - Duplicate detection
        - Outlier detection
        - Group-by aggregations
        - Time series plots
        - **And 20+ more built-in templates!**
        
        **🤖 Custom AI Prompts (Requires FREE API Key):**
        - "Show top 10 products by revenue with percentage"
        - "Create monthly sales trend with forecast"
        - "Find customers with unusual purchasing patterns"
        - "Compare categories side-by-side with statistical tests"
        - "Generate executive summary report"
        
        **💡 No Credit Card Ever Required!**
        """)
    st.stop()

# Load
try:
    with st.spinner("📖 Reading and validating data..."):
        df = load_data(uploaded_file)
        valid, msg = validate_dataframe(df)
        if not valid:
            st.error(f"❌ {msg}")
            st.stop()
        st.session_state.df = df
except Exception as e:
    st.error(f"❌ Failed to load file\n{str(e)}")
    with st.expander("🔍 Traceback"):
        st.code(traceback.format_exc())
    st.stop()

# Quick stats
cols = st.columns(3)
cols[0].metric("Rows", f"{len(df):,}")
cols[1].metric("Columns", df.shape[1])
cols[2].metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

with st.expander("👀 Data preview", expanded=False):
    n = st.slider("Rows to show", 5, min(600, len(df)), 120, key="preview_rows")
    st.dataframe(df.head(n), use_container_width=True)
    
    st.caption("**Column Types:**")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Non-Null': df.count().values,
        'Unique': [df[col].nunique() if len(df) < 100000 else "N/A" for col in df.columns]
    })
    st.dataframe(col_info, use_container_width=True, height=200)

st.divider()

# ────────────────────────────────────────────────
# Prompt selection
# ────────────────────────────────────────────────
st.subheader("2. Choose or write an analysis")

try:
    suggestions = suggest_prompts(df, max_suggestions=25)
except Exception:
    suggestions = ["Summarize this dataset"]

left, right = st.columns([3.5, 1.4])

current_prompt = None

with left:
    tab1, tab2 = st.tabs(["⚡ Suggested (Instant - No AI)", "✍️ Custom (Requires AI)"])

    with tab1:
        selected = st.selectbox(
            "Pick a pre-built analysis (works without any API keys)",
            suggestions,
            key="suggested_prompt",
            help="These run instantly without API calls - just template-based code"
        )
        current_prompt = selected
        st.session_state.active_tab = 0

    with tab2:
        if not available_providers:
            st.warning("🔑 **Custom AI prompts require an API key**")
            st.info("👈 Add a FREE API key in the sidebar to unlock custom analysis")
            st.markdown("""
            Meanwhile, try the **Suggested prompts** tab - they work instantly without any setup!
            """)
        else:
            custom_text = st.text_area(
                "Describe your analysis in plain English",
                height=120,
                placeholder="Examples:\n• Show monthly revenue trends\n• Top 10 customers by total spend\n• Compare sales across regions\n• Find outliers in price column",
                key="custom_prompt",
                help="Uses FREE AI APIs (Gemini/Groq/HuggingFace)"
            )
            if custom_text.strip():
                current_prompt = custom_text.strip()
                st.session_state.active_tab = 1

with right:
    st.markdown("**Actions**")
    run_clicked = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    if st.button("⟳ Refresh", use_container_width=True):
        st.rerun()
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.success("History cleared!")

if current_prompt:
    st.caption("📝 **Current request:**")
    st.code(current_prompt, language=None)

# ────────────────────────────────────────────────
# Execute analysis
# ────────────────────────────────────────────────
if run_clicked and current_prompt:
    with st.spinner("🔄 Preparing analysis..."):
        try:
            # STEP 1: Try template first (NO AI - ALWAYS FIRST)
            code = prompt_to_code(current_prompt, df)

            if code:
                # ✅ TEMPLATE MATCH - Run immediately WITHOUT AI
                st.info("✅ Using built-in template (no AI needed)", icon="⚡")
                if show_code:
                    with st.expander("📄 Generated code (template)"):
                        st.code(code, "python")
                
                # Execute template code directly
                st.subheader("📊 Results")
                with st.spinner("⚙️ Running code..."):
                    result = run_code(df, code)
                
                # Display results
                if result["type"] == "dataframe":
                    st.dataframe(result["df"], use_container_width=True)
                    csv_bytes = result["df"].to_csv(index=False).encode("utf-8")
                    fname = f"result_{datetime.now():%Y%m%d_%H%M%S}.csv"
                    st.download_button("💾 Download as CSV", csv_bytes, fname, "text/csv")

                elif result["type"] == "image":
                    st.image(result["path"], use_column_width=True)
                    with open(result["path"], "rb") as imgf:
                        st.download_button(
                            "💾 Download Chart",
                            imgf,
                            file_name=f"chart_{datetime.now():%Y%m%d_%H%M%S}.png",
                            mime="image/png"
                        )

                elif result["type"] == "text":
                    if "error" in result["output"].lower() or "❌" in result["output"]:
                        st.error("**Output:**")
                        st.code(result["output"])
                    else:
                        st.success("**Output:**")
                        st.text(result["output"])
                else:
                    st.info(f"Result type: {result.get('type', 'unknown')}")
                    st.write(result)
                
                # Add to history
                st.session_state.history.append({
                    "timestamp": datetime.now(),
                    "prompt": current_prompt[:90] + ("..." if len(current_prompt) > 90 else ""),
                    "result_type": result["type"],
                    "used_ai": False,
                    "provider": None
                })
                
                st.success("✅ Analysis complete!", icon="🎉")

            else:
                # STEP 2: No template match - TRY AI (if available)
                if not available_providers:
                    st.error("❌ This analysis requires AI, but no API keys are configured!", icon="🔑")
                    st.info("**To use custom prompts:**\n1. Get a FREE API key (see sidebar)\n2. Add to `.env` file\n3. Restart the app")
                    st.warning("**Or:** Try one of the **Suggested prompts** - they work without any API keys!")
                    st.stop()

                df_summary = get_df_summary(df)

                st.subheader(f"🤖 {available_providers[selected_provider]['name']} is generating code...")
                status_container = st.status(f"Calling {selected_provider.upper()} API...", expanded=True)
                live_output = st.empty()

                full_response = ""
                start_ts = time.time()
                token_count = 0
                max_retries = 2  # Allow continuation attempts

                for attempt in range(max_retries + 1):
                    for token in ask_llm(
                        prompt=current_prompt,
                        df_info=df_summary,
                        provider=selected_provider,
                        model=selected_model,
                        timeout=llm_timeout
                    ):
                        full_response += token
                        token_count += 1

                        if token_count % 5 == 0:
                            live_output.markdown(full_response + " ▌")

                        # Check for error messages
                        if token.startswith("["):
                            status_container.update(label=f"❌ {token}", state="error", expanded=False)
                            live_output.error(token)
                            
                            # Suggest alternatives on error
                            other_providers = [k for k in available_providers.keys() if k != selected_provider]
                            if other_providers:
                                st.info(f"💡 **Try switching to:** {', '.join([available_providers[p]['name'] for p in other_providers])}")
                            else:
                                st.info("💡 **Get more FREE API keys** from the sidebar guide!")
                            st.stop()

                    code = extract_python_code(full_response)
                    
                    # Check if code is complete
                    if code and not _is_code_incomplete(code):
                        break  # ✅ Code is complete
                    
                    # Need continuation
                    if attempt < max_retries:
                        current_prompt = _continuation_prompt()
                        if time.time() - start_ts > llm_timeout + 30:
                            raise TimeoutError(f"Generation exceeded timeout ({llm_timeout}s)")
                    else:
                        break  # Max retries reached

                live_output.markdown(full_response)
                status_container.update(label="✅ Code generation complete", state="complete", expanded=False)

                if not code:
                    st.error("❌ Could not extract valid Python code from AI response")
                    with st.expander("🔍 Raw AI output"):
                        st.text(full_response)
                    st.info("💡 **Try:** Different model or simpler prompt")
                    st.stop()

                if show_code:
                    with st.expander("📄 AI-generated code"):
                        st.code(code, "python")

                # STEP 3: Execute AI-generated code
                st.subheader("📊 Results")
                
                with st.spinner("⚙️ Running code..."):
                    result = run_code(df, code)

                # STEP 4: Display results
                if result["type"] == "dataframe":
                    st.dataframe(result["df"], use_container_width=True)
                    csv_bytes = result["df"].to_csv(index=False).encode("utf-8")
                    fname = f"result_{datetime.now():%Y%m%d_%H%M%S}.csv"
                    st.download_button("💾 Download as CSV", csv_bytes, fname, "text/csv")

                elif result["type"] == "image":
                    st.image(result["path"], use_column_width=True)
                    with open(result["path"], "rb") as imgf:
                        st.download_button(
                            "💾 Download Chart",
                            imgf,
                            file_name=f"chart_{datetime.now():%Y%m%d_%H%M%S}.png",
                            mime="image/png"
                        )

                elif result["type"] == "text":
                    if "error" in result["output"].lower() or "❌" in result["output"]:
                        st.error("**Output:**")
                        st.code(result["output"])
                    else:
                        st.success("**Output:**")
                        st.text(result["output"])

                else:
                    st.info(f"Result type: {result.get('type', 'unknown')}")
                    st.write(result)

                # Add to history
                st.session_state.history.append({
                    "timestamp": datetime.now(),
                    "prompt": current_prompt[:90] + ("..." if len(current_prompt) > 90 else ""),
                    "result_type": result["type"],
                    "used_ai": True,
                    "provider": selected_provider
                })

                st.success("✅ Analysis complete!", icon="🎉")

        except TimeoutError as te:
            st.error(f"⏱️ **Timeout:** {str(te)}")
            st.info("**Try:** Simpler prompt or different provider")

        except Exception as exc:
            st.error(f"❌ **Error during analysis:**\n{str(exc)}")
            with st.expander("🔍 Full error details"):
                st.code(traceback.format_exc())
            st.info("**Common fixes:**\n- Try a suggested prompt (no AI needed)\n- Check API keys in sidebar\n- Try different provider/model")

# ────────────────────────────────────────────────
# History panel
# ────────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    with st.expander(f"📜 Recent analyses ({len(st.session_state.history)})"):
        for idx, entry in enumerate(reversed(st.session_state.history[-10:]), 1):
            if entry.get('used_ai') and entry.get('provider'):
                badge = f"🤖 {entry['provider'].upper()}"
            elif entry.get('used_ai'):
                badge = "🤖 AI"
            else:
                badge = "⚡ Template"
            st.markdown(
                f"**{idx}.** [{entry['timestamp'].strftime('%H:%M:%S')}] "
                f"{badge} → `{entry['result_type']}`\n"
                f"_{entry['prompt']}_"
            )

st.divider()

# Footer
col1, col2 = st.columns([2, 1])
with col1:
    st.caption("💡 **Suggested prompts work instantly** • Custom prompts need FREE API (optional)")
with col2:
    st.caption("Built by **FelicityTech** • [LinkedIn](https://www.linkedin.com/in/solomon-eniola-adegoke/)")
