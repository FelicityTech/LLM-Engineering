# analyst.py
"""
Core logic for FelicityTech AI Data Analyst.
NOW SUPPORTS MULTIPLE 100% FREE APIs (NO CREDIT CARD NEEDED):
- Google Gemini API (FREE - Best for data analysis)
- Groq API (FREE - Super fast)
- Hugging Face Inference API (FREE - Many models)
"""

import io
import tempfile
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import sys
import os
import json
import re
from typing import Dict, List, Optional, Union, Generator, Any
import requests
from dotenv import load_dotenv

# Optional scientific libraries
try:
    import duckdb
except ImportError:
    duckdb = None

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

load_dotenv()

# ────────────────────────────────────────────────
# FREE API Configuration
# ────────────────────────────────────────────────
class FreeAPIConfig:
    """Configuration for multiple free API providers"""
    
    # Google Gemini (100% FREE - No credit card)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    GEMINI_MODELS = [
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash",
        "gemini-1.5-pro"
    ]
    
    # Groq (100% FREE - No credit card)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    GROQ_MODELS = [
        "llama-3.3-70b-versatile",
        "deepseek-r1-distill-llama-70b",
        "llama-3.1-8b-instant"
    ]
    
    # Hugging Face (100% FREE - No credit card)
    HF_API_KEY = os.getenv("HF_API_KEY")
    HF_BASE_URL = "https://api-inference.huggingface.co/models"
    HF_MODELS = [
        "meta-llama/Llama-3.2-3B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "Qwen/Qwen2.5-7B-Instruct"
    ]


class AnalystConfig:
    """Global configuration constants"""
    MAX_ROWS_PREVIEW = 1000
    MAX_SUGGESTIONS = 8
    DEFAULT_TIMEOUT = 120
    DEFAULT_PROVIDER = "gemini"  # Best free option
    PLOT_DPI = 150
    PLOT_FIGSIZE = (10, 6)


# ────────────────────────────────────────────────
# Multi-Provider LLM Client
# ────────────────────────────────────────────────
class FreeAPIClient:
    """Unified client for multiple free API providers"""
    
    def __init__(self):
        self.providers = self._get_available_providers()
    
    def _get_available_providers(self) -> Dict[str, Dict]:
        """Check which API keys are configured"""
        providers = {}
        
        if FreeAPIConfig.GEMINI_API_KEY:
            providers["gemini"] = {
                "name": "Google Gemini",
                "models": FreeAPIConfig.GEMINI_MODELS,
                "status": "✅ Ready"
            }
        
        if FreeAPIConfig.GROQ_API_KEY:
            providers["groq"] = {
                "name": "Groq (Super Fast)",
                "models": FreeAPIConfig.GROQ_MODELS,
                "status": "✅ Ready"
            }
        
        if FreeAPIConfig.HF_API_KEY:
            providers["huggingface"] = {
                "name": "Hugging Face",
                "models": FreeAPIConfig.HF_MODELS,
                "status": "✅ Ready"
            }
        
        return providers
    
    def get_status_message(self) -> str:
        """Get human-readable status of available APIs"""
        if not self.providers:
            return "❌ No API keys configured. Please add keys to .env file."
        
        msg = f"✅ {len(self.providers)} Free API(s) Available:\n"
        for key, info in self.providers.items():
            msg += f"  • {info['name']}: {info['status']}\n"
        return msg
    
    def stream_gemini(self, prompt: str, df_info: str, model: str, timeout: int) -> Generator[str, None, None]:
        """Stream from Google Gemini API (FREE)"""
        api_key = FreeAPIConfig.GEMINI_API_KEY
        if not api_key:
            yield "[ERROR] GEMINI_API_KEY not found in .env file"
            return
        
        system_content = f"""You are an expert data analyst. Generate COMPLETE Python code using pandas and matplotlib.

CRITICAL RULES:
1. Return code inside ```python ``` blocks
2. DataFrame is named 'df'
3. For tables: assign to 'result' variable
4. For charts: create with plt, do NOT call plt.show()
5. Generate ENTIRE code - don't stop midway

### DATASET INFO:
{df_info}

### USER REQUEST:
{prompt}"""

        url = f"{FreeAPIConfig.GEMINI_BASE_URL}/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{
                "parts": [{"text": system_content}]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 4096
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=timeout, stream=False)
            response.raise_for_status()
            
            data = response.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                yield text
            else:
                yield "[ERROR] No response from Gemini"
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                yield "[RATE-LIMIT] Gemini free tier limit reached. Try Groq or wait 1 minute."
            elif e.response.status_code == 400:
                yield f"[ERROR] Bad request to Gemini: {e.response.text}"
            else:
                yield f"[ERROR] Gemini API error {e.response.status_code}: {str(e)}"
        except Exception as e:
            yield f"[ERROR] {type(e).__name__}: {str(e)}"
    
    def stream_groq(self, prompt: str, df_info: str, model: str, timeout: int) -> Generator[str, None, None]:
        """Stream from Groq API (FREE & FAST)"""
        api_key = FreeAPIConfig.GROQ_API_KEY
        if not api_key:
            yield "[ERROR] GROQ_API_KEY not found in .env file"
            return
        
        system_prompt = f"""You are an expert data analyst. Generate COMPLETE Python code using pandas and matplotlib.

CRITICAL RULES:
1. Return code inside ```python ``` blocks
2. DataFrame is named 'df'
3. For tables: assign to 'result' variable
4. For charts: create with plt, do NOT call plt.show()
5. Generate ENTIRE code - don't stop midway

### DATASET INFO:
{df_info}"""

        url = f"{FreeAPIConfig.GROQ_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"### USER REQUEST:\n{prompt}\n\nGenerate COMPLETE Python code:"}
            ],
            "temperature": 0.3,
            "max_tokens": 4096,
            "stream": True
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue
                            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                yield "[RATE-LIMIT] Groq free tier limit reached. Try Gemini or wait."
            else:
                yield f"[ERROR] Groq API error {e.response.status_code}: {str(e)}"
        except Exception as e:
            yield f"[ERROR] {type(e).__name__}: {str(e)}"
    
    def stream_huggingface(self, prompt: str, df_info: str, model: str, timeout: int) -> Generator[str, None, None]:
        """Stream from Hugging Face Inference API (FREE)"""
        api_key = FreeAPIConfig.HF_API_KEY
        if not api_key:
            yield "[ERROR] HF_API_KEY not found in .env file"
            return
        
        system_content = f"""You are an expert data analyst. Generate COMPLETE Python code.

Dataset Info:
{df_info}

User Request: {prompt}

Generate complete Python code inside ```python ``` blocks. Use 'df' for DataFrame, assign results to 'result' variable."""

        url = f"{FreeAPIConfig.HF_BASE_URL}/{model}"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "inputs": system_content,
            "parameters": {
                "max_new_tokens": 2048,
                "temperature": 0.3,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                yield data[0].get("generated_text", "")
            elif isinstance(data, dict) and "generated_text" in data:
                yield data["generated_text"]
            else:
                yield "[ERROR] Unexpected response from Hugging Face"
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 503:
                yield "[ERROR] Model is loading (try again in 20 seconds) or try a different model"
            elif e.response.status_code == 429:
                yield "[RATE-LIMIT] HuggingFace rate limit. Try Gemini or Groq."
            else:
                yield f"[ERROR] HF API error {e.response.status_code}: {str(e)}"
        except Exception as e:
            yield f"[ERROR] {type(e).__name__}: {str(e)}"
    
    def stream(self, provider: str, prompt: str, df_info: str, model: str, timeout: int) -> Generator[str, None, None]:
        """Route to appropriate provider"""
        if provider == "gemini":
            yield from self.stream_gemini(prompt, df_info, model, timeout)
        elif provider == "groq":
            yield from self.stream_groq(prompt, df_info, model, timeout)
        elif provider == "huggingface":
            yield from self.stream_huggingface(prompt, df_info, model, timeout)
        else:
            yield f"[ERROR] Unknown provider: {provider}"


# Global client instance
free_api_client = FreeAPIClient()


# ────────────────────────────────────────────────
# Data Loading (unchanged)
# ────────────────────────────────────────────────
def _looks_like_csv(raw_bytes: bytes) -> bool:
    try:
        sample = raw_bytes[:1024].decode(errors="ignore")
    except Exception:
        return False
    return "," in sample and "\n" in sample


def load_data(file_or_path) -> pd.DataFrame:
    """Load data from file path or Streamlit UploadedFile"""
    try:
        if isinstance(file_or_path, (str, Path)):
            p = Path(file_or_path)
            if not p.exists():
                raise FileNotFoundError(f"File not found: {p}")
            s = p.suffix.lower()

            if s == ".csv":
                return pd.read_csv(p, encoding='utf-8', low_memory=False)
            if s in {".xls", ".xlsx"}:
                return pd.read_excel(p, engine='openpyxl')
            if s == ".json":
                return pd.read_json(p)
            if s == ".parquet":
                return pd.read_parquet(p)
            if s == ".feather":
                return pd.read_feather(p)
            if s in {".h5", ".hdf5", ".hdf"}:
                with pd.HDFStore(p, mode='r') as store:
                    keys = store.keys()
                    if not keys:
                        raise ValueError("HDF5 file has no datasets")
                    return pd.read_hdf(p, key=keys[0])
            if s == ".xml":
                return pd.read_xml(p)
            if s == ".txt":
                for sep in ['\t', ',', '|']:
                    try:
                        return pd.read_csv(p, sep=sep, encoding='utf-8')
                    except:
                        continue
                return pd.read_csv(p, encoding='utf-8', low_memory=False)
            return pd.read_csv(p, encoding='utf-8', low_memory=False)

        # Streamlit UploadedFile
        name = getattr(file_or_path, "name", "")
        suffix = Path(name).suffix.lower()
        raw = file_or_path.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        bio = io.BytesIO(raw)

        if suffix == ".csv" or (not suffix and _looks_like_csv(raw)):
            bio.seek(0)
            return pd.read_csv(bio, encoding='utf-8', low_memory=False)
        if suffix in {".xls", ".xlsx"}:
            bio.seek(0)
            return pd.read_excel(bio, engine='openpyxl')
        if suffix == ".json":
            bio.seek(0)
            return pd.read_json(bio)
        if suffix == ".parquet":
            bio.seek(0)
            return pd.read_parquet(bio)
        if suffix == ".feather":
            bio.seek(0)
            return pd.read_feather(bio)
        if suffix in {".h5", ".hdf5", ".hdf"}:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            try:
                with pd.HDFStore(tmp_path, mode='r') as store:
                    keys = store.keys()
                    if not keys:
                        raise ValueError("HDF5 file has no datasets")
                    return pd.read_hdf(tmp_path, key=keys[0])
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        if suffix == ".xml":
            bio.seek(0)
            return pd.read_xml(bio)
        if suffix == ".txt":
            bio.seek(0)
            for sep in ['\t', ',', '|']:
                try:
                    return pd.read_csv(bio, sep=sep, encoding='utf-8')
                except:
                    bio.seek(0)
            return pd.read_csv(bio, encoding='utf-8', low_memory=False)

        bio.seek(0)
        try:
            return pd.read_csv(bio, encoding='utf-8', low_memory=False)
        except:
            bio.seek(0)
            return pd.read_json(bio)

    except Exception as e:
        raise ValueError(f"Failed to load data: {str(e)}")


# ────────────────────────────────────────────────
# Column Type Detection & Prompts (unchanged)
# ────────────────────────────────────────────────
def _detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = []

    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            datetime_cols.append(c)
        else:
            try:
                sample = df[c].dropna().astype(str).iloc[:min(20, len(df))]
                if len(sample) > 0:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().sum() >= max(1, len(sample) // 2):
                        datetime_cols.append(c)
            except:
                pass

    categorical = []
    for c in df.columns:
        if c not in numeric + datetime_cols:
            nunique = df[c].nunique(dropna=True)
            if 1 < nunique <= 50:
                categorical.append(c)

    text = [c for c in df.columns if c not in numeric + datetime_cols + categorical
            and df[c].dtype == 'object']

    return {
        "numeric": numeric,
        "datetime": datetime_cols,
        "categorical": categorical,
        "text": text
    }


def suggest_prompts(df: pd.DataFrame, max_suggestions: int = None) -> List[str]:
    if max_suggestions is None:
        max_suggestions = AnalystConfig.MAX_SUGGESTIONS

    types = _detect_column_types(df)
    numeric = types["numeric"]
    datetime_cols = types["datetime"]
    categorical = types["categorical"]

    suggestions = [
        "Summarize the dataset in 5 bullet points."
    ]

    if df.isnull().sum().sum() > 0:
        suggestions.append("Show columns with missing values and their percentages.")

    if categorical:
        col = categorical[0]
        suggestions.append(f"Show the top 10 counts for the categorical column '{col}'.")
        if len(categorical) >= 2:
            suggestions.append(f"Create a bar chart comparing '{categorical[0]}' and '{categorical[1]}'.")

    if numeric:
        suggestions.append("Show summary statistics for numeric columns.")
        col = numeric[0]
        suggestions.append(f"Create a histogram of '{col}'.")
        if len(numeric) >= 2:
            suggestions.append(f"Create a scatter plot comparing '{numeric[0]}' vs '{numeric[1]}'.")

    if datetime_cols and numeric:
        suggestions.append(f"Create a time series line plot of '{numeric[0]}' by '{datetime_cols[0]}'.")

    if len(numeric) >= 2:
        suggestions.append("Show the correlation matrix heatmap for numeric columns.")

    return suggestions[:max_suggestions]


# ────────────────────────────────────────────────
# Template Code Generation (unchanged from your original)
# ────────────────────────────────────────────────
def prompt_to_code(prompt: str, df: pd.DataFrame) -> Optional[str]:
    """Match common requests to templates"""
    p = prompt.strip().lower()

    if "summarize the dataset" in p or p.startswith("summarize"):
        code = textwrap.dedent("""
            info = []
            info.append(f"📊 Rows: {len(df):,} | Columns: {len(df.columns)}")
            info.append(f"💾 Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            dtypes = df.dtypes.value_counts()
            info.append("🔍 Column types: " + ", ".join([f"{str(k)}: {v}" for k,v in dtypes.items()]))
            missing = df.isnull().sum()
            total_missing = missing.sum()
            if total_missing > 0:
                info.append(f"⚠ Total missing: {total_missing:,}")
            else:
                info.append("✅ No missing values")
            result = "\\n".join(info)
        """)
        return code

    if "missing values" in p and "percentage" in p:
        code = textwrap.dedent("""
            missing = df.isnull().sum()
            missing_pct = (missing / len(df) * 100).round(2)
            result = pd.DataFrame({
                'column': missing.index,
                'missing_count': missing.values,
                'missing_percentage': missing_pct.values
            })
            result = result[result['missing_count'] > 0].sort_values('missing_count', ascending=False)
        """)
        return code

    if "summary statistics" in p:
        return "result = df.select_dtypes(include=['number']).describe().T.round(2)"

    if "histogram" in p:
        m = re.search(r"'([^']+)'", prompt)
        col = m.group(1) if m else None
        if col and col in df.columns:
            code = textwrap.dedent(f"""
                plt.figure(figsize=(10, 6))
                plt.hist(df['{col}'].dropna(), bins=30, edgecolor='black', alpha=0.7)
                plt.title('Distribution of {col}')
                plt.xlabel('{col}')
                plt.ylabel('Frequency')
                plt.grid(alpha=0.3)
                plt.tight_layout()
                result_img_path = None
            """)
            return code

    if "scatter plot" in p:
        m = re.findall(r"'([^']+)'", prompt)
        if len(m) >= 2 and all(c in df.columns for c in m[:2]):
            xcol, ycol = m[0], m[1]
            code = textwrap.dedent(f"""
                plt.figure(figsize=(10, 6))
                plt.scatter(df['{xcol}'], df['{ycol}'], alpha=0.6)
                plt.title('{ycol} vs {xcol}')
                plt.xlabel('{xcol}')
                plt.ylabel('{ycol}')
                plt.grid(alpha=0.3)
                plt.tight_layout()
                result_img_path = None
            """)
            return code

    if "correlation" in p and "heatmap" in p:
        code = textwrap.dedent("""
            try:
                import seaborn as sns
                corr = df.select_dtypes(include=['number']).corr()
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
                plt.title('Correlation Matrix')
                plt.tight_layout()
            except ImportError:
                corr = df.select_dtypes(include=['number']).corr()
                plt.figure(figsize=(10, 8))
                plt.imshow(corr, cmap='coolwarm')
                plt.colorbar()
                plt.title('Correlation Matrix')
            result_img_path = None
        """)
        return code

    return None


# ────────────────────────────────────────────────
# Code Execution (unchanged)
# ────────────────────────────────────────────────
def run_code(df: pd.DataFrame, code: str) -> Dict[str, Any]:
    local_ns = {
        "pd": pd,
        "np": np,
        "df": df.copy(),
        "plt": plt,
        "sns": None
    }

    if "sns" in code.lower():
        try:
            import seaborn as sns
            local_ns["sns"] = sns
        except ImportError:
            pass

    old_stdout = sys.stdout
    stdout_buf = io.StringIO()
    sys.stdout = stdout_buf

    try:
        exec(code, {}, local_ns)

        if "result_img_path" in local_ns and local_ns["result_img_path"]:
            return {"type": "image", "path": local_ns["result_img_path"]}

        if plt.get_fignums():
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                plt.savefig(f.name, bbox_inches="tight", dpi=AnalystConfig.PLOT_DPI)
                plt.close("all")
                return {"type": "image", "path": f.name}

        if "result" in local_ns:
            res = local_ns["result"]
            if isinstance(res, pd.DataFrame):
                return {"type": "dataframe", "df": res}
            return {"type": "text", "output": str(res)}

        out = stdout_buf.getvalue().strip()
        if out:
            return {"type": "text", "output": out}

        return {"type": "text", "output": "✅ Execution finished"}

    except Exception as e:
        import traceback
        return {"type": "text", "output": f"❌ Error:\n{traceback.format_exc()}"}

    finally:
        sys.stdout = old_stdout
        plt.close("all")


# ────────────────────────────────────────────────
# Data Summary
# ────────────────────────────────────────────────
def get_df_summary(df: pd.DataFrame) -> str:
    summary = []
    summary.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    summary.append(f"Columns: {', '.join(df.columns[:20])}")

    types = _detect_column_types(df)
    if types["numeric"]:
        summary.append(f"Numeric: {', '.join(types['numeric'][:10])}")
    if types["categorical"]:
        summary.append(f"Categorical: {', '.join(types['categorical'][:10])}")
    if types["datetime"]:
        summary.append(f"DateTime: {', '.join(types['datetime'][:5])}")

    return "\n".join(summary)


# ────────────────────────────────────────────────
# NEW: Unified LLM Streaming Function
# ────────────────────────────────────────────────
def ask_llm(
    prompt: str,
    df_info: Optional[str] = None,
    provider: str = "gemini",
    model: str = None,
    timeout: int = None
) -> Generator[str, None, None]:
    """Stream response from selected free API provider"""
    
    if timeout is None:
        timeout = AnalystConfig.DEFAULT_TIMEOUT
    
    # Auto-select first available provider if not specified
    if provider not in free_api_client.providers:
        available = list(free_api_client.providers.keys())
        if not available:
            yield "[ERROR] No API keys configured. Please add at least one to .env file"
            return
        provider = available[0]
    
    # Auto-select first model if not specified
    if model is None:
        model = free_api_client.providers[provider]["models"][0]
    
    yield from free_api_client.stream(provider, prompt, df_info or "", model, timeout)


# ────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────
def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    if df.empty:
        return False, "DataFrame is empty"
    if len(df.columns) == 0:
        return False, "DataFrame has no columns"
    if len(df) > 1_000_000:
        return False, f"DataFrame too large ({len(df):,} rows)"
    return True, "OK"


def extract_python_code(llm_output: str) -> Optional[str]:
    if "```python" in llm_output:
        try:
            return llm_output.split("```python")[1].split("```")[0].strip()
        except IndexError:
            pass

    if "```" in llm_output:
        try:
            code = llm_output.split("```")[1].split("```")[0].strip()
            lines = code.split("\n")
            if lines and lines[0].strip().lower() in ["python", "py"]:
                return "\n".join(lines[1:]).strip()
            return code
        except IndexError:
            pass

    return None


def get_api_status() -> str:
    """Get status of all configured APIs"""
    return free_api_client.get_status_message()


def get_available_providers() -> Dict[str, Dict]:
    """Get dict of available providers and their models"""
    return free_api_client.providers