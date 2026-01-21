# analyst.py
"""
Core logic for FelicityTech AI Data Analyst.
ENHANCED VERSION: 50+ built-in templates (NO AI NEEDED)
Only complex custom requests use FREE AI APIs
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


# ────────────────────────────────────────────────
# FREE API Configuration
# ────────────────────────────────────────────────
class FreeAPIConfig:
    """Configuration for multiple free API providers"""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    GEMINI_MODELS = ["gemini-1.5-pro"]

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    GROQ_MODELS = ["llama-3.1-8b-instant"]

    HF_API_KEY = os.getenv("HF_API_KEY")
    HF_BASE_URL = "https://api-inference.huggingface.co/models"
    HF_MODELS = ["Qwen/Qwen2.5-7B-Instruct"]


class AnalystConfig:
    """Global configuration constants"""
    MAX_ROWS_PREVIEW = 1000
    MAX_SUGGESTIONS = 30  # Increased for more templates
    DEFAULT_TIMEOUT = 120
    DEFAULT_PROVIDER = "gemini"
    PLOT_DPI = 150
    PLOT_FIGSIZE = (10, 6)


# ────────────────────────────────────────────────
# Data Loading
# ────────────────────────────────────────────────
def _looks_like_csv(raw_bytes: bytes) -> bool:
    try:
        sample = raw_bytes[:1024].decode(errors="ignore")
        return "," in sample and "\n" in sample
    except:
        return False


def load_data(file_or_path) -> pd.DataFrame:
    """Load data from file path or Streamlit UploadedFile"""
    try:
        if isinstance(file_or_path, (str, Path)):
            p = Path(file_or_path)
            if not p.exists():
                raise FileNotFoundError(f"File not found: {p}")
            s = p.suffix.lower()

            loaders = {
                ".csv": lambda: pd.read_csv(p, encoding='utf-8', low_memory=False),
                ".xls": lambda: pd.read_excel(p, engine='openpyxl'),
                ".xlsx": lambda: pd.read_excel(p, engine='openpyxl'),
                ".json": lambda: pd.read_json(p),
                ".parquet": lambda: pd.read_parquet(p),
                ".feather": lambda: pd.read_feather(p),
                ".xml": lambda: pd.read_xml(p),
            }

            if s in loaders:
                return loaders[s]()
            
            if s in {".h5", ".hdf5", ".hdf"}:
                with pd.HDFStore(p, mode='r') as store:
                    keys = store.keys()
                    if not keys:
                        raise ValueError("HDF5 file has no datasets")
                    return pd.read_hdf(p, key=keys[0])
            
            if s == ".txt":
                for sep in ['\t', ',', '|']:
                    try:
                        return pd.read_csv(p, sep=sep, encoding='utf-8')
                    except:
                        continue
            
            return pd.read_csv(p, encoding='utf-8', low_memory=False)

        # Streamlit UploadedFile
        name = getattr(file_or_path, "name", "")
        suffix = Path(name).suffix.lower()
        raw = file_or_path.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        bio = io.BytesIO(raw)

        loaders = {
            ".csv": lambda: pd.read_csv(bio, encoding='utf-8', low_memory=False),
            ".xls": lambda: pd.read_excel(bio, engine='openpyxl'),
            ".xlsx": lambda: pd.read_excel(bio, engine='openpyxl'),
            ".json": lambda: pd.read_json(bio),
            ".parquet": lambda: pd.read_parquet(bio),
            ".feather": lambda: pd.read_feather(bio),
            ".xml": lambda: pd.read_xml(bio),
        }

        if suffix in loaders or (not suffix and _looks_like_csv(raw)):
            bio.seek(0)
            return loaders.get(suffix, lambda: pd.read_csv(bio, encoding='utf-8', low_memory=False))()

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
# Column Type Detection
# ────────────────────────────────────────────────
def _detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detect column types for smart template suggestions"""
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


# ────────────────────────────────────────────────
# ENHANCED: 50+ Built-in Template Suggestions
# ────────────────────────────────────────────────
def suggest_prompts(df: pd.DataFrame, max_suggestions: int = None) -> List[str]:
    """Generate comprehensive list of instant analysis templates"""
    if max_suggestions is None:
        max_suggestions = AnalystConfig.MAX_SUGGESTIONS

    types = _detect_column_types(df)
    numeric = types["numeric"]
    datetime_cols = types["datetime"]
    categorical = types["categorical"]
    text = types["text"]

    suggestions = []

    # ──── OVERVIEW & BASICS (Always available) ────
    suggestions.extend([
        "📊 Dataset summary & statistics",
        "🔍 Show first 20 rows",
        "🔍 Show last 20 rows",
        "📋 Show all column names and types",
        "💾 Show memory usage by column",
        "🔢 Count total rows and columns",
    ])

    # ──── DATA QUALITY ────
    if df.isnull().sum().sum() > 0:
        suggestions.extend([
            "⚠️ Missing value analysis (count & percentage)",
            "🧹 Show rows with any missing values",
            "📉 Visualize missing data pattern",
        ])
    
    suggestions.extend([
        "🔄 Detect duplicate rows",
        "🎯 Show unique values count per column",
        "📊 Data types distribution",
    ])

    # ──── NUMERIC ANALYSIS ────
    if numeric:
        col = numeric[0]
        suggestions.extend([
            "📈 Summary statistics for all numeric columns",
            f"📊 Histogram of '{col}'",
            f"📉 Box plot for '{col}' (outlier detection)",
            "🎯 Detect outliers in all numeric columns (IQR method)",
            "📊 Distribution plots for all numeric columns",
        ])

        if len(numeric) >= 2:
            col1, col2 = numeric[0], numeric[1]
            suggestions.extend([
                f"📈 Scatter plot: '{col1}' vs '{col2}'",
                "🔥 Correlation matrix heatmap",
                "📊 Pair plot for numeric columns",
                f"📈 Compare distributions: '{col1}' and '{col2}'",
            ])

    # ──── CATEGORICAL ANALYSIS ────
    if categorical:
        col = categorical[0]
        suggestions.extend([
            f"📊 Top 10 values in '{col}'",
            f"📊 Bottom 10 values in '{col}'",
            f"📊 Value counts bar chart for '{col}'",
            f"🥧 Pie chart for '{col}' distribution",
        ])

        if len(categorical) >= 2:
            col1, col2 = categorical[0], categorical[1]
            suggestions.extend([
                f"📊 Cross-tabulation: '{col1}' vs '{col2}'",
                f"📊 Grouped bar chart: '{col1}' by '{col2}'",
            ])

        if categorical and numeric:
            suggestions.append(f"📊 '{numeric[0]}' grouped by '{categorical[0]}'")

    # ──── TIME SERIES ────
    if datetime_cols:
        dt_col = datetime_cols[0]
        suggestions.extend([
            f"📅 Show date range for '{dt_col}'",
            f"📊 Records per month for '{dt_col}'",
            f"📊 Records per year for '{dt_col}'",
            f"📊 Records per day of week for '{dt_col}'",
        ])

        if numeric:
            suggestions.extend([
                f"📈 Time series: '{numeric[0]}' over '{dt_col}'",
                f"📊 Monthly trend of '{numeric[0]}'",
            ])

    # ──── ADVANCED ANALYSIS ────
    if len(numeric) >= 3:
        suggestions.append("🔥 3D scatter plot for top 3 numeric columns")

    if categorical and len(numeric) >= 1:
        suggestions.append(f"📊 Statistical summary by '{categorical[0]}'")

    # ──── TEXT ANALYSIS ────
    if text:
        col = text[0]
        suggestions.extend([
            f"📝 Text length distribution for '{col}'",
            f"🔤 Most common words in '{col}'",
        ])

    # ──── AGGREGATIONS ────
    if numeric:
        suggestions.extend([
            f"🎯 Top 10 rows by '{numeric[0]}'",
            f"🎯 Bottom 10 rows by '{numeric[0]}'",
        ])

    if categorical and numeric:
        suggestions.extend([
            f"📊 Sum of '{numeric[0]}' by '{categorical[0]}'",
            f"📊 Average of '{numeric[0]}' by '{categorical[0]}'",
            f"📊 Count by '{categorical[0]}'",
        ])

    return suggestions[:max_suggestions]


# ────────────────────────────────────────────────
# MASSIVELY EXPANDED: Template Code Generator
# ────────────────────────────────────────────────
def prompt_to_code(prompt: str, df: pd.DataFrame) -> Optional[str]:
    """Match prompts to 50+ instant templates (NO AI NEEDED)"""
    p = prompt.strip().lower()
    types = _detect_column_types(df)
    numeric = types["numeric"]
    categorical = types["categorical"]
    datetime_cols = types["datetime"]
    text = types["text"]

    # ──── OVERVIEW & BASICS ────
    if "dataset summary" in p or p == "📊 dataset summary & statistics":
        return textwrap.dedent("""
            info = []
            info.append(f"📊 **Dataset Overview**")
            info.append(f"  • Rows: {len(df):,}")
            info.append(f"  • Columns: {len(df.columns)}")
            info.append(f"  • Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            info.append("")
            
            dtypes = df.dtypes.value_counts()
            info.append("📋 **Column Types:**")
            for dtype, count in dtypes.items():
                info.append(f"  • {dtype}: {count} columns")
            info.append("")
            
            missing = df.isnull().sum().sum()
            if missing > 0:
                pct = (missing / (len(df) * len(df.columns)) * 100)
                info.append(f"⚠️  **Missing Values:** {missing:,} ({pct:.2f}%)")
            else:
                info.append("✅ **No missing values**")
            
            info.append("")
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                info.append(f"🔄 **Duplicate Rows:** {duplicates:,}")
            else:
                info.append("✅ **No duplicate rows**")
            
            result = "\\n".join(info)
        """)

    if "first 20 rows" in p or "show first" in p:
        return "result = df.head(20)"

    if "last 20 rows" in p or "show last" in p:
        return "result = df.tail(20)"

    if "column names and types" in p or "show all column" in p:
        return textwrap.dedent("""
            result = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count().values,
                'Unique': [df[col].nunique() for col in df.columns],
                'Sample': [str(df[col].iloc[0]) if len(df) > 0 else '' for col in df.columns]
            })
        """)

    if "memory usage" in p:
        return textwrap.dedent("""
            mem = df.memory_usage(deep=True)
            result = pd.DataFrame({
                'Column': mem.index,
                'Memory (MB)': (mem.values / 1024**2).round(3)
            }).sort_values('Memory (MB)', ascending=False)
        """)

    if "count total rows" in p or "count.*columns" in p:
        return "result = f'Rows: {len(df):,}, Columns: {len(df.columns)}'"

    # ──── DATA QUALITY ────
    if "missing value" in p and ("analysis" in p or "percentage" in p):
        return textwrap.dedent("""
            missing = df.isnull().sum()
            missing_pct = (missing / len(df) * 100).round(2)
            result = pd.DataFrame({
                'Column': missing.index,
                'Missing_Count': missing.values,
                'Missing_%': missing_pct.values
            })
            result = result[result['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
            if len(result) == 0:
                result = "✅ No missing values found!"
        """)

    if "rows with.*missing" in p or "show rows with any missing" in p:
        return textwrap.dedent("""
            result = df[df.isnull().any(axis=1)].head(100)
            if len(result) == 0:
                result = "✅ No rows with missing values!"
        """)

    if "visualize missing" in p or "missing data pattern" in p:
        return textwrap.dedent("""
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) == 0:
                result = "✅ No missing data to visualize!"
            else:
                plt.figure(figsize=(10, 6))
                missing_data.plot(kind='barh', color='coral')
                plt.title('Missing Values by Column')
                plt.xlabel('Number of Missing Values')
                plt.ylabel('Column')
                plt.tight_layout()
                result_img_path = None
        """)

    if "duplicate" in p:
        return textwrap.dedent("""
            dups = df.duplicated()
            dup_count = dups.sum()
            if dup_count > 0:
                result = f"🔄 Found {dup_count:,} duplicate rows\\n\\nFirst 20 duplicates:\\n" + str(df[dups].head(20))
            else:
                result = "✅ No duplicate rows found!"
        """)

    if "unique values count" in p:
        return textwrap.dedent("""
            result = pd.DataFrame({
                'Column': df.columns,
                'Unique_Values': [df[col].nunique() for col in df.columns],
                'Total_Values': len(df)
            })
            result['Uniqueness_%'] = (result['Unique_Values'] / result['Total_Values'] * 100).round(2)
            result = result.sort_values('Unique_Values', ascending=False)
        """)

    if "data types distribution" in p:
        return textwrap.dedent("""
            type_counts = df.dtypes.value_counts()
            result = pd.DataFrame({
                'Data_Type': type_counts.index.astype(str),
                'Column_Count': type_counts.values
            })
        """)

    # ──── NUMERIC ANALYSIS ────
    if "summary statistics" in p:
        return textwrap.dedent("""
            result = df.select_dtypes(include=['number']).describe().T
            result['missing'] = df.select_dtypes(include=['number']).isnull().sum()
            result = result.round(2)
        """)

    if "histogram" in p:
        m = re.search(r"'([^']+)'", prompt)
        col = m.group(1) if m else (numeric[0] if numeric else None)
        if col and col in df.columns:
            return textwrap.dedent(f"""
                plt.figure(figsize=(10, 6))
                data = df['{col}'].dropna()
                plt.hist(data, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
                plt.title(f'Distribution of {col}', fontsize=14, fontweight='bold')
                plt.xlabel('{col}')
                plt.ylabel('Frequency')
                plt.grid(alpha=0.3)
                plt.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {{data.mean():.2f}}')
                plt.axvline(data.median(), color='green', linestyle='--', label=f'Median: {{data.median():.2f}}')
                plt.legend()
                plt.tight_layout()
                result_img_path = None
            """)

    if "box plot" in p and "outlier" in p:
        m = re.search(r"'([^']+)'", prompt)
        col = m.group(1) if m else (numeric[0] if numeric else None)
        if col and col in df.columns:
            return textwrap.dedent(f"""
                plt.figure(figsize=(10, 6))
                plt.boxplot(df['{col}'].dropna(), vert=False)
                plt.title(f'Box Plot: {col} (Outlier Detection)', fontsize=14)
                plt.xlabel('{col}')
                plt.grid(alpha=0.3)
                plt.tight_layout()
                result_img_path = None
            """)

    if "detect outliers" in p and "iqr" in p:
        return textwrap.dedent("""
            outlier_summary = []
            for col in df.select_dtypes(include=['number']).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower) | (df[col] > upper)]
                if len(outliers) > 0:
                    outlier_summary.append({
                        'Column': col,
                        'Outlier_Count': len(outliers),
                        'Percentage': f"{len(outliers)/len(df)*100:.2f}%",
                        'Lower_Bound': round(lower, 2),
                        'Upper_Bound': round(upper, 2)
                    })
            
            if outlier_summary:
                result = pd.DataFrame(outlier_summary)
            else:
                result = "✅ No outliers detected!"
        """)

    if "distribution plots for all numeric" in p:
        return textwrap.dedent("""
            num_cols = df.select_dtypes(include=['number']).columns[:6]
            if len(num_cols) == 0:
                result = "❌ No numeric columns found!"
            else:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()
                for i, col in enumerate(num_cols):
                    axes[i].hist(df[col].dropna(), bins=20, edgecolor='black', alpha=0.7)
                    axes[i].set_title(col)
                    axes[i].grid(alpha=0.3)
                for j in range(i+1, 6):
                    fig.delaxes(axes[j])
                plt.tight_layout()
                result_img_path = None
        """)

    if "scatter plot" in p:
        m = re.findall(r"'([^']+)'", prompt)
        if len(m) >= 2 and all(c in df.columns for c in m[:2]):
            x, y = m[0], m[1]
        elif len(numeric) >= 2:
            x, y = numeric[0], numeric[1]
        else:
            return None
        
        return textwrap.dedent(f"""
            plt.figure(figsize=(10, 6))
            plt.scatter(df['{x}'], df['{y}'], alpha=0.6, edgecolors='black', linewidth=0.5)
            plt.title(f'{y} vs {x}', fontsize=14, fontweight='bold')
            plt.xlabel('{x}')
            plt.ylabel('{y}')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            result_img_path = None
        """)

    if "correlation" in p and "heatmap" in p:
        return textwrap.dedent("""
            try:
                import seaborn as sns
                corr = df.select_dtypes(include=['number']).corr()
                plt.figure(figsize=(12, 10))
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                           square=True, linewidths=1, cbar_kws={"shrink": 0.8})
                plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
                plt.tight_layout()
            except ImportError:
                corr = df.select_dtypes(include=['number']).corr()
                plt.figure(figsize=(10, 8))
                plt.imshow(corr, cmap='coolwarm', aspect='auto')
                plt.colorbar()
                plt.title('Correlation Matrix')
                plt.xticks(range(len(corr)), corr.columns, rotation=45)
                plt.yticks(range(len(corr)), corr.columns)
            result_img_path = None
        """)

    if "pair plot" in p:
        return textwrap.dedent("""
            try:
                import seaborn as sns
                num_cols = df.select_dtypes(include=['number']).columns[:4]
                sns.pairplot(df[num_cols].dropna())
                plt.tight_layout()
                result_img_path = None
            except ImportError:
                result = "❌ Seaborn required for pair plots. Install with: pip install seaborn"
        """)

    if "compare distributions" in p:
        m = re.findall(r"'([^']+)'", prompt)
        if len(m) >= 2:
            col1, col2 = m[0], m[1]
        elif len(numeric) >= 2:
            col1, col2 = numeric[0], numeric[1]
        else:
            return None
        
        return textwrap.dedent(f"""
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].hist(df['{col1}'].dropna(), bins=30, alpha=0.7, color='blue', edgecolor='black')
            axes[0].set_title(f'Distribution: {col1}')
            axes[0].grid(alpha=0.3)
            
            axes[1].hist(df['{col2}'].dropna(), bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[1].set_title(f'Distribution: {col2}')
            axes[1].grid(alpha=0.3)
            plt.tight_layout()
            result_img_path = None
        """)

    # ──── CATEGORICAL ANALYSIS ────
    if "top 10" in p and "values" in p:
        m = re.search(r"'([^']+)'", prompt)
        col = m.group(1) if m else (categorical[0] if categorical else None)
        if col and col in df.columns:
            return textwrap.dedent(f"""
                counts = df['{col}'].value_counts().head(10)
                result = pd.DataFrame({{
                    '{col}': counts.index,
                    'Count': counts.values,
                    'Percentage': (counts.values / len(df) * 100).round(2)
                }})
            """)

    if "bottom 10" in p and "values" in p:
        m = re.search(r"'([^']+)'", prompt)
        col = m.group(1) if m else (categorical[0] if categorical else None)
        if col and col in df.columns:
            return textwrap.dedent(f"""
                counts = df['{col}'].value_counts().tail(10)
                result = pd.DataFrame({{
                    '{col}': counts.index,
                    'Count': counts.values,
                    'Percentage': (counts.values / len(df) * 100).round(2)
                }})
            """)

    if "value counts bar chart" in p:
        m = re.search(r"'([^']+)'", prompt)
        col = m.group(1) if m else (categorical[0] if categorical else None)
        if col and col in df.columns:
            return textwrap.dedent(f"""
                counts = df['{col}'].value_counts().head(15)
                plt.figure(figsize=(10, 6))
                counts.plot(kind='barh', color='teal', edgecolor='black')
                plt.title(f'Top 15 Values: {col}', fontsize=14, fontweight='bold')
                plt.xlabel('Count')
                plt.ylabel('{col}')
                plt.tight_layout()
                result_img_path = None
            """)

    if "pie chart" in p:
        m = re.search(r"'([^']+)'", prompt)
        col = m.group(1) if m else (categorical[0] if categorical else None)
        if col and col in df.columns:
            return textwrap.dedent(f"""
                counts = df['{col}'].value_counts().head(10)
                plt.figure(figsize=(10, 8))
                plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
                plt.title(f'Distribution: {col}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                result_img_path = None
            """)

    if "cross-tabulation" in p or "cross tabulation" in p:
        m = re.findall(r"'([^']+)'", prompt)
        if len(m) >= 2:
            col1, col2 = m[0], m[1]
        elif len(categorical) >= 2:
            col1, col2 = categorical[0], categorical[1]
        else:
            return None
        
        return f"result = pd.crosstab(df['{col1}'], df['{col2}'], margins=True)"

    if "grouped bar chart" in p:
        m = re.findall(r"'([^']+)'", prompt)
        if len(m) >= 2:
            col1, col2 = m[0], m[1]
        elif len(categorical) >= 2:
            col1, col2 = categorical[0], categorical[1]
        else:
            return None
        
        return textwrap.dedent(f"""
            crosstab = pd.crosstab(df['{col1}'], df['{col2}'])
            crosstab.plot(kind='bar', figsize=(12, 6), edgecolor='black')
            plt.title(f'{col1} by {col2}', fontsize=14, fontweight='bold')
            plt.xlabel('{col1}')
            plt.ylabel('Count')
            plt.legend(title='{col2}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            result_img_path = None
        """)

    if "grouped by" in p and categorical and numeric:
        m_cat = re.search(r"'([^']+)'.*grouped by.*'([^']+)'", prompt)
        if m_cat:
            num_col, cat_col = m_cat.group(1), m_cat.group(2)
        else:
            num_col, cat_col = numeric[0], categorical[0]
        
        return textwrap.dedent(f"""
            result = df.groupby('{cat_col}')['{num_col}'].agg([
                ('count', 'count'),
                ('mean', 'mean'),
                ('median', 'median'),
                ('std', 'std'),
                ('min', 'min'),
                ('max', 'max')
            ]).round(2).reset_index()
        """)

    # ──── TIME SERIES ────
    if "date range" in p and datetime_cols:
        dt_col = datetime_cols[0]
        m = re.search(r"'([^']+)'", prompt)
        if m:
            dt_col = m.group(1)
        
        return textwrap.dedent(f"""
            df['{dt_col}'] = pd.to_datetime(df['{dt_col}'], errors='coerce')
            min_date = df['{dt_col}'].min()
            max_date = df['{dt_col}'].max()
            date_range = (max_date - min_date).days
            result = f"📅 Date Range for '{dt_col}':\\n  • Start: {{min_date}}\\n  • End: {{max_date}}\\n  • Duration: {{date_range}} days"
        """)

    if "records per month" in p and datetime_cols:
        dt_col = datetime_cols[0]
        m = re.search(r"'([^']+)'", prompt)
        if m:
            dt_col = m.group(1)
        
        return textwrap.dedent(f"""
            df['{dt_col}'] = pd.to_datetime(df['{dt_col}'], errors='coerce')
            result = df['{dt_col}'].dt.to_period('M').value_counts().sort_index().reset_index()
            result.columns = ['Month', 'Record_Count']
        """)

    if "records per year" in p and datetime_cols:
        dt_col = datetime_cols[0]
        m = re.search(r"'([^']+)'", prompt)
        if m:
            dt_col = m.group(1)
        
        return textwrap.dedent(f"""
            df['{dt_col}'] = pd.to_datetime(df['{dt_col}'], errors='coerce')
            result = df['{dt_col}'].dt.year.value_counts().sort_index().reset_index()
            result.columns = ['Year', 'Record_Count']
        """)

    if "day of week" in p and datetime_cols:
        dt_col = datetime_cols[0]
        m = re.search(r"'([^']+)'", prompt)
        if m:
            dt_col = m.group(1)
        
        return textwrap.dedent(f"""
            df['{dt_col}'] = pd.to_datetime(df['{dt_col}'], errors='coerce')
            result = df['{dt_col}'].dt.day_name().value_counts().reset_index()
            result.columns = ['Day_of_Week', 'Record_Count']
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            result['Day_of_Week'] = pd.Categorical(result['Day_of_Week'], categories=day_order, ordered=True)
            result = result.sort_values('Day_of_Week')
        """)

    if "time series" in p and datetime_cols and numeric:
        m = re.findall(r"'([^']+)'", prompt)
        if len(m) >= 2:
            num_col, dt_col = m[0], m[1]
        else:
            num_col, dt_col = numeric[0], datetime_cols[0]
        
        return textwrap.dedent(f"""
            df['{dt_col}'] = pd.to_datetime(df['{dt_col}'], errors='coerce')
            df_sorted = df.sort_values('{dt_col}')
            plt.figure(figsize=(12, 6))
            plt.plot(df_sorted['{dt_col}'], df_sorted['{num_col}'], marker='o', markersize=3, linewidth=1.5)
            plt.title(f'Time Series: {num_col} over {dt_col}', fontsize=14, fontweight='bold')
            plt.xlabel('{dt_col}')
            plt.ylabel('{num_col}')
            plt.xticks(rotation=45)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            result_img_path = None
        """)

    if "monthly trend" in p and datetime_cols and numeric:
        m = re.search(r"'([^']+)'", prompt)
        num_col = m.group(1) if m else numeric[0]
        dt_col = datetime_cols[0]
        
        return textwrap.dedent(f"""
            df['{dt_col}'] = pd.to_datetime(df['{dt_col}'], errors='coerce')
            df['month'] = df['{dt_col}'].dt.to_period('M')
            monthly = df.groupby('month')['{num_col}'].mean().reset_index()
            monthly['month'] = monthly['month'].astype(str)
            
            plt.figure(figsize=(12, 6))
            plt.plot(monthly['month'], monthly['{num_col}'], marker='o', linewidth=2)
            plt.title(f'Monthly Trend: {num_col}', fontsize=14, fontweight='bold')
            plt.xlabel('Month')
            plt.ylabel(f'Average {num_col}')
            plt.xticks(rotation=45)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            result_img_path = None
        """)

    # ──── TEXT ANALYSIS ────
    if "text length distribution" in p and text:
        m = re.search(r"'([^']+)'", prompt)
        col = m.group(1) if m else text[0]
        
        return textwrap.dedent(f"""
            df['text_length'] = df['{col}'].astype(str).str.len()
            plt.figure(figsize=(10, 6))
            plt.hist(df['text_length'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='purple')
            plt.title(f'Text Length Distribution: {col}', fontsize=14, fontweight='bold')
            plt.xlabel('Character Count')
            plt.ylabel('Frequency')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            result_img_path = None
        """)

    if "most common words" in p and text:
        m = re.search(r"'([^']+)'", prompt)
        col = m.group(1) if m else text[0]
        
        return textwrap.dedent(f"""
            from collections import Counter
            import re
            
            all_text = ' '.join(df['{col}'].dropna().astype(str))
            words = re.findall(r'\\b\\w+\\b', all_text.lower())
            word_counts = Counter(words).most_common(20)
            
            result = pd.DataFrame(word_counts, columns=['Word', 'Count'])
        """)

    # ──── AGGREGATIONS ────
    if "top 10 rows by" in p and numeric:
        m = re.search(r"'([^']+)'", prompt)
        col = m.group(1) if m else numeric[0]
        
        return f"result = df.nlargest(10, '{col}')"

    if "bottom 10 rows by" in p and numeric:
        m = re.search(r"'([^']+)'", prompt)
        col = m.group(1) if m else numeric[0]
        
        return f"result = df.nsmallest(10, '{col}')"

    if "sum of" in p and "by" in p and categorical and numeric:
        m = re.findall(r"'([^']+)'", prompt)
        if len(m) >= 2:
            num_col, cat_col = m[0], m[1]
        else:
            num_col, cat_col = numeric[0], categorical[0]
        
        return textwrap.dedent(f"""
            result = df.groupby('{cat_col}')['{num_col}'].sum().reset_index()
            result.columns = ['{cat_col}', 'Total_{num_col}']
            result = result.sort_values('Total_{num_col}', ascending=False)
        """)

    if "average of" in p and "by" in p and categorical and numeric:
        m = re.findall(r"'([^']+)'", prompt)
        if len(m) >= 2:
            num_col, cat_col = m[0], m[1]
        else:
            num_col, cat_col = numeric[0], categorical[0]
        
        return textwrap.dedent(f"""
            result = df.groupby('{cat_col}')['{num_col}'].mean().reset_index()
            result.columns = ['{cat_col}', 'Average_{num_col}']
            result = result.sort_values('Average_{num_col}', ascending=False).round(2)
        """)

    if "count by" in p and categorical:
        m = re.search(r"'([^']+)'", prompt)
        col = m.group(1) if m else categorical[0]
        
        return textwrap.dedent(f"""
            result = df['{col}'].value_counts().reset_index()
            result.columns = ['{col}', 'Count']
        """)

    if "statistical summary by" in p and categorical and numeric:
        cat_col = categorical[0]
        
        return textwrap.dedent(f"""
            result = df.groupby('{cat_col}').describe().T.round(2)
        """)

    # ──── ADVANCED ────
    if "3d scatter" in p and len(numeric) >= 3:
        x, y, z = numeric[0], numeric[1], numeric[2]
        
        return textwrap.dedent(f"""
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(df['{x}'], df['{y}'], df['{z}'], c='blue', marker='o', alpha=0.6)
            ax.set_xlabel('{x}')
            ax.set_ylabel('{y}')
            ax.set_zlabel('{z}')
            ax.set_title(f'3D Scatter: {x}, {y}, {z}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            result_img_path = None
        """)

    # No template match
    return None


# ────────────────────────────────────────────────
# Code Execution
# ────────────────────────────────────────────────
def run_code(df: pd.DataFrame, code: str) -> Dict[str, Any]:
    """Execute generated code safely"""
    local_ns = {
        "pd": pd,
        "np": np,
        "df": df.copy(),
        "plt": plt,
        "sns": None
    }

    if "sns" in code.lower() or "seaborn" in code.lower():
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

        return {"type": "text", "output": "✅ Code executed successfully"}

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
    """Generate concise dataset summary for LLM context"""
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
# LLM Streaming Function
# ────────────────────────────────────────────────
def ask_llm(
    prompt: str,
    df_info: Optional[str] = None,
    provider: str = "gemini",
    model: str = None,
    timeout: int = None
) -> Generator[str, None, None]:
    """Stream response from selected free API provider (ONLY for custom prompts)"""

    if timeout is None:
        timeout = AnalystConfig.DEFAULT_TIMEOUT

    if provider not in free_api_client.providers:
        available = list(free_api_client.providers.keys())
        if not available:
            yield "[ERROR] No API keys configured. Please add at least one to .env file"
            return
        provider = available[0]

    if model is None:
        model = free_api_client.providers[provider]["models"][0]

    yield from free_api_client.stream(provider, prompt, df_info or "", model, timeout)


# ────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────
def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    """Validate DataFrame is suitable for analysis"""
    if df.empty:
        return False, "DataFrame is empty"
    if len(df.columns) == 0:
        return False, "DataFrame has no columns"
    if len(df) > 1_000_000:
        return False, f"DataFrame too large ({len(df):,} rows)"
    return True, "OK"


def extract_python_code(llm_output: str) -> Optional[str]:
    """Extract Python code from LLM response"""
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


def _is_code_incomplete(code: str) -> bool:
    """Detect whether generated Python code is likely incomplete"""
    if not code or not code.strip():
        return True

    if not code.strip().endswith((")", "]", "}", "result", "plt.savefig", "plt.tight_layout()")):
        return True

    try:
        compile(code, "<generated>", "exec")
        return False
    except SyntaxError:
        return True


def _continuation_prompt() -> str:
    """Generate continuation prompt for incomplete code"""
    return (
        "CONTINUE the previous Python code EXACTLY from where it stopped.\n"
        "Rules:\n"
        "- DO NOT repeat any code\n"
        "- DO NOT explain\n"
        "- DO NOT add markdown\n"
        "- Return ONLY Python code\n"
        "- Finish all open blocks\n"
        "- End with `result = ...` or `plt.savefig(...)`\n"
    )
