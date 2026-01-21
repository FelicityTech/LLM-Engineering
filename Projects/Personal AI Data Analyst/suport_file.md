# 📁 Supported File Formats - FelicityTech AI Data Analyst

## Overview

The FelicityTech AI Data Analyst supports **8 different file formats** for maximum flexibility in your data analysis workflow.

---

## 📊 Supported Formats

### 1. CSV (Comma-Separated Values)
**Extensions:** `.csv`

**Best for:**
- Simple tabular data
- Data exchange between systems
- Lightweight storage

**Example:**
```csv
name,age,city
John,30,New York
Jane,25,London
```

**Pros:**
- ✅ Universal compatibility
- ✅ Human-readable
- ✅ Small file size

**Cons:**
- ❌ No data type preservation
- ❌ Encoding issues possible

---

### 2. Excel
**Extensions:** `.xlsx`, `.xls`

**Best for:**
- Business data
- Reports from Excel
- Data with formatting

**Pros:**
- ✅ Preserves formatting
- ✅ Multiple sheets support
- ✅ Wide adoption

**Cons:**
- ❌ Larger file size
- ❌ Slower for big data

**Note:** Only the first sheet is loaded by default.

---

### 3. JSON (JavaScript Object Notation)
**Extensions:** `.json`

**Best for:**
- API responses
- Nested/hierarchical data
- Configuration files

**Example:**
```json
[
  {"name": "John", "age": 30, "city": "New York"},
  {"name": "Jane", "age": 25, "city": "London"}
]
```

**Pros:**
- ✅ Handles nested data
- ✅ Preserves data types
- ✅ Human-readable

**Cons:**
- ❌ Can be verbose
- ❌ Not ideal for large datasets

---

### 4. Parquet
**Extensions:** `.parquet`

**Best for:**
- Big data processing
- Data warehousing
- High-performance analytics

**Pros:**
- ✅ **Excellent compression**
- ✅ **Very fast read/write**
- ✅ Preserves data types perfectly
- ✅ Columnar storage (efficient queries)

**Cons:**
- ❌ Not human-readable
- ❌ Requires pyarrow library

**Recommended for files > 100MB**

---

### 5. Feather
**Extensions:** `.feather`

**Best for:**
- Fast data exchange
- Temporary storage
- Python-R interoperability

**Pros:**
- ✅ **Extremely fast**
- ✅ Preserves data types
- ✅ Simple format

**Cons:**
- ❌ Not for long-term storage
- ❌ Limited compression
- ❌ Less common than Parquet

---

### 6. HDF5 (Hierarchical Data Format)
**Extensions:** `.h5`, `.hdf5`, `.hdf`

**Best for:**
- Scientific data
- Large arrays
- Complex hierarchical data
- Time series with metadata

**Pros:**
- ✅ Handles huge datasets
- ✅ Efficient for numerical data
- ✅ Stores metadata
- ✅ Supports compression

**Cons:**
- ❌ More complex format
- ❌ Requires tables library
- ❌ File can become corrupted

**Note:** If multiple datasets exist in the file, the first one is loaded.

---

### 7. XML (Extensible Markup Language)
**Extensions:** `.xml`

**Best for:**
- Configuration files
- Web service responses
- Structured documents
- Legacy systems

**Example:**
```xml
<data>
  <record>
    <name>John</name>
    <age>30</age>
    <city>New York</city>
  </record>
</data>
```

**Pros:**
- ✅ Self-describing
- ✅ Handles hierarchical data
- ✅ Wide industry support

**Cons:**
- ❌ Verbose
- ❌ Slower parsing
- ❌ Larger file size

---

### 8. TXT (Text Files)
**Extensions:** `.txt`

**Best for:**
- Tab-delimited data
- Pipe-delimited data
- Simple data exports
- Log files

**Supported delimiters:**
- Tab (`\t`)
- Comma (`,`)
- Pipe (`|`)

**Example:**
```
name	age	city
John	30	New York
Jane	25	London
```

**Pros:**
- ✅ Simple format
- ✅ Universal compatibility
- ✅ Lightweight

**Cons:**
- ❌ No data type information
- ❌ Delimiter detection needed

---

## 🔄 Format Conversion Examples

### CSV to Parquet (for better performance)
```python
import pandas as pd

# Read CSV
df = pd.read_csv('large_file.csv')

# Save as Parquet (much smaller and faster)
df.to_parquet('large_file.parquet')
```

### Excel to CSV
```python
df = pd.read_excel('data.xlsx')
df.to_csv('data.csv', index=False)
```

### JSON to Parquet
```python
df = pd.read_json('data.json')
df.to_parquet('data.parquet')
```

---

## 📈 Performance Comparison

| Format | Read Speed | File Size | Compression | Type Preservation |
|--------|-----------|-----------|-------------|-------------------|
| CSV | Moderate | Large | None | Poor |
| Excel | Slow | Large | Moderate | Good |
| JSON | Moderate | Large | None | Good |
| **Parquet** | **Fast** | **Small** | **Excellent** | **Perfect** |
| **Feather** | **Very Fast** | Moderate | Good | **Perfect** |
| HDF5 | Fast | Small | Good | Perfect |
| XML | Slow | Very Large | None | Good |
| TXT | Moderate | Large | None | Poor |

**Recommendation:** For files > 100MB, convert to Parquet for best performance.

---

## 🛠️ Troubleshooting

### Issue: "Failed to load data"

**For CSV/TXT files:**
```python
# Try different encodings
df = pd.read_csv('file.csv', encoding='latin-1')
df = pd.read_csv('file.csv', encoding='utf-8-sig')

# Try different delimiters
df = pd.read_csv('file.txt', sep='\t')
df = pd.read_csv('file.txt', sep='|')
```

**For Excel files:**
```bash
# Install openpyxl
pip install openpyxl
```

**For Parquet files:**
```bash
# Install pyarrow
pip install pyarrow
```

**For HDF5 files:**
```bash
# Install tables (PyTables)
pip install tables
```

**For XML files:**
```bash
# Install lxml
pip install lxml
```

### Issue: "File too large"

**Solution: Sample the data**
```python
# Read only first 100,000 rows
df = pd.read_csv('huge_file.csv', nrows=100000)

# Or sample randomly
df = pd.read_csv('huge_file.csv')
df_sample = df.sample(n=50000, random_state=42)
df_sample.to_csv('sample.csv', index=False)
```

### Issue: HDF5 "No dataset found"

**Solution: Specify key**
```python
# List available keys
import pandas as pd
with pd.HDFStore('file.h5', 'r') as store:
    print(store.keys())

# Read specific key
df = pd.read_hdf('file.h5', key='/data')
```

---

## 💡 Best Practices

### 1. Choose the Right Format

**For daily work:**
- CSV - Simple data sharing
- Excel - Business reports
- JSON - API data

**For performance:**
- Parquet - Large datasets
- Feather - Fast temporary storage
- HDF5 - Scientific/numerical data

### 2. File Size Guidelines

| Data Size | Recommended Format |
|-----------|-------------------|
| < 10 MB | CSV, Excel, JSON |
| 10-100 MB | CSV, Parquet |
| 100 MB - 1 GB | **Parquet** |
| > 1 GB | **Parquet, HDF5** |

### 3. Data Type Preservation

**Formats that preserve types:**
- ✅ Parquet (best)
- ✅ Feather
- ✅ HDF5
- ✅ JSON
- ⚠️ Excel (partial)
- ❌ CSV (loses types)
- ❌ TXT (loses types)

### 4. Compression Tips

```python
# Parquet with compression
df.to_parquet('file.parquet', compression='gzip')

# HDF5 with compression
df.to_hdf('file.h5', key='data', complevel=9)

# CSV with gzip
df.to_csv('file.csv.gz', compression='gzip', index=False)
```

---

## 🎯 Quick Reference

### Reading Files in Python

```python
import pandas as pd

# CSV
df = pd.read_csv('file.csv')

# Excel
df = pd.read_excel('file.xlsx')

# JSON
df = pd.read_json('file.json')

# Parquet
df = pd.read_parquet('file.parquet')

# Feather
df = pd.read_feather('file.feather')

# HDF5
df = pd.read_hdf('file.h5', key='data')

# XML
df = pd.read_xml('file.xml')

# TXT (tab-delimited)
df = pd.read_csv('file.txt', sep='\t')
```

---

## 📚 Additional Resources

- [Pandas I/O Documentation](https://pandas.pydata.org/docs/user_guide/io.html)
- [Parquet Format](https://parquet.apache.org/)
- [HDF5 Documentation](https://www.hdfgroup.org/)
- [PyArrow Documentation](https://arrow.apache.org/docs/python/)

---

<div align="center">

**FelicityTech AI Data Analyst**

Created by [Solomon Eniola Adegoke](https://www.linkedin.com/in/solomon-eniola-adegoke/)

</div>