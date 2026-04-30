import json

file_path = r'c:\Users\USER\Documents\FelicityTech\LLM-Engineering\Explainable AI in Python\MAE-SHAP-Kernel_explainer.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for cell in data.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        for i, line in enumerate(source):
            if 'X = insurance.drop(columns=["charges"]' in line or "X = insurance.drop(columns=['charges']" in line:
                if i == 0 or 'insurance = insurance.dropna()' not in source[i-1]:
                    source.insert(i, "insurance = insurance.dropna()\n")
                break

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)
