"""
System Check Script for AI Data Analyst
Run this to diagnose installation issues
"""

import sys
import subprocess
import os
from pathlib import Path

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_python():
    print_section("1. Python Installation")
    print(f"✅ Python Version: {sys.version}")
    print(f"✅ Python Path: {sys.executable}")
    return True

def check_pip():
    print_section("2. Pip Installation")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Pip: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Pip Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Pip not found: {e}")
        return False

def check_packages():
    print_section("3. Required Packages")
    required = {
        "streamlit": "1.31.0",
        "pandas": "2.0.0",
        "numpy": "1.24.0",
        "matplotlib": "3.7.0",
        "seaborn": "0.12.0",
        "openpyxl": "3.1.0",
        "pyarrow": "14.0.0",
        "tables": "3.8.0",
        "lxml": "4.9.0",
        "scipy": "1.11.0"
    }
    
    all_ok = True
    for package, min_version in required.items():
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "show", package],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version_line = [l for l in result.stdout.split('\n') if l.startswith('Version:')]
                if version_line:
                    version = version_line[0].split(':')[1].strip()
                    print(f"✅ {package}: {version}")
                else:
                    print(f"✅ {package}: installed (version unknown)")
            else:
                print(f"❌ {package}: NOT INSTALLED (required: {min_version}+)")
                all_ok = False
        except Exception as e:
            print(f"❌ {package}: Error checking - {e}")
            all_ok = False
    
    if not all_ok:
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements.txt")
    
    return all_ok

def check_ollama():
    print_section("4. Ollama Installation (Optional)")
    
    # Try common Windows paths
    possible_paths = [
        "ollama",
        os.path.expanduser("~\\AppData\\Local\\Programs\\Ollama\\ollama.exe"),
        "C:\\Program Files\\Ollama\\ollama.exe",
        os.path.expanduser("~\\AppData\\Local\\Ollama\\ollama.exe"),
    ]
    
    ollama_found = False
    ollama_path = None
    
    for path in possible_paths:
        try:
            result = subprocess.run([path, "--version"], 
                                  capture_output=True, 
                                  text=True,
                                  timeout=5)
            if result.returncode == 0:
                print(f"✅ Ollama found: {path}")
                print(f"   Version: {result.stdout.strip()}")
                ollama_found = True
                ollama_path = path
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        except Exception:
            continue
    
    if not ollama_found:
        print("❌ Ollama not found")
        print("   Install from: https://ollama.ai")
        print("   This is OPTIONAL - app works without it")
        return False
    
    # Check for models
    try:
        result = subprocess.run([ollama_path, "list"],
                              capture_output=True,
                              text=True,
                              timeout=10)
        if result.returncode == 0:
            models = result.stdout.strip().split('\n')[1:]  # Skip header
            if models and models[0]:
                print(f"✅ Models installed:")
                for model in models[:5]:  # Show first 5
                    print(f"   - {model.split()[0]}")
            else:
                print("⚠️  No models installed")
                print("   Run: ollama pull llama3.1")
    except Exception as e:
        print(f"⚠️  Could not check models: {e}")
    
    return True

def check_files():
    print_section("5. Project Files")
    
    required_files = ["analyst.py", "app.py", "requirements.txt"]
    all_found = True
    
    for file in required_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            print(f"✅ {file} ({size:,} bytes)")
        else:
            print(f"❌ {file} - NOT FOUND")
            all_found = False
    
    return all_found

def suggest_fixes():
    print_section("6. Suggested Actions")
    
    print("\n📋 To fix missing packages:")
    print("   pip install -r requirements.txt")
    print("   (or: python -m pip install -r requirements.txt)")
    
    print("\n🤖 To install Ollama (optional):")
    print("   1. Visit https://ollama.ai")
    print("   2. Download Windows installer")
    print("   3. Run installer")
    print("   4. Restart Command Prompt")
    print("   5. Run: ollama pull llama3.1")
    
    print("\n▶️  To start the app:")
    print("   streamlit run app.py")
    
    print("\n🆘 If issues persist:")
    print("   1. Try using a virtual environment:")
    print("      python -m venv venv")
    print("      venv\\Scripts\\activate")
    print("      pip install -r requirements.txt")
    print("   2. Run as Administrator")
    print("   3. Check Windows Firewall settings")

def main():
    print("\n" + "🧠 FELICITYTECH AI DATA ANALYST - SYSTEM CHECK".center(60))
    print("="*60)
    print("Created by Solomon Eniola Adegoke".center(60))
    print("="*60)
    
    results = {
        "Python": check_python(),
        "Pip": check_pip(),
        "Packages": check_packages(),
        "Files": check_files(),
        "Ollama": check_ollama()  # Optional
    }
    
    print_section("7. Summary")
    
    critical_ok = all([results["Python"], results["Pip"], results["Packages"], results["Files"]])
    
    if critical_ok:
        print("✅ ALL CRITICAL COMPONENTS READY!")
        if results["Ollama"]:
            print("✅ Ollama available for custom queries")
        else:
            print("⚠️  Ollama not installed (optional)")
            print("   You can still use 13+ built-in analyses")
        print("\n🚀 Ready to start: streamlit run app.py")
    else:
        print("❌ SOME COMPONENTS MISSING")
        print("\n   Missing:")
        for name, status in results.items():
            if not status and name != "Ollama":
                print(f"   - {name}")
    
    suggest_fixes()
    
    print("\n" + "="*60)
    print("Check complete!".center(60))
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCheck cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()