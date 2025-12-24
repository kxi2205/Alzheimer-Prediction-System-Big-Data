#!/usr/bin/env python3
"""
Pre-Deployment Validation Script
Checks if the app is ready for Streamlit Cloud deployment
"""

import sys
from pathlib import Path
import json

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def check_mark(passed):
    return f"{Colors.GREEN}✓{Colors.END}" if passed else f"{Colors.RED}✗{Colors.END}"

def check_requirements_file():
    """Check if requirements.txt exists and is properly formatted"""
    print(f"{Colors.BOLD}1. Checking requirements.txt...{Colors.END}")
    
    project_root = Path(__file__).parent
    req_file = project_root / 'requirements.txt'
    
    if not req_file.exists():
        print(f"  {check_mark(False)} requirements.txt not found in root directory")
        return False
    
    print(f"  {check_mark(True)} requirements.txt exists")
    
    # Check for essential packages
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'scikit-learn',
        'xgboost', 'joblib', 'plotly'
    ]
    
    with open(req_file, 'r') as f:
        content = f.read().lower()
    
    missing = []
    for package in required_packages:
        if package in content:
            print(f"  {check_mark(True)} {package} found")
        else:
            print(f"  {check_mark(False)} {package} MISSING")
            missing.append(package)
    
    return len(missing) == 0

def check_model_files():
    """Check if model files exist"""
    print(f"\n{Colors.BOLD}2. Checking model files...{Colors.END}")
    
    project_root = Path(__file__).parent
    model_path = project_root / 'outputs' / 'models' / 'xgboost_model.pkl'
    
    if not model_path.exists():
        print(f"  {check_mark(False)} Model file not found: {model_path}")
        return False
    
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  {check_mark(True)} Model file exists: {model_path.name}")
    print(f"  {check_mark(True)} Model size: {size_mb:.2f} MB")
    
    if size_mb > 100:
        print(f"  {Colors.YELLOW}⚠ Warning: Model > 100MB, consider Git LFS{Colors.END}")
    
    return True

def check_config_files():
    """Check configuration files"""
    print(f"\n{Colors.BOLD}3. Checking configuration files...{Colors.END}")
    
    project_root = Path(__file__).parent
    
    files_to_check = {
        'streamlit_app.py': True,
        '.streamlit/config.toml': False,
        'outputs/config/final_features.json': False,
        '.gitattributes': False
    }
    
    all_critical = True
    
    for file_path, critical in files_to_check.items():
        full_path = project_root / file_path
        exists = full_path.exists()
        symbol = check_mark(exists)
        status = "" if exists else " - MISSING" + (" (CRITICAL)" if critical else "")
        print(f"  {symbol} {file_path}{status}")
        
        if critical and not exists:
            all_critical = False
    
    return all_critical

def check_imports():
    """Test if all required packages can be imported"""
    print(f"\n{Colors.BOLD}4. Testing package imports...{Colors.END}")
    
    packages_to_test = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'joblib': 'joblib',
        'plotly': 'plotly'
    }
    
    all_imported = True
    
    for import_name, package_name in packages_to_test.items():
        try:
            __import__(import_name)
            print(f"  {check_mark(True)} {package_name}")
        except ImportError:
            print(f"  {check_mark(False)} {package_name} - Cannot import")
            all_imported = False
    
    return all_imported

def check_file_paths():
    """Check that code uses relative paths"""
    print(f"\n{Colors.BOLD}5. Checking for absolute paths in code...{Colors.END}")
    
    project_root = Path(__file__).parent
    app_file = project_root / 'streamlit_app.py'
    
    if not app_file.exists():
        print(f"  {check_mark(False)} streamlit_app.py not found")
        return False
    
    with open(app_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for common absolute path patterns
    issues = []
    
    if 'C:\\' in content or 'C:/' in content:
        issues.append("Windows absolute paths found (C:\\)")
    
    if '/Users/' in content or '/home/' in content:
        issues.append("Unix absolute paths found (/Users/ or /home/)")
    
    if issues:
        for issue in issues:
            print(f"  {check_mark(False)} {issue}")
        print(f"  {Colors.YELLOW}  Use Path(__file__).parent for relative paths{Colors.END}")
        return False
    else:
        print(f"  {check_mark(True)} No absolute paths detected")
        return True

def check_git_status():
    """Check git status (if git is available)"""
    print(f"\n{Colors.BOLD}6. Checking git repository status...{Colors.END}")
    
    project_root = Path(__file__).parent
    git_dir = project_root / '.git'
    
    if not git_dir.exists():
        print(f"  {Colors.YELLOW}⚠ Not a git repository{Colors.END}")
        return True
    
    try:
        import subprocess
        
        # Check for uncommitted changes
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        if result.returncode != 0:
            print(f"  {Colors.YELLOW}⚠ Could not check git status{Colors.END}")
            return True
        
        uncommitted = result.stdout.strip()
        
        if uncommitted:
            print(f"  {Colors.YELLOW}⚠ Uncommitted changes detected:{Colors.END}")
            for line in uncommitted.split('\n')[:5]:  # Show first 5
                print(f"    {line}")
            if len(uncommitted.split('\n')) > 5:
                print(f"    ... and {len(uncommitted.split('\n')) - 5} more")
            print(f"\n  {Colors.YELLOW}  Commit and push before deploying{Colors.END}")
            return False
        else:
            print(f"  {check_mark(True)} No uncommitted changes")
            return True
            
    except FileNotFoundError:
        print(f"  {Colors.YELLOW}⚠ Git not found in PATH{Colors.END}")
        return True

def main():
    print_header("Streamlit Cloud Pre-Deployment Validation")
    
    checks = [
        ("Requirements file", check_requirements_file),
        ("Model files", check_model_files),
        ("Configuration files", check_config_files),
        ("Package imports", check_imports),
        ("File paths", check_file_paths),
        ("Git status", check_git_status)
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"  {check_mark(False)} Error during check: {str(e)}")
            results.append((name, False))
    
    # Summary
    print_header("Validation Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  {name:.<40} {status}")
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} checks passed{Colors.END}\n")
    
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All checks passed! Ready for deployment.{Colors.END}\n")
        print("Next steps:")
        print("1. Commit and push all changes:")
        print("   git add .")
        print("   git commit -m 'Ready for Streamlit deployment'")
        print("   git push origin main")
        print("\n2. Deploy on Streamlit Cloud:")
        print("   - Go to https://share.streamlit.io")
        print("   - Click 'New app'")
        print("   - Select your repository")
        print("   - Set main file: streamlit_app.py")
        print("   - Click 'Deploy'\n")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some checks failed. Please fix the issues above.{Colors.END}\n")
        print("See STREAMLIT_TROUBLESHOOTING.md for detailed help.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
