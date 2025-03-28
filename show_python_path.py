# Create a file show_python_path.py with this content:
import sys
import site
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")
print(f"User site-packages: {site.USER_SITE}")