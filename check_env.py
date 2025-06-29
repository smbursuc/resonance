import sys
import numpy
import matplotlib # Since matplotlib was in your traceback
import os

print(f"--- Environment Check ---")
print(f"Python Executable:    {sys.executable}")
print(f"Python Version:       {sys.version.split()[0]}") # Just the version number
print(f"Current Working Dir:  {os.getcwd()}" if 'os' in sys.modules else "os module not imported yet for CWD") # Add import os if not already there

print("\n--- sys.path ---")
for p in sys.path:
    print(f"  {p}")

print(f"\n--- NumPy ---")
print(f"NumPy Version:        {numpy.__version__}")
print(f"NumPy Location:       {numpy.__file__}")

print(f"\n--- Matplotlib ---")
print(f"Matplotlib Version:   {matplotlib.__version__}")
print(f"Matplotlib Location:  {matplotlib.__file__}")

# Try the problematic import sequence if possible, or a part of it
try:
    from numpy.core import umath
    from numpy.core._ufunc_config import _no_nep50_warning # This was near the error
    print("\nSuccessfully imported parts related to the error from NumPy.")
except ImportError as e:
    print(f"\nFailed during specific NumPy internal import test: {e}")