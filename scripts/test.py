"""Test if matplotlib can save figures to the network path"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from RCP_analysis.python.functions.config_loading import *

# Setup
FIG_DIR = METADATA_ROOT.parent / "figures" / "time_calibration"
FIG_DIR.mkdir(parents=True, exist_ok=True)

test_path = FIG_DIR / "test_figure.png"

print(f"Testing figure save to: {test_path}")
print(f"Directory exists: {FIG_DIR.exists()}")

# Create simple plot
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
ax.set_title("Test Figure")

# Try to save
try:
    plt.savefig(test_path, dpi=150)
    print(f"✓ savefig completed")
    
    if test_path.exists():
        print(f"✓ File exists: {test_path.stat().st_size} bytes")
    else:
        print(f"✗ File not found after save!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    plt.close()