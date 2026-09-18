import os

# Prevent OpenBLAS/MKL thread memory exhaustion crashes on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import uvicorn

# Ensure 'src' is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main():
    print("=" * 70)
    print("STARTING PRECISION OBSERVABILITY CONSOLE")
    print("Access the dashboard at: http://127.0.0.1:8000")
    print("=" * 70)
    uvicorn.run("dashboard.server:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
