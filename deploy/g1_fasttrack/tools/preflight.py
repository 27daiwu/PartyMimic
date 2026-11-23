#!/usr/bin/env python3
"""READ-ONLY preflight = tools/diag.py --write-pass. Needs --entry-fsm N (the fsm observed by `bin/ft diag`)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag
if __name__ == "__main__":
    sys.exit(diag.main(sys.argv[1:] + ["--write-pass"]))
