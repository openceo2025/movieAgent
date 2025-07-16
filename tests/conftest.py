import sys
from pathlib import Path

# Ensure project root is on sys.path so movie_agent can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
