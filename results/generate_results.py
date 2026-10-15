"""
Convenience entry-point alias for results/generate_results_visuals.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(BASE_DIR, "generate_results_visuals.py")

if __name__ == "__main__":
    import generate_results_visuals
    generate_results_visuals.main()
