#!/usr/bin/env python3
"""
Test script for model_application.py scaling functionality
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_application import test_scaling_consistency

if __name__ == "__main__":
    print("Running scaling consistency test...")
    try:
        test_scaling_consistency()
        print("✓ All tests passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        sys.exit(1)
