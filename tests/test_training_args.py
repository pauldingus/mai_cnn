#!/usr/bin/env python3
"""
Test script to verify training arguments export/import functionality
"""

import pickle
import os
import sys
import tempfile

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Test the training arguments functionality
def test_training_args():
    print("Testing training arguments export/import...")
    
    # Create test training arguments (matching ConvNeXt_transfer.py)
    training_args = {
        'scaling': 'standard',
        'per_image_scaling': True,
        'do_augmentation': True,
        'do_clipping': True,
        'lower_clip': 0,
        'upper_clip': 40,
    }
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save training arguments
        training_args_file = os.path.join(temp_dir, 'training_args.pkl')
        with open(training_args_file, 'wb') as f:
            pickle.dump(training_args, f)
        print(f"✓ Saved training arguments to {training_args_file}")
        
        # Load training arguments
        with open(training_args_file, 'rb') as f:
            loaded_args = pickle.load(f)
        print(f"✓ Loaded training arguments: {loaded_args}")
        
        # Verify they match
        assert loaded_args == training_args, "Training arguments don't match!"
        print("✓ Training arguments match!")
        
        # Test Args class
        from model_application import Args
        
        # Test creating Args from training arguments
        args = Args.from_training_args(loaded_args)
        print(f"✓ Created Args object: scaling={args.scaling}, per_image_scaling={args.per_image_scaling}")
        print(f"  do_clipping={args.do_clipping}, lower_clip={args.lower_clip}, upper_clip={args.upper_clip}")
        
        # Verify values
        assert args.scaling == 'standard'
        assert args.per_image_scaling == True
        assert args.do_clipping == True
        assert args.lower_clip == 0
        assert args.upper_clip == 40
        print("✓ Args object values are correct!")
        
    print("✓ All tests passed!")

if __name__ == "__main__":
    test_training_args()
