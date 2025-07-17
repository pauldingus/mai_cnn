# Siamese Fusion Model (siamese_fusion.py)

## Overview

This script implements a Siamese neural network with temporal fusion for market detection in satellite imagery. The model processes 7-day sequences of satellite images to predict whether a market is present at any time during the week.

## Architecture

### Day Encoder (Siamese Branch)
- **Input**: Single day image (128x128x1)
- **Architecture**: 3 Conv2D blocks with BatchNormalization and MaxPooling
- **Filters**: [16, 32, 64]
- **Output**: 128-dimensional feature vector

### Temporal Fusion
- **TimeDistributed**: Applies the same day encoder to all 7 days
- **Conv1D Layers**: 2 stacked 1D convolutions for temporal pattern recognition
- **Global Pooling**: GlobalMaxPooling1D to aggregate temporal features
- **Output**: Single sigmoid activation for binary market detection

## Input Format

The model expects input in the format produced by `data_loader.py`:
- **Shape**: (batch_size, 128, 128, 7)
- **Format**: Channels-last, where the last dimension contains the 7 days
- **Preprocessing**: The model internally reshapes to (batch_size, 7, 128, 128, 1) for TimeDistributed processing

## Training Setup

The script follows the same structure as `ConvNeXt_transfer.py`:

### Training Arguments
```python
training_args = {
    'scaling': 'standard',
    'per_image_scaling': True,
    'do_augmentation': True,
    'do_clipping': True,
    'lower_clip': 0,
    'upper_clip': 40,
}
```

### Data Pipeline
- **Train/Val/Test Split**: 70/15/15
- **Batch Size**: 16
- **Sample Size**: 1000 images for scaling estimation
- **Augmentation**: Random flips and rotations during training

### Training Configuration
- **Optimizer**: Adam (lr=1e-4)
- **Loss**: Binary crossentropy
- **Epochs**: 30
- **Callbacks**: 
  - EarlyStopping (patience=5)
  - ReduceLROnPlateau (patience=3, factor=0.5)
  - ModelCheckpoint (saves best model)

## Usage

### Direct Execution
```bash
python models/siamese_fusion/siamese_fusion.py
```

### With SLURM
```bash
sbatch run_siamese_fusion.sbatch
```

### Environment Variables
- `SCRATCH`: Base directory for model outputs (default: "/scratch/users/pdingus")

## Output Structure

The script creates the following directory structure:
```
models/siamese_fusion_anymarket/YYYYMMDD_HHMM/
├── checkpoints/
│   └── best_model.keras          # Best model checkpoint
├── history.pkl                   # Training history
├── scaler.pkl                    # Data scaler
└── training_args.pkl             # Training arguments for model_application.py
```

## Model Application

The trained model can be used with `model_application.py` for inference:

```bash
python model_application.py \
    --country Nigeria \
    --model-path models/siamese_fusion_anymarket/YYYYMMDD_HHMM \
    --drop-threshold 0.1
```

## Differences from ConvNeXt Transfer Learning

1. **Architecture**: Custom Siamese network vs. pre-trained ConvNeXt
2. **Temporal Processing**: Explicit 7-day sequence processing vs. treating as 7-channel image
3. **Model Size**: Smaller, custom architecture vs. large pre-trained model
4. **Training Time**: Faster training from scratch vs. transfer learning

## Performance Considerations

- **Memory**: More memory-efficient than ConvNeXt due to smaller model size
- **Training Speed**: Faster per epoch but may require more epochs
- **Temporal Modeling**: Better at capturing temporal patterns due to dedicated 1D convolutions
- **Interpretability**: More interpretable architecture with explicit day-by-day processing
