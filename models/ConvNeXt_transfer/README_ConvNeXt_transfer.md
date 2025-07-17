# ConvNeXt Transfer Learning Model (ConvNeXt_transfer.py)

## Overview

This script implements a transfer learning approach using the pre-trained ConvNeXt-Tiny model for market detection in satellite imagery. The model processes 7-day sequences of satellite images (treated as 7-channel images) to predict whether a market is present.

## Architecture

### Base Model
- **Pre-trained Model**: ConvNeXt-Tiny from ImageNet
- **Input Adaptation**: Custom Conv2D layer to convert 7-channel input to 3-channel
- **Transfer Learning**: Two-stage training (frozen → fine-tuning)

### Architecture Details
1. **Input Layer**: (128, 128, 7) - 7-day satellite image sequence
2. **Channel Adapter**: Conv2D(3, (1,1)) + BatchNorm + ReLU to convert 7→3 channels
3. **ConvNeXt Backbone**: Pre-trained ConvNeXt-Tiny (frozen initially)
4. **Classification Head**: GlobalAveragePooling2D + Dense(1, sigmoid)

### Training Strategy
- **Stage 1**: Train with frozen ConvNeXt backbone (30 epochs)
- **Stage 2**: Fine-tune entire model with lower learning rate (15 epochs)

## Input Format

The model expects input in the format produced by `data_loader.py`:
- **Shape**: (batch_size, 128, 128, 7)
- **Format**: Channels-last, where the last dimension contains the 7 days
- **Processing**: Treats the 7 days as 7 channels in a single image

## Training Setup

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
- **Caching**: Datasets are cached and prefetched for performance

### Training Configuration

#### Stage 1 (Frozen Backbone)
- **Optimizer**: Adam (default learning rate)
- **Loss**: Binary crossentropy
- **Epochs**: 30
- **Trainable**: Only channel adapter and classification head

#### Stage 2 (Fine-tuning)
- **Optimizer**: Adam (lr=1e-5)
- **Loss**: Binary crossentropy  
- **Epochs**: 15
- **Trainable**: Entire model including ConvNeXt backbone

#### Callbacks
- EarlyStopping (patience=5)
- ReduceLROnPlateau (patience=3, factor=0.5)
- ModelCheckpoint (saves best model)

## Usage

### Direct Execution
```bash
python models/ConvNeXt_transfer/ConvNeXt_transfer.py
```

### With SLURM
```bash
sbatch run_ConvNeXt_training.sbatch
```

### Environment Variables
- `SCRATCH`: Base directory for model outputs (default: "/scratch/users/pdingus")

## Output Structure

The script creates the following directory structure:
```
models/ConvNeXt_transfer/YYYYMMDD_HHMM/
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
    --model-path models/ConvNeXt_transfer/YYYYMMDD_HHMM \
    --drop-threshold 0.1
```

## Differences from Siamese Fusion Model

1. **Architecture**: Pre-trained ConvNeXt vs. custom Siamese network
2. **Temporal Processing**: Treats 7 days as channels vs. explicit sequence processing
3. **Model Size**: Large pre-trained model vs. smaller custom architecture
4. **Training Strategy**: Transfer learning vs. training from scratch
5. **Training Time**: Longer per epoch but potentially fewer epochs needed

## Performance Considerations

- **Memory**: More memory-intensive due to large ConvNeXt backbone
- **Training Speed**: Slower per epoch but leverages pre-trained features
- **Feature Quality**: Benefits from ImageNet pre-training for spatial features
- **Generalization**: Potentially better generalization due to pre-trained features
- **Computational Cost**: Higher inference cost due to model complexity

## Advantages of Transfer Learning Approach

1. **Pre-trained Features**: Leverages spatial features learned from ImageNet
2. **Faster Convergence**: Often converges faster due to good initialization
3. **Better Performance**: Typically achieves higher accuracy on small datasets
4. **Robustness**: More robust to overfitting with limited training data

## Model Architecture Visualization

```
Input (128,128,7)
    ↓
Conv2D(3,(1,1)) + BN + ReLU  [Channel Adapter]
    ↓
ConvNeXt-Tiny Backbone       [Pre-trained, ImageNet]
    ↓
GlobalAveragePooling2D
    ↓
Dense(1, sigmoid)            [Market Probability]
```

## Hardware Requirements

- **GPU Memory**: Minimum 8GB recommended for batch_size=16
- **Training Time**: ~2-4 hours on modern GPU (depending on dataset size)
- **Storage**: ~500MB for model checkpoints and artifacts

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch_size or use gradient accumulation
2. **Slow Training**: Ensure dataset caching is enabled
3. **Poor Convergence**: Check learning rate schedule and data quality

### Performance Tuning
- Adjust learning rates for each training stage
- Experiment with different ConvNeXt variants (Small, Base)
- Consider mixed precision training for speed improvements
