#!/usr/bin/env python3
"""
model_application.py

Script for running CNN predictions on candidate locations using Earth Engine.
Adapted for use with sbatch on computing clusters.
"""

import os
import sys
import argparse
import logging
import pickle
import ee
import geemap
import requests
import pandas as pd
import numpy as np
import geopandas as gpd
import keras
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster
import matplotlib.pyplot as plt
import traceback
import subprocess
import concurrent.futures
import time
from io import BytesIO
from shapely.geometry import mapping
from keras.applications.convnext import ConvNeXtTiny, LayerScale
from data_loader import TFDatasetBuilder, read_tif, replace_invalid_and_crop
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# =============================================================
# Logging configuration
# =============================================================

# Set up logging with custom filter
class LogFilter(logging.Filter):
    """Custom filter to exclude unwanted log messages."""
    def filter(self, record):
        # Filter out messages we don't want to see
        unwanted_messages = [
            "Created 1 records",
            "Created 2 records", 
            "Created 3 records",
            "Created 4 records",
            "Created 5 records",
            "Created 6 records",
            "Created 7 records",
            "Created 8 records",
            "Created 9 records"
        ]
        
        # Check if any unwanted message pattern is in the log message
        for unwanted in unwanted_messages:
            if unwanted in str(record.getMessage()):
                return False
        
        # Also filter out messages that match the pattern "Created X records"
        import re
        if re.search(r'Created \d+ records?', str(record.getMessage())):
            return False
            
        return True

# Create custom filter
log_filter = LogFilter()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_application.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Apply filter to all handlers
for handler in logging.getLogger().handlers:
    handler.addFilter(log_filter)

# Also apply to third-party loggers that might be noisy
for logger_name in ['geemap', 'ee', 'googleapiclient', 'urllib3']:
    third_party_logger = logging.getLogger(logger_name)
    third_party_logger.addFilter(log_filter)
    third_party_logger.setLevel(logging.WARNING)  # Reduce verbosity

logger = logging.getLogger(__name__)


# =============================================================
# Setup
# =============================================================

class Args:
    def __init__(self, do_clipping=True, lower_clip=0, upper_clip=40, scale=10, 
                 scaling='standard', per_image_scaling=True):
        self.do_clipping = do_clipping
        self.lower_clip = lower_clip
        self.upper_clip = upper_clip
        self.scale = scale
        self.scaling = scaling
        self.per_image_scaling = per_image_scaling
    
    @classmethod
    def from_training_args(cls, training_args, scale=10):
        """
        Create Args object from training arguments dictionary.
        
        Args:
            training_args: Dictionary of training arguments
            scale: Scale parameter for downloads (default: 10)
            
        Returns:
            Args object with parameters from training
        """
        if training_args is None:
            # Return default args
            return cls()
        
        return cls(
            do_clipping=training_args.get('do_clipping', True),
            lower_clip=training_args.get('lower_clip', 0),
            upper_clip=training_args.get('upper_clip', 40),
            scale=scale,
            scaling=training_args.get('scaling', 'standard'),
            per_image_scaling=training_args.get('per_image_scaling', True)
        )


def initialize_earth_engine():
    """Initialize Earth Engine with proper backend for cluster environment."""
    try:
        # Fallback to standard initialization
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        logger.info("Earth Engine initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Earth Engine: {e}")
        sys.exit(1)


# =============================================================
# Primary Functions
# =============================================================

def cnn_predictions_country(candidate_locs_folder, model_path, model, country_name, cityMask, args, drop_threshold=0.1, show_high_preds=False):
    """Main function to generate CNN predictions for a country."""
    logger.info(f"Starting predictions for {country_name}")
    
    # Get list of diffImgs for this country
    asset_list = ee.data.listAssets(f'{candidate_locs_folder}/S2/diffImgs/')
    diffImg_list = [asset['id'] for asset in asset_list['assets'] if 'norm' not in asset['id']]
    diffImg_all = ee.ImageCollection(f'{candidate_locs_folder}/diffImgAll').reduce(ee.Reducer.mean())
    diffImg_norm = ee.ImageCollection(f'{candidate_locs_folder}/diffImgAll_norm5k').reduce(ee.Reducer.mean())

    logger.info(f"Found {len(diffImg_list)} diffImg assets to process")

    # Load existing predictions CSV if it exists
    predictions_file = f'{model_path}/all_predictions_{country_name}.csv'
    try:
        master_gdf = pd.read_csv(predictions_file)
        master_gdf = gpd.GeoDataFrame(master_gdf, geometry=gpd.GeoSeries.from_wkt(master_gdf['geometry']))
        logger.info("Loaded existing predictions CSV")
    except Exception:
        logger.info("Could not load predictions CSV, will create new")
        master_gdf = None

    # For each diffImg, make predictions and export to GEE
    for i, diffImg_id in enumerate(diffImg_list):
        diffImg_name = diffImg_id.split('/')[-1]
        logger.info(f"Processing {i+1}/{len(diffImg_list)}: {diffImg_name}")

        try:
            # If predictions do not exist already, make them
            if (master_gdf is None or diffImg_name not in master_gdf['diffImg_name'].values):
                
                diffImg = ee.Image(diffImg_id)
                predictions_ee, predictions_gdf = predictions_from_diffImg(
                    diffImg, diffImg_all, diffImg_norm, model, cityMask, args, show_high_preds
                )
                predictions_gdf['diffImg_name'] = diffImg_name
                
                # Drop predictions below the threshold
                if 'prediction' in predictions_gdf.columns:
                    initial_count = len(predictions_gdf)
                    predictions_gdf = predictions_gdf[predictions_gdf['prediction'] >= drop_threshold]
                    logger.info(f"Kept {len(predictions_gdf)}/{initial_count} predictions above threshold {drop_threshold}")

                # Combine into master_gdf
                if master_gdf is None:
                    master_gdf = predictions_gdf
                else:
                    master_gdf = pd.concat([master_gdf, predictions_gdf], ignore_index=True)

                # Save predictions to CSV
                master_gdf.to_csv(predictions_file, index=False)
                logger.info(f"Saved predictions to {predictions_file}")
            else:
                logger.info(f"Predictions already exist for {diffImg_name}, skipping")
        
        except Exception as e:
            logger.error(f'Error with {diffImg_name}: {e}')
            traceback.print_exc()

    return master_gdf


def predictions_from_diffImg(diffImg, diffImg_all, diffImg_norm, model, cityMask, args, show_high_preds=False):
    """Generate predictions from a single diffImg."""
    tile_size = 128 
    pixel_size = 10
    tile_length = pixel_size * tile_size
    image_geo = diffImg.geometry().difference(cityMask.geometry(), 25)

    # Create an offset grid covering the diffImg
    grid = image_geo.coveringGrid(proj=ee.Projection('EPSG:4326').atScale(tile_length))
    offset_grid = image_geo.buffer(tile_size*pixel_size*0.5) \
        .coveringGrid(proj=ee.Projection('EPSG:4326') \
        .translate(tile_length, tile_length) \
        .atScale(tile_length))
    total_grid = grid.merge(offset_grid)

    # Filter grid cells by pixel values in diffImg
    high_pixels_grid = diffImg_norm.select('max_all_norm_mean').reduceRegions(
        collection=total_grid,
        reducer=ee.Reducer.max(),
        scale=10,
    ).filter(ee.Filter.gt('max', 3))

    # Convert high_pixels_grid to GeoJSON
    high_pixels_grid_geojson = high_pixels_grid.getInfo()
    logger.info(f'Number of high pixel grid cells: {len(high_pixels_grid_geojson["features"])}')

    # Convert GeoJSON to GeoPandas DataFrame
    gdf = gpd.GeoDataFrame.from_features(high_pixels_grid_geojson['features'])

    predictions_gdf = predict_gdf_batch_threaded(gdf, diffImg_all, model, args, show_high_preds=show_high_preds)

    predictions_ee = gdf_to_ee_feature_collection(predictions_gdf)
    
    return predictions_ee, predictions_gdf


def get_image_data(image, geometry, scale=10):
    """Download image data from Earth Engine."""
    # Clip the image to the specified geometry
    clipped_image = image.clip(geometry).unmask(ee.Image.constant(0))

    original_names = ['weekday_0_mean', 'weekday_1_mean', 'weekday_2_mean',
                      'weekday_3_mean', 'weekday_4_mean', 'weekday_5_mean', 'weekday_6_mean']
    new_names = ['weekday_0', 'weekday_1', 'weekday_2', 
                 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6']
    
    # Get the image data as a numpy array
    url = clipped_image.select(original_names, new_names).getDownloadURL({
        'scale': scale, 
        'format': 'NPY',
        'bands': [
            {'id':'weekday_0', 'scale':scale},
            {'id':'weekday_1', 'scale':scale},
            {'id':'weekday_2', 'scale':scale},
            {'id':'weekday_3', 'scale':scale},
            {'id':'weekday_4', 'scale':scale},
            {'id':'weekday_5', 'scale':scale},
            {'id':'weekday_6', 'scale':scale}
            ]
        })

    response = requests.get(url)
    data = np.load(BytesIO(response.content))

    # Reshape the data to have the shape (7, 128, 128)
    new_data = data[['weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6']]
    new_data = new_data.view((float, len(new_data.dtype.names))).reshape(data.shape[0], data.shape[1], 7)
    
    # Check if the data shape is at least (_, 128, 128)
    if new_data.shape[0] < 128 or new_data.shape[1] < 128:
        # Calculate the padding sizes and pad with 0s
        pad_width = ((0, 0), (0, max(0, 128 - new_data.shape[0])), (0, max(0, 128 - new_data.shape[1])))
        new_data = np.pad(new_data, pad_width, mode='constant', constant_values=0)

    new_data = new_data[:128, :128, :]

    if new_data.shape[0] != 128 or new_data.shape[1] != 128 or new_data.shape[2] != 7:
        return np.zeros((128, 128, 7), dtype=new_data.dtype)

    return new_data


def robust_get_image_data(args):
    """Robust version of get_image_data with retry logic."""
    image, geometry, scale, max_attempts = args
    for attempt in range(max_attempts):
        try:
            return get_image_data(image, geometry, scale=scale)
        except Exception as e:
            wait = 2 ** attempt  # exponential backoff
            logger.warning(f"Error downloading image: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    logger.error("Failed to download image after several attempts")
    return None


def display_diffImg(arr, title_prefix="Diff Image", save_path=None):
    """Display all 7 days (channels) of a (H, W, 7) or (1, H, W, 7) array as subplots."""
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    fig, axes = plt.subplots(1, 7, figsize=(18, 3))
    for i in range(7):
        axes[i].imshow(arr[:, :, i], cmap='viridis', vmin=0, vmax=0.5)
        axes[i].set_title(f"{title_prefix} - Day {i+1}")
        axes[i].axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")
    plt.close()  # Close figure to free memory


def predict_gdf_batch_threaded(gdf, image, model, args, max_workers=3, show_high_preds=False):
    """Make predictions on a GeoDataFrame using multithreading."""
    processed_data_list = []
    scale = getattr(args, 'scale', 10)
    max_attempts = 4

    def gdf_row_to_args(i):
        ee_fc = geemap.geopandas_to_ee(gdf.iloc[[i]].set_crs("EPSG:4326"))
        geometry = ee_fc.geometry()
        return (image, geometry, scale, max_attempts)

    # Create arguments for each row
    all_args = [gdf_row_to_args(i) for i in range(len(gdf))]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(robust_get_image_data, all_args),
            total=len(gdf),
            desc="Downloading image patches"
        ))

    # Preprocessing retrieved data
    for i, data in enumerate(results):
        if data is not None:
            # Use per-image scaling preprocessing
            processed_data = preprocess_new_data_per_image(data, args)
            processed_data_list.append(processed_data)
        else:
            processed_data_list.append(np.zeros((128,128,7)))

    if processed_data_list:
        batch_data = np.stack(processed_data_list)
        predictions = model.predict(batch_data)
        gdf['prediction'] = predictions

        # Display high predictions if requested
        if show_high_preds:
            for idx, pred in enumerate(predictions):
                if pred > 0.8:
                    save_path = f"high_pred_{idx}_{pred[0]:.3f}.png"
                    display_diffImg(batch_data[idx], save_path=save_path)

    return gdf


def gdf_to_ee_feature_collection(gdf):
    """Convert GeoDataFrame to Earth Engine FeatureCollection."""
    features = []
    for _, row in gdf.iterrows():
        geom = ee.Geometry(mapping(row['geometry']))
        feature = ee.Feature(geom, row.drop('geometry').to_dict())
        features.append(feature)
    return ee.FeatureCollection(features)


def get_candidate_loc_folders(country_names):
    """Get candidate location folders from GCP projects."""
    logger.info("Retrieving candidate location folders from GCP")
    
    # Get the list of candidate loc projects from GCP
    command = '''gcloud projects list --filter="projectId ~ '.*-candidate-locs'"'''
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        logger.error(f"Failed to list GCP projects: {result.stderr}")
        return {}
    
    output = result.stdout.strip().split("\n")

    # Split each line into columns and create a list of dictionaries
    data = []
    for line in output[1:]:  # Skip the header row
        cols = line.split()
        if len(cols) >= 3:
            data.append({
                "PROJECT_ID": cols[0],
                "NAME": cols[1],
                "PROJECT_NUMBER": cols[2]
            })

    # Convert the data into a Pandas DataFrame
    candidate_loc_projects = pd.DataFrame(data)

    # Get a complete list of candidate loc folders
    all_folders = []
    for project_id in candidate_loc_projects['PROJECT_ID']:
        all_folders.append(f'projects/{project_id}/assets')
        try:
            assetList = ee.data.listAssets(f'projects/{project_id}/assets')
            folders = [a['id'] for a in assetList['assets'] if a['type'] == 'FOLDER']
            candidate_loc_folders = [f for f in folders if '-candidate-locs' in f.split('/')[-1]]
            all_folders = all_folders + candidate_loc_folders
        except Exception as e:
            logger.warning(f"Could not list assets for project {project_id}: {e}")

    # Create a lookup for candidate locs based on country names
    candidate_loc_lookup = {}
    for country in country_names:
        country_string = country.lower().replace(' ','')
        folder_string = country_string + '-candidate-locs'
        matching_folder = [f for f in all_folders if folder_string in f.split('/')[-2:]]
        
        # Special cases
        if country == 'Nigeria':
            matching_folder = ['projects/nigeria-candidate-locs/assets']
        elif country == 'Senegal':
            matching_folder = ['projects/sudan-candidate-locs/assets/senegal-candidate-locs']
        elif country == 'GuineaBissau':
            matching_folder = ['projects/ethiopia-candidate-locs/assets/guinea-bissau-candidate-locs']
        
        if matching_folder:
            candidate_loc_lookup[country] = matching_folder[0]
            logger.info(f"Found candidate loc folder for {country}: {matching_folder[0]}")
        else:
            logger.warning(f'No candidate loc folder found for {country}')

    return candidate_loc_lookup


def preprocess_new_data_per_image(arr, args):
    """
    Preprocess a new ndarray for prediction with per-image scaling.
    This matches the preprocessing used in ConvNeXt_transfer.py.
    
    Args:
        arr: Input ndarray of shape (H, W, 7).
        args: Args object containing scaling parameters.
    
    Returns:
        Preprocessed ndarray of shape (H, W, 7).
    """
    # Convert to shape (7, H, W) for consistency with data_loader
    arr = np.transpose(arr, (2, 0, 1))
    
    # Replace invalid values and crop
    arr = replace_invalid_and_crop(arr, 128)

    # Clip values
    if args.do_clipping:
        arr = np.clip(arr, args.lower_clip, args.upper_clip)

    # Apply per-image scaling
    if args.per_image_scaling and args.scaling == 'standard':
        # Flatten the array for scaling
        flat = arr.flatten().reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(flat)
        flat = scaler.transform(flat)
        arr = flat.reshape(arr.shape)
    elif args.per_image_scaling and args.scaling == 'robust':
        from sklearn.preprocessing import RobustScaler
        flat = arr.flatten().reshape(-1, 1)
        scaler = RobustScaler()
        scaler.fit(flat)
        flat = scaler.transform(flat)
        arr = flat.reshape(arr.shape)

    # Convert back to channels-last format (H, W, 7)
    arr = np.transpose(arr, (1, 2, 0))
    
    return arr.astype(np.float32)


def load_training_args(model_path):
    """
    Load training arguments from the model directory.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Dictionary of training arguments, or None if not found
    """
    training_args_file = os.path.join(model_path, 'training_args.pkl')
    
    if os.path.exists(training_args_file):
        try:
            with open(training_args_file, 'rb') as f:
                training_args = pickle.load(f)
            logger.info(f"Loaded training arguments from {training_args_file}")
            logger.info(f"Training arguments: {training_args}")
            return training_args
        except Exception as e:
            logger.warning(f"Could not load training arguments from {training_args_file}: {e}")
            return None
    else:
        logger.info(f"Training arguments file not found at {training_args_file}")
        return None


# =============================================================
# Tests and Main
# =============================================================

def test_scaling_consistency(model_path=None):
    """
    Test function to ensure that our preprocessing matches ConvNeXt_transfer.py exactly.
    Uses images from data/training_data_S2 to validate scaling.
    
    Args:
        model_path: Optional path to model directory to load training arguments from
    """
    logger.info("Running scaling consistency test...")
    
    # Load training arguments if model_path is provided
    if model_path:
        training_args = load_training_args(model_path)
        test_args = Args.from_training_args(training_args)
        logger.info(f"Using training arguments from {model_path}")
    else:
        # Create test args matching default ConvNeXt_transfer.py settings
        test_args = Args(
            do_clipping=True,
            lower_clip=0,
            upper_clip=40,
            scaling='standard',
            per_image_scaling=True
        )
        logger.info("Using default test arguments")
    
    # Load a sample image from training data
    try:
        # Read CSV to get image paths
        csv_path = "./data/training_data_S2/image_metadata.csv"
        df = pd.read_csv(csv_path)
        
        # Get first few image paths for testing
        test_paths = df['image_file_path'].head(3).tolist()
        
        logger.info(f"Testing with {len(test_paths)} images")
        
        # Test each image
        for i, path in enumerate(test_paths):
            logger.info(f"Testing image {i+1}: {path}")
            
            # Method 1: Using direct preprocessing (how ConvNeXt_transfer.py does it)
            # This matches the _read_and_transform method in TFDatasetBuilder
            
            # Get a single image using the builder's method
            arr1 = read_tif(path)
            arr1 = replace_invalid_and_crop(arr1, 128)
            if test_args.do_clipping:
                arr1 = np.clip(arr1, test_args.lower_clip, test_args.upper_clip)
            
            # Apply per-image scaling as in data_loader
            flat1 = arr1.flatten().reshape(-1, 1)
            scaler1 = StandardScaler()
            scaler1.fit(flat1)
            flat1 = scaler1.transform(flat1)
            arr1_scaled = flat1.reshape(arr1.shape)
            
            # Convert to channels-last for comparison
            arr1_final = np.transpose(arr1_scaled, (1, 2, 0))
            
            # Method 2: Using our new preprocessing function
            arr2 = read_tif(path)
            arr2 = np.transpose(arr2, (1, 2, 0))  # Convert to (H, W, 7)
            arr2_final = preprocess_new_data_per_image(arr2, test_args)
            
            # Compare results
            mean_diff = np.mean(np.abs(arr1_final - arr2_final))
            max_diff = np.max(np.abs(arr1_final - arr2_final))
            
            logger.info(f"  Image {i+1} differences - Mean: {mean_diff:.8f}, Max: {max_diff:.8f}")
            
            # Check statistics
            arr1_mean = np.mean(arr1_final)
            arr1_std = np.std(arr1_final)
            arr2_mean = np.mean(arr2_final)
            arr2_std = np.std(arr2_final)
            
            logger.info(f"  Method 1 - Mean: {arr1_mean:.6f}, Std: {arr1_std:.6f}")
            logger.info(f"  Method 2 - Mean: {arr2_mean:.6f}, Std: {arr2_std:.6f}")
            
            # Verify scaling worked (should be close to 0 mean, 1 std)
            if abs(arr2_mean) > 1e-6:
                logger.warning(f"  Mean is not close to 0: {arr2_mean}")
            if abs(arr2_std - 1.0) > 1e-6:
                logger.warning(f"  Std is not close to 1: {arr2_std}")
            
            # Assert they are very close
            assert mean_diff < 1e-6, f"Methods differ too much: {mean_diff}"
            assert max_diff < 1e-5, f"Methods differ too much: {max_diff}"
            
        logger.info("✓ Scaling consistency test passed!")
        
    except Exception as e:
        logger.error(f"Scaling consistency test failed: {e}")
        raise


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run CNN predictions on candidate locations')
    parser.add_argument('--country', type=str, default='Nigeria', help='Country name to process')
    parser.add_argument('--model-path', type=str, default='models/ConvNeXt_transfer/20250617_2101', 
                       help='Path to model directory')
    parser.add_argument('--drop-threshold', type=float, default=0.1, 
                       help='Drop predictions below this threshold')
    parser.add_argument('--show-high-preds', action='store_true', default=False, 
                       help='Save images of high predictions')
    parser.add_argument('--max-workers', type=int, default=3, 
                       help='Number of worker threads for downloads')
    parser.add_argument('--do-clipping', action='store_true', default=True,
                       help='Apply clipping to input data')
    parser.add_argument('--lower-clip', type=float, default=0,
                       help='Lower clipping bound')
    parser.add_argument('--upper-clip', type=float, default=40,
                       help='Upper clipping bound')
    parser.add_argument('--test-scaling', action='store_true',
                       help='Run scaling consistency test and exit')
    parser.add_argument('--ignore-training-args', action='store_true',
                       help='Ignore training arguments from model directory and use command line args')
    
    args_parsed = parser.parse_args()
    
    # Run test if requested
    if args_parsed.test_scaling:
        try:
            test_scaling_consistency(args_parsed.model_path)
            logger.info("Test completed successfully")
            return
        except Exception as e:
            logger.error(f"Test failed: {e}")
            sys.exit(1)
    
    logger.info(f"Starting model application for {args_parsed.country}")
    logger.info(f"Model path: {args_parsed.model_path}")
    
    # Initialize Earth Engine
    initialize_earth_engine()
    
    # Load model
    try:
        model = keras.models.load_model(
            f'{args_parsed.model_path}/checkpoints/best_model.keras',
            custom_objects={"ConvNeXtTiny": ConvNeXtTiny, "LayerScale": LayerScale}
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Load training arguments from model directory
    training_args = None
    if not args_parsed.ignore_training_args:
        training_args = load_training_args(args_parsed.model_path)
    else:
        logger.info("Ignoring training arguments from model directory (--ignore-training-args)")
    
    # Create args object from training arguments or defaults
    if training_args is not None:
        args = Args.from_training_args(training_args)
        logger.info("Using training arguments from model directory")
    else:
        # Fallback to command line arguments
        args = Args(
            do_clipping=args_parsed.do_clipping,
            lower_clip=args_parsed.lower_clip,
            upper_clip=args_parsed.upper_clip,
        )
        logger.info("Using default/command line arguments")
    
    logger.info(f"Using arguments: scaling={args.scaling}, per_image_scaling={args.per_image_scaling}, "
               f"do_clipping={args.do_clipping}, lower_clip={args.lower_clip}, upper_clip={args.upper_clip}")
    
    # Get candidate location folders
    candidate_loc_lookup = get_candidate_loc_folders([args_parsed.country])
    
    if args_parsed.country not in candidate_loc_lookup:
        logger.error(f"No candidate location folder found for {args_parsed.country}")
        sys.exit(1)
    
    candidate_locs_folder = candidate_loc_lookup[args_parsed.country]
    logger.info(f"Using candidate location folder: {candidate_locs_folder}")
    
    # Load city mask
    try:
        cityMask = ee.FeatureCollection(f'{candidate_locs_folder}/cityMask')
        logger.info("City mask loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load city mask: {e}")
        sys.exit(1)
    
    # Run predictions
    try:
        gdf_predictions = cnn_predictions_country(
            candidate_locs_folder, 
            args_parsed.model_path, 
            model, 
            args_parsed.country, 
            cityMask, 
            args,
            drop_threshold=args_parsed.drop_threshold,
            show_high_preds=args_parsed.show_high_preds
        )
        logger.info(f"Completed predictions for {args_parsed.country}")
        logger.info(f"Total predictions: {len(gdf_predictions) if gdf_predictions is not None else 0}")
    except Exception as e:
        logger.error(f"Failed to generate predictions: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
