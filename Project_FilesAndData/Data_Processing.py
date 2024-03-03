import configparser
import glob
import logging
import os
from typing import Callable, List, Tuple
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import netCDF4 as nc
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# In order to get any of this to work, ensure proper requirements are installed.
# To get data folders//hurricane folders navigate to "https://www.ncei.noaa.gov/products/hurricane-satellite-data".
# Wrote using the 'HURSAT Tatiana' netCDF4 file from the 2016 hurricanes folder.
# 
# 

# Setup configuration and pull pathing
script_dir = os.path.dirname(__file__)
config_path = os.path.join(script_dir, r'C:\Users\payto\Desktop\Computer_Science\CS_1070\config.ini')
config = configparser.ConfigParser()
config.read(config_path)

folder_path = config["Paths"]["folder_path"]
output_folder = config['Paths']['output_folder']
base_folder = config['Paths']['base_folder']

# Setup logging for errors
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# SQL Functions that aren't used currently
def create_db_connection(db_string):
    """
    Create a database connection using SQLAlchemy.

    Parameters:
    db_string (str): Database connection string.

    Returns:
    engine: SQLAlchemy engine object.
    """
    engine = create_engine(db_string)
    return engine
def save_data_to_db(df, table_name, engine):
    """
    Save a pandas DataFrame to a PostgreSQL table.

    Parameters:
    df (pd.DataFrame): DataFrame to be saved.
    table_name (str): Name of the table where data will be saved.
    engine: SQLAlchemy engine object.
    """
    df.to_sql(table_name, engine, if_exists='append', index=False)


def assign_ids_to_netcdf_files(folder_path):
    """
    Assigns a unique ID to each NetCDF file in the specified folder.

    Parameters:
    folder_path (str): Path to the folder containing NetCDF files.

    Returns:
    dict: A dictionary mapping file paths to unique IDs.
    """
    file_ids = {}
    for idx, file in enumerate(glob.glob(os.path.join(folder_path, '*.nc'))):
        file_ids[file] = idx
    return file_ids

def parse_filename(file_path: str) -> dict:
    """
    Parse the filename to extract the observation date, time, and hurricane name.

    Parameters:
    file_path (str): Path to the NetCDF file.

    Returns:
    dict: Dictionary containing the observation date, time, and hurricane name.
    """
    filename = os.path.basename(file_path)
    parts = filename.split('.')

    if len(parts) < 12:
        logging.warning(f"Filename format is not as expected: {filename}")
        return {}

    # Extracting required data from the filename
    storm_name = parts[1]
    year_observed = parts[2]
    month_observed = parts[3].zfill(2)
    day_observed = parts[4].zfill(2)
    time_observed = f"{parts[5][:2]}:{parts[5][2:]}"

    return {
        'ObservationDate': f"{year_observed}-{month_observed}-{day_observed}",
        'ObservationTime': time_observed,
        'HurricaneName': storm_name
    }

def process_netcdf(file_path: str, file_id: int) -> Tuple[pd.DataFrame, dict]:
    """
    Processes a single NetCDF file to extract spatial data and interpolated data.

    Parameters:
    file_path (str): Path to the NetCDF file.
    file_id (int): Identifier for the NetCDF file.

    Returns:
    Tuple[pd.DataFrame, dict]: 
        - A DataFrame containing spatial data (latitude, longitude).
        - A dictionary with interpolated data.
    """
    try:
        with Dataset(file_path, 'r') as dataset:

            # Check for necessary variables in the dataset
            required_vars = ['lat', 'lon', 'NomDate', 'NomTime', 'WindSpd', 'CentPrs', 'bt_eye', 'bt_eyewall', 'eye_prob',
                              'eye_comp', 'rad_eye', 'archer_lat', 'archer_lon', 'archer_combo_score']
            missing_vars = [var for var in required_vars if var not in dataset.variables]
            if missing_vars:
                logging.warning(f"Missing variables {missing_vars} in file {file_path}")
                return pd.DataFrame(), {}

            # Extract Spatial and Temporal Data
            lat = dataset.variables['lat'][:]
            lon = dataset.variables['lon'][:]
            if len(lat) != len(lon):
                logging.warning(f"Length of latitude and longitude arrays do not match in file {file_path}")
                return pd.DataFrame(), {}

            file_info = parse_filename(file_path)
            observation_date = file_info['ObservationDate']
            observation_time = file_info['ObservationTime']
            hurricane_name = file_info['HurricaneName']
            spatial_temporal_data = {
                'HurricaneName': [hurricane_name] * len(lat),
                'FileID': [file_id] * len(lat),
                'Latitude': lat,
                'Longitude': lon,
                'ObservationDate': [observation_date] * len(lat),
                'ObservationTime': [observation_time] * len(lat),
            }
            spatial_df = pd.DataFrame(spatial_temporal_data)

            # Extract Interpolated Data
            interpolated_data = {
                'HurricaneName': hurricane_name,
                'FileID': file_id,
                'observation_date':[observation_date],
                'observation_time': [observation_time],
                'WindSpeed': dataset.variables['WindSpd'][:][0],
                'CentralPressure': dataset.variables['CentPrs'][:][0],
                'bt_eye': dataset.variables['bt_eye'][:][0],
                'bt_eyewall': dataset.variables['bt_eyewall'][:][0],
                'EyeProbability': dataset.variables['eye_prob'][:][0],
                'EyeCompleteness': dataset.variables['eye_comp'][:][0],
                'EyeRadius': dataset.variables['rad_eye'][:][0],
                'ArcherLat': dataset.variables['archer_lat'][:][0],
                'ArcherLon': dataset.variables['archer_lon'][:][0],
                'ArcherComboScore': dataset.variables['archer_combo_score'][:][0]
            }

            return spatial_df, interpolated_data

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return pd.DataFrame(), {}
    except OSError as e:
        logging.error(f"Error opening file {file_path}: {e}")
        return pd.DataFrame(), {}

def extract_and_save_images(file_path: str, output_folder: str, file_id):
    """
    Extracts and saves image data from a NetCDF file using the unique file ID.

    Parameters:
    file_path (str): Path to the NetCDF file.
    output_folder (str): Directory to save the images.
    hurricane_name (str): Name of the hurricane.
    file_id (int): Unique identifier for the NetCDF file.
    """
    try:
        with Dataset(file_path, 'r') as dataset:
            file_info = parse_filename(file_path)
            hurricane_name = file_info['HurricaneName']
            unique_identifier = os.path.basename(file_path).split('.')[0]
            image_vars = ['IRWIN', 'VSCHN', 'IRWVP', 'IRSPL', 'IRNIR']
            for var in image_vars:
                if var in dataset.variables:
                    image_data = dataset.variables[var][0, :, :]
                    output_file_path = os.path.join(output_folder, f'{hurricane_name}_{unique_identifier}_{var}_{file_id}.png')
                    plt.imshow(image_data, cmap='gray')
                    plt.axis('off')
                    plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    logging.info(f"Saved image to {output_file_path}")
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except OSError as e:
        logging.error(f"Error opening file {file_path}: {e}")

def process_single_hurricane_folder(folder_path: str, process_netcdf: Callable[[str, int], Tuple[pd.DataFrame, dict]]) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Process all NetCDF files in a single hurricane folder, returning both spatial data and interpolated data.

    Parameters:
    folder_path (str): Path to the folder containing NetCDF files.
    process_netcdf_function (Callable): Function to process individual NetCDF files.

    Returns:
    Tuple[pd.DataFrame, List[dict]]: 
        - A DataFrame of combined spatial data.
        - A list of dictionaries for the interpolated data.
    """
    all_spatial_data = []
    all_interpolated_data = []

    file_ids = assign_ids_to_netcdf_files(folder_path)

    for file_path, file_id in file_ids.items():
        spatial_df, interpolated_data = process_netcdf(file_path, file_id)
        if not spatial_df.empty:
            all_spatial_data.append(spatial_df)
        if interpolated_data:
            all_interpolated_data.append(interpolated_data)

    combined_spatial_df = pd.concat(all_spatial_data, ignore_index=True) if all_spatial_data else pd.DataFrame()
    return combined_spatial_df, all_interpolated_data

def process_images_in_single_folder(folder_path: str, output_folder: str):
    """
    Process all NetCDF files in a single hurricane folder for image extraction and saving.

    Parameters:
    folder_path (str): Path to the folder containing NetCDF files.
    output_folder (str): Directory to save the images.
    hurricane_name (str): Name of the hurricane.
    """
    file_ids = assign_ids_to_netcdf_files(folder_path)

    for file_path, file_id in file_ids.items():
        extract_and_save_images(file_path, output_folder, file_id)

def process_multiple_hurricane_folders(base_folder: str) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Process multiple hurricane folders containing NetCDF files, returning both spatial data and interpolated data.

    Parameters:
    base_folder (str): Path to the base folder containing hurricane folders.

    Returns:
    Tuple[pd.DataFrame, List[dict]]: 
        - A DataFrame of combined spatial data from all hurricanes.
        - A list of dictionaries for the interpolated data.
    """
    all_spatial_data = []
    all_interpolated_data = []

    for hurricane_folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, hurricane_folder)
        spatial_df, interpolated_data = process_single_hurricane_folder(folder_path, process_netcdf)

        # Append the hurricane name to each interpolated data dictionary
        for data in interpolated_data:
            data['Hurricane'] = hurricane_folder

        if not spatial_df.empty:
            all_spatial_data.append(spatial_df)
        all_interpolated_data.extend(interpolated_data)  # Use extend to flatten the list

    combined_spatial_df = pd.concat(all_spatial_data, ignore_index=True) if all_spatial_data else pd.DataFrame()
    return combined_spatial_df, all_interpolated_data

def process_images_in_multiple_folders(base_folder: str, output_folder: str):
    """
    Process multiple hurricane folders containing NetCDF files for image extraction and saving.

    Parameters:
    base_folder (str): Path to the base folder containing hurricane folders.
    output_base_folder (str): Base directory to save the images for each hurricane.
    """
    for hurricane_folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, hurricane_folder)
        if os.path.isdir(folder_path):
            hurricane_output_folder = os.path.join(output_folder, hurricane_folder)
            if not os.path.exists(hurricane_output_folder):
                os.makedirs(hurricane_output_folder)
            process_images_in_single_folder(folder_path, hurricane_output_folder)

# Setup_IDs
file_ids = assign_ids_to_netcdf_files(folder_path)

# Processing the data
# For numerical data
# hurricane_data = process_single_hurricane_folder(folder_path, process_netcdf)
# interpolated_df = pd.DataFrame(hurricane_data[1])

spatial_data, interpolated_data = process_multiple_hurricane_folders(base_folder)

spatial_data.to_csv('combined_spatial_data.csv', index=False)

interpolated_data = pd.DataFrame(interpolated_data)
interpolated_data.to_csv('combined_interpolated_data.csv', index=False)

# For image data
# process_images_in_single_folder(folder_path, output_image_folder)

# process_images_in_multiple_folders(base_folder, output_folder)

# Save to CSV files
# hurricane_data[0].to_csv('spatial_data.csv', index=False)
# interpolated_df.to_csv('interpolated_data.csv', index=False)
