import json
import pandas as pd
import yaml
from geopy.geocoders import Nominatim
from geopy import Point
from geopy.distance import geodesic
import numpy as np
import csv
import ast  # Ensure ast is imported for literal_eval
import os

def get_charger_data():
    """
    Retrieves charger data from a JSON file and filters it for charging stations located in London.

    Returns:
        pd.DataFrame: A DataFrame containing the ID, latitude, and longitude of the chargers in London.
    """

    # From JSON file
    with open('environment/data/Ontario_Charger_Dataset.json') as file:
        data = json.load(file)
    charger_data = []
    for station in [item for item in data['fuel_stations'] if item['city'] == 'London']:
        charger_id = station['id']
        charger_lat = round(station['latitude'], 5)
        charger_long = round(station['longitude'], 5)
        charger_data.append([charger_id, charger_lat, charger_long])
    charger_info = pd.DataFrame(charger_data, columns=['id', 'latitude', 'longitude'])

    return charger_info

def get_charger_list(chargers, org_lat, org_long, dest_lat, dest_long, num_of_chargers):

    """
    Generates a list of charger locations around the origin, destination, and midway point.

    Parameters:
        chargers (list): List of chargers with their latitude and longitude.
        org_lat (float): Latitude of the origin.
        org_long (float): Longitude of the origin.
        dest_lat (float): Latitude of the destination.
        dest_long (float): Longitude of the destination.
        num_of_chargers (int): Number of chargers to select around each point.

    Returns:
        list: A list of unique chargers around the origin, destination, and midway point, padded to ensure
              the list contains num_of_chargers * 3 chargers.
    """

    list_of_chargers = [(i, lat, long) for i, (lat, long) in enumerate(chargers)]

    # Calculate the midway point between origin and destination
    midway_lat = (org_lat + dest_lat) / 2
    midway_long = (org_long + dest_long) / 2

    # Get list of chargers around origin, destination, and midway point
    org_chargers = get_closest_chargers(org_lat, org_long, num_of_chargers, list_of_chargers)
    dest_chargers = get_closest_chargers(dest_lat, dest_long, num_of_chargers, list_of_chargers)
    midway_chargers = get_closest_chargers(midway_lat, midway_long, num_of_chargers, list_of_chargers)

    # Combine list
    combined_list = org_chargers + dest_chargers + midway_chargers

    # Remove duplicates and pad list so that there's always num_of_chargers * 3
    unique_list = list(set(combined_list))
    padding_chargers = get_closest_chargers(midway_lat, midway_long, num_of_chargers, list_of_chargers, unique_list)

    # Combine and append lists
    return unique_list + padding_chargers


def load_config_file(fname=None):
    # Created by Santiago 26/04/2024
    """
    Helper method to load the configuration parameters from yaml file first the default yaml file is
    loaded and the upadted with the new most updated configuration yaml file given by fname.

    Parameters:
        fname (string) path to the yaml config file

    Returns:
        parameters (python dict)

    """

    with open(fname) as config:
        parameters = yaml.safe_load(config)

    return parameters

def get_closest_chargers(current_lat, current_long, return_num, charger_list, existing_list=None):

    """
    Returns a list of charging stations which are closest to the current coordinates

    Parameters:
        current_lat: Latitude to use as the origin
        current_long: Longitude to use as the origin
        return_num: Amount of charging stations to return
        charger_list: A list of tuples (id, lat, long)

    Returns:
        A filtered list of tuples (id, lat, long)
    """

    # Calculate the distances and travel times between the current coordinates and all charger locations
    distances_and_times = []
    origin = (current_lat, current_long)
    for charger in charger_list:
        destination = (charger[1], charger[2])
        distance, time = get_distance_and_time(origin, destination)
        distances_and_times.append((charger[0], distance, time))

    # Sort the distances in ascending order
    distances_and_times.sort(key=lambda x: x[1])

    # Get the closest charging stations up to the desired return_num
    closest_chargers = []

    if existing_list is None:
        for i in range(min(return_num, len(distances_and_times))):
            charger_id = distances_and_times[i][0]
            charger_lat = charger_list[charger_id][1]
            charger_long = charger_list[charger_id][2]
            closest_chargers.append((charger_id, charger_lat, charger_long))

    else: # Need more chargers because there was duplicates
        counter = 0
        i = return_num - 1
        while counter + len(existing_list) < return_num * 3:
            i += 1
            charger_id = distances_and_times[i][0]
            charger_lat = charger_list[charger_id][1]
            charger_long = charger_list[charger_id][2]
            if (charger_id, charger_lat, charger_long) in existing_list:
                continue
            else:
                closest_chargers.append((charger_id, charger_lat, charger_long))
                counter += 1

    return closest_chargers

def get_distance_and_time(origin, destination):

    """
    Gets the distance and travel time between two coordinates using geodesic distance and an assumed average speed.

    Parameters:
        origin (tuple): A tuple containing the latitude and longitude of the origin.
        destination (tuple): A tuple containing the latitude and longitude of the destination.

    Returns:
        tuple: A tuple containing the distance in kilometers and the travel time in seconds.
    """

    # Assume average driving speed of 60 km/h
    average_speed = 60

    # Calculate the distance between origin and destination using geodesic distance
    distance = geodesic(origin, destination).km

    round(distance, 3)

    # Calculate the travel time based on the distance and average speed
    # Time = distance / speed
    time = round((distance / average_speed) * 3600, 3)  # time in seconds

    return distance, time

def get_org_dest_coords(center, radius, org_angle):

    """
    Calculates the coordinates of origin and destination points on the circumference of a circle
    with a given center and radius.

    Parameters:
        center (tuple): A tuple containing the latitude and longitude of the center point.
        radius (float): The radius of the circle in kilometers.
        org_angle (float): The angle in radians for the origin point on the circle's circumference.

    Returns:
        np.ndarray: A 1D array containing the latitude and longitude of the origin and destination points,
                    rounded to 4 decimal places.
    """

    lat, long = center
    km_in_degrees = radius / 111.11  # Approximation of km to degrees

    # Radius for origin point equals to the maximum radius
    org_radius = km_in_degrees

    # Calculate coordinates of origin point on the circle's circumference
    org_lat = lat + org_radius * np.sin(org_angle)
    org_long = long + org_radius * np.cos(org_angle)

    # Angle in radians for destination point
    dest_angle = (org_angle + np.pi) % (2 * np.pi)  # add π radians (180 degrees) to get the opposite point

    # Radius for destination point equals to the maximum radius
    dest_radius = km_in_degrees

    # Calculate coordinates of destination point on the circle's circumference
    dest_lat = lat + dest_radius * np.sin(dest_angle)
    dest_long = long + dest_radius * np.cos(dest_angle)

    # Generating an array of origin coordinates and dest cordinates
    # Rounded to 4 decimals for grid discretization
    orig_dest_coord = np.round(np.array([org_lat, org_long, dest_lat, dest_long]), decimals=4)

    return orig_dest_coord.transpose()

def read_excel_data(file_path, sheet_name):

    """
    Retrieves data from an Excel file and returns it as a DataFrame.

    Parameters:
        file_path (str): The path to the Excel file.
        sheet_name (str): The name of the sheet to read from the Excel file.

    Returns:
        pandas.DataFrame: A DataFrame containing the data from the specified sheet.
    """

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

def save_to_csv(data, filename, append=False):

    mode = 'a' if append else 'w'
    file_exists = os.path.isfile(filename)
    if isinstance(data[0], dict):
        keys = data[0].keys()  # Extract the headers from the first dictionary
        with open(filename, mode, newline='') as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            if not append or not file_exists:
                writer.writeheader()  # Write header if not appending or file doesn't exist
            writer.writerows(data)  # Write data rows
    elif isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
        if not append or not file_exists:
            # First write, include header
            df.to_csv(filename, index=False)
        else:
            df.to_csv(filename, mode=mode, header=False, index=False)
        
    else:
        # If data is a list of lists or other structure
        with open(filename, mode, newline='') as file:
            writer = csv.writer(file)
            if not append or not file_exists:
                writer.writerow(data[0])  # Write header if not appending or file doesn't exist
            writer.writerows(data)  # Write rows for lists of lists

# def save_to_csv(data, filename, append=False):
#     mode = 'a' if append else 'w'
#     file_exists = os.path.isfile(filename)

#     def flush_file(file_obj):
#         file_obj.flush()
#         os.fsync(file_obj.fileno())

#     if isinstance(data[0], dict):
#         keys = data[0].keys()  # Extract the headers from the first dictionary
#         with open(filename, mode, newline='', encoding='utf-8') as file:
#             writer = csv.DictWriter(file, fieldnames=keys)
#             if not append or not file_exists:
#                 writer.writeheader()
#             writer.writerows(data)
#             flush_file(file)

#     elif isinstance(data, np.ndarray):
#         df = pd.DataFrame(data)
#         with open(filename, mode, newline='', encoding='utf-8') as file:
#             if not append or not file_exists:
#                 df.to_csv(file, index=False)
#             else:
#                 df.to_csv(file, header=False, index=False)
#             flush_file(file)
#     data = None


def read_csv_data(filename, columns=None):
    try:
        if columns is None:
            # Read the CSV file with the 'path' column as a string
            df = pd.read_csv(filename, converters={'path': str})
        else:
            df = pd.read_csv(filename, converters={'path': str}, usecols=columns)

        # Parse the 'path' column using ast.literal_eval
        if 'path' in df.columns:
            df['path'] = df['path'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        return df

    except FileNotFoundError:
        print(f"File {filename} not found.")
        return pd.DataFrame()  # Return an empty DataFrame
    except pd.errors.EmptyDataError:
        print(f"File {filename} is empty.")
        return pd.DataFrame()  # Return an empty DataFrame
    except Exception as e:
        print(f"An error occurred while reading {filename}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame

def save_to_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, cls=NumpyEncoder)

def load_from_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32)):  # Handle numpy integer types
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):  # Handle numpy float types
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def load_from_csv(filename):

    """
    Loads data from a CSV file and attempts to parse each item as a Python literal.

    Parameters:
        filename (str): The name of the file from which the data will be loaded.

    Returns:
        list: A list of tuples, where each tuple represents a row of data from the CSV file.
    """

    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            converted_row = []
            for item in row:
                try:
                    # Attempt to parse the item as a Python literal (e.g., list, int, float)
                    converted_item = ast.literal_eval(item)
                    converted_row.append(converted_item)
                except (ValueError, SyntaxError):
                    # Keep as string if conversion fails
                    converted_row.append(item)
            data.append(tuple(converted_row))
