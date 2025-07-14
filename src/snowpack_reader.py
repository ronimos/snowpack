# -*- coding: utf-8 -*-
"""
snowpack_reader.py
===================

A hardware-adaptive reader for SNOWPACK .pro files using xarray.

This module provides the `SnowpackProfile` class for parsing, processing, 
and analyzing snow profile data from SNOWPACK `.pro` files. It is designed 
to seamlessly leverage available hardware—CPU or GPU—for optimal performance.

Key Features
------------
- Parses station metadata and profile time series from SNOWPACK `.pro` files.
- Stores data as an `xarray.Dataset` backed by either NumPy (CPU) or CuPy (GPU).
- Slices data by date range using the `slice()` method.
- Calculates critical snowpack properties such as `rc_flat` using vectorized methods.
- Supports flexible profile summarization (min, max, weighted mean, custom functions).
- Enables analysis of specific snowpack sections (e.g., slabs above weak layers).

Hardware Acceleration
---------------------
- **GPU Support:** If an NVIDIA GPU and `cupy` are available, numerical computations 
  are offloaded to the GPU for faster processing. Install with:
  `pip install cupy-cuda12x` (CUDA Toolkit required).
- **CPU Fallback:** Falls back to NumPy if no GPU is detected.

Typical Workflow
----------------
1. Instantiate `SnowpackProfile` with a `.pro` file path.
2. Access parsed data via the `.data` attribute (an xarray Dataset).
3. Use `slice()` to select a date range, then chain analysis methods like
   `get_profile_summary()` or `find_layer_by_criteria()`.

Repository: https://github.com/ronimos/snowpack
Last Updated: July 8, 2025
Author: Ron Simenhois
"""


import logging
import re
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

# --- Hardware-Adaptive Array Library ---
# This block detects if a GPU is available and chooses the appropriate library.
# All subsequent code uses the 'xp' alias for array operations, making the
# script hardware-agnostic.

try:
    import cupy as xp
    _ = xp.arange(1)
    GPU_AVAILABLE = True
    print("✅ GPU detected. Using cupy for accelerated calculations.")
except (ImportError, RuntimeError):
    import numpy as xp
    GPU_AVAILABLE = False
    print("ℹ️ No GPU or cupy found. Falling back to CPU using numpy.")
# ---

import pandas as pd
import xarray as xr
from tqdm import tqdm
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

logger = logging.getLogger(__name__)

# --- Static Mappings ---

# Maps header keys from .pro files to human-readable dictionary keys.
# Source: https://snowpack.slf.ch/doc-release/html/snowpackio.html
HEADER_MAP = {
    'Altitude=': 'altitude',
    'Latitude=': 'latitude',
    'Longitude=': 'longitude',
    'SlopeAngle=': 'slopeAngle',
    'SlopeAzi=': 'slopeAzi',
    'StationName=': 'stationName'
}

# Static mapping of parameter codes to human-readable names.
# Source: https://snowpack.slf.ch/doc-release/html/snowpackio.html
PARAM_CODES = {
    "0500": "timestamp", 
    "0501": "height", 
    "0504": "element_ID",
    "0502": "density", 
    "0503": "temperature", 
    "0506": "lwc",
    "0508": "dendricity", 
    "0509": "sphericity", 
    "0510": "coord_number",
    "0511": "bond_size", 
    "0512": "grain_size", 
    "0513": "grain_type",
    # "0514": "sh_at_surface",
    "0515": "ice_content", 
    "0516": "air_content",
    "0517": "stress", 
    "0518": "viscosity", 
    "0520": "temperature_gradient",
    "0523": "viscous_deformation_rate", 
    # "0530": "stab_indices",
    "0531": "stab_deformation_rate", 
    "0532": "sn38", 
    "0533": "sk38",
    "0534": "hand_hardness", 
    "0535": "opt_equ_grain_size",
    "0601": "shear_strength", 
    "0602": "gs_difference", 
    "0603": "hardness_difference",
    "0604": "ssi", 
    "1501": "height_nodes", 
    "1532": "sn38_nodes",
    "1533": "sk38_nodes", 
    "0540": "date_of_birth",
    "0607": "accumulated_temperature_gradient", 
    "9998": "depth",
    "9999": "rc_flat" # Placeholder for calculated rc_flat
}

# --- End Static Mappings ---

class SnowpackProfile:
    """
    Reads, parses, and represents a SNOWPACK .pro file.

    This class handles the entire lifecycle of a .pro file, from reading and
    parsing to performing hardware-accelerated numerical computations.

    Attributes:
        filename (str): The path to the input .pro file.
        metadata (Dict): Station parameters parsed from the file header.
        data (Optional[xr.Dataset]): An xarray Dataset containing all profile
            data. The underlying arrays will be `cupy` arrays if a GPU is
            used, otherwise they will be `numpy` arrays.
    """

    def __init__(self, filename: str, _load_data: bool = True):
        """
        Initializes the reader and, by default, processes the specified file.

        Args:
            filename (str): The full path to the .pro file.
            _load_data (bool, optional): If False, initializes an empty object
                without reading the file. Used internally. Defaults to True.
        """
        self.filename: str = filename
        self.metadata: Dict = {}
        self.data: Optional[xr.Dataset] = None
        if _load_data:
            self._read_profile()

    def __len__(self) -> int:
        """Returns the number of profiles (timestamps) in the dataset."""
        if self.data is None:
            return 0
        return len(self.data.timestamp)

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the object."""
        device = "GPU" if GPU_AVAILABLE else "CPU"
        return f"<SnowpackProfile(filename='{self.filename}', profiles={len(self)}, device='{device}')>"

    def _read_profile(self):
        """Orchestrates the reading and parsing of the entire .pro file."""
        if not Path(self.filename).exists():
            logger.error(f"File not found: {self.filename}")
            return
        in_header, in_data = False, False
        temp_profiles: List[Dict] = []
        current_ts_data: Dict = {}

        with open(self.filename, 'r') as f:
            for line in f: # No tqdm here for production speed
                line = line.strip()
                if not line: continue
                if line == '[STATION_PARAMETERS]': 
                    in_header, in_data = True, False
                elif line == '[DATA]': 
                    in_header, in_data = False, True
                elif line.startswith('['): 
                    in_header, in_data = False, False
                elif in_header: 
                    self._parse_header_line(line)
                elif in_data:
                    is_new_ts, timestamp_key = self._is_new_timestamp_line(line)
                    if is_new_ts:
                        if current_ts_data: temp_profiles.append(current_ts_data)
                        current_ts_data = {'timestamp': timestamp_key}
                    else:
                        self._parse_data_line(line, current_ts_data)

        if current_ts_data: temp_profiles.append(current_ts_data)
        if not temp_profiles:
            logger.warning(f"No data was parsed from file: {self.filename}")
            return

        self._create_dataset_from_profiles(temp_profiles)
        if self.data is not None:
            self._compute_and_add_depth()
            self._compute_and_add_rc_flat_vectorized()

    def _create_dataset_from_profiles(self, profiles: List[Dict]):
        """Converts parsed data into an xarray.Dataset."""
        timestamps = pd.to_datetime([p['timestamp'] for p in profiles], format='%d.%m.%Y %H:%M:%S', errors='coerce')
        valid_indices = ~pd.isna(timestamps)
        profiles = [p for i, p in enumerate(profiles) if valid_indices[i]]
        timestamps = timestamps.dropna()
        if not profiles: return

        all_params = sorted(list(set(key for p in profiles for key in p if key != 'timestamp')))
        max_layers = max((len(p.get('height', [])) for p in profiles), default=0)
        data_vars = {param: (("timestamp", "layer_index"), np.full((len(profiles), max_layers), np.nan, dtype=np.float32)) for param in all_params}

        for i, profile in enumerate(profiles):
            num_layers = len(profile.get('height', []))
            for param, (dims, arr) in data_vars.items():
                if param in profile:
                    values = profile.get(param)
                    if values is not None:
                        arr[i, :num_layers] = np.array(values)[:num_layers]

        if GPU_AVAILABLE:
            for param, (dims, arr) in data_vars.items():
                data_vars[param] = (dims, xp.asarray(arr))
        
        self.data = xr.Dataset(data_vars, coords={'timestamp': timestamps, 'layer_index': np.arange(max_layers)})
        self.data = self.data.sortby('timestamp')

    def _parse_header_line(self, line: str):
        """Parses a single line from the [STATION_PARAMETERS] section."""
        for key, value in HEADER_MAP.items():
            if line.startswith(key):
                self.metadata[value] = line.split('=', 1)[1].strip()

    def _is_new_timestamp_line(self, line: str) -> Tuple[bool, Optional[str]]:
        """Checks if a data line marks the beginning of a new profile."""
        parts = line.split(',', 1)
        return (True, parts[1]) if parts[0] == "0500" else (False, None)

    def _parse_data_line(self, line: str, current_ts_data: Dict):
        """Parses a single data line containing layer data for a parameter."""
        parts = line.split(',')
        param_name = PARAM_CODES.get(parts[0])
        if param_name:
            current_ts_data[param_name] = np.array(parts[2:], dtype=float)

    def _compute_and_add_depth(self):
        """
        Calculates the depth of each layer from the snow surface and adds it
        to the dataset.
        """
        if 'height' not in self.data.data_vars:
            logger.warning("Cannot calculate depth without 'height' variable.")
            return
        
        height = self.data['height']
        # Get the max height for each individual timestamp
        total_height = height.max(dim='layer_index', skipna=True)
        # Broadcast the subtraction
        depth = total_height - height
        
        # Use hardware-specific `where` to avoid `astype` error with cupy
        if GPU_AVAILABLE:
            # Use cupy's native where on the raw arrays
            final_depth_data = xp.where(height.notnull().data, depth.data, xp.nan)
            self.data['depth'] = xr.DataArray(final_depth_data, dims=depth.dims, coords=depth.coords)
        else:
            # The xarray method is fine for numpy
            self.data['depth'] = depth.where(height.notnull())           
        


    def _compute_and_add_rc_flat_vectorized(self):
        """
        Calculates rc_flat for all profiles in a single vectorized operation.

        This method leverages the detected hardware (CPU or GPU) via the `xp`
        array alias to perform array-based arithmetic. This is the primary
        computationally intensive step that benefits from GPU acceleration.
        It corrects the data at the source by setting the physically
        meaningless surface `rc_flat` value to a large number.
        """
        required_vars = {'density', 'grain_size', 'shear_strength', 'height'}
        if not required_vars.issubset(self.data.data_vars):
            logger.warning(f"Skipping rc_flat calculation due to missing variables.")
            return

        # --- Physical Constants ---
        RHO_ICE, GS_0, G, A, B = 917.0, 0.00125, 9.81, 4.6e-9, -2.0

        # --- Data Preparation ---
        # These are xarray.DataArray objects wrapping either numpy or cupy arrays
        height, density, grain_size, shear_strength = (
            self.data['height'], self.data['density'],
            self.data['grain_size'], self.data['shear_strength']
        )

        # --- Calculations (performed on either CPU or GPU via `xp` alias) ---
        height_of_bottom = height.shift(layer_index=1, fill_value=0)
        thick = height - height_of_bottom
        layer_load = (density * thick * G)
        load = layer_load.reindex(layer_index=layer_load.layer_index[::-1]).cumsum(dim='layer_index').reindex(layer_index=layer_load.layer_index)

        total_thick_above = self.data['height'].max(dim='layer_index', skipna=True) - height
        
        rho_sl_raw = load / (total_thick_above * G)
        
        if GPU_AVAILABLE:
            rho_sl_numpy = rho_sl_raw.to_numpy()
            rho_sl_cpu_da = xr.DataArray(rho_sl_numpy, dims=rho_sl_raw.dims, coords=rho_sl_raw.coords)
            rho_sl_filled_cpu = rho_sl_cpu_da.bfill(dim='layer_index').ffill(dim='layer_index')
            rho_sl_gpu = xp.asarray(rho_sl_filled_cpu.values)
            rho_sl = xr.DataArray(rho_sl_gpu, dims=rho_sl_raw.dims, coords=rho_sl_raw.coords)
        else:
            rho_sl = rho_sl_raw.bfill(dim='layer_index').ffill(dim='layer_index')

        tau_p = shear_strength * 1000.0
        gs = grain_size * 0.001
        e_prime = 5.07e9 * (rho_sl / RHO_ICE)**5.13 / (1 - 0.2**2)
        dsl_over_sigman = 1.0 / (G * rho_sl)
        term1_under = A * (density / RHO_ICE * gs / GS_0)**B
        term2_under = 2 * tau_p * e_prime * dsl_over_sigman

        term1_data = term1_under.data.copy()
        term1_data[term1_data <= 0] = xp.nan
        term1_sqrt = xp.sqrt(term1_data)
        term1 = xr.DataArray(term1_sqrt, dims=term1_under.dims, coords=term1_under.coords)

        term2_data = term2_under.data.copy()
        term2_data[term2_data <= 0] = xp.nan
        term2_sqrt = xp.sqrt(term2_data)
        term2 = xr.DataArray(term2_sqrt, dims=term2_under.dims, coords=term2_under.coords)
        
        rc_flat_combined = term1 * term2
        rc_flat_filled_data = xp.nan_to_num(rc_flat_combined.data, nan=9999.0)
        rc_flat_da = xr.DataArray(rc_flat_filled_data, dims=rc_flat_combined.dims, coords=rc_flat_combined.coords)

        # Set rc_flat for the surface layer to 9999.0 as it is not physically meaningful
        max_heights = height.max(dim='layer_index', skipna=True)
        is_not_surface = height < max_heights
        
        # --- FIX for cupy incompatibility with xarray.where() ---
        # This final .where() call also needs to be handled carefully on the GPU.
        if GPU_AVAILABLE:
            # Use cupy's native where on the raw arrays
            final_data = xp.where(is_not_surface.data, rc_flat_da.data, 9999.0)
            rc_flat_da = xr.DataArray(final_data, dims=rc_flat_da.dims, coords=rc_flat_da.coords)
        else:
            # The xarray method is fine for numpy
            rc_flat_da = rc_flat_da.where(is_not_surface, 9999.0)
        # --- End Fix ---

        # Add the final calculated data back to the main dataset
        self.data['rc_flat'] = rc_flat_da.transpose('timestamp', 'layer_index')


    def slice(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> 'SnowpackProfile':
        """
        Creates a new SnowpackProfile object containing a slice of the data.
        This version uses boolean masking after normalizing dates to midnight
        to ensure robust slicing across different data sources.
        """
        if self.data is None or self.data.timestamp.size == 0:
            return self

        # Normalize the dataset's timestamps to midnight for a clean date comparison
        timestamps = pd.to_datetime(self.data.timestamp.values).normalize()
        
        # Create boolean masks, normalizing the boundary dates as well
        start_mask = timestamps >= pd.to_datetime(start_date).normalize() if start_date else True
        end_mask = timestamps <= pd.to_datetime(end_date).normalize() if end_date else True

        combined_mask = start_mask & end_mask
        
        if not np.any(combined_mask):
            logger.warning(f"No data found in the date range {start_date} to {end_date} for file {self.filename}")
            sliced_data = self.data.isel(timestamp=slice(0, 0)) # Create an empty slice
        else:
            # Use the boolean mask to select the data using isel
            sliced_data = self.data.isel(timestamp=np.where(combined_mask)[0])

        new_profile = SnowpackProfile(self.filename, _load_data=False)
        new_profile.data = sliced_data
        new_profile.metadata = self.metadata
        
        return new_profile

    def save_as_netcdf(self, output_path: str):
        """
        Saves the profile's xarray.Dataset to a NetCDF file for fast reloading.
        This method handles the conversion from GPU (CuPy) to CPU (NumPy) arrays
        before saving, as required by the underlying NetCDF library.

        Args:
            output_path (str): The destination file path for the .nc file.
        """
        if self.data is not None and self.data.timestamp.size > 0:
            try:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                
                data_to_save = self.data
                if GPU_AVAILABLE:
                    cpu_data = xr.Dataset(attrs=self.data.attrs)
                    for var_name, data_array in self.data.data_vars.items():
                        cpu_data[var_name] = (data_array.dims, data_array.get())
                    
                    coords_dict = {}
                    for coord_name, coord_val in self.data.coords.items():
                        if hasattr(coord_val.data, 'get'):
                             coords_dict[coord_name] = (coord_val.dims, coord_val.data.get())
                        else:
                             coords_dict[coord_name] = coord_val
                    cpu_data = cpu_data.assign_coords(coords_dict)
                    data_to_save = cpu_data
                
                data_to_save.to_netcdf(output_path)
                logger.debug(f"Successfully saved profile to NetCDF: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save NetCDF file to {output_path}: {e}", exc_info=True)
        else:
            logger.warning(f"No data to save for NetCDF file: {output_path}")
    
    def get_profile_summary(
        self,
        parameters_to_calculate: Dict[str, Any],
        from_height: Optional[float] = None,
        above_or_below: str = 'above',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extracts summary statistics for specified parameters.

        This method processes the profile closest to noon for each day. It first
        transfers the necessary data slice from GPU to CPU (if applicable), then
        performs summary calculations using pandas.

        Args:
            parameters_to_calculate (Dict[str, Any]): A dictionary mapping a
                new column name to a calculation type. The type can be a string
                ('min', 'max', 'mean', etc.) or a callable function that accepts
                a DataFrame and returns a scalar.
            from_height (Optional[float]): Height in meters to slice the snowpack.
                If None, the entire profile is used. Defaults to None.
            above_or_below (str): Section to analyze ('above' or 'below')
                the `from_height`. Defaults to 'above'.
            start_date (Optional[str]): Start date for the summary
                (e.g., 'YYYY-MM-DD'). Defaults to None.
            end_date (Optional[str]): End date for the summary
                (e.g., 'YYYY-MM-DD'). Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame with a 'date' index and columns for each
            requested summary statistic.
        """
        sliced_profile = self.slice(start_date, end_date)
        if sliced_profile.data is None or sliced_profile.data.timestamp.size == 0:
            return pd.DataFrame()
        
        data_in_range = sliced_profile.data
        
        if GPU_AVAILABLE:
            data_in_range_cpu = data_in_range.as_numpy()
            full_df = data_in_range_cpu.to_dataframe().reset_index()
        else:
            full_df = data_in_range.to_dataframe().reset_index()

        # --- The rest of the function uses pandas and is CPU-bound ---
        noon_time = full_df['timestamp'].dt.normalize() + pd.Timedelta(hours=12)
        full_df['time_from_noon'] = (full_df['timestamp'] - noon_time).abs()
        closest_indices = full_df.loc[full_df.groupby(full_df['timestamp'].dt.date)['time_from_noon'].idxmin()]
        noon_df = closest_indices.copy()
        if noon_df.empty:
            return pd.DataFrame()
        noon_df['date'] = noon_df['timestamp'].dt.normalize()

        summary_list = []
        for ts in noon_df['timestamp']:
            summary_row = {'date': ts.normalize()}
            profile_layers = full_df[full_df['timestamp'] == ts].copy()
            if from_height is not None:
                profile_layers = profile_layers[profile_layers['height'] > from_height] if above_or_below == 'above' else profile_layers[profile_layers['height'] <= from_height]
            if not profile_layers.empty:
                profile_layers = profile_layers.sort_values('height').copy()
                profile_layers['thickness'] = profile_layers['height'].diff()
                base_h = from_height if (from_height is not None and above_or_below == 'above') else 0
                if not profile_layers.empty:
                    first_row_index = profile_layers.index[0]
                    profile_layers.loc[first_row_index, 'thickness'] = np.float32(profile_layers['height'].iloc[0] - base_h)

            for name, calc in parameters_to_calculate.items():
                if callable(calc):
                    try:
                        summary_row[name] = calc(profile_layers)
                    except Exception as e:
                        logger.warning(f"Custom function for '{name}' failed: {e}")
                    continue

                # Parse the parameter and calculation type from the input dictionary
                if isinstance(calc, str):
                    param, calc_type = name.split('-')[0], calc
                elif isinstance(calc, tuple):
                    param, calc_type = calc
                else:
                    continue # Skip unknown formats

                if param in profile_layers:
                    series = profile_layers[param].dropna()
                    if series.empty:
                        continue
                    if param == 'hand_hardness':
                        series = series.abs()  # Ensure hand hardness is treated as absolute value
                    
                    if calc_type == 'min':
                        summary_row[name] = series.min()
                        if 'height' in profile_layers.columns:
                           summary_row[f"{name}-height"] = profile_layers.loc[series.idxmin()]['height']
                    elif calc_type == 'max':
                        summary_row[name] = series.max()
                        if 'height' in profile_layers.columns:
                           summary_row[f"{name}-height"] = profile_layers.loc[series.idxmax()]['height']
                    elif calc_type == 'mean':
                        summary_row[name] = series.mean()
                    elif calc_type == 'median':
                        summary_row[name] = series.median()
                    elif calc_type in ['weighted_mean', 'weighted_sum']:
                        weights = profile_layers.loc[series.index]['thickness']
                        weighted_sum_val = (series * weights).sum()
                        if calc_type == 'weighted_sum':
                            summary_row[name] = weighted_sum_val
                        elif weights.sum() > 0:
                            summary_row[name] = weighted_sum_val / weights.sum()
            summary_list.append(summary_row)

        if not summary_list:
            return pd.DataFrame()
        return pd.DataFrame(summary_list).set_index('date')


    def find_layer_by_criteria(
        self,
        criteria: Dict[str, str],
        search_from: str = 'top',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Finds the layer that best matches a set of prioritized criteria.

        For each day in a date range, this function scores each layer based on
        how many criteria it meets. The order of criteria in the input dictionary
        is used to assign priority, breaking ties. It returns a DataFrame
        summarizing the best-matching layer for each day, including the values
        of the parameters that were searched for.

        Args:
            criteria (Dict[str, str]): An ordered dictionary where keys are
                parameter names or 'depth', and values are condition strings
                (e.g., {'density': '< 150', 'depth': '> 0.2',
                'grain_size': '1.0 to 2.0'}). The order determines priority.
            search_from (str): If a tie remains after prioritization, this
                direction is used to select the final layer. Can be 'top' or
                'bottom'. Defaults to 'top'.
            start_date (Optional[str]): The start date for the search period.
            end_date (Optional[str]): The end date for the search period.

        Returns:
            pd.DataFrame: A DataFrame with the date as the index and columns for
            'height', 'matching_criteria_count', 'matching_parameters', and a
            column for each parameter in the input criteria.
        
        Raises:
            ValueError: If an invalid operator or search direction is provided.
        """
        sliced_profile = self.slice(start_date, end_date)
        if sliced_profile.data is None or sliced_profile.data.timestamp.size == 0:
            return pd.DataFrame()
        
        data_in_range = sliced_profile.data

        if GPU_AVAILABLE:
            data_in_range_cpu = data_in_range.as_numpy()
            full_df = data_in_range_cpu.to_dataframe().reset_index()
        else:
            full_df = data_in_range.to_dataframe().reset_index()

        noon_time = full_df['timestamp'].dt.normalize() + pd.Timedelta(hours=12)
        full_df['time_from_noon'] = (full_df['timestamp'] - noon_time).abs()
        closest_indices = full_df.loc[full_df.groupby(full_df['timestamp'].dt.date)['time_from_noon'].idxmin()]
        noon_df = closest_indices.copy()
        if noon_df.empty:
            return pd.DataFrame()
        
        results_list = []
        op_pattern = re.compile(r'([<>=!]+)\s*(\S+)')
        
        ordered_criteria = list(criteria.items())
        num_criteria = len(ordered_criteria)

        for ts in tqdm(noon_df['timestamp'], desc="Finding Layers by Criteria"):
            daily_result = {'date': ts.normalize()}
            df = full_df[full_df['timestamp'] == ts].copy()
            
            if df.empty:
                continue

            if 'depth' in criteria and 'height' in df.columns:
                total_snow_height = df['height'].max()
                df['depth'] = total_snow_height - df['height']
            
            if 'hand_hardness' in criteria and 'hand_hardness' in df.columns:
                df['hand_hardness'] = df['hand_hardness'].abs()

            criteria_masks = {}
            score = pd.Series(0, index=df.index, dtype=int)
            for i, (param, condition) in enumerate(ordered_criteria):
                weight = 2**(num_criteria - 1 - i)

                if param not in df.columns:
                    logger.warning(f"Parameter '{param}' not found in profile. Skipping criterion.")
                    continue
                
                condition_mask = pd.Series(False, index=df.index)

                if ' to ' in condition:
                    try:
                        low_str, high_str = condition.split(' to ', 1)
                        low, high = float(low_str), float(high_str)
                        condition_mask = (df[param] >= low) & (df[param] <= high)
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid 'between' format for '{param}': '{condition}'. Skipping.")
                        continue
                else: 
                    match = op_pattern.match(condition)
                    if not match:
                        logger.warning(f"Invalid condition format for '{param}': '{condition}'. Skipping.")
                        continue
                    
                    op, value_str = match.groups()
                    try: value = float(value_str)
                    except ValueError:
                        logger.warning(f"Could not convert value '{value_str}' to float. Skipping.")
                        continue
                    
                    if op == '<': condition_mask = (df[param] < value)
                    elif op == '>': condition_mask = (df[param] > value)
                    elif op == '<=': condition_mask = (df[param] <= value)
                    elif op == '>=': condition_mask = (df[param] >= value)
                    elif op == '==': condition_mask = (df[param] == value)
                    elif op == '!=': condition_mask = (df[param] != value)
                    else: raise ValueError(f"Unsupported operator '{op}' in criteria.")
                
                criteria_masks[param] = condition_mask.fillna(False)
                score += criteria_masks[param].astype(int) * weight

            max_score = score.max()

            if max_score == 0:
                daily_result['height'] = None
                daily_result['matching_criteria_count'] = 0
                daily_result['matching_parameters'] = ''
                for param, _ in ordered_criteria:
                    daily_result[param] = None
            else:
                best_matching_layers = df[score == max_score]
                
                target_layer_index = None
                if search_from == 'top':
                    target_layer_index = best_matching_layers.index[-1]
                elif search_from == 'bottom':
                    target_layer_index = best_matching_layers.index[0]
                else:
                    raise ValueError("Argument 'search_from' must be either 'top' or 'bottom'.")

                target_layer = df.loc[target_layer_index]
                daily_result['height'] = target_layer['height']
                
                matched_params = [
                    param for param, mask in criteria_masks.items() 
                    if target_layer_index is not None and mask.loc[target_layer_index]
                ]
                daily_result['matching_criteria_count'] = len(matched_params)
                daily_result['matching_parameters'] = ', '.join(matched_params)
                
                # --- Add the actual parameter values from the found layer ---
                for param, _ in ordered_criteria:
                    if param in target_layer:
                        daily_result[param] = target_layer[param]
                    else:
                        daily_result[param] = None
            
            results_list.append(daily_result)

        if not results_list:
            return pd.DataFrame()
        
        return pd.DataFrame(results_list).set_index('date')

def read_snowpack(pro_file_path: str) -> Optional[SnowpackProfile]:
    """
    Reads snowpack data, prioritizing a cached NetCDF file over the raw .pro file.

    If a .nc file corresponding to the .pro file exists, it is loaded directly.
    If not, the .pro file is parsed, and a new .nc file is created for future use.
    This "parse-once, read-many" approach dramatically speeds up the pipeline.

    Args:
        pro_file_path (str): The full path to the raw .pro file.

    Returns:
        Optional[SnowpackProfile]: A SnowpackProfile object with the loaded data,
                                   or None if both reading methods fail.
    """
    pro_path = Path(pro_file_path)
    nc_path = pro_path.with_suffix('.nc')

    if nc_path.exists():
        try:
            data = xr.open_dataset(nc_path)
            profile = SnowpackProfile(str(pro_path), _load_data=False)
            profile.data = data
            logger.debug(f"Loaded snowpack data from cached NetCDF: {nc_path}")
            return profile
        except Exception as e:
            logger.warning(f"Could not read cached NetCDF file {nc_path}, falling back to .pro. Error: {e}")

    try:
        profile = SnowpackProfile(str(pro_path))
        if profile.data is not None and profile.data.timestamp.size > 0:
            profile.save_as_netcdf(str(nc_path))
        return profile
    except Exception as e:
        logger.error(f"Failed to read and process .pro file {pro_path}: {e}")
        return None
          

if __name__ == '__main__':
    # --- Example Usage ---
    # This block demonstrates a multi-step analysis using the new slice() method.
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    from glob import glob
    import random
    
    try:
        pro_files = glob("../data/snowpack/output/2024/**/*.pro", recursive=True)
        if not pro_files:
            raise FileNotFoundError("No .pro files found. Please update the glob path.")
        
        pro_path = random.choice(pro_files)
        logger.info(f"Selected file for demonstration: {pro_path}")
        
    except (FileNotFoundError, IndexError) as e:
        logger.error(f"Error finding a file to test: {e}. Please ensure files exist.")
        pro_path = None

    if pro_path:
        try:
            reader = SnowpackProfile(pro_path)
            print(f"\n{str(reader)}\n")

            # --- Example 1: Get a summary for a specific date range ---
            print("--- Example 1: Getting summary for February 2025 ---")
            feb_profile = reader.slice(start_date='2025-02-01', end_date='2025-02-28')
            
            summary_df = feb_profile.get_profile_summary(
                parameters_to_calculate={'height-max': ('height', 'max')}
            )
            print("Max snow height for each day in February:")
            print(summary_df.head())

            # --- Example 2: Find layers by criteria in the sliced data ---
            print("\n--- Example 2: Finding weak layers in February 2025 ---")
            weak_layer_criteria = {
                'depth': '30 to 100',
                'rc_flat': '< 0.2',
                'density': '< 230',
            }
            found_layers_df = feb_profile.find_layer_by_criteria(criteria=weak_layer_criteria)
            
            if not found_layers_df.empty:
                print("Found layers matching criteria in February:")
                print(found_layers_df)
            else:
                logger.info("No layers found matching criteria in February.")

            # --- Example 3: Chained analysis to find slab properties above a weak layer ---
            print("\n--- Example 3: Chained analysis for slab properties ---")
            
            # First, get the location of the weakest layer for each day in our sliced profile
            weak_layer_locations = feb_profile.get_profile_summary(
                parameters_to_calculate={'rc_flat-min': ('rc_flat', 'min')}
            )
            weak_layer_locations.rename(columns={'rc_flat-min-height': 'weak_layer_height'}, inplace=True)
            
            slab_analysis_results = []
            # Iterate through each day where a weak layer was found
            for date, row in tqdm(weak_layer_locations.iterrows(), total=weak_layer_locations.shape[0], desc="Analyzing Slabs"):
                weak_layer_height = row['weak_layer_height']
                if pd.isna(weak_layer_height):
                    continue
                
                # For each day, get a profile for that single day to analyze the slab
                single_day_profile = reader.slice(start_date=date, end_date=date)
                
                slab_summary = single_day_profile.get_profile_summary(
                    from_height=weak_layer_height,
                    above_or_below='above',
                    parameters_to_calculate={
                        'slab_density_weighted_mean': ('density', 'weighted_mean'),
                        'slab_log_hardness_mean': lambda df: (2**df['hand_hardness'] * df['thickness']).mean() if not df.empty and 'hand_hardness' in df else None
                    }
                )
                if not slab_summary.empty:
                    slab_analysis_results.append(slab_summary)

            if slab_analysis_results:
                slab_df = pd.concat(slab_analysis_results)
                final_df = weak_layer_locations.join(slab_df, how='inner')
                print("\nCombined Daily Weak Layer and Slab Analysis for February:")
                print(final_df.head())
            else:
                print("\nCould not perform slab analysis.")

        except Exception as e:
            logger.error(f"An error occurred during analysis: {e}", exc_info=True)
