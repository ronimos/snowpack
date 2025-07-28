# Technical Plan: Adapting `snowpack_reader.py` to the AvaCollabra Data Model

## 1. Introduction:

This document outlines the technical plan for refactoring the existing `snowpack_reader.py` script to align with the specifications detailed in the [`development_plan.md`](https://gitlab.com/avacollabra/postprocessing/py-base-initial-dump/-/blob/main/development_plan.md?ref_type=heads). The current reader is a high-performance, single-file parser that produces a time-series dataset for one location. The development plan specifies a more powerful, multi-dimensional data model designed for scalability and interoperability across different data sources (SNOWPACK, CROCUS, etc.).

The core of this work involves shifting from a file-centric paradigm to a data-centric one. This means evolving the reader from a tool that processes one file at a time into a comprehensive package that can manage and analyze gridded snow profile data from multiple locations, slopes, and model realizations simultaneously. This refactoring is essential for achieving the project's goals of scalability, collaborative development, and support for large-scale operational forecasting.

### 1.5. Proposed Project Structure

To facilitate a modular and maintainable codebase, the functionality of the original `snowpack_reader.py` will be broken down into a structured package. This approach separates concerns, making the project easier to develop, test, and extend.

```
snowpack_array/
├── __init__.py
├── dataset.py          # Defines the core SnowProfileDataset class.
├── io.py               # Contains all file parsers (read_pro, read_smet) and writers (to_pro).
│
├── analysis.py         # Contains high-level analysis functions (find_layer_by_criteria).
│
├── registry.py         # Holds the canonical variable name registry and metadata.
│
└── utils.py            # Optional: For helper functions like vectorized computations.

tests/
├── __init__.py
├── data/               # Contains small sample files for testing.
│   ├── sample.pro
│   └── sample.smet
├── test_io.py          # Tests for all parsing and writing functions.
└── test_analysis.py    # Tests for analysis functions.
```

* **`dataset.py`**: The heart of the package, defining the main user-facing `SnowProfileDataset` class that wraps the `xarray.Dataset`.
* **`io.py`**: All logic related to reading from and writing to files. This keeps data format specifics isolated.
* **`analysis.py`**: High-level scientific and post-processing functions that operate on `SnowProfileDataset` objects.
* **`registry.py`**: Centralizes the mapping of variable names, units, and metadata, ensuring consistency across the package.

This structure directly supports the goals of the development plan by creating a clean, collaborative, and testable environment.

## 2. Core Architectural Changes

The most significant changes are architectural, focusing on adopting the new internal data model.

### 2.1. From `SnowpackProfile` Class to a Data Model Wrapper

The current `SnowpackProfile` class is designed to represent the contents of a single `.pro` file. The development plan calls for a new class implementation that **wraps an `xarray.Dataset`** via composition.

**Required Work:**

1.  **Create a New Wrapper Class:** Introduce a new primary class (e.g., `SnowProfileDataset`) that will contain the `xarray.Dataset` as its main attribute. This class will be the main object users interact with.
2.  **Deprecate `SnowpackProfile` as a User-Facing Object:** The logic within `SnowpackProfile` for parsing files will be extracted and moved into dedicated parser functions (see Section 3). The class itself will be retired or repurposed as an internal utility.

### 2.2. Adopting the Multi-Dimensional Data Model

The development plan's central feature is a gridded data structure with five primary dimensions: `(location, time, slope, realization, layer)`. The current reader only produces a dataset with `(timestamp, layer_index)`.

**Required Work:**

1.  **Modify Parser Outputs:** All parser functions must be updated to produce `xarray.Dataset` objects that conform to the new 5D structure. When reading a single `.pro` file, the `location`, `slope`, and `realization` dimensions will have a size of 1.
2.  **Implement Coordinate Variables:** The parsers must populate the coordinate variables specified in the plan. For example, when parsing a `.pro` file, the `location` dimension will have coordinates for `latitude`, `longitude`, and `altitude` extracted from the file header.
3.  **Handle Data Sparsity:** Implement the specified strategies for handling sparse data, particularly along the `layer` dimension. This involves padding profiles with `NaN` values to a fixed maximum layer count to maintain a regular grid.
4.  **Add Profile Status Tracking:** Introduce the `profile_status` data variable as specified. The parser will need to determine the status (e.g., "valid profile with snow") and record it in the dataset.

## 3. Refactoring Parsing and I/O

The file reading logic needs to be extracted from the `SnowpackProfile` class and structured into the modular `read_*` functions defined in the development plan.

### 3.1. Implementing Standalone `read_*` Functions

The plan specifies a clear API with functions like `read_pro()`, `read_smet()`, and a high-level `read()` wrapper.

**Required Work:**

1.  **Create `read_pro()`:** Extract the line-by-line parsing logic from the `SnowpackProfile._read_profile()` method and place it inside a new `read_pro()` function. This function will take a file path and return a `SnowProfileDataset` object compliant with the new 5D data model. It must also implement the specified arguments like `datetime_start` and `include_smet`.
2.  **Implement Surface Hoar Logic:** As specified in the plan, ensure the `read_pro` function correctly identifies surface hoar information from the file header and represents it as the topmost layer in the profile, filling missing properties with `NaN`.
3.  **Create `read()` Wrapper:** Implement the high-level `read()` function. This function will be responsible for:
    * Iterating through file paths and directories.
    * Auto-detecting the file type and calling the appropriate parser (e.g., `read_pro`).
    * Merging the datasets returned by the individual parsers into a single, consolidated `SnowProfileDataset`. This involves correctly assigning coordinates to the `location` and `realization` dimensions to avoid conflicts.

### 3.2. Standardizing Variable Names

The hardcoded `PARAM_CODES` dictionary in `snowpack_reader.py` must be replaced with the canonical name registry proposed in the development plan.

**Required Work:**

1.  **Create the Name Registry:** Implement a separate module or configuration file for the variable name registry.
2.  **Consider Header Definitions:** The parser must first check for a `[HEADER]` section within the `.pro` file. This section can define custom variable names for the parameter codes. The parser should prioritize these names before falling back to the hardcoded `PARAM_CODES` mapping.
3.  **Update Parsers:** Modify the parsers to use the name registry to translate all found variable names (whether from the header or the hardcoded map) to their canonical names (e.g., "rho", "density" -> "density").
4.  **Attach Metadata:** The parsers must attach the corresponding `standard_name` and `si_units` from the registry as attributes to each data variable in the `xarray.Dataset`.

### 3.3. Implementing `to_*()` Write Methods

To enable interoperability, the `SnowProfileDataset` class will include methods for exporting data into various formats, as specified in the development plan.

**Required Work:**

1.  **Implement `.to_netcdf()`:** This method will be the primary way to save the internal data structure. It will be a thin wrapper around `xarray`'s native `.to_netcdf()` functionality. The implementation will reuse the logic from the existing `save_as_netcdf` method, ensuring that data is correctly converted from GPU (CuPy) to CPU (NumPy) arrays before writing.
2.  **Implement `.to_pro()` and `.to_smet()`:** These methods will be more complex as they require converting the multi-dimensional `xarray.Dataset` back into a text-based format. The work involves:
    * **Iteration:** The methods must iterate through each individual profile within the dataset (i.e., for each unique combination of `location`, `time`, `slope`, and `realization`).
    * **Data Formatting:** For each profile, the data for each variable must be formatted back into the specific string format required by the `.pro` and `.smet` files, including parameter codes and comma-separated values.
    * **File Handling:** As the dataset can contain many profiles, these methods must accept a directory path as an argument. They will be responsible for creating a logical file-naming convention and writing each profile to a separate file within the specified directory.

## 4. Adapting Analysis and Utility Methods

The existing high-level analysis methods in `snowpack_reader.py` are powerful but are built for a single-profile timeseries. They must be refactored to work with the new multi-dimensional data model.

### 4.1. Updating Analysis Functions

The methods `get_profile_summary` and `find_layer_by_criteria` need to be adapted.

**Required Work:**

1.  **Generalize Method Logic:** Rewrite these functions to operate on a multi-dimensional `SnowProfileDataset`. Instead of processing a single timeseries, they should use `xarray`'s `groupby()` or `apply_ufunc()` capabilities to perform the analysis for each profile (i.e., for each unique combination of `location`, `time`, `slope`, and `realization`).
2.  **Preserve Vectorization:** The core logic within these functions should remain as vectorized as possible to leverage the performance benefits of `xarray` and NumPy/CuPy.

### 4.2. Vectorized Computations (`rc_flat`, `depth`)

The existing vectorized calculations are a major strength of `snowpack_reader.py` and align well with the new model.

**Required Work:**

1.  **Verify Broadcasting:** These methods (`_compute_and_add_depth`, `_compute_and_add_rc_flat_vectorized`) will require minimal changes. The primary task is to verify that `xarray` correctly broadcasts the calculations across the new `location`, `slope`, and `realization` dimensions. The use of the `xp` alias for NumPy/CuPy should be maintained.

### 4.3. Caching and NetCDF I/O

The existing `.nc` caching mechanism is valuable and should be preserved.

**Required Work:**

1.  **Adapt `save_as_netcdf`:** The existing `save_as_netcdf` method will now be a method of the new `SnowProfileDataset` class (e.g., `.to_netcdf()`). It will save the entire multi-dimensional dataset.
2.  **Update `read_snowpack` (now `read_netcdf`):** The logic for checking for a cached `.nc` file will be moved into the new `read_netcdf()` function, as specified in the development plan.

### 4.4. Concatenating and Merging Datasets

The development plan relies on native `xarray` functionality for combining datasets. The work required is primarily in the implementation of the high-level `read()` wrapper and in providing clear documentation.

**Required Work:**

1.  **Implement `xr.concat` in `read()` Wrapper:** The `read()` function must use `xr.concat` to combine the list of individual `xarray.Dataset` objects returned by the parsers into a single dataset. It needs to correctly handle concatenation along the `location` and `realization` dimensions.
2.  **Document Merging Workflows:** Create tutorials that clearly demonstrate how to use `xr.merge`, `.update()`, and `.combine_first()` for common use cases, such as updating a forecast with nowcast data.

### 4.5. Visualization Support

While the package is not a dedicated visualization library, it must provide the necessary hooks and helpers to enable easy plotting, as specified in the plan.

**Required Work:**

1.  **Develop a Profile Plotting Interface:** Create a method within the `SnowProfileDataset` class (e.g., `.plot.profile()`) that can take a single profile (a dataset subsetted to a single location, time, slope, and realization) and generate a stratigraphy plot using a library like NiViz or snowpat. This may involve writing a converter to translate the `xarray.Dataset` slice into the format expected by the plotting library.
2.  **Implement 2D Map Reshaping:** Create a helper function or method that can reshape scalar data variables (e.g., snow depth) from the 1D `location` dimension into a 2D grid for map-based plotting. This function will need to handle both regular grids and irregular meshes (requiring a user-provided geometry file).

## 5. Summary of Key Tasks

1.  **Architectural Shift:**
    * Create a new `SnowProfileDataset` class to wrap `xarray.Dataset`.
    * Restructure the output of all parsers to a `(location, time, slope, realization, layer)` dimensional model.
2.  **Parser Refactoring:**
    * Extract parsing logic into standalone `read_pro()` and `read_smet()` functions.
    * Implement a high-level `read()` wrapper for multi-file and multi-format ingestion.
    * Replace `PARAM_CODES` with a canonical variable name registry.
    * Implement `to_*` methods for writing data to various formats.
3.  **Analysis Method Adaptation:**
    * Rewrite `get_profile_summary` and `find_layer_by_criteria` to use `xarray.groupby()` on the new data model.
    * Verify that existing vectorized calculations broadcast correctly.
4.  **New Feature Implementation:**
    * Add the `profile_status` data variable.
    * Enrich the dataset with CF-compliant metadata for timezones and coordinate reference systems.
    * Implement helpers and documentation for data manipulation and visualization.

## 6. Package Integrity and Testing

To ensure the reliability and correctness of the refactored package, a comprehensive testing strategy must be implemented as outlined in the development plan.

**Required Work:**

1.  **Establish Test Infrastructure:** Create a `tests/` directory at the root of the project. This will contain subdirectories for unit tests, integration tests, and a `data/` directory for small, versioned sample files (`.pro`, `.smet`, etc.) used by the tests.
2.  **Implement Unit Tests:** Write `pytest` unit tests for each core component, including:
    * Each parser function (`read_pro`, `read_smet`).
    * The canonical name registry logic.
    * Key analysis methods, testing them with known inputs and expected outputs.
3.  **Implement Integration Tests:** Create tests for workflows that involve multiple components, such as verifying that the high-level `read()` wrapper can correctly parse multiple file types and merge them into a valid `SnowProfileDataset`.
4.  **Enable Doctests:** Add example code to the docstrings of all public-facing functions and methods. Enable `pytest`'s doctest module to ensure these examples are automatically tested and remain correct as the code evolves.
5.  **Set Up Continuous Integration (CI):** Configure a CI pipeline (e.g., using GitLab CI or GitHub Actions) that automatically runs the full test suite (including doctests) on every commit and pull request. This will provide immediate feedback to developers and prevent regressions.
