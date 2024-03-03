# Hurricane Data Analysis Using HURSAT Dataset

## Introduction

This project aims to analyze key characteristics of hurricanes and make predictions with them using the HURSAT-B1 dataset. The dataset includes satellite data centered on historical tropical cyclones, providing a source of data for understanding hurricane dynamics and completing comprehensive analysis on hurricanes.

## Dataset Overview

- **HURSAT-B1 Dataset**: The data used in this analysis is part of the [HURSAT_B1 Dataset](https://www.ncei.noaa.gov/products/hurricane-satellite-data) from the National Center for Environmental Information - NOAA. It contains raw satellite observations from the ISCCP B1 data, focused on tropical cyclones from 1978 to 2016. 
- **Resolution and Coverage**: The data is gridded to approximately 8km resolution and available at 3-hour intervals.
- **Parameters**: Includes infrared and visible satellite channels, with brightness temperatures and other meteorological variables.

## Features of the Project

- Data cleaning and preprocessing to ensure high-quality input for machine learning models.
- Application of machine learning to predict hurricane attributes like wind speed and central pressure.
- Comprehensive analysis of hurricane data, focusing on wind speed, central pressure, and eye characteristics.
- Application of advanced analysis techniques to predict Hurricane Categories in innovative way.

## Techniques Used

### Data Cleaning and Preprocessing

- **Handling Missing Values**: Employed various techniques to handle missing data in 'WindSpeed' and 'CentralPressure' columns.
- **Outlier Detection**: Identified and treated outliers in the dataset using statistical methods.
- **Data Transformation**: Standardized and normalized data to prepare for machine learning models.

### Machine Learning and Deep Learning

- **Dense Neural Network**: Utilized in different scenarios to classify hurricanes with data other than windspeed.
- **RandomForestRegressor**: Implemented for imputing missing values in key columns, ensuring data integrity.
- **Predictive Modeling**: Utilized advanced machine learning algorithms to predict hurricane characteristics.

### Data Visualization

- Developed visualizations to analyze data trends and relationships.
- Employed libraries like Matplotlib and Seaborn for generating charts and graphs.

## Data Model

The project leverages a detailed data model derived from the HURSAT-B1 dataset. It includes:

- **Schema Design**: Detailed database schema focusing on key hurricane attributes.
- **Data Flow**: Demonstrates how data is processed, cleaned, and utilized in machine learning models.

## Findings

- The project highlights significant correlations and predictions regarding hurricane characteristics.
- Insights into the predictability of hurricane features like wind speed and central pressure based on historical data.
- Insights into hurricane categorization with untraditional variables.

## Installation and Usage

```bash
# Clone the repository
git clone https://github

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter notebooks
jupyter notebook Data_Cleaning.ipynb
