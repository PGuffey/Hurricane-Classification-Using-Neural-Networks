import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from math import sqrt
from scipy.interpolate import UnivariateSpline
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report


def check_missing_data(df):
    return df.isnull().sum()

def check_data_types(df):
    return df.dtypes

def check_duplicates(df):
    return df.duplicated().sum()

def check_outliers(df, columns):
    outlier_dict = {}
    for column in columns:
        # Convert column to numeric, coercing non-numeric values to NaN
        numeric_column = pd.to_numeric(df[column], errors='coerce')
        
        # Handle NaN values if they were created during coercion
        # Here we're dropping them, but you may choose to fill them with mean/median or some other method
        numeric_column = numeric_column.dropna()

        # Calculate Q1, Q3, and IQR for outlier detection
        Q1 = numeric_column.quantile(0.25)
        Q3 = numeric_column.quantile(0.75)
        IQR = Q3 - Q1

        # Calculate the number of outliers using the IQR method
        outlier_count = ((numeric_column < (Q1 - 1.5 * IQR)) | (numeric_column > (Q3 + 1.5 * IQR))).sum()
        outlier_dict[column] = outlier_count
    return outlier_dict

def plot_trends(df, column_name):
    plt.figure(figsize=(14, 7)) 
    plt.plot(df[column_name], marker='o', linestyle='-', label=column_name) 
    plt.title(f'Trend of {column_name} over Time') 
    plt.xlabel('Index') 
    plt.ylabel(column_name) 
    plt.legend() 
    plt.show() 

def spline_interpolation(df, column_name, lower_bound=None, upper_bound=None):
    x = df.index
    y = df[column_name].dropna()
    x_non_null = y.index
    
    spline = UnivariateSpline(x_non_null, y)
    interpolated_values = spline(x)

    if lower_bound is not None:
        interpolated_values[interpolated_values < lower_bound] = lower_bound
    if upper_bound is not None:
        interpolated_values[interpolated_values > upper_bound] = upper_bound

    df[column_name] = interpolated_values
    
    return df

def gentle_slope_spline_interpolation(df, column_name, lower_bound):
    """
    Spline interpolation that decreases less and less as it gets closer to the lower bound.
    """
    x = df.index

    y = df[column_name].dropna()
    x_non_null = y.index

    spline = UnivariateSpline(x_non_null, y)

    interpolated_values = spline(x)

    for i in range(len(interpolated_values)):
        if interpolated_values[i] < lower_bound:
            distance_to_lower_bound = lower_bound - interpolated_values[i]
            interpolated_values[i] += distance_to_lower_bound * (1 - np.exp(-distance_to_lower_bound / 10))

    df[column_name] = interpolated_values
    
    return df

def replace_with_mean(df, column_name):
    """
    Replace null or NaN values in a column with the mean of the non-null values.
    """
    mean_value = df[column_name].mean()
    df[column_name].fillna(mean_value, inplace=True)
    
    return df

def classify_hurricane_category(wind_speed):
    """
    Classify the hurricane category based on the Saffir-Simpson scale using wind speed.
    Includes a Category 0 for wind speeds below Category 1 threshold.
    """
    if wind_speed < 74:
        return 0
    elif wind_speed <= 95:
        return 1
    elif wind_speed <= 110:
        return 2
    elif wind_speed <= 129:
        return 3
    elif wind_speed <= 156:
        return 4
    elif wind_speed >= 157:
        return 5
    else:
        return None

def convert_to_datetime(row):
    date_str = row['observation_date'].strip("[]'")
    time_str = row['observation_time'].strip("[]'")
    return datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M')

def train_impute_evaluate_rf(data):
    # Train Random Forest models
    model_rf_wind = RandomForestRegressor()
    model_rf_pressure = RandomForestRegressor()
    
    # Check and train model for WindSpeed if there are valid training samples
    if not data.dropna(subset=['WindSpeed', 'CentralPressure']).empty:
        train_data_windspeed = data.dropna(subset=['WindSpeed', 'CentralPressure'])
        model_rf_wind.fit(train_data_windspeed[['CentralPressure']], train_data_windspeed['WindSpeed'])
        
        # Calculate MSE for WindSpeed model
        mse_wind = mean_squared_error(train_data_windspeed['WindSpeed'], model_rf_wind.predict(train_data_windspeed[['CentralPressure']]))
    else:
        mse_wind = None

    # Check and train model for CentralPressure if there are valid training samples
    if not data.dropna(subset=['WindSpeed', 'CentralPressure']).empty:
        train_data_pressure = data.dropna(subset=['WindSpeed', 'CentralPressure'])
        model_rf_pressure.fit(train_data_pressure[['WindSpeed']], train_data_pressure['CentralPressure'])
        
        # Calculate MSE for CentralPressure model
        mse_pressure = mean_squared_error(train_data_pressure['CentralPressure'], model_rf_pressure.predict(train_data_pressure[['WindSpeed']]))
    else:
        mse_pressure = None

    # Impute missing values if possible
    missing_wind = data['WindSpeed'].isna() & data['CentralPressure'].notna()
    if not data.loc[missing_wind].empty:
        data.loc[missing_wind, 'WindSpeed'] = model_rf_wind.predict(data.loc[missing_wind, ['CentralPressure']])

    missing_pressure = data['CentralPressure'].isna() & data['WindSpeed'].notna()
    if not data.loc[missing_pressure].empty:
        data.loc[missing_pressure, 'CentralPressure'] = model_rf_pressure.predict(data.loc[missing_pressure, ['WindSpeed']])

    return data, mse_wind, mse_pressure


interpolated_data = pd.read_csv(r'C:\Users\payto\Desktop\combined_interpolated_data.csv')
spatial_data = pd.read_csv(r'C:\Users\payto\Desktop\Computer_Science\CS_1070\cs_1070_final_project\Data_Files\spatial_data.csv')

# Ensure correct data types, especially for numeric columns // Sovled 'MaskedConstant' TypeError
interpolated_data['WindSpeed'] = pd.to_numeric(interpolated_data['WindSpeed'], errors='coerce')
interpolated_data['CentralPressure'] = pd.to_numeric(interpolated_data['CentralPressure'], errors='coerce')

interpolated_data.replace('--', np.nan, inplace=True)

interpolated_data['WindSpeed'] = interpolated_data['WindSpeed'].astype(float)
interpolated_data['CentralPressure'] = interpolated_data['CentralPressure'].astype(float)

mean_eye_probability_by_hurricane = interpolated_data.groupby('HurricaneName')['EyeProbability'].transform('mean')
interpolated_data['EyeProbability'] = interpolated_data['EyeProbability'].fillna(mean_eye_probability_by_hurricane)

interpolated_data, mse_wind, mse_pressure = train_impute_evaluate_rf(interpolated_data)

# print(f"Imputed dataset with MSE for WindSpeed: {mse_wind}, MSE for CentralPressure: {mse_pressure}")

interpolated_data = interpolated_data.dropna(subset=['WindSpeed', 'CentralPressure'])

# print(interpolated_data.isnull().sum())

interpolated_data['HurricaneCategory'] = interpolated_data['WindSpeed'].apply(classify_hurricane_category)

interpolated_data['datetime'] = interpolated_data.apply(convert_to_datetime, axis=1)

interpolated_data['month'] = interpolated_data['datetime'].dt.month
interpolated_data['hour'] = interpolated_data['datetime'].dt.hour

grouped = interpolated_data.groupby('HurricaneName')

interpolated_data['WindSpeed_Change'] = grouped['WindSpeed'].diff()
interpolated_data['CentralPressure_Change'] = grouped['CentralPressure'].diff()

window_size = 3
interpolated_data['WindSpeed_RollingMean'] = grouped['WindSpeed'].rolling(window=window_size).mean().reset_index(level=0, drop=True)
interpolated_data['CentralPressure_RollingMean'] = grouped['CentralPressure'].rolling(window=window_size).mean().reset_index(level=0, drop=True)
interpolated_data['WindSpeed_RollingStd'] = grouped['WindSpeed'].rolling(window=window_size).std().reset_index(level=0, drop=True)
interpolated_data['CentralPressure_RollingStd'] = grouped['CentralPressure'].rolling(window=window_size).std().reset_index(level=0, drop=True)


# print("Descriptive Statistics for Interpolated Data:\n", interpolated_data.describe())
# print("\nDescriptive Statistics for Spatial Data:\n", spatial_data.describe())



# Select features and target variable (excluding wind speed)
# Select features and target variable

interpolated_data = interpolated_data.dropna()

features = ["CentralPressure","bt_eye","bt_eyewall","EyeProbability","EyeCompleteness","EyeRadius","ArcherLat","ArcherLon",
            "ArcherComboScore","WindSpeed_Change","CentralPressure_Change",
            "WindSpeed_RollingMean","CentralPressure_RollingMean","WindSpeed_RollingStd","CentralPressure_RollingStd"]

# ['CentralPressure', 'bt_eye', 'bt_eyewall', 'EyeProbability', 
#             'EyeCompleteness', 'ArcherLat', 'ArcherLon']

X = interpolated_data[features]
y = interpolated_data['HurricaneCategory']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE()
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

y_train_cat = to_categorical(y_train, num_classes=6)
y_test_cat = to_categorical(y_test, num_classes=6)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(len(features),)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax')) # 6 neurons for categories 1-5 and an additional one for 0 or any other class

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(X_train, y_train_cat, 
                    validation_data=(X_test, y_test_cat),
                    epochs=50, 
                    batch_size=32)

model.evaluate(X_test, y_test_cat)

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test, y_pred_labels))

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model accuracy over epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model loss over epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()




# # Feature Importance
# feature_importances = grid_search.best_estimator_.feature_importances_

# # Normalize feature importances
# normalized_importances = feature_importances / np.sum(feature_importances)

# # Calculate Intensity Scores (Example using mean of normalized weighted features)
# interpolated_data['IntensityScore'] = (interpolated_data[features] * normalized_importances).mean(axis=1)
# interpolated_data['IntensityScore'] = 1 + 4 * (interpolated_data['IntensityScore'] - interpolated_data['IntensityScore'].min()) / (interpolated_data['IntensityScore'].max() - interpolated_data['IntensityScore'].min())

# # Save or display the results
# # data.to_csv('hurricane_intensity_scores.csv')
# print(interpolated_data[['HurricaneName', 'IntensityScore']])

# 7. Feature Importance Extraction and Intensity Score Calculation

# # Extract feature importances and normalize
# feature_importances = best_model.feature_importances_
# normalized_importances = feature_importances / np.sum(feature_importances)

# # Calculate Intensity Scores based on these importances
# interpolated_data['IntensityScore'] = (interpolated_data[features] * normalized_importances).sum(axis=1)


# # Assuming you have the normalized_importances from the Random Forest model
# # and your dataset 'data' with the required features

# # Step 1: Calculate Intensity Scores
# # Calculate the weighted sum of the features for each hurricane
# interpolated_data['IntensityScore'] = (interpolated_data[features] * normalized_importances).sum(axis=1)

# # Step 2: Scale the Scores to a 1-5 Range
# # Here, I am using a simple linear scaling. You might want to use a different scaling method.
# min_score, max_score = interpolated_data['IntensityScore'].min(), interpolated_data['IntensityScore'].max()
# interpolated_data['ScaledIntensityScore'] = 1 + 4 * (interpolated_data['IntensityScore'] - min_score) / (max_score - min_score)

# # Step 3: Categorize the Hurricanes
# # Define categories based on the Scaled Intensity Score
# def categorize_hurricane(score):
#     if score <= 2:
#         return 'Category 1'
#     elif score <= 3:
#         return 'Category 2'
#     elif score <= 4:
#         return 'Category 3'
#     elif score <= 4.5:
#         return 'Category 4'
#     else:
#         return 'Category 5'

# interpolated_data['HurricaneCategoryNoWind'] = interpolated_data['ScaledIntensityScore'].apply(categorize_hurricane)

# # Display or save the results
# print(interpolated_data[['HurricaneName', 'HurricaneCategory', 'ScaledIntensityScore']])
# interpolated_data.to_csv('Combined_Interpolated_data_Pre.csv')




# #  ---------------------------------------
# # Setting the sequence length
# sequence_length = 4  # Considering shorter hurricanes, 4 data points (12 hours)

# # Extracting unique hurricane names
# hurricane_names = interpolated_data['HurricaneName'].unique()

# # Placeholder for sequences (features) and corresponding targets (continuous score)
# sequences = []
# targets = []

# for name in hurricane_names:
#     # Extracting data for each hurricane
#     hurricane_specific_data = interpolated_data[interpolated_data['HurricaneName'] == name]
    
#     # Ensuring the data is in chronological order
#     hurricane_specific_data.sort_values('datetime', inplace=True)

#     # Scaling the features
#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(hurricane_specific_data[features])

#     # Creating the TimeseriesGenerator for each hurricane
#     generator = TimeseriesGenerator(scaled_features, scaled_features[:,0], length=sequence_length, batch_size=1)

#     # Appending generated sequences and their targets
#     for i in range(len(generator)):
#         x, y = generator[i]
#         sequences.append(x[0])
#         targets.append(y[0])

# # Converting to numpy arrays
# sequences = np.array(sequences)
# targets = np.array(targets)

# # Splitting the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=0.2, random_state=42)

# # Model architecture
# model = Sequential([
#     LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
#     LeakyReLU(alpha=0.01),
#     LSTM(50),
#     LeakyReLU(alpha=0.01),
#     Dense(1)
# ])

# # Compile the model
# model.compile(optimizer=Adam(), loss='mean_squared_error')

# # Train the model
# model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# # Save the model
# model.save('hurricane_intensity_model.h5')

# # Load the model (for future use)
# loaded_model = load_model('hurricane_intensity_model.h5')






# SQL Database Creation Code (Although I don't know how we'd be able to share it):
# ---------------------------------------------------------------------------------------
# # Database connection parameters
# db_username = 'username'
# db_password = 'password'
# db_host = 'host'
# db_name = 'database_name'
# db_type = 'postgresql'  # or 'mysql', 'sqlite', etc.

# # SQLAlchemy engine
# engine = create_engine(f'{db_type}://{db_username}:{db_password}@{db_host}/{db_name}')

# # Save DataFrames to SQL tables
# spatial_df.to_sql('spatial_data', engine, if_exists='replace', index=False)
# interpolated_df.to_sql('interpolated_data', engine, if_exists='replace', index=False)
# ---------------------------------------------------------------------------------------
