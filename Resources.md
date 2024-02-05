## Libraries, Modules, and APIs

- These are overview resources that could be potentially useful when implemented into code in order to gather, process, or predict from data

|Name| Resource Link    | Description |Usage| Initial Notes |
|---| --- | ---|---|---|
|NOAA - Hurricane Satellite Data|https://www.ncei.noaa.gov/products/hurricane-satellite-data|Government satellite imagery data source for project. More specific link to data is : https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/ |Hurricane Database Source|Project Data source|
|NumPy|https://numpy.org/|High performance python math library|Inclusion with models|Default inclusion, dependency for everything|
|Pandas|https://pandas.pydata.org/|Data handling library|Data cleaning and analysis|Default inclusion|
|SciKit-Learn|https://scikit-learn.org/stable/|Basic and efficient tools for machine learning|Model performance, tuning, accuracy scoring, and basic ML applications|Useful for testing and tuning|
|TensorFlow|https://www.tensorflow.org/ | High level machine learning library with massive library of easy-to-train models|We would use this after data has been prepared for model use | PyTorch might be better but we'll see, TF has better deployment speed |
|PyTorch|https://pytorch.org/ | Similar to TF, large Python library of machine learning models | Excels with computer vision, utilize on satellite data |I think this is a good option for the large model library we use but I need to look more into it |
|Keras|https://keras.io/ | API for TF (and now PyTorch, conveniently) that allows for fast and extremely easy deployment and training of neural networks | This is the fastest way to train and test the CNN we will use for analysis of satellite imagery|Absolutely should be implemented |
|SciPy|https://scipy.org/|Scientific computing library for Python for model analysis and accuracy|We can implement this to optimize calculations and model hyper-parameters within the system|Side tool for optimization and tuning|
|OpenCV|https://opencv.org/|Open source computer vision Python library|Analysis of satellite imagery|Open source means good docs|
|AWS|https://aws.amazon.com/free/database/ |Free Amazon databases |Utilize these to store all the aggregated data that we can collect |We can check out other providers I just think AWS is the safest/most reliable bet |
|NASA APIs|https://api.nasa.gov/|NASA REST API's including weather and satellite imagery live datasets|We can use some of this data to provide contextual information for the prediction models|Useful|
|NASA FIRMS API|https://firms.modaps.eosdis.nasa.gov/map|NASA live fire location data including REST API|We can use this to provide some live fire data|NOTE: NASA's TOS requires this API and data to not be used for life/property preservation because of inaccuracies, so it's usefulness might be limited|
|Sentinel Hub|https://www.sentinel-hub.com/|Live API for satellite imagery|Best characteristics is that it offers high-res imagery (10m) and near-infrared imaging (useful for fire detection)|Not exact or perfectly live, but still useful|
|EarthCache|https://skywatch.com/earthcache/get-access/|Live API for satellite imagery|Aggregates imagery from multiple sources at high-res and multiple outputs|Again, not perfectly live but still useful|
|Google Earth|https://earth.google.com/web/|API for access to all high-res Google Earth Data|Same uses as other datasets|Same drawbacks as well as being less up-to-date|
|SkyWatch Free Satellite Data|https://skywatch.com/free-sources-of-satellite-data/|More sources of satellite imagery as well as API's|Similar applications, different output types|All free sources|
|AirNow|https://docs.airnowapi.org/|AirNow API for live fire and smoke data|More geolocation contextual data to feed to potential ensemble/aggregate model that takes more than satellite data|Can get coordinates for reported fire and smoke sightings in given location|
|ArcGIS|https://developers.arcgis.com/python/|Another geolocation DB for satellite geo data and map visualization, we can use ArcGIS layers to find fire and thermal hotspot data|Probably better and more accurate than AirNow|Not sure about accuracy|
|Landfire|https://www.landfire.gov/index.php|Large database of numerous layers of fire data|Contextual data including location, type, environment, weather data|We can use this to correlate fire data with predictive features|


