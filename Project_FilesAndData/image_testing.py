import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np



# Replace with your file paths
file_path1 = "C:\\Users\\payto\\OneDrive\\Desktop\\HURSAT_Tatiana\\2016040S16155.TATIANA.2016.02.09.0300.25.HIM-8.020.hursat-b1.v06.nc"
file_path2 = "C:\\Users\\payto\\OneDrive\\Desktop\\HURSAT_Tatiana\\2016040S16155.TATIANA.2016.02.09.0600.25.HIM-8.021.hursat-b1.v06.nc"

# Open the first NetCDF file and extract data
dataset1 = nc.Dataset(file_path1)
irwin_data1 = dataset1.variables['IRWIN'][0,:,:]  # Adjust if necessary
dataset1.close()

# Open the second NetCDF file and extract data
dataset2 = nc.Dataset(file_path2)
irwin_data2 = dataset2.variables['IRWIN'][0,:,:]  # Adjust if necessary
dataset2.close()

# Create a figure with 2 subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Adjust the figure size as needed

# Display the first image
axs[0].imshow(irwin_data1, cmap='gray')
axs[0].set_title('First IRWIN Image')
axs[0].axis('off')  # To turn off axis

# Display the second image
axs[1].imshow(irwin_data2, cmap='gray')
axs[1].set_title('Second IRWIN Image')
axs[1].axis('off')  # To turn off axis

# Show the plot
plt.show()