import numpy as np
from matplotlib import pyplot as plt
import Image_Preprocessing as images
from Image_Prediction_Model import ConvLSTM

image_directory_path = '/Users/zrdav/Documents/CSCI_1070/cs_1070_final_project/Data_Files/hursat-analysis-data'

x_train, y_train, x_val, y_val, train_dataset, val_dataset = images.construct_dataset(image_directory_path, 'IRWIN')

#Uncomment if training model ->
'''
model = ConvLSTM(x_train, y_train, x_val, y_val, 20, 5)
model.fit()
model.save_model('/Users/zrdav/Documents/CSCI_1070/cs_1070_final_project/Models/model.keras')
'''

loaded_model = ConvLSTM.load_model('/Users/zrdav/Documents/CSCI_1070/cs_1070_final_project/Models/model.keras', x_train, y_train, x_val, y_val, 20, 5)

example = val_dataset[np.random.choice(range(len(val_dataset)), size=1)[0]]

# Pick the first/last ten frames from the example.
frames = example[:10, ...]
original_frames = example[10:, ...]

#Create 10 new frames using the prediction model on the evaluation sample
for _ in range(10):
    print('predicting')
    new_prediction = loaded_model.predict(np.expand_dims(frames, axis=0))
    new_prediction = np.squeeze(new_prediction, axis=0)
    predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)

    # Extend the set of prediction frames.
    frames = np.concatenate((frames, predicted_frame), axis=0)

fig, axes = plt.subplots(2, 10, figsize=(20, 4))

# Plot the original frames.
for idx, ax in enumerate(axes[0]):
    ax.imshow(np.squeeze(original_frames[idx]), cmap="gray")
    ax.set_title(f"OFrame {idx}")
    ax.axis("off")

# Plot the new frames.
new_frames = frames[10:, ...]
for idx, ax in enumerate(axes[1]):
    ax.imshow(np.squeeze(new_frames[idx]), cmap="gray")
    ax.set_title(f"PFrame {idx}")
    ax.axis("off")

plt.show()