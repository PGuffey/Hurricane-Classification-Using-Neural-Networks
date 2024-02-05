import numpy as np
import cv2 as cv
import os
import glob
import time

image_directory_path = '/Users/zrdav/Documents/CSCI_1070/cs_1070_final_project/Data_Files/hursat-analysis-data'

#Walk file path and find all .png images, sort image sequences, resize to interpretable values
def get_images_from_directory(directory, imaging_type):
    resized_images = []
    pattern = '*' + imaging_type + '*' + '.png'
    for filename in sorted(glob.glob(os.path.join(directory, pattern))):
        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        img_resized = cv.resize(img, (64, 64), interpolation=cv.INTER_CUBIC)
        
        img_array = np.array(img_resized)
        resized_images.append(img_array)

    sequence_array = np.array(resized_images)
    return sequence_array

#Get n = max num of subsequences and split sequence into n subsequences
def split_sequence_into_subsequences(sequence, subsequence_len):
    num_sequences = len(sequence) // subsequence_len
    subsequences = sequence[:num_sequences * subsequence_len].reshape(
        num_sequences, subsequence_len, sequence.shape[1], sequence.shape[2]
    )
    return subsequences

#Get all files into time sequences, split each sequence into subsequences of 50 frames if data is HURSAT data
def construct_dataset(directory_path, imaging_type):
    folders = [
        get_images_from_directory(os.path.join(directory_path, folder), imaging_type)
        for folder in os.listdir(directory_path)
        if folder.startswith('HURSAT')
    ]
    
    subsequences = []
    subsequence_len = 50

    for folder in folders:
        # Split each sequence into subsequences while maintaining sequence integrity
        sequence_subsequences = split_sequence_into_subsequences(folder, subsequence_len)
        subsequences.append(sequence_subsequences)

    stacked_subsequences = np.concatenate(subsequences, axis=0)

    return stacked_subsequences

#Normalize pixel values between 0 and 1 
def normalize_image(image):
    normalized_image = cv.normalize(image, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
    return normalized_image

#Blur images to avoid overfitting to noise
def blur_image(image):
    bilateral_blur = cv.bilateralFilter(image, 5, 15, 20)
    return bilateral_blur

#Canny edge detection
def detect_edges(image):
    canny = cv.Canny(image, 150, 175)
    dilated = cv.dilate(canny, (3,3), iterations=1)
    return dilated

#Use OpenCV to visualize frame sequences
def play_time_series(images, delay, id=None):
    for frame in images:
        cv.imshow(f'{id}', frame)

        key = cv.waitKey(delay)
        if key == ord('q'):
            break
    cv.destroyAllWindows()

data = construct_dataset(image_directory_path, 'IRWIN')


# folder_ids = [folder for folder in os.listdir(image_directory_path) if folder.startswith('HURSAT')]
# for id, folder in enumerate(data):
#     id = folder_ids[id]
#     blurred = [blur_image(image) for image in folder]
#     normalized = [normalize_image(image) for image in blurred]
#     #play_time_series(normalized, id, 33)