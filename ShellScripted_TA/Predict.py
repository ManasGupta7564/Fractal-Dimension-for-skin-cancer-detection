from skimage.feature import hog
import os, ctypes
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray, rgb2lab
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd

# HOG Parameters
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
IMG_SIZE = (64, 64)  # Resize all images to 64x64 for consistency

def extract_hog_features(image_path):
    # Read the image
    img = imread(image_path)
    # Resize the image
    img_resized = resize(img, IMG_SIZE, anti_aliasing=True)
    # Convert to grayscale
    img_gray = rgb2gray(img_resized)
    # Extract HOG features
    features = hog(
        img_gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm='L2-Hys',
        visualize=False,
        channel_axis=None  # For grayscale images
    )
    return features


def color_thermometers(image_path, block_size=(3, 3)):
    # Read and resize the image
    img = imread(image_path)
    img_resized = resize(img, (64, 64), anti_aliasing=True)  # Resize to a standard size

    # Convert the image to LAB color space
    img_lab = rgb2lab(img_resized)

    # Image dimensions
    height, width, _ = img_lab.shape
    block_height, block_width = block_size

    # Initialize a list to store the average color for each block
    color_values = []

    # Iterate over the image in blocks
    for i in range(0, height, block_height):
        for j in range(0, width, block_width):
            # Define the block
            block = img_lab[i:i + block_height, j:j + block_width]
            # Calculate the mean color for the block (mean across each channel)
            mean_color = np.mean(block, axis=(0, 1))  # Mean of each channel: L, A, B
            color_values.extend(mean_color)

    return np.array(color_values)



def preprocess_images(image_paths):
    processed_images = []
    for path in tqdm(image_paths):
        img = imread(path)  # Load image
        img_resized = resize(img, IMG_SIZE, anti_aliasing=True)  # Resize
        img_gray = rgb2gray(img_resized)  # Convert to grayscale [0,1]
        processed_images.append(img_gray)
    return np.array(processed_images, dtype=np.float32)  # shape: (n_samples, 64, 64)

def algorithm1(image_path, max_bits_per_feature=5):
    # Step 1: Load and flatten single image
    print("Loading image...")
    img = preprocess_images([image_path])  # shape (1, H, W)
    img_flat = img.reshape(-1)             # shape (n_pixels,)
    n_pixels = img_flat.shape[0]
    mb = max_bits_per_feature

    # Step 2: Compute thresholds per pixel
    print("Computing thresholds per pixel...")
    thresholds_per_pixel = []
    for pixel_idx in range(n_pixels):
        feature_value = img_flat[pixel_idx]
        u = np.array([feature_value])  # single value for this pixel

        if len(u) > mb:
            s = len(u) / mb
            thresholds = [u[int(i * s)] for i in range(1, mb)]
        else:
            thresholds = u[1:] if len(u) > 1 else []

        thresholds_per_pixel.append(np.array(thresholds, dtype=img_flat.dtype))

    # Step 3: Booleanize this image
    print("Booleanizing...")
    bool_features_list = []
    for pixel_idx in range(n_pixels):
        feature_value = img_flat[pixel_idx]
        thresholds = thresholds_per_pixel[pixel_idx]
        if len(thresholds) == 0:
            continue
        bool_cols = (feature_value > thresholds).astype(np.uint8)[None, :]  # shape (1, n_thresholds)
        bool_features_list.append(bool_cols)

    if len(bool_features_list) > 0:
        booleanized_features = np.hstack(bool_features_list)  # shape (1, total_bits)
    else:
        booleanized_features = np.zeros((1, 0), dtype=np.uint8)


    return booleanized_features
def load_model(lib, load_dir, num_classes):
    lib.tm_load.argtypes = [ctypes.c_char_p]
    lib.tm_load.restype = ctypes.c_void_p
    tmachines = []
    for cls in range(num_classes):
        fname = os.path.join(load_dir, f"tm_class{cls}.bin").encode('ascii')
        tm = lib.tm_load(fname)
        if not tm:
            raise RuntimeError(f"tm_load failed for class {cls}")
        tmachines.append(tm)
    return tmachines

def predict_single(lib, tmachines, img, features):
    # Flatten input
    Xi_test = (ctypes.c_int * features)(*img.flatten().tolist())
    # Scores from each class machine
    class_sums = [lib.tm_score(tm, Xi_test) for tm in tmachines]
    return class_sums

def quantile_binning(data, n_bins=4):
    binned_data = []
    for column in data.columns:
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        column_binned = discretizer.fit_transform(data[[column]]).astype(int)
        binned_data.append(column_binned)
    binned_data = np.hstack(binned_data)
    return pd.DataFrame(binned_data, columns=data.columns)


def booleanize(binned_data):
    booleanized_data = []
    column_names = []
    for column in binned_data.columns:
        unique_bins = sorted(binned_data[column].unique())
        for bin_value in unique_bins:
            boolean_column = (binned_data[column] == bin_value).astype(int)
            booleanized_data.append(boolean_column)
            column_names.append(f"{column}_bin_{bin_value}")
    booleanized_data = np.column_stack(booleanized_data)
    return pd.DataFrame(booleanized_data, columns=column_names)


def booleanize_array(num_array, n_bins):
    num_array = pd.DataFrame(num_array)
    num_bin = quantile_binning(num_array, n_bins)
    num_final = booleanize(num_bin)
    return num_final.to_numpy()


def main():
    img_path = input("Path to image: ")
    num_classes = int(input("Number of Classes: "))

    models = {}
    shared_lib = {}
    methods = ["tmachines_hog", "tmachines_3x3", "tmachines_4x4", "tmachines_thresh"]
    bool_data = {}

    print("Select the methods you want to use for predictions (press y/n)")
    for i in methods:
        use = input(f"{i}: ")
        if use == "y":
            lib_path = input("Shared lib path: ")
            ta_states = input("TA state path: ")

            lib = ctypes.CDLL(lib_path)
            models[i] = load_model(lib, ta_states, num_classes)
            shared_lib[i] = lib

            if i == "tmachines_hog":
                bool_data[i] = booleanize_array(extract_hog_features(img_path))
            elif i == "tmachines_3x3":
                bool_data[i] = booleanize_array(color_thermometers(img_path, block_size=(3, 3)))
            elif i == "tmachines_4x4":
                bool_data[i] = booleanize_array(color_thermometers(img_path, block_size=(4, 4)))
            else:
                bool_data[i] = algorithm1(img_path)

    # Aggregate votes
    total_votes = [0] * num_classes
    for i in models.keys():
        feature_len = bool_data[i].shape[1]
        class_sum = predict_single(shared_lib[i], models[i], bool_data[i], feature_len)
        for c in range(num_classes):
            total_votes[c] += class_sum[c]

    print("Aggregated votes per class:", total_votes)
    best_class = max(range(num_classes), key=lambda c: total_votes[c])
    print(f"Predicted class: {best_class}, Votes: {total_votes[best_class]}")

if __name__ == "__main__" :
    main()