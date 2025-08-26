import struct as st
import numpy as np
import os


# ---------- Helper Functions ----------
def read_idx_images(filename):
    """Read MNIST image files in IDX format (.ubyte)."""
    with open(filename, 'rb') as f:
        # Read magic number
        magic = st.unpack('>I', f.read(4))[0]
        if magic != 2051:
            raise ValueError(f"Magic number mismatch in image file: {magic}")

        n_images = st.unpack('>I', f.read(4))[0]
        n_rows = st.unpack('>I', f.read(4))[0]
        n_cols = st.unpack('>I', f.read(4))[0]

        print(f"Found {n_images} images of size {n_rows}x{n_cols}")

        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(n_images, n_rows, n_cols)
    return images


def read_idx_labels(filename):
    """Read MNIST label files in IDX format (.ubyte)."""
    with open(filename, 'rb') as f:
        # Read magic number
        magic = st.unpack('>I', f.read(4))[0]
        if magic != 2049:
            raise ValueError(f"Magic number mismatch in label file: {magic}")

        n_labels = st.unpack('>I', f.read(4))[0]
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


# ---------- Paths ----------
DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

train_images_file = os.path.join(DATA_DIR, "train-images-idx3-ubyte")
train_labels_file = os.path.join(DATA_DIR, "train-labels-idx1-ubyte")
test_images_file = os.path.join(DATA_DIR, "t10k-images-idx3-ubyte")
test_labels_file = os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte")

# ---------- Load Data ----------
x_train = read_idx_images(train_images_file)
y_train = read_idx_labels(train_labels_file)
x_test = read_idx_images(test_images_file)
y_test = read_idx_labels(test_labels_file)

print("✅ Shapes:")
print("x_train:", x_train.shape, "y_train:", y_train.shape)
print("x_test:", x_test.shape, "y_test:", y_test.shape)

# ---------- Save as NumPy ----------
np.save(os.path.join(DATA_DIR, "x_train.npy"), x_train)
np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
np.save(os.path.join(DATA_DIR, "x_test.npy"), x_test)
np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)

print(f"✅ Saved NumPy arrays into {DATA_DIR}/")
