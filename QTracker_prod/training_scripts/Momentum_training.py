
import uproot
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam  # Use legacy Adam optimizer for M1/M2 Mac compatibility
import argparse
import shutil
import glob
import os
import sys
import re

os.makedirs("models/MOMENTA/", exist_ok=True)
ith_Model = None

def get_next_ith_model():
    # Find all existing mup and mum model files
    model_files = glob.glob("models/MOMENTA/mom_model_*_*.h5")
    existing_pairs = set()

    pattern = re.compile(r'mom_model_(mup|mum)_(\d+)\.h5$')

    for f in model_files:
        match = pattern.search(os.path.basename(f))
        if match:
            tag = match.group(1)
            index = int(match.group(2))
            if index not in existing_pairs:
                existing_pairs.add(index)

    # Now search for the next available index where neither file exists
    ith_Model = 0
    while True:
        mup_path = f"models/MOMENTA/mom_model_mup_{ith_Model}.h5"
        mum_path = f"models/MOMENTA/mom_model_mum_{ith_Model}.h5"
        if not os.path.exists(mup_path) and not os.path.exists(mum_path):
            return ith_Model
        ith_Model += 1

ith_Model = get_next_ith_model()
print(f"Next model number (clean slot for mup and mum): {ith_Model}")

# Function to load data from multiple ROOT files
def load_data(root_files):
    hit_arrays_list = []
    targets_list = []
    
    for root_file in root_files:
        with uproot.open(root_file) as file:
            tree = file["tree"]

            # Read in HitArray (with drift distances) and target variables
            hit_arrays = np.array(tree["HitArray"].array(library="np"), dtype=np.float32)
            gpx = tree["gpx"].array(library="ak").to_numpy().astype(np.float32)
            gpy = tree["gpy"].array(library="ak").to_numpy().astype(np.float32)
            gpz = tree["gpz"].array(library="ak").to_numpy().astype(np.float32)

            # Ensure they have the same number of samples
            assert hit_arrays.shape[0] == gpx.shape[0] == gpy.shape[0] == gpz.shape[0], \
                f"Shape mismatch in {root_file}: HitArray={hit_arrays.shape}, gpx={gpx.shape}, gpy={gpy.shape}, gpz={gpz.shape}"

            # Zero out irrelevant slots (both elementID and driftDistance)
            hit_arrays[:, 7:12] = 0     # unused station-1
            hit_arrays[:, 55:58] = 0    # DP-1
            hit_arrays[:, 59:62] = 0    # DP-2

            # Ensure driftDistance is zeroed out where elementID is zero
            hit_arrays[hit_arrays[:, :, 0] == 0, 1] = 0  # Zero out driftDistance where elementID is zero
            # Stack targets properly
            targets = np.column_stack((gpx, gpy, gpz))

            # Append to lists
            hit_arrays_list.append(hit_arrays)
            targets_list.append(targets)
    
    # Concatenate all data
    X = np.concatenate(hit_arrays_list, axis=0)
    y = np.concatenate(targets_list, axis=0)
    
    return X, y

# Build the model to accept a single 3D input
def build_model(input_shape):
    # Input layer for the 3D array
    input_layer = Input(shape=input_shape)

    # Flatten the input to pass through dense layers
    x = Flatten()(input_layer)

    # Dense layers
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)

    # Output layer for gpx, gpy, gpz
    output = Dense(3, activation="linear")(x)

    # Define the model
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model

# Train the model
def train_model(root_files, output_h5, epochs=100, batch_size=32):
    # Load data
    X, y = load_data(root_files)

    # Build model
    model = build_model(input_shape=(62, 2))  # Input shape is (62, 2)

    # Train model
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

    # Save model
    # Extract base and extension
    base_name, ext = os.path.splitext(os.path.basename(output_h5))
    save_path = f"models/MOMENTA/{base_name}_{ith_Model}{ext}"
    model.save(save_path)
    print(f"Model also saved to {save_path}")


    #Making a copy of the code.
    script_name = sys.argv[0]
    backup_dir = "models/MOMENTA/BackUps"

    os.makedirs(backup_dir, exist_ok=True)
    base_name = os.path.basename(script_name)
    backup_name = f"models/MOMENTA/BackUps/{os.path.splitext(base_name)[0]}_{ith_Model}.py"  # Append ith_model before the extension

    shutil.copy(script_name, backup_name)
    print(f"Backup created: {backup_name}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a DNN to predict gpx, gpy, gpz from HitArray.")
    parser.add_argument("input_root_1", type=str, help="Path to the first input ROOT file (e.g., mup).")
    parser.add_argument("input_root_2", type=str, help="Path to the second input ROOT file (e.g., mum).")

    parser.add_argument("--output", type=str, default="./models/MOMENTA/mom_model", help="Base name of the output model file.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Train with first file → mup model
    train_model(
        [args.input_root_1],
        args.output + "_mup.h5",
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Train with second file → mum model
    train_model(
        [args.input_root_2],
        args.output + "_mum.h5",
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    print(f"Next model number: {ith_Model}")


    
