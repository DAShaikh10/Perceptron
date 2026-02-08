"""
@Author: DAShaikh10
"""

import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import Perceptron


# Read the command-line arguments.
argparser = argparse.ArgumentParser(
    description="Single perceptron implementation capable of fitting over linearly separable dataset."
    "Train, test, and save different models."
)
argparser.add_argument("--epochs", "-e", type=int, default=10, help="Number of training epochs. Defaults to 10.")
argparser.add_argument("--data_path", "-d", type=str, required=True, help="Path to the CSV data file.")
argparser.add_argument(
    "--model_path",
    "-m",
    type=str,
    default="./saved_models/DATA_FILE_NAME_perceptron.pkl",
    help="Path to save the trained model. Defaults to './saved_models/`DATA_FILE_NAME`_perceptron.pkl'.",
)
argparser.add_argument(
    "--verbose", "-v", action="store_true", help="If set, show visualizations before, during, and after training."
)

# Parse the arguments.
args = argparser.parse_args()
epochs = args.epochs
data_file_path = args.data_path
verbose = args.verbose

# Read the dataset file name and use that as the model file name.
file_name = data_file_path.split("/")[-1].split(".")[0]
model_file_path = args.model_path.replace("DATA_FILE_NAME", file_name)

# Load the dataset.
print(f"INFO: Loading dataset from '{data_file_path}' ...")
try:
    data = pd.read_csv(data_file_path)
except FileNotFoundError:
    print(f"ERROR: Data file '{data_file_path}' not found.")
    sys.exit()

print(f"INFO: Dataset loaded. Shape: {data.shape}")

# Separate features (X) and labels (y).
X = data.drop("y", axis=1).to_numpy(dtype=np.int8)  # Drop the label column to get the features.
y = data["y"].to_numpy(dtype=np.int8)  # Get the label column.

input_size = X.shape[1]

# Visualize the dataset.
if verbose and input_size == 2:
    plt.figure(figsize=(6, 4))
    plt.title(f"Input Data: {file_name.upper()}")
    plt.grid(True)

    print("\nINFO: Visualizing Input Data ...")
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", label="Class 0", alpha=0.7)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="red", label="Class 1", alpha=0.7)

    plt.legend(title="Classes")
    plt.xlabel("Feature 1 (x1)")
    plt.ylabel("Feature 2 (x2)")
    plt.xticks(np.arange(0, 2, step=1))
    plt.yticks(np.arange(0, 2, step=1))
    plt.show()

# Instantiate the perceptron model.
# Input size is the number of features in the dataset.
model = Perceptron(input_size)

# Train the model.
model.fit(X, y, epochs=epochs, verbose=verbose)
print(f"\nDEBUG: Weights: {model.weights} (w0 is bias weight)")

# Test the model.
print(r"\nINFO: Testing on all data ...")
CORRECT_COUNT = 0
for inputs, label in zip(X, y):
    prediction = model.predict(inputs)
    if label == prediction:
        CORRECT_COUNT += 1
    print(f"Input: {inputs}, Expectation: {label}, Prediction: {prediction}")

print(f"\nINFO: Accuracy: {CORRECT_COUNT / len(y) * 100:.2f}%")

# Save the trained model.
print(f"\nDEBUG: Saving trained model to '{model_file_path}' ...")
model.save(model_file_path)
print("DEBUG: Model saved successfully.")
