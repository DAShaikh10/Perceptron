"""
@Author: DAShaikh10
"""

import argparse

import numpy as np
import pandas as pd

from models import Perceptron

# Read the command-line arguments.
argparser = argparse.ArgumentParser(description="Test trained single perceptron model(s).")
argparser.add_argument("--data_path", "-d", type=str, required=True, help="Path to the CSV data file.")
argparser.add_argument(
    "--model_path",
    "-m",
    type=str,
    required=True,
    help="Path to the trained model. Supports pickle for loading the model.",
)

# Parse the arguments.
args = argparser.parse_args()
data_file_path = args.data_path
model_file_path = args.model_path

# Load the dataset.
print(f"INFO: Loading dataset from '{data_file_path}' ...")
try:
    data = pd.read_csv(data_file_path)
except FileNotFoundError:
    print(f"ERROR: Data file '{data_file_path}' not found.")
    exit()

print(f"INFO: Dataset loaded. Shape: {data.shape}")

# Separate features (X) and labels (y).
X = data.drop("y", axis=1).to_numpy(dtype=np.float16)  # Drop the label column to get the features.
y = data["y"].to_numpy(dtype=np.float16)  # Get the label column.

# Input size is the number of features in the dataset.
input_size = X.shape[1]

# Instantiate the perceptron model.
model = Perceptron(input_size)

# Load the trained model.
print(f"\nDEBUG: Loading trained model '{model_file_path}' ...")
model.load(model_file_path)
print(f"INFO: Model loaded successfully. Weights: {model.weights} (w0 is bias weight)")

# Test the model.
print(f"\nINFO: Testing on all data ...")
correct_count = 0
for inputs, label in zip(X, y):
    prediction = model.predict(inputs)
    if label == prediction:
        correct_count += 1
    print(f"Input: {inputs}, Expectation: {label}, Prediction: {prediction}")

print(f"\nINFO: Accuracy: {correct_count / len(y) * 100:.2f}%")
