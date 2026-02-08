"""
@Author: DAShaikh10
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

from activations import HeavisideStepActivation


class Perceptron:
    """
    A single-layer perceptron implementation.
    Capable of fitting over linearly separable datasets. Fits the model using the Perceptron Learning Algorithm (PLA).
    Uses a heaviside step activation function.

    We use the following notation throughout the code:
    - Inputs: x = [x0, x1, x2, ..., xn]
    - Weights: w = [w0, w1, w2, ..., wn] where w0 is the bias weight. We initialize the weights as a zero vector.
    - Weighted sum: z = w0 + w1*x1 + w2*x2 + ... + wn*xn
    - Output: y = activate(z)
    """

    def __init__(self, input_size: int) -> None:
        # We start with random set of weights. For simplicity, we initialize them to zero here.
        self.weights = np.zeros(input_size + 1, dtype=np.int8)  # +1 for the bias weight

        # We use the heaviside step activation function.
        self._activate = HeavisideStepActivation.activate

    def fit(self, train_data, labels, epochs: int = 10, verbose: bool = True) -> None:
        """
        Train the perceptron using the Perceptron Learning Algorithm (PLA).
        We use the heaviside step activation function here.
        Optionally, visualize the decision boundary after each epoch for 2D data and the error history after training.

        Args:
            train_data: The input training data (features).
            labels: The corresponding labels for the training data.
            epochs: Number of epochs to train the model.
            verbose: If True, show visualizations during training.

        Returns:
            None
        """

        error_history = []
        visualize = verbose and train_data.shape[1] == 2
        if visualize:
            plt.ion()

        print(f"\nINFO: Starting training (Epochs: {epochs}) ...")
        for epoch in range(epochs):
            num_errors = 0
            for x, label in zip(train_data, labels):
                # Make a prediction.
                prediction = self.predict(x)

                # Calculate the error.
                error = self.loss(prediction, label)
                if error != 0:
                    num_errors += 1

                    # Update weights based on the prediction error.
                    self.update_weights(x, error)

            print(f"DEBUG: Epoch {epoch + 1}/{epochs}, Number of errors: {num_errors}")

            if visualize:
                plot_title = f"Epoch: {epoch + 1}/{epochs} | Errors: {num_errors}"
                # Record the number of errors for this epoch for visualization.
                error_history.append(num_errors)
                self.plot_decision_boundary(train_data, labels, title=plot_title)
                plt.pause(0.5)

            if num_errors == 0:
                print("\nINFO: Training complete - Early Stoppage! - no errors found.")
                if visualize:
                    plt.pause(0.5)
                break

        # Plot the error history after training.
        if visualize:
            plt.ioff()
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, len(error_history) + 1), error_history, marker="o")
            plt.title("Perceptron Training Progress (Errors per Epoch)")
            plt.xlabel("Epoch")
            plt.ylabel("Number of Misclassifications")
            plt.xticks(np.arange(1, len(error_history) + 1, step=1))
            plt.yticks(np.arange(0, len(train_data) + 1, step=1))
            plt.grid(True)
            plt.show()

    def load(self, file_path: str) -> None:
        """
        Load the model weights from a file. We use pickle for simplicity.

        Args:
            file_path: The path to the file from which the weights will be loaded.

        Returns:
            None
        """

        with open(file_path, "rb") as file:
            self.weights = pickle.load(file)

    def loss(self, predicted, actual) -> np.int8:
        """
        Simple difference loss function.
        `loss = actual - predicted`

        Args:
            predicted: The predicted output from the model.
            actual: The actual target output.

        Returns:
            The difference between actual and predicted values.
        """

        return actual - predicted

    def predict(self, inputs) -> np.int8:
        """
        Make a prediction for the given inputs.
        The prediction is made by calculating the weighted sum of inputs and applying the activation function.
        `prediction = activate(w0*1 + w1*x1 + ... + wn*xn)`

        Args:
            inputs: The input features for which to make the prediction.

        Returns:
            The predicted output (0 or 1).
        """

        # Calculate the weighted sum i.e. dot product of the inputs and weights.
        # For higher dimension matrices, matrix multiplication with transposed weights is better.
        # Alternatively, we can write `np.dot(self.weights[1:], inputs)`.
        weighted_sum = self.weights[0] + self.weights[1:] @ inputs

        # Apply the activation function to get the final prediction.
        return self._activate(weighted_sum)

    def save(self, file_path: str) -> None:
        """
        Save the model weights to a file. We use pickle for simplicity.

        Args:
            file_path: The path to the file where the weights will be saved.

        Returns:
            None
        """

        with open(file_path, "wb") as file:
            pickle.dump(self.weights, file)

    def update_weights(self, inputs, error) -> None:
        """
        Update the weights based on the prediction error.
        The logic is based on the Perceptron Learning Rule:
        If error != 0:
            w0 = w0 + 1
            wi = wi + xi for all i
        Else:
            w0 = w0 - 1
            wi = wi - xi for all i
        Args:
            inputs: The input features.
            error: The prediction error (actual - predicted).

        Returns:
            None
        """

        if error > 0:
            # Update bias weight.
            # Bias weight update rule: w0 = w0 + 1. Since bias input is always 1.
            self.weights[0] += 1

            # Update other weights. Note that we are performing vector addition here.
            # Weight update rule: w = w + x
            self.weights[1:] += inputs
        else:
            # Update bias weight.
            # Bias weight update rule: w0 = w0 - 1. Since bias input is always 1.
            self.weights[0] -= 1

            # Update other weights. Note that we are performing vector subtraction here.
            # Weight update rule: w = w - x
            self.weights[1:] -= inputs

    def plot_decision_boundary(self, X, y, title=None) -> None:
        """
        Plots the decision boundary for 2D data.
        This function will only work if the data has 2 features.

        Args:
            X: The input data (features).
            y: The corresponding labels.
            title: Optional title for the plot.

        Returns:
            None
        """

        plt.clf()

        # Get weights.
        w0, w1, w2 = self.weights

        # Plot data points.
        plt.scatter(X[y == 0, 0], X[y == 0, 1], color="blue", label="Class 0", alpha=0.7)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color="red", label="Class 1", alpha=0.7)

        # Calculate the decision line.
        x1_min = np.min(X[:, 0]) - 1
        x1_max = np.max(X[:, 0]) + 1
        x_values = np.linspace(x1_min, x1_max, 50)

        if w2 != 0:
            y_values = (-w1 * x_values - w0) / w2
            plt.plot(x_values, y_values, color="green", linestyle="--", label="Decision Boundary")
        elif w1 != 0:
            vertical_line_x = -w0 / w1
            plt.axvline(x=vertical_line_x, color="green", linestyle="--", label="Decision Boundary")

        plt.title(title or "Perceptron Decision Boundary (2D)")
        plt.xlabel("Feature 1 (x1)")
        plt.ylabel("Feature 2 (x2)")
        plt.legend()
        plt.grid(True)
        plt.xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
        plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
