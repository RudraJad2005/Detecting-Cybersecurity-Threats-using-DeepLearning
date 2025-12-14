# Detecting Cybersecurity Threats using Deep Learning

This project implements a deep learning model to detect suspicious cybersecurity events based on system logs. It uses the BETH dataset to train a neural network to classify events as suspicious or benign.

## Dataset

The project uses the BETH dataset, which contains the following features:

| Column | Description |
|--------|-------------|
| `processId` | Unique identifier for the process |
| `threadId` | ID for the thread spawning the log |
| `parentProcessId` | Label for the process spawning this log |
| `userId` | ID of user spawning the log |
| `mountNamespace` | Mounting restrictions the process log works within |
| `argsNum` | Number of arguments passed to the event |
| `returnValue` | Value returned from the event log |
| `sus_label` | Binary label (1: suspicious, 0: not suspicious) |

## Project Structure

- `model.ipynb`: Jupyter notebook containing the data loading, preprocessing, model definition, training, and evaluation code.
- `labelled_train.csv`: Training dataset.
- `labelled_test.csv`: Testing dataset.
- `labelled_validation.csv`: Validation dataset.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- torch
- torchmetrics

## Model Architecture

The model is a simple Feedforward Neural Network (MLP) built with PyTorch:
- **Input Layer**: Matches the number of features
- **Hidden Layer 1**: 8 neurons, ReLU activation
- **Hidden Layer 2**: 4 neurons, ReLU activation
- **Output Layer**: 1 neuron, Sigmoid activation (for binary classification)

## Usage

1. Ensure you have the required libraries installed.
2. Open `model.ipynb` in Jupyter Notebook or VS Code.
3. Run the cells to train and evaluate the model.
