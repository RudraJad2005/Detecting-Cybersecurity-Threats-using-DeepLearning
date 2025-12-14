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

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Detecting-Cybersecurity-Threats-using-DeepLearning.git
   cd Detecting-Cybersecurity-Threats-using-DeepLearning
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   - **Windows:**
     ```bash
     .venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

4. **Install the required packages:**
   ```bash
   pip install pandas scikit-learn torch torchmetrics
   ```

5. **Launch Jupyter Notebook or open in VS Code:**
   ```bash
   jupyter notebook model.ipynb
   ```
   Or open the project folder in VS Code and run the notebook cells.

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
