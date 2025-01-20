# Digit Recognition using CNN

This project aims to build a Convolutional Neural Network (CNN) model to classify handwritten digits (0-9) from the MNIST dataset. The goal is to train a deep learning model that can predict the digit labels from images of handwritten numbers.

## Table of Contents
- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Data Description](#data-description)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview
This project uses the MNIST dataset, a collection of 28x28 pixel grayscale images of handwritten digits. The objective is to preprocess the data, train a CNN model, and evaluate its performance. The model is then used to predict the labels of unseen test images.

## Tech Stack
- **Python 3.x**
- **TensorFlow/Keras** for deep learning
- **NumPy** for data manipulation
- **Pandas** for data processing
- **Matplotlib/Seaborn** for data visualization
- **Scikit-learn** for machine learning utilities
- **OpenCV** for image processing (optional)

## Getting Started

To run the code in this repository, you will need the following:

### Prerequisites
- Python 3.x
- Jupyter Notebook
- Required libraries:
  - TensorFlow
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn

You can install the necessary packages using the following command:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

### Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/digit-recognition-cnn.git
cd digit-recognition-cnn
```

### Open the Jupyter Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `index.ipynb` and follow the steps outlined in the notebook to run the project.

## Usage

1. **Data Preprocessing**  
   - The dataset is loaded from a CSV file.
   - Each image is reshaped to 28x28 pixels and normalized.
   - The data is split into training and validation sets.

2. **Model Training**  
   - A Convolutional Neural Network (CNN) is built and compiled with Adam optimizer and categorical cross-entropy loss.
   - Data augmentation is used to improve generalization during training.

3. **Model Evaluation**  
   - The model's performance is evaluated on the validation set, with metrics such as accuracy and loss plotted.
   
4. **Prediction**  
   - The trained model is used to make predictions on the test dataset.
   - The results are saved to a CSV file for submission.

## Data Description

The dataset consists of the following:
- **Train Set**: 60,000 grayscale images of handwritten digits (28x28 pixels each) with corresponding labels (0-9).
- **Test Set**: 10,000 grayscale images without labels, used for evaluation.

## Model Architecture

The CNN model consists of the following layers:
1. **Input Layer**: 28x28 grayscale images.
2. **Conv2D Layer 1**: 64 filters, 3x3 kernel, ReLU activation.
3. **BatchNormalization**: Normalizes output from the previous layer.
4. **MaxPooling2D**: Reduces the spatial dimensions.
5. **Dropout**: Prevents overfitting.
6. **Conv2D Layer 2**: 128 filters, 3x3 kernel, ReLU activation.
7. **MaxPooling2D**: Reduces the spatial dimensions.
8. **Flatten Layer**: Converts 2D matrix to a 1D vector.
9. **Dense Layer**: 512 units, ReLU activation.
10. **Dropout**: Prevents overfitting.
11. **Output Layer**: 10 units (for digits 0-9) with softmax activation.

## Results

The model achieved a validation accuracy of **99.56%** after training, demonstrating strong performance on handwritten digit recognition tasks.

### Sample Accuracy and Loss Plots
- Training and validation accuracy are plotted over epochs.
- Training and validation loss are plotted over epochs.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and create a pull request. We are open to suggestions for improvements, bug fixes, and feature additions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the creators of the MNIST dataset for providing a well-curated dataset for handwritten digit recognition.
- Thanks to TensorFlow and Keras for their excellent deep learning frameworks.
