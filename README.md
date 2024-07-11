
# Manga vs Comic Image Classification

This project demonstrates the use of a Convolutional Neural Network (CNN) to classify images into Manga and classic comic categories. The model is trained using PyTorch and employs various techniques such as data augmentation and transfer learning to enhance performance.

## Project Structure



## Summary

This project utilizes a CNN to classify images as either Manga or classic comic. The model is built using PyTorch and includes the following components:
- Convolutional layers to extract features from images.
- Max-pooling layers to reduce the dimensionality of feature maps.
- Fully connected layers for classification.
- Dropout layers to prevent overfitting.

The dataset is divided into training, validation, and test sets, with images evenly distributed between Manga and classic comic categories.

## Installation

To run this project, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/samuel29102002/Abgabe_ML_Comic_Manga.git
    cd Abgabe_ML_Comic_Manga
    ```

2. **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the dataset:**
    - Place your dataset in the `data` directory, organized as shown in the project structure.


## Usage

- **Training:** The training script trains the CNN model on the training dataset. The best model is saved in the `models` directory.
- **Validation:** The validation script evaluates the model on the validation dataset and provides performance metrics.
- **Testing:** The testing script assesses the model's performance on the test dataset and generates predictions.

## Visualization

The project includes visualization of training and validation accuracy and loss over epochs. Additionally, GradCAM is used to visualize the model's focus areas in images, providing interpretability to the model's predictions.

## Acknowledgements

- The dataset used in this project is sourced from Kaggle.
- Weights & Biases is used for experiment tracking and hyperparameter tuning.

For any questions or issues, please open an issue on the project's GitHub page.
