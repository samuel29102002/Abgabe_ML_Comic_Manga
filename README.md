
# Manga vs Comic Image Classification

This project demonstrates the use of a Convolutional Neural Network (CNN) to classify images into Manga and classic comic categories. The model is trained using PyTorch and employs various techniques such as data augmentation and transfer learning to enhance performance.

## Project Structure

```bash
Abgabe_ML_Comic_Manga
│   README.md
│   requirements.txt
│   best_model.pth
│   checkpoint.pt
│   main.ipynb
│   .gitattributes
│
├───data
│   ├───train
│   ├───validation
│   └───test
│
├───doku
│   ├───Comic-Manga-PR.pdf
│   └───ML.pptx
│
├───graphics
│   ├───confusionmatrix.png
│   ├───data.png
│   ├───gradcam1.png
│   └───gradcam2.png

```

## Summary

This project utilizes a CNN to classify images as either Manga or classic comic. The model is built using PyTorch and includes the following components:
- Convolutional layers to extract features from images.
- Max-pooling layers to reduce the dimensionality of feature maps.
- Fully connected layers for classification.
- Dropout layers to prevent overfitting.

The dataset is divided into training, validation, and test sets, with images evenly distributed between Manga and classic comic categories.

## Directories and Files

- **data/**: Contains the dataset used for training, validation, and testing.
  - **test/**: Directory containing the test dataset.
  - **train/**: Directory containing the training dataset.
  - **validation/**: Directory containing the validation dataset.

- **doku/**: Documentation related to the project.
  - **Report_ML.pdf**: PDF document detailing the Comic-Manga project.
  - **ML.pptx**: PowerPoint presentation about the project.

- **grafics/**: Directory containing graphical outputs from the project.
  - **confusionmatrix.png**: Confusion matrix image generated from the model's performance.
  - **data.png**: Image showing the dataset distribution.
  - **gradcam1.png**: GradCAM visualization of the first sample.
  - **gradcam2.png**: GradCAM visualization of the second sample.

- **.gitattributes**: Git attributes file for managing text file attributes.

- **best_model.pth**: Saved state of the best-performing model during training.

- **checkpoint.pt**: Checkpoint file used to save intermediate training states of the model.

- **main.ipynb**: Jupyter Notebook containing the main code for the project, including data preprocessing, model training, and evaluation.


- **requirements.txt**: List of required Python packages and their versions needed to run the project.


## Installation

To run this project, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/samuel29102002/Abgabe_ML_Comic_Manga.git
    cd Abgabe_ML_Comic_Manga
    ```

2. **Create a virtual environment:**
    ```bash
    python3.9 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```


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
