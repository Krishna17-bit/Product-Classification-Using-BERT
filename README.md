# Product-Classification-Using-BERT

## Introduction
This project demonstrates a prototype machine learning model for classifying dummy products into predefined categories using BERT. The objective is to show the ability to quickly prototype a solution for product classification and basic machine learning principles.

## Data Generation
We generated a dataset of dummy products with attributes such as name, description, price, and category. The data generation process involves creating synthetic data for five categories: Electronics, Clothing, Home Decor, Books, and Sports.

## Data Preprocessing
Performed basic preprocessing steps including tokenization of text attributes and encoding of categorical attributes. The BERT tokenizer was used to encode product descriptions.

## Model Development
Developed a BERT-based text classification model using TensorFlow. The model was trained and evaluated on the generated dataset. Key steps include:

1. Loading the BERT tokenizer and model.
2. Encoding product descriptions.
3. Splitting the data into training and validation sets.
4. Defining and compiling the model.
5. Training the model with early stopping.
6. Evaluating the model's performance.

## Model Evaluation
The model's performance was evaluated using basic evaluation metrics. Below are the visualizations of training and validation loss and accuracy.

## Conclusion
The prototype model demonstrated the ability to classify products into predefined categories with moderate accuracy. There is room for improvement, such as increasing the dataset size, fine-tuning the model, and using more advanced text preprocessing techniques.

## How to Run

### Prerequisites
- Python 3.9 or above
- TensorFlow
- Pandas
- NumPy
- scikit-learn
- transformers
- matplotlib

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
2. **Navigate to the project directory**:

bash
cd product-classification
3.**Install the dependencies**:
pip install -r requirements.txt
4.**Run the data generation script**:
python data_generation.py
5.**Run the model training and evaluation script**:
python model_training.py
### Dependencies
pandas
numpy
scikit-learn
tensorflow
transformers
matplotlib
Files in the Repository
Product-Classification-Using-BERT.ipynb: Script for generating the dummy dataset and training and evaluating the BERT model.
requirements.txt: List of dependencies.
dummy_products.csv: The generated dummy dataset.
README.md: This documentation file.

### Assumptions and Limitations
The dataset is synthetic and may not represent real-world data accurately.
The model was trained for a limited number of epochs for demonstration purposes.
Further fine-tuning and hyperparameter optimization are needed for better performance.


