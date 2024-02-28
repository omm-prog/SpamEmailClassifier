# Spam Email Classifier

## Overview

This Python script implements a Spam Email Classifier using natural language processing (NLP) techniques and machine learning. The code uses scikit-learn for data preprocessing, feature extraction, and model evaluation. The classifier is trained on a labeled dataset of emails to distinguish between spam and non-spam (ham) messages.

## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/spam-email-classifier.git
   cd spam-email-classifier
   pip install pandas scikit-learn nltk
   python -m nltk.downloader stopwords
   python spam_classifier.py
## Script Explanation

### Data Preprocessing:

- **Reads a CSV dataset:**
  - Reads a CSV dataset containing 'text' and 'label' columns.

### Text Preprocessing:

- **Applies text preprocessing techniques, including:**
  - Removing special characters.
  - Converting text to lowercase.
  - Eliminating stop words.

### Feature Extraction:

- **TF-IDF for Feature Extraction:**
  - Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction.
  - Converts text into numerical features.

### Model Training:

- **Multinomial Naive Bayes Classifier:**
  - Uses the Multinomial Naive Bayes classifier as an example.
  - Splits the data into training and testing sets.
  - Trains the model on the training data.

### Model Evaluation:

- **Predictions and Metrics:**
  - Predicts labels for the test set.
  - Calculates metrics:
    - Accuracy.
    - Precision.
    - Recall.
    - F1 score.

- **Confusion Matrix:**
  - Generates a confusion matrix for detailed evaluation.

    
## Results

- **Accuracy:** 0.9507
- **Precision:** 0.9033
- **Recall:** 0.9249
- **F1 Score:** 0.9140
- **Confusion Matrix:**
Where:
[[713 29]
[ 22 271]]

  
