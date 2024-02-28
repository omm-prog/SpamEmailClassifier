# SpamEmailClassifier
This Python project implements a Spam Email Classifier using python

# Key Features:

# Data Preprocessing: 
The code cleans and preprocesses the email text, including removing special characters, converting to lowercase, and eliminating stop words.

# Feature Extraction:
The project employs TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction. This technique transforms the raw text into numerical features, capturing the importance of words in the dataset.

# Model Training: 
A Multinomial Naive Bayes classifier is used for training, though the code is flexible enough to try other classifiers available in scikit-learn.

#Model Evaluation: 
The trained model is evaluated using metrics such as accuracy, precision, recall, and F1 score. A confusion matrix provides insights into the classifier's performance.

# Usage:

# Replace the "path of your dataset" with the actual path to your dataset CSV file.
Run the script to train the model and evaluate its performance on a test set.
