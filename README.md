
# Ensemble Learning-Driven Cyberbullying Detection on Social Media Platforms

## Overview
This project implements an ensemble machine learning approach to detect and classify offensive and non-offensive comments from social media data. The ensemble model combines three classifiers: Large Scale Pinball Twin SVM, Integrated Fuzzy Decision Tree, and Geographical Random Forest to improve the accuracy and robustness of the predictions.

## Table of Contents
- [Project Description](#project-description)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Overview](#model-overview)
- [Sentiment Analysis](#sentiment-analysis)
- [Preprocessing](#preprocessing)
- [Model Building](#model-building)
- [Evaluation](#evaluation)
- [Saving and Loading Models](#saving-and-loading-models)
- [Prediction](#prediction)
- [Results](#results)
- [Performance Metrics](#performance-metrics)
- [Confusion Matrix](#confusion-matrix)
- [ROC Curve](#roc-curve)
- [Word Clouds](#word-clouds)
- [Contact](#contact)

## Project Description
This project detects cyberbullying by classifying social media comments as offensive or non-offensive using text data from a dataset of comments labeled with `tagging` (1 for offensive, 0 for non-offensive). The ensemble learning model is built using various machine learning techniques and achieves high performance based on evaluation metrics such as accuracy, precision, recall, and ROC-AUC.

## Installation
1. Clone the repository or download the code.
2. Install the required libraries using pip:

    ```bash
    pip install -r requirements.txt
    ```

    Required libraries:
    - `numpy`
    - `pandas`
    - `matplotlib`
    - `seaborn`
    - `nltk`
    - `sklearn`
    - `keras`
    - `gensim`
    - `pickle`
    - `wordcloud`
    - `textblob`

3. Mount Google Drive if using Google Colab for loading/saving the dataset and models.

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

## Dataset
The dataset used for this project is stored in Google Drive and consists of 20,001 comments labeled as offensive or non-offensive (`tagging` column). The dataset is cleaned and preprocessed to remove punctuation, stopwords, and perform lemmatization.

Dataset location: `/content/drive/MyDrive/Srinath (shameena)/Suspicious Communication on Social Platforms.csv`

## Model Overview
This project uses an ensemble of three machine learning models:
- **Large Scale Pinball Twin SVM**
- **Integrated Fuzzy Decision Tree**
- **Geographical Random Forest**

The ensemble model aggregates the predictions of each base model using soft voting.

## Sentiment Analysis
A basic sentiment analysis is performed using the TextBlob library to calculate the polarity of the comments. A pie chart visualization shows the distribution of positive and negative sentiments in the dataset.

## Preprocessing
The text data is preprocessed using the following steps:
1. Lowercasing the text.
2. Tokenizing and removing punctuation.
3. Removing stopwords using NLTK.
4. Lemmatizing the tokens based on their POS tags.

Word embeddings are trained using Word2Vec from the preprocessed comments to generate class embeddings for offensive and non-offensive labels.

## Model Building
Three classifiers are used to build the ensemble:
1. **Large Scale Pinball Twin SVM**: A support vector machine with pinball loss function for imbalanced classification.
2. **Integrated Fuzzy Decision Tree**: A decision tree classifier optimized for fuzzy decisions.
3. **Geographical Random Forest**: A random forest model for ensemble decision-making.

These models are combined into a Voting Classifier using soft voting to predict offensive and non-offensive comments.

## Evaluation
Each model is evaluated based on:
- Accuracy
- Precision
- Recall
- Specificity
- ROC-AUC
- Error rate

Confusion matrices and ROC curves are plotted for each model.

## Saving and Loading Models
Models and vectorizers are saved using `pickle` for later use.

```python
# Save models
with open(Parent_path + "/ensemble_model.pkl", 'wb') as f:
    pickle.dump(ensemble_model, f)

# Load models
with open(Parent_path + "/ensemble_model.pkl", "rb") as f:
    ensemble_model = pickle.load(f)
```

## Prediction
After training the model, new text samples are processed and classified as offensive or non-offensive. The following steps are followed:
1. Preprocess the new text data.
2. Vectorize using the pre-trained vectorizer.
3. Predict using the trained ensemble model.

Sample predictions are printed in the console.

## Results
The following metrics were obtained for the ensemble model:
- **Accuracy**: 87.08%
- **Precision**: 77.18%
- **Recall**: 95.43%
- **Specificity**: 81.64%
- **ROC AUC**: 96.53%
- **Error Rate**: 12.92%


## Confusion Matrix
A confusion matrix is plotted for each model to visualize the classification performance.

## ROC Curve
The ROC curve for each model is plotted to analyze the trade-off between the True Positive Rate and False Positive Rate.

## Performance Metrics
The key metrics for evaluating the model are printed for easy reference. The metrics include accuracy, precision, recall, specificity, ROC-AUC, and error rate.
