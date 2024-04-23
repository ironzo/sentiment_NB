
# Sentiment Analysis using Naive Bayes

## Overview
This project implements a Naive Bayes classifier from scratch to perform sentiment analysis on movie reviews. The dataset used is from [IMDb dataset sentiment analysis in CSV format on Kaggle](https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format).

## Dependencies
- Python 3.x
- NumPy
- Pandas
- NLTK

## Dataset
The dataset includes a training set and a testing set containing movie reviews alongside their sentiment labels (0 or 1).

## Preprocessing
Preprocessing steps include:
- Removing stopwords
- Tokenization
- Stemming

## Model
The Naive Bayes classifier is implemented from scratch. It calculates the probability of each class based on word frequencies and uses these probabilities to predict the sentiment of movie reviews.

## Evaluation
The model achieved an accuracy of 86.06% on the test dataset.

## Usage
1. Install required Python libraries.
2. Download the dataset from Kaggle and place it in the appropriate directory.
3. Run the Jupyter notebook to train the model and make predictions.

## Future Work
Potential improvements include:
- Extending preprocessing steps to include bi-grams or tri-grams.
- Experimenting with different types of smoothing techniques.

## License
This project is open-sourced under the MIT license.
