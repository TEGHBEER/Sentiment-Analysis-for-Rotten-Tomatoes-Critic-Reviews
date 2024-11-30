# Sentiment Analysis for Movie Reviews

## Project Overview

This project focuses on **Sentiment Analysis** of movie reviews sourced from Rotten Tomatoes, utilizing both **Natural Language Processing (NLP)** techniques and **Machine Learning** models. The primary goal is to classify reviews as either **positive** or **negative** based on the review content. We employed two main approaches:

1. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**: A lexicon-based sentiment analysis model.
2. **Logistic Regression**: A machine learning model using **TF-IDF** (Term Frequency-Inverse Document Frequency) for feature extraction.

By analyzing over **957,000 reviews**, the project offers insights into public and critical sentiment toward movies and presents opportunities for automating review classification for use by studios, critics, and audiences.

## Key Features

- **Data Preprocessing**: Handling missing values, encoding sentiment labels, and text preprocessing.
- **Sentiment Prediction**: Leveraging VADER and Logistic Regression models to predict review sentiment.
- **Performance Evaluation**: Models are evaluated using **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
- **Insights**: The project highlights trends in movie review sentiment and provides valuable data-driven insights.

## Dataset Description

The dataset consists of **957,050 movie reviews** from Rotten Tomatoes, including the following key features:
- **critic_name**: Name of the critic who wrote the review.
- **top_critic**: Binary value indicating whether the critic is a "Top Critic."
- **publisher_name**: Name of the publication where the review appeared.
- **review_type**: The sentiment label of the review, either **Fresh** (positive) or **Rotten** (negative).
- **review_score**: Numeric score assigned by the critic.
- **review_date**: Date the review was published.
- **review_content**: The actual text of the review, which serves as the input for sentiment analysis.

The dataset was preprocessed to handle missing values and encoded the sentiment labels (`Fresh` as **1** for positive and `Rotten` as **0** for negative). The cleaned dataset consists of **872,671 reviews** after removing entries with missing or incomplete data.

## Objective

The primary objective of the project is to:
- **Build sentiment prediction models** capable of classifying movie reviews as positive or negative based on their content.
- Apply both **lexicon-based** and **machine learning** approaches and compare their performance.
- Extract meaningful insights into critic and public sentiment through analysis.

## Project Structure

The repository contains the following files:

- **`Sentiment Analysis for Movie Reviews.ipynb`**: Jupyter Notebook containing the entire codebase for data preprocessing, model building, evaluation, and visualization.
- **`Report.pdf`**: A detailed report documenting the project's objectives, methodology, dataset, models used, evaluation metrics, and insights derived from the analysis.
- **`README.md`**: This file providing a comprehensive overview of the project.

## Approach

### 1. Data Preprocessing

- **Handling Missing Values**: Missing entries in critical columns such as `review_content` and `review_type` were handled by dropping incomplete rows.
- **Sentiment Label Encoding**: The `review_type` column was converted into binary values for sentiment analysis:
  - `Fresh` -> 1 (positive)
  - `Rotten` -> 0 (negative)

### 2. Sentiment Analysis Methods

#### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **VADER** is a lexicon and rule-based sentiment analysis tool specifically designed for social media text. We applied VADER to the review text to compute a **compound sentiment score** for each review. Reviews with a compound score ≥ 0 were classified as **positive** (1), and those with a score < 0 were classified as **negative** (0).
- **Accuracy**: The VADER model achieved an accuracy of **64%**.

#### Logistic Regression (Machine Learning)
- **TF-IDF Vectorization**: To convert text into numerical features, we used **TF-IDF vectorization**, which measures the importance of words relative to all reviews.
- **Training the Model**: We trained a Logistic Regression model on **80% of the dataset** and tested its performance on the remaining **20%**.
- **Accuracy**: The Logistic Regression model achieved an accuracy of **74%**.

### 3. Model Evaluation

Both models were evaluated using the following metrics:
- **Accuracy**: The proportion of correct predictions.
- **Precision**: The percentage of true positive predictions among all positive predictions.
- **Recall**: The percentage of true positive predictions among all actual positives.
- **F1-Score**: The harmonic mean of precision and recall, balancing both metrics.

#### Logistic Regression Performance:
- **Precision (Positive Sentiment)**: 0.76
- **Recall (Positive Sentiment)**: 0.86
- **F1-Score (Positive Sentiment)**: 0.80
- **Overall Accuracy**: 74%

#### VADER Performance:
- **Precision (Positive Sentiment)**: 0.67
- **Recall (Positive Sentiment)**: 0.82
- **F1-Score (Positive Sentiment)**: 0.74
- **Overall Accuracy**: 64%

### 4. Comparative Insights

- **Logistic Regression** outperforms **VADER** across all key metrics, showing higher accuracy and better precision, recall, and F1-score.
- While **VADER** is quick and computationally light, it is less accurate than the machine learning approach, particularly in detecting negative reviews.

## Results & Conclusions

- **Logistic Regression** is the preferred model for this sentiment analysis task, offering better overall performance compared to **VADER**.
- The project successfully demonstrates the application of machine learning to text-based sentiment analysis, offering valuable insights for **content creators**, **studios**, and **audiences** to understand public and critical reception of movies.

## Future Work

- **Model Improvement**: Explore other machine learning algorithms such as **Random Forest**, **SVM**, or **Neural Networks** to improve sentiment classification accuracy, especially for negative reviews.
- **Sentiment Trends Analysis**: Further analysis could explore **temporal trends** in sentiment over time or by specific genres or directors.
- **Real-time Sentiment Analysis**: Implement real-time **review scraping** and sentiment analysis for **new movie releases**.


## Requirements

To run the project, ensure you have the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `vaderSentiment`
  - `matplotlib`
  - `seaborn`
  - `nltk`


