# fairsight.py
FairSight: A Python framework for detecting and mitigating bias in large language model datasets.


# FairSight: An Adaptive Bias Detection and Mitigation Framework for Large Language Model Datasets

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Download necessary NLTK resources (if not already downloaded)
try:
    nltk.data.find('sentiment/')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class FairSight:

    def __init__(self, data):
        self.data = data
        self.sia = SentimentIntensityAnalyzer()
        self.vectorizer = TfidfVectorizer(max_features=5000)  # Example: limit features

    def detect_representation_bias(self, demographic_column):
        """Detects representation bias in a demographic column."""
        counts = self.data[demographic_column].value_counts()
        print("Representation Bias:")
        print(counts)
        plt.figure(figsize=(8, 6))
        sns.barplot(x=counts.index, y=counts.values)
        plt.title(f"Distribution of {demographic_column}")
        plt.show()

    def detect_sentiment_bias(self, text_column, demographic_column):
        """Detects sentiment bias towards different demographic groups."""
        sentiments = []
        for text in self.data[text_column]:
            sentiments.append(self.sia.polarity_scores(text)['compound'])
        self.data['sentiment'] = sentiments
        sentiment_by_group = self.data.groupby(demographic_column)['sentiment'].mean()
        print("\nSentiment Bias:")
        print(sentiment_by_group)
        plt.figure(figsize=(8, 6))
        sns.barplot(x=sentiment_by_group.index, y=sentiment_by_group.values)
        plt.title(f"Average Sentiment by {demographic_column}")
        plt.show()

    def mitigate_representation_bias(self, demographic_column, target_distribution=None):
        """Mitigates representation bias by resampling."""
        if target_distribution is None:
            target_distribution = self.data[demographic_column].value_counts().max()
        groups = self.data[demographic_column].unique()
        balanced_data = pd.DataFrame()
        for group in groups:
            group_data = self.data[self.data[demographic_column] == group]
            if len(group_data) < target_distribution:
                balanced_group = group_data.sample(target_distribution, replace=True, random_state=42)
            else:
                balanced_group = group_data.sample(target_distribution, random_state=42)
            balanced_data = pd.concat([balanced_data, balanced_group])
        return balanced_data

    def evaluate_model_fairness(self, text_column, demographic_column, balanced_data=None):
        """Evaluates fairness of a model trained on the data."""
        if balanced_data is None:
            data = self.data
        else:
            data = balanced_data

        X = self.vectorizer.fit_transform(data[text_column])
        y = data[demographic_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"\nModel Accuracy: {accuracy}")

        # Example: Simple fairness check (can be expanded)
        from sklearn.metrics import classification_report
        print(classification_report(y_test, model.predict(X_test), zero_division=0))

# Example Usage (replace with your data)
data = pd.DataFrame({
    'text': [
        "This is a great product.",
        "I hate this.",
        "It's okay.",
        "Excellent!",
        "Terrible experience.",
        "Good job.",
        "The food was bad.",
        "This is amazing!",
        "It was alright.",
        "I love this!"
    ],
    'gender': ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female'],
    'race': ['white','black','white','black','white', 'black', 'white', 'black', 'white', 'black']
})

fair_sight = FairSight(data)

# Detect and mitigate representation bias
fair_sight.detect_representation_bias('gender')
balanced_data = fair_sight.mitigate_representation_bias('gender')
print("\nBalanced Data:")
print(balanced_data['gender'].value_counts())

# Detect sentiment bias
fair_sight.detect_sentiment_bias('text','gender')

# Evaluate model fairness
fair_sight.evaluate_model_fairness('text', 'gender')
fair_sight.evaluate_model_fairness('text', 'gender', balanced_data)

fair_sight.detect_representation_bias('race')
fair_sight.detect_sentiment_bias('text', 'race')
balanced_race_data = fair_sight.mitigate_representation_bias('race')
fair_sight.evaluate_model_fairness('text', 'race', balanced_race_data)
