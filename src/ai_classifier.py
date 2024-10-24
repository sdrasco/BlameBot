# src/ai_classifier.py

import numpy as np
import pandas as pd
import hdbscan
from gensim.models import FastText
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from openai_utils import OpenAIUsageTracker
import re
import ast
import json
from datetime import datetime, timedelta
import os

class AIClassifier:
    def __init__(self, data, openai_tracker, seed=846):
        self.data = data
        self.openai_tracker = openai_tracker
        self.model = None
        self.seed = seed

        # Sort the data to ensure consistent order
        self.data = self.data.sort_values(by=['Description']).reset_index(drop=True)

        # Set random seed for reproducibility
        np.random.seed(self.seed)

        # Define custom keywords for important categories
        self.category_keywords = {
            'travel': ['travel', 'hotel', 'lufthansa', 'klm', 'baonline', 'easyjet', 'air', 'airline', 'scotrail', 'hertz', 'westin', 'booking', 'airport', 'parking', 'swinton'],
            'groceries': ['groceries', 'marks', 'spencer', 'morrisons', 'balgove', 'larder', 'margiotta', 'waitrose', 'boots', 'bowhouse'],
            'utilities': ['octopus', 'energy', 'doorstepglassrecycling', 'starlink', 'talktalk'],
            'alcohol': ['alcohol', 'majestic', 'yapp', 'whisky', 'whiskey', 'yamazaki', 'beer', 'wine', 'gin', 'weisse', 'champagne', 'taittinger'],
            'tax': ['tax', 'fife', 'edinburgh', 'council'],
            'legal': ['legal', 'thorntons', 'burness', 'legl', 'turcan', 'connell'],
            'pets': ['pet', 'vet', 'animal', 'cat', 'litter', 'canin'],
            'entertainment': ['cinema', 'netflix', 'spotify', 'theatre', 'concert'],
            'dining': ['restaurant', 'dining', 'cafe', 'bar', 'pub'],
            'local transport': ['uber', 'lyft', 'taxi', 'bus', 'subway', 'metro'],
            'health': ['doctor', 'pharmacy', 'hospital', 'clinic', 'health'],
            'clothing': ['clothes', 'apparel', 'fashion', 'shoe', 'wear'],
            'technology': ['apple', 'microsoft', 'google', 'openai', 'electronics', 'gadgets', 'tech'],
            'home': ['ikea', 'furniture', 'decor', 'home', 'garden', 'bed'],
            'insurance': ['insurance', 'policy', 'premium'],
            'charity': ['donation', 'charity', 'ngo', 'fundraiser'],
            'investments': ['stock', 'bond', 'investment', 'portfolio', 'dividend'],
            'fees': ['fee', 'charge', 'penalty', 'interest'],
        }

    def train_fasttext_model(self):
        """
        Trains a FastText model on the transaction descriptions.
        """
        sentences = [desc.split() for desc in self.data['Description'].values]
        self.model = FastText(
            sentences,
            vector_size=50,
            window=2,
            min_count=1,
            workers=4,
            seed=self.seed
        )

    def get_weighted_sentence_vector(self, sentence):
        """
        Generates a weighted sentence vector using the FastText model and category keywords.
        """
        words = sentence.split()
        weighted_vectors = []

        for word in words:
            if word in self.model.wv:
                # Default weight is 1 for regular words
                weight = 1
                # Increase weight if the word is a keyword
                for category, keywords in self.category_keywords.items():
                    if word in keywords:
                        weight = 5
                        break  # Stop checking other categories
                # Multiply the word vector by its weight
                weighted_vectors.append(weight * self.model.wv[word])

        # Calculate the average vector for the sentence
        return np.mean(weighted_vectors, axis=0) if weighted_vectors else np.zeros(self.model.vector_size)

    def vectorize_descriptions(self):
        """
        Applies weighted vectorization to the transaction descriptions.
        """
        self.data['FastText_Vector'] = self.data['Description'].apply(self.get_weighted_sentence_vector)
        self.vector_df = pd.DataFrame(self.data['FastText_Vector'].tolist(), index=self.data.index)

    def combine_features(self):
        """
        Combines vectorized descriptions with scaled amounts and date features.
        """
        # Extract date features
        self.data['Year'] = self.data['Date'].dt.year
        self.data['Month'] = self.data['Date'].dt.month
        self.data['Day'] = self.data['Date'].dt.day
        self.data['Hour'] = self.data['Date'].dt.hour
        self.data['DayOfWeek'] = self.data['Date'].dt.weekday  # Monday=0, Sunday=6

        # Normalize 'Hour' as cyclical feature
        self.data['Hour_Sin'] = np.sin(2 * np.pi * self.data['Hour'] / 24)
        self.data['Hour_Cos'] = np.cos(2 * np.pi * self.data['Hour'] / 24)

        # Initialize a scaler
        scaler = MinMaxScaler()

        # Scale date features
        date_features = ['Year', 'Month', 'Day', 'Hour', 'DayOfWeek']
        self.data[date_features] = scaler.fit_transform(self.data[date_features])

        # Combine features
        self.combined_features = pd.concat(
            [self.vector_df, self.data[['Amount_Scaled']], self.data[date_features]],
            axis=1
        ).fillna(0)

        # Normalize the combined features
        combined_features_normalized = normalize(self.combined_features, norm='l2', axis=1)

        # Convert back to DataFrame
        self.combined_features = pd.DataFrame(
            combined_features_normalized,
            columns=self.combined_features.columns,
            index=self.combined_features.index
        )

    def apply_clustering(self, model="HDBSCAN"):
        """
        Applies clustering to the combined features based on the chosen model.
        
        Parameters:
        model (str): The clustering model to apply. Options are "HDBSCAN", "DBSCAN", "KMEANS".
                     Default is "HDBSCAN".
        """

        if model == "HDBSCAN":
            # Apply HDBSCAN Clustering
            clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
            cluster_labels = clusterer.fit_predict(self.combined_features)

        elif model == "DBSCAN":
            # Apply DBSCAN Clustering
            clusterer = DBSCAN(eps=0.045, min_samples=3)
            cluster_labels = clusterer.fit_predict(self.combined_features)

        elif model == "KMEANS":
            # Apply KMEANS Clustering
            SEED = 42  # Should get this from elsewhere
            clusterer = KMeans(n_clusters=30, n_init='auto', random_state=SEED)
            cluster_labels = clusterer.fit_predict(self.combined_features)

        else:
            raise ValueError(f"Invalid model '{model}' chosen. Please choose from 'HDBSCAN', 'DBSCAN', or 'KMEANS'.")

        self.data['Cluster_Label'] = cluster_labels

    def post_clustering_rules(self):
        """
        Applies post-clustering rules to assign categories based on keywords.
        """
        for category, keywords in self.category_keywords.items():
            keyword_mask = self.data['Description'].str.contains('|'.join(keywords), case=False)
            self.data.loc[keyword_mask, 'Category'] = category
            self.data.loc[keyword_mask, 'Description'] = category

    def create_crude_names(self):
        """
        Generates initial crude names for clusters using common keywords.
        """
        crude_names = {}
        for cluster in self.data['Cluster_Label'].unique():
            # Extract descriptions for the current cluster
            cluster_descriptions = self.data[self.data['Cluster_Label'] == cluster]['Description']

            # Use FastText for keyword extraction
            keywords = [
                word for desc in cluster_descriptions for word in desc.split() if word in self.model.wv
            ]
            common_keywords = pd.Series(keywords).value_counts().head(10).index.tolist()
            crude_names[cluster] = ", ".join(common_keywords)

        # Sort crude_names
        self.crude_names = self.sort_dict(crude_names)

    def ai_clarification(self):
        """
        Uses OpenAI's API to refine the category names.
        """
        prompt = (
            "The following are descriptions of the transaction clusters:\n\n"
            f"{self.crude_names}\n\n"
            "Your job is to generate concise, intuitive, and meaningful category names for each cluster. "
            "Each name should clearly reflect the type of spending represented by the transactions. "
            "Please follow these guidelines:\n"
            "- Each category name should be a maximum of two to three words.\n"
            "- For any category name that is already three or fewer words, you can reduce the number of words, but you cannot increase the number of words.\n"
            "- Do not use ambiguous terms such as 'General', 'essentials', 'more', 'misc', 'Miscellaneous', 'Retail', 'Shopping', 'Services', or 'Bills'.\n"
            "- The names should be short enough to fit in a word cloud (i.e., succinct and clear).\n"
            "- Focus on clarity and specificity, making sure the names are easy to understand.\n"
            "Output the results in a Python dictionary format, where each key is the cluster number and each value is the category name, like this:\n"
            "{-1: 'category name', 0: 'another category name', ...}. "
            "Provide only the dictionary in the output, without any additional text."
        )

        # Call the OpenAI API
        response = self.openai_tracker.chat_completion(
                model="gpt-4o",
                messages=[
                        {"role": "system", "content": "You are a financial expert tasked with refining budget category names for different clusters of transactions. "},
                        {"role": "user", "content": prompt}
                ]
            )

        # Extract the response content
        response_text = response.choices[0].message.content

        # Use regex to extract the dictionary from the response
        match = re.search(r'{.*?}', response_text, re.DOTALL)

        if match:
            ai_names_str = match.group(0)
            ai_names = ast.literal_eval(ai_names_str)  # Safely parse the dictionary

            # Sort keys and convert to int
            ai_names = self.sort_dict(ai_names)
            self.crude_names = self.sort_dict(self.crude_names)

            # Make a single dictionary for output to json file
            clarification_data = {
                'AI_names': ai_names,
                'crude_names': self.crude_names
            }

            # Write categories to file
            with open("clarification.json", 'w') as file:
                json.dump(clarification_data, file, indent=4)
                print("\nUpdated category mappings in clarification.json\n")

            self.ai_names = ai_names
        else:
            print("\nFailed to extract budget categories from the response.\n")
            self.ai_names = {}

    def use_ai_clarification(self):
        """
        Checks for existing clarification data or uses AI to clarify category names.
        """
        clarification_file = 'clarification.json'
        use_ai = True

        # Attempt to load and compare cluster names from the JSON file
        if os.path.exists(clarification_file):
            with open(clarification_file, 'r') as file:
                clarification_data = json.load(file)

            # Extract crude_names from the loaded data
            file_crude_names = self.sort_dict(clarification_data.get('crude_names', {}))

            # Keep old AI_names if new and old crude_names match
            if file_crude_names == self.crude_names:
                self.ai_names = self.sort_dict(clarification_data.get('AI_names', {}))
                print("\nUsed budget categories in clarification.json. No need to update.\n")
                use_ai = False

        if use_ai:
            print("Using AI to clarify spending category names.")
            self.ai_clarification()

        # Using the new dictionary, label the transactions
        self.data['Category'] = self.data['Cluster_Label'].map(self.ai_names)

    def sort_dict(self, d):
        """
        Converts dictionary keys to integers and returns a new dictionary sorted by these keys.
        """
        # Convert keys to integers
        int_key_dict = {int(k): v for k, v in d.items()}
        # Sort the dictionary by keys
        return dict(sorted(int_key_dict.items()))

    def process(self):
        """
        Executes all steps to classify transactions and assign categories.
        """
        self.train_fasttext_model()
        self.vectorize_descriptions()
        self.combine_features()
        self.apply_clustering()
        self.post_clustering_rules()
        self.create_crude_names()
        self.use_ai_clarification()