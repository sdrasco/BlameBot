import sys
import subprocess
import os
import re
import ast
import json
import hdbscan
import base64
import pdfkit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from openai import OpenAI
from bs4 import BeautifulSoup
from glob import glob
from gensim.models import FastText
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from wordcloud import WordCloud
from datetime import datetime, timedelta
SEED = 846 # random seed

class AccountProcessor:
    def __init__(self, data_directory):
        """
        Initialize with the directory containing account CSV files.
        """
        self.data_directory = data_directory
        self.data = None

    def load_and_validate_data(self):
        """
        Load CSV files, use the first file's columns as the standard, 
        validate subsequent files, and combine into a single DataFrame.
        """
        # Find all the statements
        csv_files = glob(os.path.join(self.data_directory, '*.CSV'))
        csv_files += glob(os.path.join(self.data_directory, '*.csv'))

        # Initialize an empty list
        dataframes = []

        # get header from the first file (check later for concistancy)
        if csv_files:
            first_file = pd.read_csv(csv_files[0])
            expected_columns = first_file.columns.tolist() 
            dataframes.append(first_file)  # Add the first file's DataFrame

            # Loop through the remaining statements
            for file in csv_files[1:]:
                # Read statement
                df = pd.read_csv(file)

                # Verify the columns
                if list(df.columns) == expected_columns:
                    dataframes.append(df)
                else:
                    print(f"File {file} has unexpected columns: {df.columns}")

        # Concatenate statements
        if dataframes:
            self.data = pd.concat(dataframes, ignore_index=True)
        else:
            raise ValueError("No valid DataFrames to concatenate after validation.")

    def clean_date(self, DateCol="Date", UK_style=True):
        """
        Build the date feature. 

        DateCol is the column name for the dates that we are converting ("Date" by dfault).

        UK_style describes the input data (True by default, should be 
        made False for accounts with US conventions)
        
        Convert dates to ISO standard (YYYY-MM-DD).

        Remove all transactions that preceed a starting cutoff date
        """
        self.data["Date"] = pd.to_datetime(self.data[DateCol], dayfirst=UK_style)  
        cutoff_date = datetime(2022, 10, 1).date()
        date_mask = self.data['Date'].dt.date >= cutoff_date
        self.data = self.data[date_mask].copy()

    def clean_description(self, ColNames):
        """
        Construct the Description feature.

        ColNames is the list of columns that we will keep, merged and normalized into Description.
        
        Merge and normalize all the text descriptors.
        """
        self.data['Description'] = self.data[ColNames].fillna('').astype(str).agg(' '.join, axis=1)
        self.data['Description'] = self.data['Description'].str.lower()

    def chop_up(self):
        #Split large sales into more frequent multiple smaller
        amount_column = 'Amount'
        max_amount=1000
        split_size=100
        new_rows = []

        for _, row in self.data.iterrows():
            amount = row[amount_column]
            if amount > max_amount:
                # Calculate the number of splits needed
                num_splits = int(amount // split_size)
                remainder = amount % split_size

                # Create rows for the splits
                for _ in range(num_splits):
                    new_row = row.copy()
                    new_row[amount_column] = split_size
                    new_rows.append(new_row)

                # Add the remainder if there's any
                if remainder > 0:
                    new_row = row.copy()
                    new_row[amount_column] = remainder
                    new_rows.append(new_row)
            else:
                new_rows.append(row)

        # Convert the list of rows back into a DataFrame
        self.data = pd.DataFrame(new_rows)

    def scale_amount(self):
        # scale sale amounts to create the Amount_Scaled feature
        scaler = StandardScaler()
        self.data['Amount_Scaled'] = scaler.fit_transform(np.log1p(self.data['Amount']).values.reshape(-1, 1)) 

    def process(self, DateCol, UK_style, DescriptionColumnNames):
        """
        Execute the complete processing sequence specific to UK bank statements.
        """
        # get and merge all the statements
        self.load_and_validate_data()

        # clean
        self.clean_date(DateCol=DateCol, UK_style=UK_style)
        self.clean_description(ColNames=DescriptionColumnNames)
        self.clean_amount()
        
        # Split large sales into more frequent multiple smaller ones
        self.chop_up()

        # scale sale amounts to create the Amount_Scaled feature
        self.scale_amount()

        # drop all the columns we don't need
        columns_to_keep = ['Date', 'Description', 'Amount', 'Amount_Scaled']
        self.data = self.data[columns_to_keep]
        return self.data
        
    def summarize(self):
        """
        Create a summary of the data.
        """
        earliest_date = self.data['Date'].min()
        latest_date = self.data['Date'].max()
        date_range = latest_date - earliest_date
        years = date_range.days / 365.25
        total_sales = self.data['Amount'].sum()
        
        summary = {
            'Date Range': f"{years:.2f} years",
            'Earliest Date': earliest_date,
            'Latest Date': latest_date,
            'Sales': len(self.data),
            'Total Sales': f"{total_sales:.0f}" 
        }
        summary_df = pd.DataFrame([summary])
        return summary_df

class uk_bank(AccountProcessor):
    def __init__(self, data_directory):
        """
        Initialize with the directory containing UK bank statements.
        """
        super().__init__(data_directory)

    def clean_amount(self):
        """
        Construct the Amount feature.
        
        Make sales positive, convert to USD.
        
        Also split large events up into many smaller ones to boost their frequency. 
        This helps clustering find categories that are rare but significant in size.
        """
        # remove deposits or the like, and make sales positive
        deposit_mask = (self.data['Amount'] <= 0) & (self.data['Category'] != 'Transfers')
        self.data = self.data[deposit_mask].copy()
        if 'Amount' in self.data.columns:
            self.data['Amount'] = self.data['Amount'].abs()
            
        # Convert GBP values to USD using our converter, and update currency label. 
        # This converts them to dollars at the time of the transactions using an exchange rate table
        # update daily. We could updated every minute, as we have minute-level data in statements,
        # but may then need a new exange rate source (current using yahoo finance API output, stored to file).
        self.converter = GBPtoUSD()
        self.data[['Amount_USD', 'Currency']] = self.data.apply(
            lambda row: pd.Series([
                self.converter.convert(row['Amount'], row['Date']) if row['Currency'] == 'GBP' else row['Amount'],
                'USD' if row['Currency'] == 'GBP' else row['Currency']
            ]), axis=1
        ) 

class CreditCardUS(AccountProcessor):
    def __init__(self, data_directory):
        """
        Initialize with the directory containing US credit card statements.
        """
        super().__init__(data_directory)

    def clean_amount(self):
        """
        Build Amount feature.
        
        Remove rows where the 'type' is 'Payment' or 'Reversal'.
        
        Make all sales positive.
        """
        # remove deposits or the like, and make sales positive
        deposit_mask = (self.data['Amount'] <= 0) & (self.data['Type'] != 'Reversal') & (self.data['Type'] != 'Payment')
        self.data = self.data[deposit_mask].copy()
        if 'Amount' in self.data.columns:
            self.data['Amount'] = self.data['Amount'].abs()

class GBPtoUSD:
    def __init__(self, data_file='gbp_usd_daily_data.csv'):
        self.data_file = data_file
        self.exchange_data = self._load_or_download_data()

    def _load_or_download_data(self):
        # If the CSV file exists, load it and get any necessary updates
        if os.path.exists(self.data_file):
            data = pd.read_csv(self.data_file, parse_dates=['Date'], index_col='Date')

            # Get the most recent date in the data
            most_recent_date = data.index.max().date()

            # Get today's date
            today = datetime.today().date()

            # Our FX data source doesn't update on the weekend.
            # If it's currently a weekend, change today's reference date to the most recent Thursday.
            if today.weekday() == 5:  # Saturday
                today -= timedelta(days=2)
            elif today.weekday() == 6:  # Sunday
                today -= timedelta(days=3)

            # If the most recent date in the file is not up to date, download missing data
            if most_recent_date < today:
                print(f"Exchange rate data is outdated. Most recent date in the file: {most_recent_date}")
                # Download missing data starting from the day after the most recent date
                start_date = most_recent_date + timedelta(days=1)
                new_data = yf.download('GBPUSD=X', interval='1d', start=start_date, end=today + timedelta(days=1))
                
                # If new data is available, append it to the existing data
                if not new_data.empty:
                    new_data = new_data[['Close']]  # Keep only the 'Close' column
                    new_data.index = pd.to_datetime(new_data.index)  # Ensure index is datetime
                    data = pd.concat([data, new_data])
                    data.to_csv(self.data_file, index=True)
                    print(f"Exchange rate data updated with new entries up to {today}.")
                else:
                    print("No new exchange data available for update.")
        else:
            # If the file does not exist, download the full dataset
            data = yf.download('GBPUSD=X', interval='1d', start='2016-01-01', end=datetime.today().date() + timedelta(days=1))
            data = data[['Close']]  # Keep only the 'Close' column
            data.to_csv(self.data_file, index=True)
            print(f"Exchange rate data downloaded and saved to '{self.data_file}'.")
        
        return data
    
    def convert(self, gbp_amount, date_time):
        """
        Convert a GBP amount to USD based on the given date using the closing price of the day.

        :param gbp_amount: Amount in GBP
        :param date_time: Date and time of the transaction (string format 'YYYY-MM-DD HH:MM:SS')
        :return: Equivalent amount in USD
        """

        # Convert the input date_time string to a pandas Timestamp object
        dt = pd.to_datetime(date_time).date()  # Use only the date part

        # Check if the date exists in the index and use the close rate for the day
        if dt in self.exchange_data.index:
            close_rate = self.exchange_data.loc[dt, 'Close']
        else:
            # Use nearest available rate if exact date is not found
            idx = self.exchange_data.index.get_indexer([dt], method='nearest')
            nearest_date = self.exchange_data.index[idx[0]]
            close_rate = self.exchange_data.loc[nearest_date, 'Close']

        # Calculate USD amount using the close rate
        usd_amount = gbp_amount * close_rate
        return usd_amount

class AmazonProcessor:
    def __init__(self, data, data_directory):
        """
        Initialize the processor with data and data directory.
        
        :param data: Main statements data (unused in this context but retained for compatibility)
        :param data_directory: Directory where data files are located
        """
        self.statements = data
        self.digital_file = glob(os.path.join(data_directory, 'Digital Items.csv'))
        self.retail_file = glob(os.path.join(data_directory, 'Retail.OrderHistory.2.csv'))
        
        # Load data
        self.retail = pd.read_csv(self.retail_file[0])
        self.digital = pd.read_csv(self.digital_file[0])
        
        # Initialize the GBP to USD converter
        self.GBPtoUSD = GBPtoUSD()
    
    def clean_retail(self):
        """
        Import and clean the amazon retail order history.
        """

        # Parse 'Ship Date' to datetime
        self.retail['Ship Date Parsed'] = pd.to_datetime(
            self.retail['Ship Date'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce'
        )

        # Parse 'Order Date' to datetime
        self.retail['Order Date Parsed'] = pd.to_datetime(
            self.retail['Order Date'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce'
        )

        # remove old transactions
        cutoff_date = datetime(2019, 1, 1).date()
        date_mask = (self.retail['Order Date Parsed'].dt.date >= cutoff_date) & (self.retail['Ship Date Parsed'].dt.date >= cutoff_date)
        self.retail = self.retail[date_mask].copy()
       
        # Drop unwanted columns and rename the ones we keep
        self.retail = self.retail[['Ship Date Parsed', 'Order Date Parsed', 'Order ID', 'Product Name', 'Total Owed']].copy()
        self.retail.rename(columns={
            'Ship Date Parsed': 'Ship Date',
            'Order Date Parsed': 'Order Date',    
            'Order ID': 'ID',
            'Product Name': 'Description',
            'Total Owed': 'Amount'
        }, inplace=True)

        # Create two datasets: one with 'Date' as 'Ship Date', one with 'Date' as 'Order Date'
        retail_ship_date = self.retail.copy()
        retail_order_date = self.retail.copy()
        retail_ship_date['Date'] = retail_ship_date['Ship Date']
        retail_order_date['Date'] = retail_order_date['Order Date']
        
        # Combine the datasets
        self.retail = pd.concat([retail_ship_date, retail_order_date], ignore_index=True)
        
        # Drop the 'Ship Date' and 'Order Date' columns to have only 'Date'
        self.retail.drop(columns=['Ship Date', 'Order Date'], inplace=True)

        # Convert all amounts from GBP to USD using the 'Date' column
        self.retail['Amount'] = self.retail.apply(
            lambda row: self.GBPtoUSD.convert(row['Amount'], row['Date']),
            axis=1
        )

        # scale sale amounts
        scaler = StandardScaler()
        self.retail['Amount_Scaled'] = scaler.fit_transform(np.log1p(self.retail['Amount']).values.reshape(-1, 1))  

    def clean_digital(self):
        """
        Import and clean the amazon digital order history.
        """
        
        # Parse 'OrderDate' to datetime, and remove orders that pre-date statements
        self.digital['OrderDate Parsed'] = pd.to_datetime(
            self.digital['OrderDate'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce'
        )
        cutoff_date = min(self.statements['Date'])  # Convert cutoff_date to a Timestamp
        date_mask = self.digital['OrderDate Parsed'] >= cutoff_date
        self.digital = self.digital[date_mask].copy()
       
        # Convert GBP to USD wherever needed
        gbp_mask = self.digital['BaseCurrencyCode'] == 'GBP'
        gbp_rows = self.digital[gbp_mask].copy()
        self.digital.loc[gbp_mask, 'OurPriceTax USD'] = gbp_rows.apply(
            lambda row: self.GBPtoUSD.convert(row['OurPriceTax'], row['OrderDate Parsed']),
            axis=1
        )

        # Drop unwanted columns and rename the ones we keep
        self.digital = self.digital[['OrderDate Parsed', 'OrderId', 'Title', 'OurPriceTax USD']].copy()
        self.digital.rename(columns={
            'OrderDate Parsed': 'Date',
            'OrderId': 'ID',
            'Title': 'Description',
            'OurPriceTax USD': 'Amount'
        }, inplace=True)

        # scale sale amounts
        scaler = StandardScaler()
        self.digital['Amount_Scaled'] = scaler.fit_transform(np.log1p(self.digital['Amount']).values.reshape(-1, 1))  

    def cross_reference(self):
        """
        Loop through all retail and digital Amazon orders. When a match is found:
        
        1) Delete the row of self.statements that is nearest in terms of fractional difference.
        2) Add the Amazon orders whose sum matched the statement['Amount'] to the statements frame, excluding the 'ID' column.
        
        Some statement transactions will be broken into several, as Amazon describes each item
        but statements describe each multi-item shipment.
        """
        
        # Define the regex pattern for matching statement Descriptions to Amazon orders
        pattern = r'amazon|amz|prime'
        
        # Perform the search to identify Amazon-related statement rows
        amazon_statements_mask = self.statements['Description'].str.contains(pattern, regex=True, case=False, na=False)
        
        # Count the number of Amazon-related statement rows
        num_matching_rows = amazon_statements_mask.sum()
        
        # Initialize counters and lists for tracking
        indices_to_delete = []  # To store indices of statements to delete
        rows_to_add = []        # To store Amazon orders to add to statements
        
        # Merge digital and retail orders
        orders = pd.concat([self.retail, self.digital], ignore_index=True)
        
        # Ensure 'Date' is in datetime format
        orders['Date'] = pd.to_datetime(orders['Date'])
        self.statements['Date'] = pd.to_datetime(self.statements['Date'])
        
        # Iterate over each unique Date in orders
        unique_days = orders['Date'].dt.date.unique()
        for day in unique_days:

            # Ensure 'day' is a datetime.date object
            if isinstance(day, pd.Timestamp):
                day = day.date()
            elif isinstance(day, np.datetime64):
                day = pd.to_datetime(day).date()

            # Filter orders for the current date
            orders_on_day = orders[orders['Date'].dt.date == day]
            
            # Iterate over each unique ID for the current date
            unique_ids = orders_on_day['ID'].unique()
            for order_id in unique_ids:
                # Sum the 'Amount' for the current Date and ID
                summed_amount = orders_on_day[orders_on_day['ID'] == order_id]['Amount'].sum()
                
                # Define the date range (±2 days)
                date_min = day - timedelta(days=2)
                date_max = day + timedelta(days=2)

                # Find matching statements in self.statements within the date range
                matching_statements = self.statements[
                    amazon_statements_mask &
                    (self.statements['Date'].dt.date >= date_min) &
                    (self.statements['Date'].dt.date <= date_max)
                ]
                
                if matching_statements.empty:
                    continue  # No statements to match on this day
                
                # Calculate the absolute relative difference between statement amounts and summed_amount
                relative_diff = (matching_statements['Amount'] - summed_amount).abs() / summed_amount
                
                # Set a threshold for relative difference (e.g., 15%)
                threshold = 0.1
                close_matches = matching_statements[relative_diff < threshold]
                
                if not close_matches.empty:
                    # Find the statement with the smallest relative difference
                    closest_match_idx = relative_diff[relative_diff < threshold].idxmin()
                    closest_diff = relative_diff[closest_match_idx]
                    
                    # Record the index to delete
                    indices_to_delete.append(closest_match_idx)
                    
                    # Extract the Amazon orders corresponding to this order_id
                    amazon_orders = orders_on_day[orders_on_day['ID'] == order_id].copy()
                    
                    # Drop the 'ID' column
                    amazon_orders = amazon_orders.drop(columns=['ID'])
                    
                    # Append the Amazon orders to the list of rows to add
                    rows_to_add.append(amazon_orders)
                    
                    #print(f"Match found for Order ID {order_id} on {day}: Statement Index {closest_match_idx} with relative diff {closest_diff:.2f}")
        
        # Delete the matched statement rows from self.statements
        if indices_to_delete:
            self.statements = self.statements.drop(index=indices_to_delete).reset_index(drop=True)
            print(f"\nDeciphered {100*len(indices_to_delete)/num_matching_rows:2.0f}% of Amazon transaction descriptions ({len(indices_to_delete)} out of {num_matching_rows}).\n")
        else:
            print("No matching amazon transactions.")
        
        # Concatenate all Amazon orders to add
        if rows_to_add:
            amazon_orders_to_add = pd.concat(rows_to_add, ignore_index=True)
            
            # Append the Amazon orders to self.statements
            self.statements = pd.concat([self.statements, amazon_orders_to_add], ignore_index=True)
        
        # scale again, should eliminate some of the earlier ones
        scaler = StandardScaler()
        self.statements['Amount_Scaled'] = scaler.fit_transform(np.log1p(self.statements['Amount']).values.reshape(-1, 1))  

        # Rename any Amazon transactions that we couldn't match to just Amazon
        amazon_statements_mask = self.statements['Description'].str.contains(pattern, regex=True, case=False, na=False)
        self.statements.loc[amazon_statements_mask, 'Description'] = 'Amazon'

    def process(self):
        """
        Get digitial and retail amazon order history, then clean it and cross-reference 
        with statements. 

        Returns:
        - The modified statements DataFrame.
        """
        
        # import and clean the order histories
        self.clean_digital()
        self.clean_retail()
        self.cross_reference()
        
        return self.statements

class AIClassifier:
    def __init__(self, data):
        self.data = data
        
        # Sort the data to ensure consistent order
        self.data = self.data.sort_values(by=['Description']).reset_index(drop=True)

        # Being extra cautious about setting random seeds that I may not have noticed.
        np.random.seed(SEED)

        # Define custom keywords for important categories
        category_keywords = {
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
            'technology': ['apple', 'microsoft', 'google', 'openai', 'chatgpt', 'electronics', 'gadgets', 'tech'],
            'home': ['ikea', 'furniture', 'decor', 'home', 'garden', 'bed'],
            'insurance': ['insurance', 'policy', 'premium'],
            'charity': ['donation', 'charity', 'ngo', 'fundraiser'],
            'investments': ['stock', 'bond', 'investment', 'portfolio', 'dividend'],
            'fees': ['fee', 'charge', 'penalty', 'interest'],
        }

        # Train FastText model on the descriptions
        sentences = [desc.split() for desc in self.data['Description'].values]
        model = FastText(
            sentences, 
            vector_size=50, 
            window=2, 
            min_count=1, 
            workers=8,
            seed=SEED
        )  # Adjust parameters as needed

        # Create sentence vectors by giving higher weight to keyword-defined categories
        def get_weighted_sentence_vector(sentence, model, category_keywords):
            words = sentence.split()
            weighted_vectors = []

            for word in words:
                if word in model.wv:
                    # Default weight is 1 for regular words
                    weight = 1  
                    # Loop through the category_keywords dictionary to give extra weight to keywords
                    for category, keywords in category_keywords.items():
                        if word in keywords:
                            weight = 5
                            sentence_weight = 5
                        
                    # Multiply the word vector by its weight
                    weighted_vectors.append(weight * model.wv[word])
            
            # Calculate the average vector for the sentence
            return np.mean(weighted_vectors, axis=0) if weighted_vectors else np.zeros(model.vector_size)

        # Apply weighted vectorization to the descriptions
        self.data['FastText_Vector'] = self.data['Description'].apply(lambda x: get_weighted_sentence_vector(x, model, category_keywords))
        self.vector_df = pd.DataFrame(self.data['FastText_Vector'].tolist(), index=self.data.index)

        # Append the Date and Ammount_Scaled features onto the word vectors, then scale the vectors
        self.combine_features()

        # Apply HDBSCAN Clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)  

        # Apply DBSCAN Clustering (alternative to HDBSCAN)
        #clusterer = DBSCAN(eps=0.045, min_samples=3)  # Adjust eps based on your data

        # Apply KMEANS Clustering (alternative to HDBSCAN)
        # clusterer = KMeans(n_clusters=30, n_init='auto', random_state=SEED)

        # get cluster labels
        cluster_labels = clusterer.fit_predict(self.combined_features)

        # Apply post-clustering rules to force certain keywords into predefined categories
        def apply_post_clustering_rules(data, category_keywords):
            for category, keywords in category_keywords.items():
                keyword_mask = data['Description'].str.contains('|'.join(keywords), case=False)
                data.loc[keyword_mask, 'Category'] = category  # Override previous category
                data.loc[keyword_mask, 'Description'] = category  # Override previous category
            return data
        self.data = apply_post_clustering_rules(self.data, category_keywords)

        # apply cluster labels
        self.data['Cluster_Label'] = cluster_labels

        # Create crude set of new cluster names from the FastText model
        crude_names = {}
        for cluster in self.data['Cluster_Label'].unique():
            # Extract descriptions for the current cluster
            cluster_descriptions = self.data[self.data['Cluster_Label'] == cluster]['Description']

            # Use FastText for keyword extraction (adjust as needed)
            keywords = [word for desc in cluster_descriptions for word in desc.split() if word in model.wv]
            common_keywords = pd.Series(keywords).value_counts().head(10).index.tolist()
            crude_names[cluster] = ", ".join(common_keywords)

        # Sort crude_names 
        crude_names = self.sort_dict(crude_names)

        # Get or create names that have been clarified by AI
        self.use_ai_clarification(crude_names)

    def combine_features(self):
        """
        Extracts and scales date features from self.data['Date'], combines them with vectorized
        descriptions and scaled amounts, normalizes the combined features, and prepares the data
        for clustering.
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

        # Combine vectorized descriptions, scaled amounts, and scaled date features
        self.combined_features = pd.concat([self.vector_df, self.data[['Amount_Scaled']], self.data[date_features]], axis=1).fillna(0)

        # Save column names and index before normalization
        column_names = self.combined_features.columns
        index_values = self.combined_features.index

        # Normalize the high dimensional vectors before clustering
        combined_features_normalized = normalize(self.combined_features, norm='l2', axis=1)

        # Convert the result back to a DataFrame with the saved column names and index
        self.combined_features = pd.DataFrame(combined_features_normalized, columns=column_names, index=index_values)

        # Convert all column names to strings to ensure compatibility with HDDBSCAN
        self.combined_features.columns = self.combined_features.columns.astype(str)

    def ai_clarification(self, crude_names):
        """
        Calls the OpenAI API to suggest concise and intuitive budget category names
        for the provided cluster descriptions. Extracts and returns the resulting 
        budget categories as a dictionary.

        Parameters:
        - crude_names (dict): A dictionary of poorly named categories.

        Returns:
        - budget_categories (dict): A dictionary of improved budget category names.
        """
                
        # Define the prompt to clarify category names.  Will send this to OpenAI API.
        prompt = (
            "You are a financial expert tasked with refining budget category names for different clusters of transactions. "
            "The following are descriptions of the transaction clusters:\n\n"
            f"{crude_names}\n\n"
            "Your job is to generate concise, intuitive, and meaningful category names for each cluster. "
            "Each name should clearly reflect the type of spending represented by the transactions. "
            "Please follow these guidelines:\n"
            "- Each category name should be a maximum of two to three words.\n"
            "- For any category name that is already three or fewer words, you can reduce the number of words, but you cannot increase the number of words.\n"
            "- Do not use ambiguous terms such as 'General', 'essentials','more', 'misc', 'Miscellaneous', 'Retail', 'Shopping', 'Services', or 'Bills'.\n"
            "- The names should be short enough to fit in a word cloud (i.e., succinct and clear).\n"
            "- Focus on clarity and specificity, making sure the names are easy to understand.\n"
            "Output the results in a Python dictionary format, where each key is the cluster number and each value is the category name, like this:\n"
            "{-1: 'category name', 0: 'another category name', ...}. "
            "Provide only the dictionary in the output, without any additional text."
        )

        # Call the OpenAI API and show usage
        client = OpenAI()
        response = client.chat.completions.create(
                model='gpt-4',
                messages=[
                        {"role": "user", "content": prompt}
                ]
            )
        
        # Extract the response content
        response_text = response.choices[0].message.content

        # Use regex to extract the dictionary from the response
        match = re.search(r'{.*?}', response_text, re.DOTALL)

        if match:
            AI_names_str = match.group(0)
            AI_names = ast.literal_eval(AI_names_str)  # Safely parse the dictionary
            
            # Sort keys and convert to int
            AI_names = self.sort_dict(AI_names)
            crude_names = self.sort_dict(crude_names)
            
            # Make a single dictionary for output to json file
            clarification_data = {
                'AI_names': AI_names,
                'crude_names': crude_names
            }

            # Write categories to file
            with open("clarification.json", 'w') as file:
                json.dump(clarification_data, file, indent=4)
                print("\nUpdated category mappings in clarification.json\n")
            return AI_names
        else:
            print("\nFailed to extract budget categories from the response.\n")
            return {}

    def use_ai_clarification(self, crude_names):
        """
        Method to handle checking for existing clarification data, and calling AI clarification if necessary.
        """
        clarification_file = 'clarification.json'

        # Initialize AI clarification flag
        USEAI = True

        # Attempt to load and compare cluster names from the JSON file
        if os.path.exists(clarification_file):
            with open(clarification_file, 'r') as file:
                clarification_data = json.load(file)

            # Extract crude_names from the loaded data
            file_crude_names = self.sort_dict(clarification_data.get('crude_names', {}))

            # Keep old AI_names if new and old crude_names match
            if file_crude_names == crude_names:
                AI_names = self.sort_dict(clarification_data.get('AI_names', {}))
                print("\nUsed budget categories in clarification.json.  No need to update.\n")
                USEAI = False

        if USEAI:
            print("Using AI to clarify spending category names.")
            AI_names = self.ai_clarification(crude_names)

        # Using the new dictionary, label the transactions
        self.data['Category'] = self.data['Cluster_Label'].map(AI_names)

    def sort_dict(self, d):
        """
        Converts dictionary keys to integers and returns a new dictionary sorted by these keys.

        Parameters:
        - d (dict): The original dictionary with string keys.

        Returns:
        - dict: A new dictionary with integer keys, sorted in ascending order.
        """
        # Convert keys to integers
        int_key_dict = {int(k): v for k, v in d.items()}
        # Sort the dictionary by keys
        return dict(sorted(int_key_dict.items()))

def shame_cloud(classifier_data, exclude_category=None, output_file=None):
    """
    Generates and optionally saves a word cloud based on the spending categories from the classifier data.

    Parameters:
    - classifier_data (pd.DataFrame): The output DataFrame from the classifier, containing 'Category' and 'Amount' columns.
    - exclude_category (str): Category to exclude from the word cloud (default is None).
    - output_file (str, optional): Filename to save the word cloud image. If None, the file will not be saved.

    Returns:
    - None: Displays the word cloud and optionally saves it as an image file.
    """
    # Group by 'Category' and sum the 'Amount' for each category
    category_totals = classifier_data.groupby('Category')['Amount'].sum()

    # Remove specified category from the totals
    category_totals = category_totals[category_totals.index != exclude_category]

    # Convert the series to a dictionary for the word cloud
    category_dict = category_totals.to_dict()

    # Generate the word cloud with adjustments for readability
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='gray', 
        colormap='Reds', 
        max_font_size=150,       # Increase max font size
        min_font_size=12,        # Set a minimum font size
        max_words=75,            # Limit the number of words to the most significant categories
        scale=6,                 # Increase scale for higher resolution
        normalize_plurals=False
    ).generate_from_frequencies(category_dict)

    # Plot the word cloud
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off axis lines and labels

    # Save the figure as a PNG file if a filename was given
    if output_file:
        plt.savefig(output_file, format='png', dpi=150, bbox_inches='tight', pad_inches=0)

    # close the word cloud
    plt.close()

    # Sort category_dict by amount and take the top categories
    top_categories = [category for category, amount in sorted(category_dict.items(), key=lambda x: x[1], reverse=True)[0:5]]
    
    # Convert the top categories into a readable summary format
    spending_habits = ", ".join(top_categories)

    # describe the family doing the spending
    family_description = "an american husband (tech guy) and wife (professor) who emigrated to scotland. They have two cats (one brown, one grey-tuxedo)."

    # Combine the family description and spending categories into the prompt
    prompt = f"A comical cartoon image depicting {family_description} The image should reflect a lifestyle in which they spend all their money on {spending_habits}. **no words**"
    
    # Ask GPT-4 to refine the prompt for DALL-E 3
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": "You are an expert at creating prompts for DALL-E 3 image generation."},
            {"role": "user", "content": f"Refine this prompt for DALL-E 3: {prompt}"}
        ]
    )

    # Extract the clarified prompt
    clarified_prompt = response.choices[0].message.content
    
    # Call the OpenAI DALL-E-3API
    response = client.images.generate(
        model="dall-e-3",
        prompt=clarified_prompt,
        n=1,
        size="1024x1024",
        quality="standard",
        response_format="url"
    )

    # Save the image
    image_url = response.data[0].url

    # If using a base64 response:
    # import base64
    # with open("family.png", "wb") as image_file:
    #     image_file.write(base64.b64decode(response['data'][0]['b64_json']))
        
    # If using a URL:
    import requests
    img_data = requests.get(image_url).content
    with open('family.png', 'wb') as handler:
        handler.write(img_data)

    print("Image saved as family.png")

def build_reports(data):

    # Convert 'Date' column to datetime if it's not already
    data['Date'] = pd.to_datetime(data['Date'])

    # Create a 'Month' column formatted as 'YYYY-MM'
    data['Month'] = data['Date'].dt.to_period('M')

    # Group by 'Month' and sum 'Amount'
    monthly_sums = data.groupby('Month')['Amount'].sum()

    # Reset index to ensure 'Month' is a column for plotting
    monthly_sums = monthly_sums.reset_index()

    # Convert 'Month' back to string for better plotting labels
    monthly_sums['Month'] = monthly_sums['Month'].astype(str)

    # Save bar chart 
    plt.figure(figsize=(12, 6))
    plt.bar(monthly_sums['Month'], monthly_sums['Amount'], color='skyblue')
    #plt.ylabel('USD')
    plt.yticks([])  # This removes the tick labels
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('monthly_sums.png', dpi=150, bbox_inches='tight')
    # Close the figure to avoid displaying it
    plt.close()

    # Calculate total and average spending
    total_spending = data['Amount'].sum()
    average_monthly_spending = monthly_sums['Amount'].mean()
    highest_spending_month = monthly_sums.loc[monthly_sums['Amount'].idxmax(), 'Month']

    # Identify top expense categories
    top_categories = data['Category'].value_counts().head(5).index.tolist()

    # Group by 'Category' and sum the 'Amount' for each category
    category_sums = classified.data.groupby('Category')['Amount'].sum()
    category_sums = category_sums.sort_values(ascending=False)

    # Calculate the number of days covered
    num_days = (data['Date'].max() - data['Date'].min()).days + 1  # Include both start and end dates

    # Prepare summary dictionary
    data_summary = {
        'Number of days covered': f"{num_days}",
        'Total Spending': f"${total_spending:,.0f}",
        'Average Monthly Spending': f"${average_monthly_spending:,.0f}",
        'Highest Spending Month': highest_spending_month,
        'Top Expense Categories': ', '.join(top_categories),
        'Spending per Category': category_sums
    }

    prompt = f"""
    Assemble a financial report for my family. Show off your dry witt in the report. Structure your report as follows:

    1. **Summary**
       - Describe the nature of this report
       - Include some or all of these details of the data considered, using a clear table:
       - Date Range: {data_summary['Number of days covered']} days
       - Total Spending: {data_summary['Total Spending']}
       - Average Monthly Spending: {data_summary['Average Monthly Spending']}
       - Highest Spending Month: {data_summary['Highest Spending Month']}
       - Spending per Category: {data_summary['Spending per Category']} (no need to show all categories)
       - Display the image 'family.png', a portrait of this family depicting their lifestyle, and talk about it.

    2. **Spending Analysis**
       - Do an analysis of the spending data in the summary, carefully looking for trends or events.
       - Display the image 'shame_cloud.png', the word cloud of shame built out of the spending category names and amounts, and talk about it
       - Describe the findings of your analysis.
       - Display the image 'monthly_sums.png', a bar chart of amounts spent each month, and talk about it
       - Don't use the image filenames in the text

    3. **Projections for Annual Costs**
       - Based on current spending trends, provide projections for annual costs. Consider factors such as potential inflation, lifestyle changes, or other likely cost changes.
       
    4. **Suggested Budget by Category**
       - Propose a more concise annual and weekly budget, consolidated to five spending categories.
       - Include a table with totals at the bottom row

    6. **Sustainability Outline**
       - Provide an assessment of the income needed to sustain the suggested budget, including pre- and post-tax amounts, stating the assumed tax rates.
    
    7. **Investment support**   
        - Explain how investments could help. Say what amount invested in medium risk securities would reduce the income needed by half.

    End the report with a footer containing a thumbnail of your image 'BlameBot_small.png' 
    that links to https://blamebot.com/ when clicked. To the right of the thumbnail, put a pearl of self wisdom about family finance in your signature self depricating dry-humor style.

    ### Design Guidelines:
    - Use a minimalist, modern layout (e.g. clean, large headers, concise sections in boxes with rounded corners and ample white space)
    - All content should be confined to the central 80% of the screen width.  
    - Ensure all numbers (such as amounts) are formatted appropriately (e.g., currency with commas, round to whole numbers).
    - Color scheme: Background should be a soft orange, Boxes should be a soft light blue, headers should be a soft red. Text should be a soft black.  
    - The colors should be muted, subtle, soft, and calming.

    ### HTML Output Requirements:
    - Provide the HTML code **without any markdown or code block formatting** (e.g., no ```html or ``` around the code).
    - Use appropriate HTML5 elements (`<section>`, `<header>`, `<table>`, etc.) to structure the document.
    - Use mathjax for equations
    - Include basic inline CSS for layout and typography
    - The images should be referenced with `<img>` tags
    - All text should be wrapped in `<p>`, `<h1>`, `<h2>`, or `<div>` tags, ensuring proper hierarchy

    Please generate the report as a single HTML document with embedded CSS. **Do not include any additional text at all outside of the HTML code.**
    """

    # Generate rough report using OpenAI's API
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are BlameBot a clever wealth manager who likes to show off their dry wit."},
            {"role": "user", "content": prompt}
        ],
    )

    # Extract the generated HTML code for parseing/polishing
    rough_report = response.choices[0].message.content

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(rough_report, 'html.parser')

    # Make sure soup object has a valid <head> element
    if soup.head is None:
        # Create a <head> tag if it doesn't exist
        soup.head = soup.new_tag('head')
        soup.insert(0, soup.head)  # Insert <head> at the beginning of the document

    # Add the Google Fonts link to the <head> section
    font_link_tag = soup.new_tag('link', href="https://fonts.googleapis.com/css2?family=Work+Sans:wght@400;600&display=swap", rel="stylesheet")
    soup.head.append(font_link_tag)

    # Add the MathJax script to the <head> section
    mathjax_script_tag = soup.new_tag('script', src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js")
    mathjax_script_tag.attrs['type'] = "text/javascript"
    mathjax_script_tag.attrs['async'] = True
    soup.head.append(mathjax_script_tag)

    # Add the font-family style to the <body> tag
    if 'style' in soup.body.attrs:
        soup.body['style'] += " font-family: 'Work Sans', Arial, sans-serif;"
    else:
        soup.body['style'] = "font-family: 'Work Sans', Arial, sans-serif;"

    # Function to convert images to Base64
    def convert_image_to_base64(image_path):
        """Converts an image to a Base64 encoded string."""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

    # Base64 encoded images
    image_map = {
        'monthly_sums.png': convert_image_to_base64('monthly_sums.png'),
        'shame_cloud.png': convert_image_to_base64('shame_cloud.png'),
        'BlameBot_small.png': convert_image_to_base64('BlameBot_small.png')
    }

    # process all image content
    for img_tag in soup.find_all('img'):
        src_value = img_tag.get('src')
        # embed Base64 images
        if src_value in image_map:
            img_tag['src'] = f"data:image/png;base64,{image_map[src_value]}"
        # refit the images, round corners, and center
        if 'style' in img_tag.attrs:
            img_tag['style'] += "border-radius: 10px;"
        else:
            img_tag['style'] = "border-radius: 10px;"
        img_tag['style'] += " max-width: 80%; height: auto;"
        img_tag['style'] += " display: block; margin: 0 auto;"
        
    # make sure paragraph text is left-justified
    for p_tag in soup.find_all('p'):
        if 'style' in p_tag.attrs:
            p_tag['style'] += " text-align: left;"
        else:
            p_tag['style'] = "text-align: left;"

    # Write the soup object to an .html file
    with open('financial_report.html', 'w') as file:
        file.write(str(soup))  
    print("Report written to 'financial_report.html'.")

    # Redact dollar amounts
    for td in soup.find_all("td"):
        if "$" in td.text:
            td.string = "[redacted]"
    for p in soup.find_all("p"):
        p.string = re.sub(r"\$\d+(?:,\d{3})*(?:\.\d{2})?", "[redacted]", p.text)

    # Save the modified content to a new HTML file
    output_path = "financial_report_redacted.html"
    with open(output_path, "w") as file:
        file.write(str(soup))
    print("Redacted report written to 'financial_report_redacted.html'.")

    # # Convert the HTML files to PDF if desired (these aren't so pretty, consider ditching)
    # html_files = [
    #     ('financial_report.html', 'financial_report.pdf'),
    #     ('financial_report_redacted.html', 'financial_report_redacted.pdf')
    # ]
    # options = {'enable-local-file-access': ''}
    # for input_html, output_pdf in html_files:
    #     pdfkit.from_file(input_html, output_pdf, options=options)
    #     print(f"PDF version of report written to '{output_pdf}'.")

#####################################################################
#                                                                   #
# Done with classes and methods. Main execution script begins here. #
#                                                                   #
#####################################################################

# Set the statement directories
us_cc_directory = '../data/us_credit_card_statements/'
uk_bank_directory = '../data/uk_bank_statements/'
amzn_directory = '../data/Amazon/'

# Process the statements
uscc = CreditCardUS(us_cc_directory)
#clean_uscc = uscc.process(DateCol='Transaction Date', UK_style=False, DescriptionColumnNames=['Description', 'Category'])
clean_uscc = uscc.process(DateCol='Transaction Date', UK_style=False, DescriptionColumnNames=['Description'])
print("\nUS credit card summary:\n")
print(uscc.summarize())
ukbank = uk_bank(uk_bank_directory)
#clean_ukbank = ukbank.process(DateCol='Date', UK_style=True, DescriptionColumnNames=['Name', 'Address', 'Description', 'Category'])
clean_ukbank = ukbank.process(DateCol='Date', UK_style=True, DescriptionColumnNames=['Name', 'Description'])
print("\nUK bank account summary:\n")
print(ukbank.summarize())

# merge the processed statements
statements = pd.concat([clean_uscc, clean_ukbank], ignore_index=True)

# cross reference amazon data
amzn = AmazonProcessor(statements, amzn_directory)
cleaned_df = amzn.process()

# apply the classifier
classified = AIClassifier(cleaned_df)

# show the shame cloud
shame_cloud(classified.data,output_file="shame_cloud.png")

classified.data['Category'].value_counts()

# Get the number of unique values in the 'Category' column
num_unique_categories = classified.data['Category'].nunique()
print(f"Defined {num_unique_categories} spending categories.")

# Group by 'Category' and sum the 'Amount' for each category
category_sums = classified.data.groupby('Category')['Amount'].sum()

# Sort and round the summed amounts in descending order
category_sums = category_sums.sort_values(ascending=False)
category_sums = category_sums.round().astype(int)

# Display the sorted summed amounts for each category
pd.set_option('display.max_rows', None)
print(f"\n{category_sums}\n")

# make the reports
build_reports(classified.data)


