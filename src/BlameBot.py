import sys
import subprocess
import os
import re
import ast
import json
import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from gensim.models import FastText
import hdbscan
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud
import yfinance as yf
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
        """
        self.data["Date"] = pd.to_datetime(self.data[DateCol], dayfirst=UK_style)  

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

    def process(self):
        """
        Execute the complete processing sequence specific to UK bank statements.
        """
        # get and merge all the statements
        self.load_and_validate_data()

        # clean
        self.clean_date(DateCol="Date", UK_style=True)
        self.clean_description()
        self.clean_amount()
        
        # drop all the columns we don't need
        columns_to_keep = ['Date', 'Description', 'Amount', 'Amount_Scaled']
        self.data = self.data[columns_to_keep]
        return self.data

    def clean_description(self):
        """
        Construct the Description feature.
        
        Merge and normalize all the text descriptors.
        """
        #self.data['Description'] = self.data[['Category','Name', 'Address', 'Description']].fillna('').astype(str).agg(' '.join, axis=1)
        self.data['Description'] = self.data[['Name', 'Address', 'Description']].fillna('').astype(str).agg(' '.join, axis=1)
        self.data['Description'] = self.data['Description'].str.lower()
        
    def clean_amount(self):
        """
        Construct the Amount feature.
        
        Make sales positive, convert to USD, and scale (standard scaling on log-transformed).
        
        Also split large events up into many smaller ones to boost their frequency. 
        This helps clustering find categories that are rare but significant in size.
        """
        # remove deposits or the like, and make sales positive
        deposit_mask = (self.data['Amount'] <= 0) & (self.data['Category'] != 'Transfers')
        self.data = self.data[deposit_mask].copy()
        if 'Amount' in self.data.columns:
            self.data['Amount'] = self.data['Amount'].abs()
            
        # All of these sales are in GBP.  Convert them to USD using our converter. 
        # This converts them to dollars at the time of the transactions using an exchange rate table
        # update daily. We could updated every minute, as we have minute-level data in statements,
        # but may then need a new exange rate source (current using yahoo finance API output, stored to file).
        self.converter = GBPtoUSD()
        self.data['Amount_USD'] = self.data.apply(
            lambda row: self.converter.convert(row['Amount'], row['Date']), axis=1
        )
        
        # Split large sales into more frequent multiple smaller ones
        self.chop_up()
        
        # scale sale amounts
        scaler = StandardScaler()
        self.data['Amount_Scaled'] = scaler.fit_transform(np.log1p(self.data['Amount']).values.reshape(-1, 1))  

class CreditCardUS(AccountProcessor):
    def __init__(self, data_directory):
        """
        Initialize with the directory containing US credit card statements.
        """
        super().__init__(data_directory)

    def process(self):
        """
        Execute the complete processing sequence specific to US credit card statements.
        """
        # get and merge all the statements
        self.load_and_validate_data()

        # clean
        self.clean_date(DateCol = "Transaction Date", UK_style=False)
        self.clean_description()
        self.clean_amount()
        
        # drop all the columns we don't need
        columns_to_keep = ['Date', 'Description', 'Amount', 'Amount_Scaled']
        self.data = self.data[columns_to_keep]
        return self.data    

    def clean_description(self):
        """
        Build the Description feature.
        
        normalize to lower case.
        """
        # normalize the description
        #self.data['Description'] = self.data[['Category', 'Description']].fillna('').astype(str).agg(' '.join, axis=1)
        self.data['Description'] = self.data['Description'].fillna('').astype(str)
        self.data['Description'] = self.data['Description'].str.lower()
        
    def clean_amount(self):
        """
        Build Amount feature.
        
        Remove rows where the 'type' is 'Payment' or 'Reversal'.
        
        Make all sales positive.
        
        Also split large events up into many smaller ones to boost their frequency. 
        This helps clustering find categories that are rare but significant in size.
        """
        # remove deposits or the like, and make sales positive
        deposit_mask = (self.data['Amount'] <= 0) & (self.data['Type'] != 'Reversal') & (self.data['Type'] != 'Payment')
        self.data = self.data[deposit_mask].copy()
        if 'Amount' in self.data.columns:
            self.data['Amount'] = self.data['Amount'].abs()

        #Split large sales into more frequent multiple smaller ones
        self.chop_up()
        
        # scale sale amounts
        scaler = StandardScaler()
        self.data['Amount_Scaled'] = scaler.fit_transform(np.log1p(self.data['Amount']).values.reshape(-1, 1))  

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
        cutoff_date = datetime(2023, 1, 1).date()
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
        
        # Parse 'OrderDate' to datetime, and remove old transactions
        self.digital['OrderDate Parsed'] = pd.to_datetime(
            self.digital['OrderDate'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce'
        )
        cutoff_date = datetime(2023, 1, 1).date()
        date_mask = self.digital['OrderDate Parsed'].dt.date >= cutoff_date
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
        print(f"\nAttempting to refine {num_matching_rows} transactions recognized as Amazon.\n")
        
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
                threshold = 0.2
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
            print(f"Refined {len(indices_to_delete)} amazon transaction descriptions by cross referencing order history data.\n")
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
    def __init__(self, data, batch_size=200):
        self.data = data
        self.batch_size = batch_size
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        # Sort the data to ensure consistent order
        self.data = self.data.sort_values(by=['Description']).reset_index(drop=True)

        # Being extra cautious about setting random seeds that I may not have noticed.
        np.random.seed(SEED)

        # Train FastText model on the descriptions
        sentences = [desc.split() for desc in self.data['Description'].values]
        model = FastText(
            sentences, 
            vector_size=100, 
            window=3, 
            min_count=2, 
            workers=8,
            seed=SEED
        )  # Adjust parameters as needed
        
        # Create sentence vectors by averaging word vectors
        def get_sentence_vector(sentence):
            words = sentence.split()
            vectors = [model.wv[word] for word in words if word in model.wv]
            return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
        
        # Apply vectorization to the descriptions
        self.data['FastText_Vector'] = self.data['Description'].apply(get_sentence_vector)
        vector_df = pd.DataFrame(self.data['FastText_Vector'].tolist(), index=self.data.index)

        # Combine vectorized descriptions with scaled amounts
        combined_features = pd.concat([vector_df, self.data[['Amount_Scaled']]], axis=1).fillna(0)

        # Convert all column names to strings to ensure compatibility with HDDBSCAN
        combined_features.columns = combined_features.columns.astype(str)

        #
        # HDBSCAN
        #
        # Apply HDBSCAN Clustering and add Cluster Labels to the Original DataFrame
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=10, 
            min_samples=4,
            cluster_selection_method='eom'
        )  
        cluster_labels = clusterer.fit_predict(combined_features)
        self.data['Cluster_Label'] = cluster_labels

        #
        # DBSCAN (alternative to HDBSCAN)
        #
        # Apply DBSCAN Clustering and add Cluster Labels to the Original DataFrame
        # clusterer = DBSCAN(eps=0.045, min_samples=3)  # Adjust eps based on your data
        # cluster_labels = clusterer.fit_predict(combined_features)
        # self.data['Cluster_Label'] = cluster_labels

        #
        # KMEANS (alternative to HDBSCAN)
        #
        # # Set the number of clusters. This needs to be specified manually for KMeans, unlike HDBSCAN.
        # n_clusters = 30  # Adjust this number based on your needs or after evaluating your data
        #
        # # Apply KMeans Clustering and add Cluster Labels to the Original DataFrame
        # clusterer = KMeans(n_clusters=n_clusters, n_init='auto', random_state=SEED)
        # cluster_labels = clusterer.fit_predict(combined_features)
        # self.data['Cluster_Label'] = cluster_labels

        # Create crude set of new cluster names from the FastText model
        crude_names = {}
        for cluster in self.data['Cluster_Label'].unique():
            # Extract descriptions for the current cluster
            cluster_descriptions = self.data[self.data['Cluster_Label'] == cluster]['Description']

            # Use FastText for keyword extraction (adjust as needed)
            keywords = [word for desc in cluster_descriptions for word in desc.split() if word in model.wv]
            common_keywords = pd.Series(keywords).value_counts().head(10).index.tolist()
            crude_names[cluster] = ", ".join(common_keywords)
        
        def sort_dict(d):
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
        
        # Sort crude_names 
        crude_names = sort_dict(crude_names)
     
        def ai_clarification(crude_names):
            """
            Calls the OpenAI API to suggest concise and intuitive budget category names
            for the provided cluster descriptions. Extracts and returns the resulting 
            budget categories as a dictionary.

            Parameters:
            - cluster_names (dict): A dictionary of poorly named categories.

            Returns:
            - budget_categories (dict): A dictionary of improved budget category names.
            """
            openai.api_key = os.getenv("OPENAI_API_KEY")

            # Define the prompt to clarify category names (edit to your liking)
            prompt = (
                "Given the following cluster descriptions:\n\n"
                f"{crude_names}\n\n"
                "Create concise and intuitive budget category names for each cluster. "
                "Output the results as a Python dictionary, where each key is the cluster number and each value is the category name, in the format: "
                "{-1: 'category name', 0: 'another category name', ...}. Only output the dictionary in this format "
                "without any extra text. Do not use the word Retail. Do not use the word Services. Do not use the word Bills."
            )

            # Call the OpenAI API (use mini if the prompt is small)
            #response = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
            response = openai.ChatCompletion.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])

            # Extract the response content
            response_text = response.choices[0].message['content']

            # Use regex to extract the dictionary from the response
            match = re.search(r'{.*?}', response_text, re.DOTALL)

            if match:
                AI_names_str = match.group(0)
                AI_names = ast.literal_eval(AI_names_str)  # Safely parse the dictionary
                
                # Sort keys and convert to int
                AI_names = sort_dict(AI_names)
                crude_names = sort_dict(crude_names)
                
                # make a single dictionary for output to json file
                clarification_data = {
                    'AI_names': AI_names,
                    'crude_names': crude_names
                }

                # write categories to file
                with open("clarification.json", 'w') as file:
                    json.dump(clarification_data, file, indent=4)
                    print("Generated new category mappings and saved to clarification.json")
                return AI_names
            else:
                print("Failed to extract budget categories from the response.")
                return {}

        # Define file path
        clarification_file = 'clarification.json'

        # Initialize AI clarification flag
        USEAI = True

        # Attempt to load and compare cluster names from the JSON file
        if os.path.exists(clarification_file):
            with open(clarification_file, 'r') as file:
                clarification_data = json.load(file)

            # Extract crude_names from the loaded data
            file_crude_names = sort_dict(clarification_data.get('crude_names', {}))

            # keep old AI_names if new and old crude_names match
            if file_crude_names == crude_names:
                AI_names = sort_dict(clarification_data.get('AI_names', {}))
                print("Loaded budget categories from clarification.json")
                USEAI = False

        if USEAI:
            print("Using GPT-4o to clarify spending category names.")
            AI_names = ai_clarification(crude_names)

        # Using the new dictionary, label the transactions
        self.data['Category'] = self.data['Cluster_Label'].map(AI_names)

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
        plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')

    # close the word cloud
    plt.close()

# Set the statement directories
us_cc_directory = '../data/us_credit_card_statements/'
uk_bank_directory = '../data/uk_bank_statements/'
amzn_directory = '../data/Amazon/'

# Process the statements
uscc = CreditCardUS(us_cc_directory)
clean_uscc = uscc.process()
print("\nUS credit card summary:\n")
print(uscc.summarize())
ukbank = uk_bank(uk_bank_directory)
clean_ukbank = ukbank.process()
print("\nUK bank account summary:\n")
print(ukbank.summarize())

# merge the processed statements
statements = pd.concat([clean_uscc, clean_ukbank], ignore_index=True)

# cross reference amazon data
amzn = AmazonProcessor(statements, amzn_directory)
cleaned_df = amzn.process()

# apply the classifier
classified = AIClassifier(cleaned_df, batch_size=200)

# show the shame cloud
shame_cloud(classified.data,output_file="shame_cloud.png")

classified.data['Category'].value_counts()

# Get the number of unique values in the 'Category' column
num_unique_categories = classified.data['Category'].nunique()
print(f"\nDefined {num_unique_categories} spending categories.")

# Group by 'Category' and sum the 'Amount' for each category
category_sums = classified.data.groupby('Category')['Amount'].sum()

# Sort the summed amounts in descending order
category_sums = category_sums.sort_values(ascending=False)

# Display the sorted summed amounts for each category
pd.set_option('display.max_rows', None)
print(f"\nCategory totals:\n{category_sums}\n")

data = classified.data

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

# Save bar chart. I prefer the style above, but saving it to file is an elaborate process.
plt.figure(figsize=(12, 6))
plt.bar(monthly_sums['Month'], monthly_sums['Amount'], color='skyblue')
plt.ylabel('USD')
#plt.yticks([])  # This removes the tick labels
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('monthly_sums.png', dpi=300, bbox_inches='tight')
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
    'Total Spending': f"${total_spending:,.2f}",
    'Average Monthly Spending': f"${average_monthly_spending:,.2f}",
    'Highest Spending Month': highest_spending_month,
    'Top Expense Categories': ', '.join(top_categories),
    'Spending per Category': category_sums
}

# Descriptions of the images
image_descriptions = """
- Image 1 ('shame_cloud.png'): A word cloud showcasing the most frequent spending categories.
- Image 2 ('monthly_sums.png'): A bar chart displaying monthly spending totals over the past year.
"""

# Create a prompt with your data summary and image descriptions (pick your style by editing)

# 
# For dry style, start prompt with this
#
# prompt = f"""
# You are a sharp and highly paid wealth manager assembling a report for my family. You are humorless and wise. 
# Based on the following financial summary and image descriptions, provide reflection and advice.

#
# For fun style, start prompt with this
#
# prompt = f"""
# You are a wise and valued old friend and financial advisor to my family. You have a dry, self-deprecating 
# sense of humor, like a disgruntled AI in an Iain Banks novel. Based on the following financial summary and 
# image descriptions, provide a humorous reflection and advice.


prompt = f"""
You are a sharp and highly paid wealth manager assembling a report for my family. You are humorless and wise. 
Based on the following financial summary and image descriptions, provide reflection and advice.

Summary:
- Date Range: {data_summary['Number of days covered']}
- Total Spending: {data_summary['Total Spending']}
- Average Monthly Spending: {data_summary['Average Monthly Spending']}
- Highest Spending Month: {data_summary['Highest Spending Month']}
- Spending per Category: {data_summary['Spending per Category']}

Images:
{image_descriptions}

Please do not repeat the summary or the image descriptions verbatim. Instead, create your own formatted report, 
with its own summary table of basic statistics or descriptors. Do not talk about yourself in the report.

You should include sections on projections for future annual costs, and a section on sustainability that describes
annual incomes needed to sustain. That section should include pre and post tax aoumnts of income, were the taxes could 
be typical of the US or UK, as some income will be from businesses in each country.

You should also include a section that proposes an annual budget.  That budget can use a smaller set of categories than the one provided.

Be verbose. Every section of the report should have some text, and every table or should be referenced and discussed.

Your output should be in the form of LaTeX code. The report should include sections 
such as Introduction, Overview of Spending, Category Analysis, and Advice.   
Assume the images are named 'shame_cloud.png' and 'monthly_sums.png'. Reference the images appropriately, and 
make sure they are large enough to see, say 90% of the document width.

IMPORTANT: Do not include any additional text at all outside of the LaTeX code. That includes any marker 
telling me that it's latex code.
"""

# Generate advice using OpenAI's GPT-4o
response = openai.ChatCompletion.create(
    model='gpt-4o',
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7
)

# Extract the generated LaTeX code
advice = response.choices[0].message['content'].strip()

# Extract the generated LaTeX code
latex_code = response.choices[0].message['content'].strip()

# Write the LaTeX code to a .tex file
with open('financial_report.tex', 'w') as file:
    file.write(latex_code)

print("Report written to 'financial_report.tex'.")

def compile_latex(tex_file, num_runs=3):
    """
    Compiles a LaTeX .tex file multiple times to resolve references.

    Parameters:
    - tex_file (str): The path to the .tex file to compile.
    - num_runs (int): Number of times to run the LaTeX compiler.

    Returns:
    - bool: True if compilation succeeded in all runs, False otherwise.
    """
    compile_command = ['pdflatex', tex_file]
    
    for run in range(1, num_runs + 1):
        print(f"Running pdflatex, pass {run}...")
        try:
            # Run the pdflatex command, suppressing stdout and stderr
            result = subprocess.run(
                compile_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error: pdflatex failed on pass {run}.")
            return False
    
    print(f"LaTeX file '{tex_file}' has been successfully compiled to 'financial_report.pdf'.")
    os.system('rm financial_report.aux'); # maybe remove this OS-specific line
    os.system('rm financial_report.log'); # maybe remove this OS-specific line
    os.system('rm financial_report.out'); # maybe remove this OS-specific line
    os.system('rm financial_report.toc'); # maybe remove this OS-specific line
    return True

# Compile the report
compile_latex('financial_report.tex');
