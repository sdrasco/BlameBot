# src/amazon_processor.py

import os
import pandas as pd
from glob import glob
from datetime import timedelta

class AmazonProcessor:
    def __init__(self, statements, data_directory):
        self.statements = statements
        self.data_directory = data_directory
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



