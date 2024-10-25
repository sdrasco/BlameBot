# src/account_processor.py

import os
import logging
import pandas as pd
import numpy as np  # Add this line
from glob import glob
from datetime import datetime
from currency_converter import GBPtoUSD
from sklearn.preprocessing import StandardScaler 

# Configure basic logging.  show warning or higher for external modules.
logging.basicConfig(
    level=logging.WARNING,  
    format='%(message)s'
)

# Create a logger for this module
logger = logging.getLogger(__name__)

# Show info level logger events for this module
logger.setLevel(logging.INFO)

class AccountProcessor:
    def __init__(self, data_directory):
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
                    logger.error(f"File {file} has unexpected columns: {df.columns}")

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

class UKBank(AccountProcessor):
    def __init__(self, data_directory):
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


