# src/currency_converter.py


import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

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




