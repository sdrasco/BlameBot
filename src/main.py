# src/main.py

import os
import logging
import numpy as np
import pandas as pd
from account_processor import UKBank, CreditCardUS
from amazon_processor import AmazonProcessor
from ai_classifier import AIClassifier
from visualization import shame_cloud
from report_generator import build_reports
from openai_utils import OpenAIUsageTracker
import openai

# Configure basic logging.  show warning or higher for external modules.
logging.basicConfig(
    level=logging.WARNING,  
    format='%(message)s'
)

# Create a logger for this module
logger = logging.getLogger(__name__)

# Show info level logger events for this module
logger.setLevel(logging.INFO)

def main():
    # Set a random seed
    SEED = 846
    np.random.seed(SEED)

    # Initialize OpenAI client and usage tracker
    openai.api_key = os.getenv('OPENAI_API_KEY')
    client = openai
    openai_tracker = OpenAIUsageTracker(client)

    # Set the statement directories
    us_cc_directory = '../data/us_credit_card_statements/'
    uk_bank_directory = '../data/uk_bank_statements/'
    amzn_directory = '../data/Amazon/'

    # Process the statements
    uscc = CreditCardUS(us_cc_directory)
    clean_uscc = uscc.process(DateCol='Transaction Date', UK_style=False, DescriptionColumnNames=['Description'])
    logger.info("\nUS credit card summary:\n")
    logger.info(uscc.summarize())

    ukbank = UKBank(uk_bank_directory)
    clean_ukbank = ukbank.process(DateCol='Date', UK_style=True, DescriptionColumnNames=['Name', 'Description'])
    logger.info("\nUK bank account summary:\n")
    logger.info(ukbank.summarize())

    # Merge the processed statements
    statements = pd.concat([clean_uscc, clean_ukbank], ignore_index=True)

    # Cross-reference Amazon data
    amzn = AmazonProcessor(statements, amzn_directory)
    cleaned_df = amzn.process()

    # Apply the classifier
    classifier = AIClassifier(cleaned_df, openai_tracker)
    classifier.process()

    # Generate the shame cloud
    shame_cloud(classifier.data, output_file="../images/shame_cloud.png")

    # Build the reports
    #build_reports(classifier.data, openai_tracker)

    # Show the total OpenAI API usage cost
    openai_tracker.calculate_total_cost()

if __name__ == "__main__":
    main()

    