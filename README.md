
# BlameBot: AI Overlord Family Finance Tool Because You Can't Math and Need More Reasons to Argue

![BlameBot Icon](BlameBot.png)

BlameBot is a Python-based tool designed to help families analyze their financial data using advanced AI techniques like ChatGPT, Natural Language Processing (NLP), and clustering. It processes financial statements from various sources, unifies the data, and generates insightful reports to better understand spending habits. And yes, it might just give you more reasons to argue.

## Features

- **Multi-Source Data Integration**: Supports processing of financial statements from US credit cards (USD) and UK bank accounts (GBP).
- **Amazon cross referencing**: Refines transaction descriptions by cross referencing Amazon order history.
- **AI-Driven Clustering**: Uses FastText embeddings and clustering algorithms (HDBSCAN, DBSCAN, or KMeans) to categorize transactions.
- **Smart Categorization**: Leverages GPT-4o to refine and clarify cluster names for intuitive spending categories.
- **Visual Insights**: Generates a word cloud and bar chart to visualize spending patterns and monthly expenditures.

## Project Structure

- `BlameBot.py`: The main Python script containing all the code for data processing, clustering, and report generation.
- `BlameBot.png`: The project logo featuring our mischievous AI overlord mascot.
- `ShameCloud.png`: An example word cloud showcasing the most frequent spending categories.
- `monthly.png`: An example bar chart displaying monthly spending totals.
- `fun_report.pdf`: Example of a humorous report output.
- `dry_report.pdf`: Example of a humorless report output.

## Getting Started

### Prerequisites

To run BlameBot, you will need:

- Python 3.x
- The required Python packages listed in `requirements.txt`
- An OpenAI API key (for GPT-4 based cluster name clarification)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/BlameBot.git
   ```
2. Navigate to the project directory:
   ```bash
   cd BlameBot
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   
### Setup

1. **Prepare Your Data:**
   - Place your US credit card CSV statements in the directory `data/us_credit_card_statements/`.
   - Place your UK bank account CSV statements in the directory `data/uk_bank_statements/`.
   - Place your Amazon order history CSV files in the directory `data/Amazon/`.
   
2. **Configure Your OpenAI API Key:**
   - Set your OpenAI API key as an environment variable:
     ```bash
     export OPENAI_API_KEY='your-openai-api-key'
     ```

### Usage

To run the tool and generate reports:

```bash
python BlameBot.py
```

## Customizing for Your Own Use

To adapt BlameBot to your specific needs, you might consider the following customizations:

### Data Format

- Ensure that your CSV files follow the expected structure with consistent columns across all files for each account type.
- Modify the `AccountProcessor` classes (`uk_bank` and `us_credit_card`) to fit the format of your statements if they differ significantly.

### Clustering Configuration

- Experiment with different clustering algorithms by modifying the `AI_FastText_Classifier` class:
  - **HDBSCAN**: Adjust parameters like `min_cluster_size` and `min_samples` to see how it affects clustering.
  - **DBSCAN**: Uncomment and adjust parameters like `eps` and `min_samples`.
  - **KMeans**: Set the number of clusters manually to explore various segmentation strategies.

### Improving Cluster Naming

- If the cluster names generated by GPT-4o are not satisfactory, tweak the `AIClarification` function's prompt in the `AI_FastText_Classifier` class.

### Visualizations

- Modify the `ShameCloud` function to customize the word cloud (e.g., change the color scheme or the maximum number of words).
- Adjust the bar chart code to fit your aesthetic or data requirements.

## Example Outputs

### Word Cloud
![Spending Categories Word Cloud](ShameCloud.png)

### Monthly Expenditures
![Monthly Expenditures Bar Chart](monthly.png)

### Reports

Reports are now minimialist, colorful, and save as both html with embedded images and as PDF files so colorful they'll blow your printing budget.  Here's are some examples, redacted to protect the reputation of the guilty:

   -[Example Report](report1.html)
   -[Another Example](report2.html)
   -[Okay, but this is the last one](report3.html)

## Fun FAQs

1. **My report looks outlandish. The images are the size of the moon!  What gives?**

   - BlameBot has full artistic license when it comes to color scheme, report layout, etc. Once in a while it halucinates a style so inovative that it isn't human redable. When that happens, try, try again.

2. **Did your artist in residence make the adorable BlameBot logo?**

   - No. As you might guess from financial wisdom, we couldn't afford a human artist. We outsourced that task to OpenAI's Dalle.

## License

This project is licensed under the GNU General Public License v3.0. See the `LICENSE` file for more details.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please fork the repository and create a pull request.

## Contact

For any questions or suggestions, please contact [Steve Drasco](mailto:steve.drasco@gmail.com).
