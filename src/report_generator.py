# src/report_generator.py

import logging
import pandas as pd
import matplotlib.pyplot as plt
import base64
import re
from bs4 import BeautifulSoup
from openai_utils import OpenAIUsageTracker

# Configure basic logging.  show warning or higher for external modules.
logging.basicConfig(
    level=logging.WARNING,  
    format='%(message)s'
)

# Create a logger for this module
logger = logging.getLogger(__name__)

# Show info level logger events for this module
logger.setLevel(logging.INFO)

def build_reports(data, openai_tracker):
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
    plt.savefig('../images/monthly_sums.png', dpi=150, bbox_inches='tight')
    # Close the figure to avoid displaying it
    plt.close()

    # Calculate total and average spending
    total_spending = data['Amount'].sum()
    average_monthly_spending = monthly_sums['Amount'].mean()
    highest_spending_month = monthly_sums.loc[monthly_sums['Amount'].idxmax(), 'Month']

    # Identify top expense categories
    top_categories = data['Category'].value_counts().head(5).index.tolist()

    # Group by 'Category' and sum the 'Amount' for each category
    category_sums = data.groupby('Category')['Amount'].sum()
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

    2. **Spending Analysis**
       - Do an analysis of the spending data in the summary, carefully looking for trends or events.
       - Display the image '../images/shame_cloud.png', the word cloud of shame built out of the spending category names and amounts, and talk about it
       - Describe the findings of your analysis.
       - Display the image '../images/monthly_sums.png', a bar chart of amounts spent each month, and talk about it
       - Don't use the image filenames in the text

    3. **Projections for Annual Costs**
       - Based on current spending trends, provide projections for annual costs. Consider factors such as potential inflation, lifestyle changes, or other likely cost changes.
       
    4. **Suggested Budget by Category**
       - Propose a more concise annual budget, consolidated to five spending categories.
       - Include a table with totals at the bottom row

    6. **Sustainability Outline**
       - Provide an assessment of the income needed to sustain the suggested budget, including pre- and post-tax amounts, stating the assumed tax rates.
    
    End the report with a footer containing a thumbnail of your image '../images/BlameBot_small.png' 
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
    - You should avoid the use of equations if at all possible, but if you must use equations, make them with mathjax
    - Include basic inline CSS for layout and typography
    - The images should be referenced with `<img>` tags
    - All text should be wrapped in `<p>`, `<h1>`, `<h2>`, or `<div>` tags, ensuring proper hierarchy

    Please generate the report as a single HTML document with embedded CSS. **Do not include any additional text at all outside of the HTML code.**
    """

    # Generate rough report using OpenAI's API
    response = openai_tracker.chat_completion(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are BlameBot a clever and humorous wealth manager who likes to show off their dry wit."},
            {"role": "user", "content": prompt}
        ],
    )

    # Extract the generated HTML code for parseing/polishing
    rough_report = response.choices[0].message.content

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(rough_report, 'html.parser')

    # Occasionaly the html output is garbled and we can't parse it.  
    # If so, bail out with apology of sorts.
    if not soup.contents or soup.contents == ['\n']:
        logger.error("BlameBot failed to write a report this time. Please doc its pay accordingly. Bad BlameBot.")
        return

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
        '../images/monthly_sums.png': convert_image_to_base64('../images/monthly_sums.png'),
        '../images/shame_cloud.png': convert_image_to_base64('../images/shame_cloud.png'),
        '../images/BlameBot_small.png': convert_image_to_base64('../images/BlameBot_small.png')
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
    with open('../html/financial_report.html', 'w') as file:
        file.write(str(soup))  
    logger.info("Report written to '../html/financial_report.html'.\n")

    # Redact dollar amounts
    for td in soup.find_all("td"):
        if "$" in td.text:
            td.string = "[redacted]"
    for p in soup.find_all("p"):
        p.string = re.sub(r"\$\d+(?:,\d{3})*(?:\.\d{2})?", "[redacted]", p.text)

    # Save the modified content to a new HTML file
    output_path = "../html/financial_report_redacted.html"
    with open(output_path, "w") as file:
        file.write(str(soup))
    logger.info("Redacted report written to '../html/financial_report_redacted.html'.\n")
