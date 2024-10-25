# src/visualization.py

import logging
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Configure basic logging.  show warning or higher for external modules.
logging.basicConfig(
    level=logging.WARNING,  
    format='%(message)s'
)

# Create a logger for this module
logger = logging.getLogger(__name__)

# Show info level logger events for this module
logger.setLevel(logging.INFO)

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

    # # Sort category_dict by amount and take the top categories
    # top_categories = [category for category, amount in sorted(category_dict.items(), key=lambda x: x[1], reverse=True)[0:5]]
    
    # # Convert the top categories into a readable summary format
    # spending_habits = ", ".join(top_categories)

    # # describe the family doing the spending
    # family_description = (
    #     "The husband is a nerdy white man with poor fashion sense and an average build."
    #     "The wife is a mixed white and asian woman with impecable taste and an atheletic build."
    #     "A brown cat with black stripes. A grey tuxedo cat."
    # )

    # # Combine the family description and spending categories into the prompt
    # prompt = (
    #     f"An image depicting {family_description} "
    #     f"The image should reflect a lifestyle in which they spend all their money on {spending_habits}. "
    #     "Focus on visual storytelling without the use of any text, letters, signs, or writing of any kind."
    # )
    
    # # Ask GPT-4 to refine the prompt for DALL-E 3
    # client = OpenAI()
    # response = openapi_usage_tracker.chat_completion(
    #     model="gpt-4o",  
    #     messages=[
    #         {"role": "system", "content": "You are an expert at creating prompts for DALL-E 3 image generation."},
    #         {"role": "user", "content": f"Refine this prompt for DALL-E 3: {prompt}"}
    #     ]
    # )

    # # Extract the clarified prompt
    # clarified_prompt = response.choices[0].message.content
    # print(clarified_prompt)
    
    # # Call the OpenAI DALL-E-3API
    # response = openapi_usage_tracker.image_create(
    #     model="dall-e-3",
    #     prompt=clarified_prompt,
    #     n=1,
    #     size="1024x1024",
    #     style="vivid",
    #     quality="standard",
    #     response_format="url"
    # )

    # # Save the image
    # image_url = response.data[0].url

    # # If using a URL:
    # import requests
    # img_data = requests.get(image_url).content
    # with open('../images/family.png', 'wb') as handler:
    #     handler.write(img_data)

    # print("Image saved as ../images/family.png")