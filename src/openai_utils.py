# src/openai_utils.py

import openai
import logging

# Configure basic logging.  show warning or higher for external modules.
logging.basicConfig(
    level=logging.WARNING,  
    format='%(message)s'
)

# Create a logger for this module
logger = logging.getLogger(__name__)

# Show info level logger events for this module
logger.setLevel(logging.INFO)

class OpenAIUsageTracker:
    def __init__(self, client):
        self.client = client  # OpenAI client instance
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_images_generated = 0

    def chat_completion(self, **kwargs):
        response = self.client.chat.completions.create(**kwargs)
        usage = response.usage
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        return response

    def image_create(self, **kwargs):
        response = self.client.Image.create(**kwargs)
        images_generated = len(response.data)
        self.total_images_generated += images_generated
        return response

    def calculate_total_cost(self):
        cost_per_input_token = 0.000005    # $ per input token
        cost_per_output_token = 0.000015   # $ per output token
        cost_per_image = 0.040             # $ per image

        total_input_cost = self.total_prompt_tokens * cost_per_input_token
        total_output_cost = self.total_completion_tokens * cost_per_output_token
        total_image_cost = self.total_images_generated * cost_per_image
        total_cost = total_input_cost + total_output_cost + total_image_cost

        logger.info(f"Total OpenAI API usage cost: ${total_cost:.4f}\n")