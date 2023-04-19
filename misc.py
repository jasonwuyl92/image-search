import pandas as pd
import random

# Function to generate random text for titles

def generate_random_images_df(filename):
    def generate_title():
        title_length = random.randint(5, 20)
        title = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=title_length))
        return title.capitalize()

    # Function to generate random image URLs
    def generate_image_url():
        url = "https://picsum.photos/200/300"  # Change the size of the image as per your requirement
        return url

    # Create a list of dictionaries with random titles and image URLs
    data = []
    for i in range(10):
        data.append({'title': generate_title(), 'IMG_URL': generate_image_url()})

    # Convert the list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(data)
    df.to_csv(filename, sep='\t', index=False)
