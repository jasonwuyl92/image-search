import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os

CLIP_MODEL_NAME = "clip-ViT-B-16"
# model = SentenceTransformer(CLIP_MODEL_NAME)
corpus_embeddings = pd.read_parquet(
        'data/patagonia_losGatos/metadata/patagonia_losGatos_embeddings.pq')

def search(input_img):
    query_embedding = model.encode(input_img)
    top_results = util.semantic_search(query_embedding,
                                       np.vstack(list(corpus_embeddings['clip-ViT-L-14-embedding'])), top_k=3)[0]

    img_folder = 'data/patagonia_losGatos/images'
    return [os.path.join(img_folder, corpus_embeddings.iloc[hit['corpus_id']]['name']) for hit in top_results]


# Define the input and output components
input_text = gr.inputs.Textbox(lines=2, placeholder="Enter some text...")

image_output1 = gr.outputs.Image(label="Output Image 1", type='filepath')
image_output2 = gr.outputs.Image(label="Output Image 2", type='filepath')
image_output3 = gr.outputs.Image(label="Output Image 3", type='filepath')

# Create the Gradio interface
iface = gr.Interface(
    fn=search,
    inputs=gr.Image(type="pil"),
    outputs=[image_output1, image_output2, image_output3],
    title="Search Similar Images",
    description="Upload an image and find similar images",
)

# Launch the Gradio interface
iface.launch(debug=True)

