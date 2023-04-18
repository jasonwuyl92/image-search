import numpy as np
import gradio as gr
from sentence_transformers import util as st_util
import pandas as pd
import os
from utils import load_models, get_image_embeddings, img_folder, model_name_to_ids

data_path = 'data/patagonia_losGatos/'

#cur_model_name = "sentence-transformer-clip-ViT-L-14"
cur_model_name = "fashion"

def search(input_img, model_name):
    query_embedding = get_image_embeddings(model_name, input_img)
    top_results = st_util.semantic_search(query_embedding,
                                       np.vstack(list(corpus_embeddings[model_name + '-embedding'])), top_k=3)[0]
    return [os.path.join(img_folder, corpus_embeddings.iloc[hit['corpus_id']]['name']) for hit in top_results]



load_models()
corpus_embeddings = pd.read_parquet(
    'data/patagonia_losGatos/metadata/patagonia_losGatos_embeddings.pq')
image_output1 = gr.outputs.Image(label="Output Image 1", type='filepath')
image_output2 = gr.outputs.Image(label="Output Image 2", type='filepath')
image_output3 = gr.outputs.Image(label="Output Image 3", type='filepath')


# Create the Gradio interface
iface = gr.Interface(
    fn=search,
    inputs=[gr.Image(type="pil"), gr.inputs.Dropdown(list(model_name_to_ids.keys()))],
    outputs=[image_output1, image_output2, image_output3],
    title="Search Similar Images",
    description="Upload an image and find similar images",
)

# Launch the Gradio interface
iface.launch(debug=True)
