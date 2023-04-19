import numpy as np
import gradio as gr
from sentence_transformers import util as st_util
import pandas as pd
import os
from utils import load_models, get_image_embeddings, img_folder, model_name_to_ids, data_path, model_names


def search(input_img, num_outputs):
    results = []
    for model_name in model_names:
        query_embedding = get_image_embeddings(model_name, input_img)
        top_results = st_util.semantic_search(query_embedding,
                                           np.vstack(list(corpus_embeddings[model_name + '-embedding'])),
                                              top_k=int(num_outputs))[0]
        results.append([os.path.join(img_folder,
                          corpus_embeddings.iloc[hit['corpus_id']]['name']) for hit in top_results])
    return results


load_models()
corpus_embeddings = pd.read_parquet(
    os.path.join(data_path, 'metadata/patagonia_losGatos_embeddings.pq'))



# Create the Gradio interface
iface = gr.Interface(
    fn=search,
    inputs=[gr.Image(type="pil"),
            gr.inputs.Number(label="Number of results", default=3)],
    outputs=[gr.Gallery(label=model_name, type='filepath') for model_name in model_names],
    title="Search Similar Images",
    description="Upload an image and find similar images",
)

# Launch the Gradio interface
iface.launch(debug=True)
