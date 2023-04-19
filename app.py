import numpy as np
import gradio as gr
from sentence_transformers import util as st_util
import pandas as pd
import os
from utils import load_models, get_image_embeddings, img_folder, data_path, model_names
from functools import partial

NUM_OUTPUTS = 4

def search(input_img, model_name):
    query_embedding = get_image_embeddings(model_name, input_img)
    top_results = st_util.semantic_search(query_embedding,
                                          np.vstack(list(corpus_embeddings[model_name + '-embedding'])),
                                          top_k=int(NUM_OUTPUTS))[0]
    return [os.path.join(img_folder,
                      corpus_embeddings.iloc[hit['corpus_id']]['name']) for hit in top_results]


load_models()
corpus_embeddings = pd.read_parquet(
    os.path.join(data_path, 'metadata/patagonia_losGatos_embeddings.pq'))


def gen_image_blocks(num_outputs):
    with gr.Row():
        row = [gr.outputs.Image(label=model_name, type='filepath') for i in range(int(num_outputs))]
    return row

with gr.Blocks() as demo:
    galleries = dict()
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.inputs.Image(type="pil", label="Input Image")
            b1 = gr.Button("Find Similar Images")
        with gr.Column(scale=3):
            for model_name in model_names:
                galleries[model_name] = gen_image_blocks(NUM_OUTPUTS)
        # galleries = []
        # for model_name in model_names:
        #     with gr.Column():
        #         gallery = gr.Gallery(label=model_name, type='filepath')
        #         galleries.append(gallery)
        #output_images = []
    for model_name in model_names:
        #number_input.change(gen_image_blocks, inputs=number_input, outputs=galleries[model_name])
        b1.click(partial(search, model_name=model_name), inputs=[image_input],
                 outputs=galleries[model_name])

demo.launch()
