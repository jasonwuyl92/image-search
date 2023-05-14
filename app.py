import os
from functools import partial

import gradio as gr
import pandas as pd

import utils
import vector_db
from utils import get_image_embedding, \
    get_image_path, model_names, download_images, generate_and_save_embeddings, get_metadata_path, url_to_image

NUM_OUTPUTS = 4


def search(input_img, model_name):
    query_embedding = get_image_embedding(model_name, input_img).tolist()
    top_results = vector_db.query_embeddings_db(query_embedding=query_embedding,
                                                dataset_name=utils.cur_dataset, model_name=model_name)
    print (top_results)
    return [utils.url_to_image(hit['metadata']['mainphotourl']) for hit in top_results['matches']]


def read_tsv_temporary_file(temp_file_wrapper):
    dataset_name = os.path.splitext(os.path.basename(temp_file_wrapper.name))[0]
    utils.set_cur_dataset(dataset_name)
    df = pd.read_csv(temp_file_wrapper.name, sep='\t')  # Read the TSV content into a pandas DataFrame
    df.to_csv(os.path.join(get_metadata_path(), dataset_name + '.tsv'), sep='\t', index=False)
    print('start downloading')
    download_images(df, get_image_path())
    generate_and_save_embeddings()
    utils.refresh_all_datasets()
    utils.set_cur_dataset(dataset_name)
    return gr.update(choices=utils.all_datasets, value=dataset_name)


def update_dataset_dropdown():
    utils.refresh_all_datasets()
    utils.set_cur_dataset(utils.all_datasets[0])
    return gr.update(choices=utils.all_datasets, value=utils.cur_dataset)


def gen_image_blocks(num_outputs):
    with gr.Row():
        row = [gr.outputs.Image(label=model_name, type='filepath') for i in range(int(num_outputs))]
    return row


with gr.Blocks() as demo:
    galleries = dict()
    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="Upload TSV File", file_types=[".tsv"])
            image_input = gr.inputs.Image(type="pil", label="Input Image")
            dataset_dropdown = gr.Dropdown(label='Datasets', choices=utils.all_datasets)
            b1 = gr.Button("Find Similar Images")
            b2 = gr.Button("Refresh Datasets")

            dataset_dropdown.select(utils.set_cur_dataset, inputs=dataset_dropdown)
            file_upload.upload(read_tsv_temporary_file, inputs=file_upload, outputs=dataset_dropdown)
            b2.click(update_dataset_dropdown, outputs=dataset_dropdown)
        with gr.Column(scale=3):
            for model_name in model_names:
                galleries[model_name] = gen_image_blocks(NUM_OUTPUTS)
    for model_name in model_names:
        b1.click(partial(search, model_name=model_name), inputs=[image_input],
                 outputs=galleries[model_name])
    b2.click(utils.refresh_all_datasets, outputs=dataset_dropdown)

demo.launch()
