from sentence_transformers import SentenceTransformer, util as st_util
from transformers import CLIPModel, CLIPProcessor
import shutil

from PIL import Image
import pandas as pd
import os
import torch
torch.set_printoptions(precision=10)
from tqdm import tqdm


model_names = ["sentence-transformer-clip-ViT-L-14", "fashion", "openai-clip"]

model_name_to_ids = {
    "sentence-transformer-clip-ViT-L-14": "clip-ViT-L-14",
    "fashion": "patrickjohncyh/fashion-clip",
    "openai-clip": "openai/clip-vit-base-patch32",
}


def get_data_path():
    directory = os.path.join('data', cur_dataset)
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
    return directory

def get_image_path():
    directory = os.path.join(get_data_path(), 'images')
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
    return directory

def get_metadata_path():
    directory = os.path.join(get_data_path(), 'metadata')
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
    return directory

def get_embeddings_path():
    metadata_path = get_metadata_path()
    return os.path.join(metadata_path, cur_dataset + '_embeddings.pq')

model_dict = dict()



def download_images(df, img_folder):
    shutil.rmtree(img_folder)
    for index, row in df.iterrows():
        try:
            st_util.http_get(row['IMG_URL'], os.path.join(img_folder,
                                                          row['title'].replace('/', '_').replace('\n', '') + '.jpg'))
        except:
            print('Error downloading image: ' + str(index) + row['title'])


def load_models():
    for model_name in model_name_to_ids:
        if model_name not in model_dict:
            model_dict[model_name] = dict()
            if model_name.startswith('sentence-transformer'):
                model_dict[model_name]['model'] = SentenceTransformer(model_name_to_ids[model_name])
            else:
                model_dict[model_name]['hf_dir'] = model_name_to_ids[model_name]
                model_dict[model_name]['model'] = CLIPModel.from_pretrained(model_name_to_ids[model_name])
                model_dict[model_name]['processor'] = CLIPProcessor.from_pretrained(model_name_to_ids[model_name])

load_models()


def get_image_embeddings(model_name, image):
    model = model_dict[model_name]['model']
    if model_name.startswith('sentence-transformer'):
        return model.encode(image)
    else:
        inputs = model_dict[model_name]['processor'](images=image, return_tensors="pt")
        image_features = model.get_image_features(**inputs).detach().numpy()[0]
        return image_features


def generate_embeddings():
    embeddings_df = pd.DataFrame()

    # Get image embeddings
    with torch.no_grad():
        for fp in tqdm(os.listdir(get_image_path()), desc="Generate embeddings for Images"):
            if fp.endswith('.jpg'):
                new_row = {'name': fp}
                for model_name in model_name_to_ids.keys():
                    new_row[f'{model_name}-embedding'] = get_image_embeddings(
                        model_name, Image.open(os.path.join(get_image_path(), fp)))
                embeddings_df = embeddings_df.append(new_row, ignore_index=True)
    return embeddings_df


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


all_datasets = get_immediate_subdirectories('data')
cur_dataset = all_datasets[0]

def set_cur_dataset(dataset):
    refresh_all_datasets()
    print(f"Setting current dataset to {dataset}")
    global cur_dataset
    cur_dataset = dataset

def refresh_all_datasets():
    global all_datasets
    all_datasets = get_immediate_subdirectories('data')
    print(f"Refreshing all datasets: {all_datasets}")
