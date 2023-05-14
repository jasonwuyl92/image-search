from sentence_transformers import SentenceTransformer, util as st_util
from transformers import CLIPModel, CLIPProcessor

from PIL import Image
import requests
import os
import torch
torch.set_printoptions(precision=10)
from tqdm import tqdm
import s3fs
from io import BytesIO
import vector_db

"sentence-transformer-clip-ViT-L-14"
model_names = ["fashion", "openai-clip"]

model_name_to_ids = {
    "sentence-transformer-clip-ViT-L-14": "clip-ViT-L-14",
    "fashion": "patrickjohncyh/fashion-clip",
    "openai-clip": "openai/clip-vit-base-patch32",
}

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]

# Define your bucket and dataset name.
S3_BUCKET = "s3://disco-io"

fs = s3fs.S3FileSystem(
    key=AWS_ACCESS_KEY_ID,
    secret=AWS_SECRET_ACCESS_KEY,
)

ROOT_DATA_PATH = os.path.join(S3_BUCKET, 'data')

def get_data_path():
    return os.path.join(ROOT_DATA_PATH, cur_dataset)

def get_image_path():
    return os.path.join(get_data_path(), 'images')

def get_metadata_path():
    return os.path.join(get_data_path(), 'metadata')

def get_embeddings_path():
    return os.path.join(get_metadata_path(), cur_dataset + '_embeddings.pq')

model_dict = dict()


def download_to_s3(url, s3_path):
    # Download the file from the URL
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Upload the file to the S3 path
    with fs.open(s3_path, "wb") as s3_file:
        for chunk in response.iter_content(chunk_size=8192):
            s3_file.write(chunk)


def remove_all_files_from_s3_directory(s3_directory):
    # List all objects in the S3 directory
    objects = fs.ls(s3_directory)

    # Remove each object
    for obj in objects:
        try:
            fs.rm(obj)
        except:
            print('Error removing file: ' + obj)

def download_images(df, img_folder):
    remove_all_files_from_s3_directory(img_folder)
    for index, row in df.iterrows():
        try:
            download_to_s3(row['IMG_URL'], os.path.join(img_folder,
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


if len(model_dict) == 0:
    print('Loading models...')
    load_models()


def get_image_embedding(model_name, image):
    """
    Takes an image as input and returns an embedding vector.
    """
    model = model_dict[model_name]['model']
    if model_name.startswith('sentence-transformer'):
        return model.encode(image)
    else:
        inputs = model_dict[model_name]['processor'](images=image, return_tensors="pt")
        image_features = model.get_image_features(**inputs).detach().numpy()[0]
        return image_features

def s3_path_to_image(fs, s3_path):
    """
    Takes an S3 path as input and returns a PIL Image object.

    Args:
        s3_path (str): The path to the image in the S3 bucket, including the bucket name (e.g., "bucket_name/path/to/image.jpg").

    Returns:
        Image: A PIL Image object.
    """
    with fs.open(s3_path, "rb") as f:
        image_data = BytesIO(f.read())
        img = Image.open(image_data)
        return img

def generate_and_save_embeddings():
    # Get image embeddings
    with torch.no_grad():
        for fp in tqdm(fs.ls(get_image_path()), desc="Generate embeddings for Images"):
            if fp.endswith('.jpg'):
                name = fp.split('/')[-1]
                for model_name in model_name_to_ids.keys():
                    s3_path = 's3://' + fp
                    vector_db.add_image_embedding_to_db(
                        embedding=get_image_embedding(model_name, s3_path_to_image(fs, s3_path)),
                        model_name=model_name,
                        dataset_name=cur_dataset,
                        path_to_image=s3_path,
                        image_name=name,
                    )


def get_immediate_subdirectories(s3_path):
    return [obj.split('/')[-1] for obj in fs.glob(f"{s3_path}/*") if fs.isdir(obj)]

all_datasets = get_immediate_subdirectories(ROOT_DATA_PATH)
cur_dataset = all_datasets[0]

def set_cur_dataset(dataset):
    refresh_all_datasets()
    print(f"Setting current dataset to {dataset}")
    global cur_dataset
    cur_dataset = dataset

def refresh_all_datasets():
    global all_datasets
    all_datasets = get_immediate_subdirectories(ROOT_DATA_PATH)
    print(f"Refreshing all datasets: {all_datasets}")
