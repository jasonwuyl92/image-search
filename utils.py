import os

from sentence_transformers import SentenceTransformer, util as st_util
from transformers import CLIPModel, CLIPProcessor

model_names = ["sentence-transformer-clip-ViT-L-14", "fashion", "openai-clip"]

model_name_to_ids = {
    "sentence-transformer-clip-ViT-L-14": "clip-ViT-L-14",
    "fashion": "patrickjohncyh/fashion-clip",
    "openai-clip": "openai/clip-vit-base-patch32",
}

data_path = 'data/patagonia_losGatos/'
img_folder = os.path.join(data_path, 'images')

model_dict = dict()

def download_images(df, img_folder):
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
