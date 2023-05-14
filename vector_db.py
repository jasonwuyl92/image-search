import pinecone
import os
import uuid

pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="us-west1-gcp")

INDEX_512_NAME = "images-512"
INDEX_768_NAME = "images-768"

index_512 = pinecone.Index(INDEX_512_NAME)
index_768 = pinecone.Index(INDEX_768_NAME)

DEV_NAMESPACE = 'disco-web-app-search-dev'
PROD_NAMESPACE = 'disco-web-app-search-prod'


def add_image_embedding_to_db(embedding, model_name, dataset_name, path_to_image, image_name):
    index = {
        512: index_512,
        768: index_768
    }[embedding.shape[0]]
    print (embedding.shape)
    index.upsert([(str(uuid.uuid4()), embedding.tolist(), {'model': model_name,
                                                           'dataset': dataset_name,
                                                           'path': path_to_image,
                                                           'image_name': image_name})])


def query_embeddings_db(query_embedding, dataset_name, model_name, top_k=4):
    index = {
        512: index_512,
        768: index_768
    }[len(query_embedding)]
    return index.query(vector=query_embedding,
                       top_k=top_k,
                       namespace=DEV_NAMESPACE,
                       include_metadata=True)
