


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import fiftyone as fo
import fiftyone.zoo as foz
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



dataset = foz.load_zoo_dataset("cifar10", shuffle=True, max_samples=1000)
model = foz.load_zoo_model("mobilenet-v2-imagenet-torch")

embeddings = dataset.compute_embeddings(model)
print(embeddings.shape)



similarity_matrix = cosine_similarity(embeddings)
print(similarity_matrix.shape)
print(similarity_matrix)




n = len(similarity_matrix)
similarity_matrix = similarity_matrix - np.identity(n)

from tqdm.auto import tqdm

id_map = [s.id for s in dataset.select_fields(["id"])]

for idx, sample in enumerate(tqdm(dataset)):
    sample["max_similarity"] = similarity_matrix[idx].max()
    sample.save()

session = fo.launch_app(dataset, remote=True)#, port="9898")

