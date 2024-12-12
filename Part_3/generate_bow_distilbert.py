import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from transformers import DistilBertTokenizer, DistilBertModel
from annoy import AnnoyIndex
import torch
import numpy as np

# Charger les métadonnées en assurant le prétraitement
metadata = pd.read_csv("data/part 3/movies_metadata.csv")

# Remplacer les valeurs manquantes dans la colonne overview
metadata["overview"] = metadata["overview"].fillna("")

# Supprimer les lignes sans titre
metadata = metadata[metadata["title"].notnull()]

# Extraire les descriptions des films
plots = metadata["overview"].tolist()

# Bag-of-Words Embeddings
vectorizer = CountVectorizer(max_features=5000)
bow_embeddings = vectorizer.fit_transform(plots).toarray()

# DistilBERT Embeddings
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model.eval()

# Désactiver CUDA si non disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

distilbert_embeddings = []

for plot in plots:
    inputs = tokenizer(plot, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    distilbert_embeddings.append(embedding)

distilbert_embeddings = np.array(distilbert_embeddings)

# Sauvegarder les embeddings
with open("bag_of_words_embeddings.pkl", "wb") as f:
    pickle.dump(bow_embeddings, f)

with open("distilbert_embeddings.pkl", "wb") as f:
    pickle.dump(distilbert_embeddings, f)

# Créer les indices Annoy
bow_annoy = AnnoyIndex(bow_embeddings.shape[1], "angular")
distilbert_annoy = AnnoyIndex(distilbert_embeddings.shape[1], "angular")

for i, (bow_vector, distilbert_vector) in enumerate(zip(bow_embeddings, distilbert_embeddings)):
    bow_annoy.add_item(i, bow_vector)
    distilbert_annoy.add_item(i, distilbert_vector)

# Construire et sauvegarder les indices
bow_annoy.build(10)
distilbert_annoy.build(10)

bow_annoy.save("bow_annoy.ann")
distilbert_annoy.save("distilbert_annoy.ann")

# Sauvegarder les métadonnées nettoyées
metadata.to_pickle("movies_metadata.pkl")