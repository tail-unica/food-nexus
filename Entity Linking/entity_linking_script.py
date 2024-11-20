"""
File con varie funzioni e dati necessari per l'entity linking
usando un approccio con bert e uno con simile a quello di bari
"""


import torch
from sentence_transformers import SentenceTransformer, util
import csv
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer


def read_first_column(file_path):
    """
    Funzione per leggere la prima colonna da un file CSV

    :param file_path: percorso del file CSV
    :return: una lista con gli elementi della prima colonna
    """
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        return [row[0] for row in reader if row]


def find_most_similar_pairs(list1, list2):
    """
    Funzione per trovare l'elemento di list2 più simile per ogni elemento di list1

    :param list1: lista di elementi
    :param list2: lista di elementi
    :return: lista di tuple degli elementi di list1 con l'elemento di list2 più simile e la sua similarità
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Carica il modello 
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2').to(device)
    #paraphrase-MiniLm-L3-v2 
    #paraphrase-MiniLM-L12-v2

    embeddings2 = model.encode(list2, convert_to_tensor=True, device=device)
    most_similar_pairs = []

    for item in list1:
        # Calcola l'embedding per l'elemento corrente di list1 e spostalo sulla GPU
        embedding1 = model.encode([item], convert_to_tensor=True, device=device)
        # Calcola la similarità coseno tra l'embedding di list1 e tutti gli embeddings di list2
        cosine_scores = util.cos_sim(embedding1, embeddings2)[0]
        # Trova l'indice dell'elemento con la similarità massima
        max_index = cosine_scores.argmax().item()
        max_score = cosine_scores[max_index].item()
        # Aggiungi la coppia con la similarità massima alla lista dei risultati
        most_similar_pairs.append((item, list2[max_index], max_score))

    return most_similar_pairs


class RecipeTransformer:
    """
    Classe che rappresenta il transformer usato per creare gli embeddings
    """
    def __init__(self, transformer_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        self.model = AutoModel.from_pretrained(transformer_name).to(self.device)
        torch.cuda.empty_cache() 

    def process_batch(self, texts_batch):
        """
        funzione che processa un batch di testi per produrre gli embeddings

        :param texts_batch: il testo da processare
        :return: gli embedding risultanti dal testo 
        """

        embeddings = []
        for text in tqdm(texts_batch, desc="Processing Titles embeddings", unit="batch"):
            batch = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
        return np.vstack(embeddings)


def compute_embeddings(text_list, transformer, batch_size=1500) :
    """
    Funzione per calcolare gli embedding di una lista di testi usando il transformer

    :param text_list: lista di testi
    :param transformer: modello transformer
    :param batch_size: dimensione del batch
    :return: matrice di embeddings
    """
    embeddings = []
    num_texts = len(text_list)

    for i in tqdm(range(0, num_texts, batch_size), desc="Calcolo embedding"):
        batch = text_list[i:i + batch_size]
        batch_embeddings = transformer.process_batch(batch)
        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)


def find_similar_by_title(input_text, entities_list, embeddings, transformer):
    """
    Funzione per trovare la ricetta più simile usando cosine similarity applicata agli embeddings

    :param input_text: una ricetta da associare a una della lista
    :param entities_list: lista di ricette
    :param embeddings: le embeddings delle ricette della seconda lista
    :param transformer: il modello transformer
    :return: la ricetta più simile ad input text e la sua similarità
    """
    
    input_embedding = transformer.process_batch([input_text])[0] #.numpy()
    similarities = np.dot(embeddings, input_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(input_embedding) + 1e-10)

    max_index = np.argmax(similarities)
    max_score = similarities[max_index]

    return entities_list[max_index], max_score