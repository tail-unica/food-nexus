"""
File con varie funzioni e dati necessari per l'entity linking
usando un approccio con bert e uno con simile a quello di bari
"""


import torch  # type: ignore
from sentence_transformers import SentenceTransformer, util  # type: ignore
import csv
import math
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer  # type: ignore


def read_first_column(file_path):
    """
    Funzione per leggere la prima colonna da un file CSV

    :param file_path: percorso del file CSV
    :return: una lista con gli elementi della prima colonna
    """
    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        return [row[0] for row in reader if row]


def find_most_similar_pairs(list1, list2):
    """
    Funzione per trovare l'elemento di list2 più simile per ogni elemento di list1

    :param list1: lista di elementi
    :param list2: lista di elementi
    :return: lista di tuple degli elementi di list1 con l'elemento di list2 più simile e la sua similarità
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Carica il modello
    model = SentenceTransformer("paraphrase-MiniLM-L3-v2").to(device)
    # paraphrase-MiniLm-L3-v2
    # paraphrase-MiniLM-L12-v2

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
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
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
        for text in tqdm(
            texts_batch, desc="Processing Titles embeddings", unit="batch"
        ):
            batch = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
        return np.vstack(embeddings)


def compute_embeddings(text_list, transformer, batch_size=1500):
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
        batch = text_list[i : i + batch_size]
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

    input_embedding = transformer.process_batch([input_text])[0]  # .numpy()
    similarities = np.dot(embeddings, input_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(input_embedding)
        + 1e-10
    )

    max_index = np.argmax(similarities)
    max_score = similarities[max_index]

    return entities_list[max_index], max_score


def read_specified_columns(
    file_path: str, elenco_colonne: list, delimiter: str = ";"
) -> list:
    """
    Funzione per leggere specifiche colonne da un file CSV

    :param file_path: percorso del file CSV
    :param elenco_colonne: lista dei nomi delle colonne da includere
    :return: una lista di tuple, ciascuna contenente i valori delle colonne specificate
    """
    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:

        reader = csv.DictReader(
            csvfile, delimiter=delimiter
        )  # Usa DictReader per accedere ai valori delle colonne per nome

        # print(reader.fieldnames)
        if not set(elenco_colonne).issubset(reader.fieldnames):  # type: ignore
            raise ValueError(
                "Una o più colonne specificate non esistono nel file CSV"
            )
        return [
            tuple(row[col] for col in elenco_colonne) for row in reader if row
        ]


def normalize_columns(data: list) -> list:
    """
    Normalizza i valori delle colonne specificate (seconda, terza e quarta) in base al valore della quinta colonna.

    :param data: Lista di tuple creata dallo script originale.
                 Ogni tupla contiene i valori delle colonne: [title, col2, col3, col4, servingSize].
    :return: Una nuova lista contenente il titolo e le colonne normalizzate.
    """
    normalized_data = []

    for row in data:
        title = row[0]  # Prima colonna: titolo
        try:
            # Conversione di serving size
            serving_size = float(row[5]) if row[4] else 0
            if serving_size == 0:
                normalized_row = [title] + [0 for i in range(1, 5)]
            else:
                normalized_row = [title] + [
                    float(row[i]) * 100 / serving_size if row[i] else 0
                    for i in range(1, 5)
                ]
            normalized_data.append(normalized_row)

        except ValueError as e:
            print(f"Errore nella normalizzazione della riga {row}: {e}")

    return normalized_data


def calculate_macronutrient_similarity(tuple1, tuple2):
    """
    Calcola l'indice di similarità per i macronutrienti tra due tuple.

    :param tuple1: tupla con i macronutrienti
    :param tuple2: tupla con i macronutrienti

    :return: indice di vicinanza delle sue tuple di macronutrienti compresa tra -0,5 e 0,5
    """
    _, carbs1, fats1, proteins1 = tuple1
    _, carbs2, fats2, proteins2 = tuple2

    # se i valori di uno dei due sono mancanti torna zero
    if (carbs1 == 0 and fats1 == 0 and proteins1 == 0) or (
        carbs2 == 0 and fats2 == 0 and proteins2 == 0
    ):
        return 0.0
    else:
        # Calcolare la distanza euclidea
        d = math.sqrt(
            (carbs1 - carbs2) ** 2
            + (fats1 - fats2) ** 2
            + (proteins1 - proteins2) ** 2
        )

        # Converte l' indice tra -0.05 e 0.05
        d_norm = d / math.sqrt(2 * 100**2)
        similarity = 0.05 - (d_norm * 0.1)
        return similarity


def find_most_similar_pairs_with_indicators(list1, list2):
    """
    Trova l'elemento di list2 più simile per ogni elemento di list1 considerando
    cosine similarity e indice di similarità per i macronutrienti.

    :param list1: lista di tuple (nome, carboidrati, grassi, proteine)
    :param list2: lista di tuple (nome, carboidrati, grassi, proteine)
    :return: lista di tuple nella forma (elemento1, elemento2, valore_similarità)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Carica il modello
    model = SentenceTransformer("paraphrase-MiniLM-L3-v2").to(device)

    # Estrarre solo i nomi per calcolare gli embeddings
    names2 = [t[0] for t in list2]

    # Calcola gli embeddings per list2
    embeddings2 = model.encode(names2, convert_to_tensor=True, device=device)
    most_similar_pairs = []

    for i, item in enumerate(list1):
        # Calcola l'embedding per il nome corrente
        embedding1 = model.encode(
            [item[0]], convert_to_tensor=True, device=device
        )
        # Calcola la cosine similarity
        cosine_scores = util.cos_sim(embedding1, embeddings2)[0]

        # Calcola il punteggio totale combinato per ogni elemento in list2
        total_scores = []
        for j, tuple2 in enumerate(list2):
            macronutrient_similarity = calculate_macronutrient_similarity(
                item, tuple2
            )
            total_score = cosine_scores[j].item() + macronutrient_similarity
            total_scores.append(total_score)

        # Trova l'indice con il punteggio totale massimo
        max_index = total_scores.index(max(total_scores))
        max_score = total_scores[max_index]

        # Aggiungi la coppia alla lista dei risultati
        most_similar_pairs.append((item[0], list2[max_index][0], max_score))

    return most_similar_pairs
