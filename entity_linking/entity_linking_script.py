"""
File with various functions and data necessary for entity linking
using an approach with BERT and one similar to the Bari approach
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
    Function to read the first column from a CSV file

    :param file_path: path to the CSV file
    :return: a list of elements from the first column
    """
    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        return [row[0] for row in reader if row]


def find_most_similar_pairs(list1, list2):
    """
    Function to find the most similar element from list2 for each element of list1

    :param list1: list of elements
    :param list2: list of elements
    :return: list of tuples containing elements from list1 and their most similar element from list2 with similarity score
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    model = SentenceTransformer("paraphrase-MiniLM-L3-v2").to(device)
    # paraphrase-MiniLm-L3-v2
    # paraphrase-MiniLM-L12-v2

    embeddings2 = model.encode(list2, convert_to_tensor=True, device=device)
    most_similar_pairs = []

    for item in list1:
        # Calculate the embedding for the current item in list1 and move it to the GPU
        embedding1 = model.encode([item], convert_to_tensor=True, device=device)
        # Calculate cosine similarity between the list1 embedding and all embeddings from list2
        cosine_scores = util.cos_sim(embedding1, embeddings2)[0]
        # Find the index of the element with the maximum similarity
        max_index = cosine_scores.argmax().item()
        max_score = cosine_scores[max_index].item()
        # Add the pair with the maximum similarity to the result list
        most_similar_pairs.append((item, list2[max_index], max_score))

    return most_similar_pairs


class RecipeTransformer:
    """
    Class representing the transformer used to create embeddings
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
        Function that processes a batch of texts to produce embeddings

        :param texts_batch: the text to process
        :return: the embeddings resulting from the text
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
    Function to compute embeddings for a list of texts using the transformer

    :param text_list: list of texts
    :param transformer: transformer model
    :param batch_size: batch size
    :return: matrix of embeddings
    """
    embeddings = []
    num_texts = len(text_list)

    for i in tqdm(
        range(0, num_texts, batch_size), desc="Calculating embeddings"
    ):
        batch = text_list[i : i + batch_size]
        batch_embeddings = transformer.process_batch(batch)
        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)


def find_similar_by_title(input_text, entities_list, embeddings, transformer):
    """
    Function to find the most similar recipe using cosine similarity applied to embeddings

    :param input_text: a recipe to match with one from the list
    :param entities_list: list of recipes
    :param embeddings: embeddings of the recipes from the second list
    :param transformer: transformer model
    :return: the most similar recipe to the input text and its similarity score
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
    Function to read specified columns from a CSV file

    :param file_path: path to the CSV file
    :param elenco_colonne: list of column names to include
    :return: a list of tuples, each containing the values from the specified columns
    """
    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:

        # Use DictReader to access column values by name
        reader = csv.DictReader(csvfile, delimiter=delimiter)

        # print(reader.fieldnames)
        if not set(elenco_colonne).issubset(reader.fieldnames):  # type: ignore
            raise ValueError(
                "One or more specified columns do not exist in the CSV file"
            )
        return [
            tuple(row[col] for col in elenco_colonne) for row in reader if row
        ]


def normalize_columns(data: list) -> list:
    """
    Normalize the values of the specified columns (second, third, and fourth) based on the value of the fifth column.

    :param data: List of tuples created by the original script.
                 Each tuple contains the values for the columns: [title, col2, col3, col4, servingSize].
    :return: A new list containing the title and normalized columns.
    """
    normalized_data = []

    for row in data:
        # First column: title
        title = row[0]
        try:
            # Convert serving size
            serving_size = float(row[4]) if row[4] else 0
            if serving_size == 0:
                normalized_row = [title] + [0 for i in range(1, 5)]
            else:
                normalized_row = [title] + [
                    float(row[i]) * 100 / serving_size if row[i] else 0
                    for i in range(1, 4)
                ]
            normalized_data.append(normalized_row)

        except ValueError as e:
            print(f"Error in normalizing row {row}: {e}")

    return normalized_data


def calculate_macronutrient_similarity(tuple1, tuple2):
    """
    Calculates the macronutrient similarity index between two tuples.

    :param tuple1: tuple with macronutrient values
    :param tuple2: tuple with macronutrient values

    :return: similarity index of the tuples' macronutrients, ranging from -0.5 to 0.5
    """
    _, carbs1, fats1, proteins1 = tuple1
    _, carbs2, fats2, proteins2 = tuple2

    # if any of the values are missing, return zero
    if (
        (carbs1 == 0 and fats1 == 0 and proteins1 == 0)
        or (carbs2 == 0 and fats2 == 0 and proteins2 == 0)
        or (
            carbs1 == ""
            or fats1 == ""
            or proteins1 == ""
            or carbs2 == ""
            or fats2 == ""
            or proteins2 == ""
        )
    ):
        return 0.0
    else:
        # Calculate the Euclidean distance
        d = math.sqrt(
            (float(carbs1) - float(carbs2)) ** 2
            + (float(fats1) - float(fats2)) ** 2
            + (float(proteins1) - float(proteins2)) ** 2
        )

        # Normalize the index between -0.05 and 0.05
        d_norm = d / math.sqrt(2 * 100**2)
        similarity = 0.05 - (d_norm * 0.1)
        return similarity


def find_most_similar_pairs_with_indicators(list1, list2):
    """
    Finds the most similar item from list2 for each item in list1, considering
    both cosine similarity and the macronutrient similarity index.

    :param list1: list of tuples (name, carbohydrates, fats, proteins)
    :param list2: list of tuples (name, carbohydrates, fats, proteins)
    :return: list of tuples in the form (item1, item2, similarity_value)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    model = SentenceTransformer("paraphrase-MiniLM-L3-v2").to(device)

    # Extract only the names to calculate embeddings
    names2 = [t[0] for t in list2]

    # Calculate embeddings for list2
    embeddings2 = model.encode(names2, convert_to_tensor=True, device=device)
    most_similar_pairs = []

    for i, item in enumerate(list1):
        # Calculate the embedding for the current name
        embedding1 = model.encode(
            [item[0]], convert_to_tensor=True, device=device
        )
        # Calculate cosine similarity
        cosine_scores = util.cos_sim(embedding1, embeddings2)[0]

        # Calculate the total combined score for each item in list2
        total_scores = []
        for j, tuple2 in enumerate(list2):
            macronutrient_similarity = calculate_macronutrient_similarity(
                item, tuple2
            )
            total_score = cosine_scores[j].item() + macronutrient_similarity
            total_scores.append(total_score)

        # Find the index with the highest total score
        max_index = total_scores.index(max(total_scores))
        max_score = total_scores[max_index]

        # Add the pair to the result list
        most_similar_pairs.append((item[0], list2[max_index][0], max_score))

    return most_similar_pairs
