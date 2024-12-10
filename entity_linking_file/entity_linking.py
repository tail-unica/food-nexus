"""
File with various functions and data necessary for entity linking
using an approach with BERT and one similar to the Bari approach
"""

import csv
import math
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
import gc


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

    embeddings2 = model.encode(list2, convert_to_tensor=True, device=device)
    most_similar_pairs = []

    for item in list1:
        # Calculate the embedding for the current item in list1 and move it to the GPU
        embedding1 = model.encode([item], convert_to_tensor=True, device=device)
        # Calculate cosine similarity between the list1 embedding and all embeddings from list2
        cosine_scores = util.cos_sim(embedding1, embeddings2)[0]
        # Find the index of the element with the maximum similarity
        max_index = cosine_scores.argmax().item()
        max_score = cosine_scores[max_index].item()  # type: ignore
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

    input_embedding = transformer.process_batch([input_text])[0]

    assert (
        input_embedding.shape[0] == embeddings.shape[1]
    ), "Embedding dimensions do not match."

    # Convert the embeddings to numpy format if necessary
    if isinstance(input_embedding, torch.Tensor):
        input_embedding = input_embedding.cpu().numpy()
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    similarities = cosine_similarity([input_embedding], embeddings)[0]  # type: ignore

    max_index = np.argmax(similarities)
    max_score = similarities[max_index]

    return entities_list[max_index], max_score


def read_specified_columns(
    file_path: str,
    elenco_colonne: list,
    delimiter: str = ";",
    encoding: str = "utf-8",
    max_cell_length: int = 1000,
) -> list:
    """
    Function to read specified columns from a CSV file, ignoring rows with overly long cell values.

    :param file_path: path to the CSV file
    :param elenco_colonne: list of column names to include
    :param delimiter: delimiter used in the CSV file (default: ';')
    :param encoding: file encoding (default: 'utf-8')
    :param max_cell_length: maximum allowed length for a cell value (default: 1000)
    :return: a list of tuples, each containing the values from the specified columns
    """
    csv.field_size_limit(1_000_000_000)

    with open(file_path, "r", newline="", encoding=encoding) as csvfile:
        # Use DictReader to access column values by name
        reader = csv.DictReader(csvfile, delimiter=delimiter)

        # Check if the specified columns exist in the file
        if not set(elenco_colonne).issubset(reader.fieldnames):  # type: ignore
            raise ValueError(
                "One or more specified columns do not exist in the CSV file"
            )

        # Filter rows and check for overly long cell values
        filtered_rows = []
        for row in reader:
            # Check if all values in the specified columns are below the length threshold
            if all(
                len(row[col]) <= max_cell_length
                for col in elenco_colonne
                if row[col]
            ):
                filtered_rows.append(tuple(row[col] for col in elenco_colonne))

        return filtered_rows


def normalize_columns(data: list) -> list:
    """
    Normalize the values of the specified columns (second, third, and fourth) based on the value of the fifth column.

    :param data: List of tuples created by the original script.
                 Each tuple contains the values for the columns: [title, col2, col3, col4, servingSize, title_normalized].
    :return: A new list containing the title and normalized columns.
    """
    normalized_data = []

    for row in data:
        # First column: title
        title = row[0]
        normalized_title = row[5]
        try:
            # Convert serving size
            serving_size = float(row[4]) if row[4] else 0
            if serving_size == 0:
                normalized_row = (
                    [title] + [0 for i in range(1, 5)] + [normalized_title]
                )
            else:
                normalized_row = (
                    [title]
                    + [
                        float(row[i]) * 100 / serving_size if row[i] else 0
                        for i in range(1, 4)
                    ]
                    + [normalized_title]
                )
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
    carbs1, fats1, proteins1 = tuple1
    carbs2, fats2, proteins2 = tuple2

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


def find_k_most_similar_pairs_with_indicators(
    list1, list2, k=1, model="paraphrase-MiniLM-L3-v2", use_indicator=False, batch_size=2
):
    """
    Finds the k most similar items from list2 for each item in list1, considering
    both cosine similarity and the macronutrient similarity index.

    :param list1: list of tuples (name, carbohydrates, fats, proteins, optional string)
    :param list2: list of tuples (name, carbohydrates, fats, proteins, optional string)
    :param k: number of most similar items to return
    :param use_indicator: whether to use indicators or not
    :return: list of tuples (item1, item2, similarity_value, indicator1, indicator2)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    model = SentenceTransformer(model, trust_remote_code=True).to(device)

    if not use_indicator:
        # Ensure each element in list1 and list2 has 4 elements by appending (0, 0, 0)
        list1 = [(item[0], 0, 0, 0, item[-1]) for item in list1]
        list2 = [(item[0], 0, 0, 0, item[-1]) for item in list2]

    # Extract only the names to calculate embeddings
    names2 = [t[0] for t in list2]

    # Calculate embeddings for list2
    embeddings2 = model.encode(names2, convert_to_tensor=True, device=device, batch_size=batch_size)
    most_similar_tuples = []

    for item in list1:
        # Calculate the embedding for the current name
        embedding1 = model.encode(
            [item[0]], convert_to_tensor=True, device=device, batch_size=batch_size
        )
        # Calculate cosine similarity
        cosine_scores = util.cos_sim(embedding1, embeddings2)[0]

        # Calculate the total combined score for each item in list2
        total_scores = []
        for j, tuple2 in enumerate(list2):
            macronutrient_similarity = calculate_macronutrient_similarity(
                item[1:4], tuple2[1:4]
            )
            total_score = cosine_scores[j].item() + macronutrient_similarity
            total_scores.append(
                (tuple2[0], total_score, tuple2[-1])
            )  # Name, score, and indicator

        # If k<0, take all items
        if k > 0:
            # Sort by score in descending order and take the top k items
            top_k_scores = sorted(
                total_scores, key=lambda x: x[1], reverse=True
            )[:k]
        else:
            top_k_scores = sorted(
                total_scores, key=lambda x: x[1], reverse=True
            )

        # Add the k most similar pairs to the result list as tuples with indicators
        for pair in top_k_scores:
            most_similar_tuples.append(
                (
                    pair[1],
                    item[0],
                    pair[0],
                )
            )

    return most_similar_tuples


def read_csv(file_path):
    """Reads a CSV file and returns a list of tuples."""
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        data = [(row["off"], row["foodkg"]) for row in reader]
    return data


def evaluate_entity_linking_method(
    data,
    model="paraphrase-MiniLM-L3-v2",
    show_progress=False,
    threshold_list=[0.5],
):
    """
    Evaluates the accuracy of an entity linking method.

    :param data: list of tuples (off, foodkg)
    :param method: entity linking method
    :param transformer: (optional) transformer for the second method
    :param embeddings: (optional) precomputed embeddings for the second method
    :return: accuracy as a percentage
    """
    # Prepare data for the entity linking method
    list1 = [(row[0], 0, 0, 0, row[0]) for row in data]  # Off entities
    list2 = [(row[1], 0, 0, 0, row[1]) for row in data]  # FoodKG entities

    # Apply the method
    k = 1  # Number of results to consider
    linked_entities = find_k_most_similar_pairs_with_indicators(
        list1, list2, k=k, model=model
    )

    # Evaluate results
    accuracy_list = []
    considered_list = []
    accuracy_considered_list = []
    for j, threshold in enumerate(threshold_list):
        correct_count = 0
        correct_considered_count = 0
        considered_count = 0
        for i, (similarity, original_off, linked_foodkg) in enumerate(
            linked_entities
        ):
            if similarity > threshold:

                considered_count += 1
            if show_progress:
                print(
                    f"Original OFF: {original_off}, Linked FoodKG: {linked_foodkg}, \n  Similarity: {similarity:.2f}, \n  Correct: {linked_foodkg.lower() == data[i][1].lower()}\n"
                )
            expected_foodkg = data[i][1]
            if linked_foodkg.lower().strip() == expected_foodkg.lower().strip():
                correct_count += 1
                if similarity > threshold:
                    correct_considered_count += 1

        accuracy = (correct_count / len(data)) * 100
        if considered_count != 0:
            accuracy_considered = (
                correct_considered_count / considered_count
            ) * 100
        else:
            accuracy_considered = 0
        accuracy_list.append(accuracy)
        considered_list.append(considered_count)
        accuracy_considered_list.append(accuracy_considered)

    tokenizer =  AutoTokenizer.from_pretrained(model)
    model1 = AutoModel.from_pretrained(model)

    vocab_size = tokenizer.vocab_size
    number_of_parameters = model1.num_parameters()

    del model1
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return (
        [model for i in range(len(threshold_list))],
        [vocab_size for i in range(len(threshold_list))],
        [number_of_parameters for i in range(len(threshold_list))],
        accuracy_list,
        accuracy_considered_list,
        considered_list,
        threshold_list,
    )
