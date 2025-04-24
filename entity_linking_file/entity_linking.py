"""
File with various functions and data necessary for entity linking
using an approach with BERT and one similar to the Bari approach
"""

import csv
import math
from math import ceil
import numpy as np
import torch
import time
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
import gc
import os

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


def compute_embeddings(text_list, transformer, batch_size=1):
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
        id = row[6]
        try:
            # Convert serving size
            serving_size = float(row[4]) if row[4] else 0
            if serving_size == 0:
                normalized_row = (
                    [title] + [0 for i in range(1, 5)] + [normalized_title] + [id]
                )
            else:
                normalized_row = (
                    [title]
                    + [
                        float(row[i]) * 100 / serving_size if row[i] else 0
                        for i in range(1, 4)
                    ]
                    + [normalized_title]+ [id]
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
        or (
            carbs1 is None
            or fats1 is None
            or proteins1 is None
            or carbs2 is None
            or fats2 is None
            or proteins2 is None
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


def calculate_batch_macronutrient_similarities(list1_macros, list2_macros):
    """
    Vectorized calculation of macronutrient similarities for all pairs.
    
    :param list1_macros: list of macronutrient tuples from list1
    :param list2_macros: list of macronutrient tuples from list2
    :return: matrix of similarity values
    """
    results = torch.zeros((len(list1_macros), len(list2_macros)))
    
    # Filter out invalid macronutrient values
    valid_list1_indices = []
    valid_list1_macros = []
    for i, (carbs1, fats1, proteins1) in enumerate(list1_macros):
        if (carbs1 == 0 and fats1 == 0 and proteins1 == 0) or \
           (carbs1 == "" or fats1 == "" or proteins1 == "") or \
           (carbs1 is None or fats1 is None or proteins1 is None):
            continue
        valid_list1_indices.append(i)
        valid_list1_macros.append((float(carbs1), float(fats1), float(proteins1)))
    
    valid_list2_indices = []
    valid_list2_macros = []
    for j, (carbs2, fats2, proteins2) in enumerate(list2_macros):
        if (carbs2 == 0 and fats2 == 0 and proteins2 == 0) or \
           (carbs2 == "" or fats2 == "" or proteins2 == "") or \
           (carbs2 is None or fats2 is None or proteins2 is None):
            continue
        valid_list2_indices.append(j)
        valid_list2_macros.append((float(carbs2), float(fats2), float(proteins2)))
    
    # If either list is empty after filtering, return the zero matrix
    if not valid_list1_macros or not valid_list2_macros:
        return results
    
    # Convert to tensors for faster computation
    list1_tensor = torch.tensor(valid_list1_macros, dtype=torch.float32)
    list2_tensor = torch.tensor(valid_list2_macros, dtype=torch.float32)
    
    # Calculate Euclidean distances using broadcasting
    # Shape: [len(valid_list1_macros), len(valid_list2_macros), 3]
    diff = list1_tensor.unsqueeze(1) - list2_tensor.unsqueeze(0)
    
    # Square the differences and sum over the last dimension
    # Shape: [len(valid_list1_macros), len(valid_list2_macros)]
    squared_distances = torch.sum(diff ** 2, dim=2)
    
    # Square root to get the Euclidean distances
    distances = torch.sqrt(squared_distances)
    
    # Normalize and convert to similarity
    normalized_distances = distances / math.sqrt(2 * 100**2)
    similarities = 0.05 - (normalized_distances * 0.1)
    
    # Fill the results tensor with the calculated similarities
    for i_idx, i in enumerate(valid_list1_indices):
        for j_idx, j in enumerate(valid_list2_indices):
            results[i, j] = similarities[i_idx, j_idx]
    
    return results


def find_k_most_similar_pairs_with_indicators(
    list1, list2, k=1, model="paraphrase-MiniLM-L3-v2", use_indicator=False, batch_size=1, device=None):
    """
    finds the k most similar items from list2 for each item in list1,
    considering both cosine similarity and the macronutrient similarity index.

    :param list1: list of tuples (name, carbohydrates, fats, proteins, id, name_normalized)
    :param list2: list of tuples (name, carbohydrates, fats, proteins, id, name_normalized)
    :param k: number of most similar items to return
    :param use_indicator: whether to use indicators or not
    :return: list of tuples (item1, item2, similarity_value, id1, id2)
    """    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    model = SentenceTransformer(model, trust_remote_code=True).to(device)

    if not use_indicator:
        # Ensure each element in list1 and list2 has 4 elements by appending (0, 0, 0)
        list1 = [(item[0], 0, 0, 0, 0, item[-1]) for item in list1]
        list2 = [(item[0], 0, 0, 0, 0, item[-1]) for item in list2]

    # Extract only the normalized names to calculate embeddings
    names1 = [t[-1] for t in list1]
    names2 = [t[-1] for t in list2]

    # Calculate embeddings for both lists at once
    embeddings1 = model.encode(sentences=names1, convert_to_tensor=True, device=device, batch_size=batch_size)
    embeddings2 = model.encode(sentences=names2, convert_to_tensor=True, device=device, batch_size=batch_size)

    # Calculate all cosine similarities at once
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    #cosine_scores = model.similarity(embeddings1, embeddings2)

    # Get macronutrient data if needed
    if use_indicator:
        list1_macros = [item[1:4] for item in list1]
        list2_macros = [item[1:4] for item in list2]
        
        # Calculate all macronutrient similarities at once
        macro_similarities = calculate_batch_macronutrient_similarities(list1_macros, list2_macros)
        
        # Add to cosine scores
        total_scores = cosine_scores + macro_similarities.to(device)
    else:
        total_scores = cosine_scores

    most_similar_tuples = []

    # Process each item to find top k matches
    for i, item1 in enumerate(list1):
        # Get scores for this item
        item_scores = total_scores[i]
        if k > 0:
            # Convert to CPU for sorting
            item_scores_cpu = item_scores.cpu()
            
            # Get indices of top k scores
            top_k_indices: torch.Tensor = torch.topk(item_scores_cpu, min(k, len(list2)), dim=0).indices
        else:
            item_scores_cpu = item_scores.cpu()
            # Sort all scores
            top_k_indices = indices = torch.arange(len(item_scores_cpu))

            
        # Add results
        for idx in top_k_indices:
            idx = idx.item()
            item2 = list2[idx]  #type: ignore
            score = item_scores_cpu[idx].item()  #type: ignore
            most_similar_tuples.append(
                (
                    score,
                    item1[0],
                    item2[0],
                    item2[-2],
                    item1[-2]
                )
            )

    # Clean up GPU memory
    del embeddings1, embeddings2, cosine_scores, total_scores
    if use_indicator and 'macro_similarities' in locals():
        del macro_similarities  #type: ignore
    torch.cuda.empty_cache()
    gc.collect()

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
    list1 = [(row[0], 0, 0, 0, row[0]) for row in data]  # off entities
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
            if(threshold == "."):
                print(threshold, original_off, linked_foodkg)
            if float(similarity) > float(threshold):

                considered_count += 1
            if show_progress:
                print(
                    f"Original OFF: {original_off}, Linked FoodKG: {linked_foodkg}, \n  Similarity: {similarity:.2f}, \n  Correct: {linked_foodkg.lower() == data[i][1].lower()}\n"
                )
            expected_foodkg = data[i][1]
            if linked_foodkg.lower().strip() == expected_foodkg.lower().strip():
                correct_count += 1
                if float(similarity) > float(threshold):
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

    tokenizer =  AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    model1 = AutoModel.from_pretrained(model, trust_remote_code=True)

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



def calcolate_embeddings(header, file_input, file_output, chunk_size, batch_size, model1):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model: SentenceTransformer = SentenceTransformer(model_name_or_path=model1, trust_remote_code=True).to(device)

    with open(file_input, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  
        list_foodkg_recipe = [row[0] for row in reader]  

    len_1 = 0
    if os.path.exists(file_output):
        with open(file_output) as f:
            len_1 = sum(1 for _ in f) - 1
        mode = "a"
        list_foodkg_recipe = list_foodkg_recipe[len_1:]  
    else:
        mode = "w"

    print(f"Modalità: {mode}, Partenza da riga: {len_1}")
    numero_chunk = ceil(len(list_foodkg_recipe)/chunk_size)
    print(f"Numero chunk: {numero_chunk}")

    start_time = time.time()

    with open(file_output, mode=mode, newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if mode == "w":
            writer.writerow(header)

        for chunk_count in range(numero_chunk):
            chunk_start_time = time.time()
            print(f"Elaborazione chunk {chunk_count+1} di {numero_chunk}...")

            start_idx = chunk_count * chunk_size
            end_idx = (chunk_count + 1) * chunk_size
            list_foodkg_recipe_temp = list_foodkg_recipe[start_idx:end_idx]

            embeddings1 = model.encode(
                sentences=list_foodkg_recipe_temp,
                convert_to_tensor=False,
                device=device,
                batch_size=batch_size
            )

            writer.writerows(zip(list_foodkg_recipe_temp, map(lambda x: x.tolist(), embeddings1)))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            chunk_time = time.time() - chunk_start_time
            remaining_chunks = numero_chunk - (chunk_count + 1)
            estimated_time_remaining = (chunk_time * remaining_chunks) / 60
            print(f"Chunk {chunk_count+1} completato in {chunk_time:.2f} sec. Tempo stimato rimanente: {estimated_time_remaining:.2f} min")

    total_time = time.time() - start_time
    print(f"Processo completato in {total_time / 60:.2f} minuti.")


def read_file_rows(file):
    with open(file, 'rb') as f:
        count = 0
        buffer_size = 1024 * 1024
        while chunk := f.read(buffer_size):
            count += chunk.count(b'\n')
    return count - 1 


def merge_embeddingaaa(header, file1, file2, file_output, chunk_size, threshold=0.85):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device: ", device)

    num_rows1 = read_file_rows(file1)
    num_rows2 = read_file_rows(file2)
    num_chunks1 = (num_rows1 + chunk_size - 1) // chunk_size
    num_chunks2 = (num_rows2 + chunk_size - 1) // chunk_size

    print(f"File 1: {num_rows1} righe ({num_chunks1} chunk)")
    print(f"File 2: {num_rows2} righe ({num_chunks2} chunk)")
    
    with open(file_output, mode="w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow([header[0], header[1], "cosine_similarity"])  

        with open(file1, newline="", encoding="utf-8") as f1:
            reader1 = csv.reader(f1)
            next(reader1)  
            
            start_time = time.time()
            
            for chunk1_idx in range(num_chunks1):
                chunk_start_time = time.time()
                print(f"Processing chunk {chunk1_idx+1}/{num_chunks1} of file 1...")

                chunk1 = [next(reader1, None) for _ in range(chunk_size)]
                chunk1 = [row for row in chunk1 if row] 
                
                names1 = [row[0] for row in chunk1]
                embeddings1 = torch.tensor([eval(row[1]) for row in chunk1]).to(device)

                with open(file=file2, newline="", encoding="utf-8") as f2:
                    reader2 = csv.reader(f2)
                    next(reader2)  
                    
                    for chunk2_idx in range(num_chunks2):
                        print(f"  Processing chunk {chunk2_idx+1}/{num_chunks2} of file 2 {chunk1_idx}/{num_chunks1} of file 1")
                        
                        chunk2 = [next(reader2, None) for _ in range(chunk_size)]
                        chunk2 = [row for row in chunk2 if row]

                        names2 = [row[0] for row in chunk2]
                        embeddings2 = torch.tensor([eval(row[1]) for row in chunk2]).to(device)

                        cosine_similarities = util.cos_sim(embeddings1, embeddings2)

                        for i, name1 in enumerate(names1):
                            for j, name2 in enumerate(names2):
                                similarity = cosine_similarities[i][j].item()
                                if similarity >= threshold:
                                    writer.writerow([name1, name2, similarity])
                        
                        del embeddings2
                        torch.cuda.empty_cache()

                del embeddings1
                torch.cuda.empty_cache()
                
                chunk_time = time.time() - chunk_start_time
                remaining_chunks = num_chunks1 - (chunk1_idx + 1)
                estimated_time_remaining = (chunk_time * remaining_chunks) / 60 
                
                print(f"Chunk {chunk1_idx+1} completato in {chunk_time:.2f} sec. Tempo stimato rimanente: {estimated_time_remaining:.2f} min")

    total_time = time.time() - start_time
    print(f"Processo completato in {total_time / 60:.2f} minuti.")




def merge_embedding(header, file1, file2, file_output, chunk_size, threshold=0.85):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device: ", device)

    num_rows1 = read_file_rows(file1)
    num_rows2 = read_file_rows(file2)
    num_chunks1 = (num_rows1 + chunk_size - 1) // chunk_size
    num_chunks2 = (num_rows2 + chunk_size - 1) // chunk_size
    total_chunks = num_chunks1 * num_chunks2
    chunks_processed = 0
    
    print(f"File 1: {num_rows1} righe ({num_chunks1} chunk)")
    print(f"File 2: {num_rows2} righe ({num_chunks2} chunk)")
    print(f"Totale combinazioni di chunk da processare: {total_chunks}")
    
    # Creiamo un file temporaneo per ogni chunk di risultati
    temp_files = []
    
    global_start_time = time.time()
    
    # Elaboriamo un chunk alla volta per entrambi i file
    for chunk1_idx in range(num_chunks1):
        chunk1_start_time = time.time()
        print(f"Processing chunk {chunk1_idx+1}/{num_chunks1} of file 1...")
        
        # Carichiamo un chunk del file 1
        with open(file1, newline="", encoding="utf-8") as f1:
            reader1 = csv.reader(f1)
            next(reader1)  # Saltiamo l'intestazione
            
            # Avanziamo fino al chunk corrente
            for _ in range(chunk1_idx * chunk_size):
                next(reader1, None)
            
            # Leggiamo il chunk corrente
            chunk1 = []
            for _ in range(chunk_size):
                row = next(reader1, None)
                if row:
                    chunk1.append(row)
        
        names1 = [row[0] for row in chunk1]
        embeddings1 = torch.tensor([eval(row[1]) for row in chunk1]).to(device)
        
        # Creiamo un file temporaneo per i risultati di questo chunk
        temp_file = f"temp_results_{chunk1_idx}_{header}.csv"
        temp_files.append(temp_file)
        
        with open(temp_file, mode="w", newline="", encoding="utf-8") as temp_out:
            temp_writer = csv.writer(temp_out)
            
            # Elaboriamo il file 2 un chunk alla volta
            for chunk2_idx in range(num_chunks2):
                chunk2_start_time = time.time()
                chunks_processed += 1
                print(f"  Processing chunk {chunk2_idx+1}/{num_chunks2} of file 2 (against chunk {chunk1_idx+1}/{num_chunks1} of file 1)")
                
                # Carichiamo un chunk del file 2
                with open(file2, newline="", encoding="utf-8") as f2:
                    reader2 = csv.reader(f2)
                    next(reader2)  # Saltiamo l'intestazione
                    
                    # Avanziamo fino al chunk corrente
                    for _ in range(chunk2_idx * chunk_size):
                        next(reader2, None)
                    
                    # Leggiamo il chunk corrente
                    chunk2 = []
                    for _ in range(chunk_size):
                        row = next(reader2, None)
                        if row:
                            chunk2.append(row)
                
                names2 = [row[0] for row in chunk2]
                embeddings2 = torch.tensor([eval(row[1]) for row in chunk2]).to(device)
                
                # Calcolo batch delle similarità
                cosine_similarities = util.cos_sim(embeddings1, embeddings2)
                
                # Utilizziamo un approccio vettorizzato per trovare le corrispondenze
                matches = torch.where(cosine_similarities >= threshold)
                
                # Scriviamo i risultati
                for idx1, idx2 in zip(matches[0].tolist(), matches[1].tolist()):
                    similarity = cosine_similarities[idx1, idx2].item()
                    temp_writer.writerow([names1[idx1], names2[idx2], similarity])
                
                # Liberiamo la memoria GPU
                del embeddings2
                del cosine_similarities
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()  # Forziamo il garbage collector
                
                # Calcoliamo il tempo stimato rimanente dopo ogni chunk di file2
                chunk2_time = time.time() - chunk2_start_time
                remaining_chunks = total_chunks - chunks_processed
                chunks_per_second = 1.0 / chunk2_time if chunk2_time > 0 else 0
                estimated_time_remaining = remaining_chunks / chunks_per_second if chunks_per_second > 0 else 0
                
                # Statistiche temporali per il chunk attuale
                elapsed_time = time.time() - global_start_time
                completion_percentage = (chunks_processed / total_chunks) * 100
                
                print(f"  Chunk {chunk2_idx+1}/{num_chunks2} (file 2) completato in {chunk2_time:.2f} sec.")
                print(f"  Progresso: {chunks_processed}/{total_chunks} chunks ({completion_percentage:.2f}%)")
                print(f"  Tempo trascorso: {elapsed_time/60:.2f} min, Tempo stimato rimanente: {estimated_time_remaining/60:.2f} min")
                print(f"  Velocità attuale: {chunks_per_second:.4f} chunks/sec")
        
        # Liberiamo la memoria
        del embeddings1
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        # Calcoliamo statistiche per l'intero chunk del file1
        chunk1_time = time.time() - chunk1_start_time
        remaining_file1_chunks = num_chunks1 - (chunk1_idx + 1)
        estimated_file1_time_remaining = chunk1_time * remaining_file1_chunks
        
        print(f"Chunk {chunk1_idx+1}/{num_chunks1} (file 1) completato in {chunk1_time/60:.2f} min.")
        print(f"Tempo stimato per completare i chunk rimanenti del file 1: {estimated_file1_time_remaining/60:.2f} min")
    
    # Combiniamo tutti i file temporanei nel file di output finale
    print("Combinazione dei risultati temporanei nel file di output finale...")
    combination_start_time = time.time()
    
    with open(file_output, mode="w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["name_file1", "name_file2", "cosine_similarity"])
        
        for i, temp_file in enumerate(temp_files):
            temp_combination_start = time.time()
            print(f"Processando file temporaneo {i+1}/{len(temp_files)}...")
            
            with open(temp_file, newline="", encoding="utf-8") as fin:
                reader = csv.reader(fin)
                for row in reader:
                    writer.writerow(row)
            
            # Rimuoviamo il file temporaneo dopo l'uso
            os.remove(temp_file)
            
            temp_combination_time = time.time() - temp_combination_start
            remaining_temp_files = len(temp_files) - (i + 1)
            estimated_combination_time = temp_combination_time * remaining_temp_files
            
            print(f"File temporaneo {i+1}/{len(temp_files)} processato in {temp_combination_time:.2f} sec.")
            print(f"Tempo stimato per completare la combinazione: {estimated_combination_time:.2f} sec.")
    
    combination_time = time.time() - combination_start_time
    total_time = time.time() - global_start_time
    
    print(f"Combinazione completata in {combination_time/60:.2f} minuti.")
    print(f"Processo complessivo completato in {total_time/60:.2f} minuti.")