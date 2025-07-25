import re
import pandas as pd
from rdflib import RDF, RDFS, XSD, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL
from sentence_transformers import SentenceTransformer, util
import ast
import gc 
import os
import time


def sanitize_for_uri(value) -> str:
    """
    Generic sanitization function for URIs

    :param value: value to sanitize

    :return: sanitized value
    """
    return re.sub(r"[^a-zA-Z0-9_]", "", str(value))

UNICA = Namespace("https://github.com/tail-unica/kgeats/")
SCHEMA = Namespace("https://schema.org/")

dict_hum = {}
dict_off = {}

hum_file = "../csv_file/pp_recipes_normalized_by_pipeline.csv"
off_file = "../csv_file/off_normalized_final.csv"
hum_off_file = "../csv_file/file_off_hummus_filtered_975.csv"

chunksize = 200000
cont_chunk = 0

for df_off_chunk in pd.read_csv(off_file, sep="\t", on_bad_lines="skip", chunksize=chunksize, low_memory=False, usecols=["product_name_normalized", "code"]):
    print(f"Processing rows off from {chunksize * cont_chunk} to {chunksize * (cont_chunk+1)}")
    
    for idx, row in df_off_chunk.iterrows():
        if(row["product_name_normalized"] is not None and row["product_name_normalized"] != ""):
            id = URIRef(value=UNICA[f"Recipe_off_{row['code']}"])
            if id is not None:
                if row["product_name_normalized"] not in dict_off:
                    dict_off[row["product_name_normalized"]] = [id]
                else: 
                    dict_off[row["product_name_normalized"]].append(id)
    cont_chunk += 1

cont_chunk = 0
for df_hum_chunk in pd.read_csv(hum_file, sep=";", on_bad_lines="skip", chunksize=chunksize, low_memory=False, usecols=["title_normalized", "recipe_id"]):
    print(f"Processing rows hummus from {chunksize * cont_chunk} to {chunksize * (cont_chunk+1)}")
    
    for idx, row in df_hum_chunk.iterrows():
        if(row["title_normalized"] is not None and row["title_normalized"] != ""):
            id = URIRef(UNICA[f"Recipe_hummus{sanitize_for_uri(row['recipe_id'])}"])
            if id is not None:
                if row["title_normalized"] not in dict_hum:
                    dict_hum[row["title_normalized"]] = [id]
                else: 
                    dict_hum[row["title_normalized"]].append(id)
    cont_chunk += 1


import numpy as np

numchunk = 0
chunksize = 10000
counter = 0
all_association = []
all_association_dictionary = {}

hum_off_file_df = pd.read_csv(hum_off_file, sep=",", low_memory=False, on_bad_lines="skip")
total_lines  = len(hum_off_file_df)

total_chunks = (total_lines // chunksize) + 1
start_total = time.time()


for df_merge_chunk in pd.read_csv(hum_off_file, sep=",", on_bad_lines="skip", chunksize=chunksize, low_memory=False, usecols=["title_normalized", "product_name_normalized"]):
    chunk_start = time.time()
    print(f"\nProcessing chunk {numchunk+1}/{total_chunks}")

    for row in df_merge_chunk.itertuples(index=False):
        title = row.title_normalized
        product = row.product_name_normalized

        if title in dict_hum and product in dict_off:
            for hum_recipe in dict_hum[title]:
                conta_ricette = 0
                for off_recipe in dict_off[product]: 
                    counter += 1
                    conta_ricette +=1
                all_association.append(conta_ricette)

                if hum_recipe not in all_association_dictionary:
                    all_association_dictionary[hum_recipe] = 1

    del df_merge_chunk
    gc.collect() 

    chunk_time = time.time() - chunk_start
    avg_time_per_chunk = (time.time() - start_total) / (numchunk + 1)
    remaining_chunks = total_chunks - (numchunk + 1)
    est_remaining = avg_time_per_chunk * remaining_chunks
    print(f"Chunk time: {chunk_time:.2f}s — Estimated remaining: {est_remaining/60:.1f} min")
    print(counter)
    numchunk += 1

total_time = time.time() - start_total
print(f"\nTotal processing time: {total_time/60:.2f} minutes")



from numpy import floating

hum_df = pd.read_csv(hum_file, sep=";", low_memory=False, on_bad_lines="skip")
total_lines  = len(hum_df)

mean_association = sum(all_association) / total_lines

print("number of recipes with an association: ", len(all_association_dictionary), "/", total_lines)

all_associated = all_association.copy()

for i in range(total_lines - len(all_association_dictionary)):
    all_associated.append(0)

median_association = np.median(all_association)
median_associated= np.median(all_associated)

print("Average number of associations: ", mean_association)
print("Median of associations: ", median_association)
print("Median of associations for all recipes: ", median_associated)
print("Percentage of associated Hummus recipes: ", (len(all_association_dictionary) * 100 / total_lines), "%")
print("Total associations: ", counter)
