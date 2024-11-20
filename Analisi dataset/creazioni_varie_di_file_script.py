"""
File che contiene script e dati necessari per la creazione di 
modelli LLM, analisi sui file CSV e sulle ontologie
e crea file usati come input per altri script
"""


import csv
import ast
import pandas as pd #type: ignore
from nltk import pos_tag, word_tokenize #type: ignore
from ast import literal_eval
import ollama #type: ignore
import re
from collections import defaultdict
import sys
from collections import Counter


#script per estrarre ricette dal csv
def estrai_ricette(input_file, output_file, nome_colonna="title", righe=100, delimitatore1=',', delimitatore2=' ') -> None:
    """
    Funzione usata per estrarre le ricette o i prodotti dal csv

    :param input_file: percorso del file CSV da cui estrarre le ricette
    :param output_file: percorso del file CSV dove salvare le ricette estratte
    :param nome_colonna: nome della colonna che contiene il titolo della ricetta
    :param righe: numero di righe da estrarre
    :param delimitatore1: delimitatore del file CSV da cui estrarre le ricette
    :param delimitatore2: delimitatore del file CSV dove salvare le ricette estratte
    :return: None
    """

    with open(input_file, mode='r', newline='', encoding='utf-8') as csv_input:
        reader = csv.DictReader(csv_input, delimiter=delimitatore1) 
        
        with open(output_file, mode='w', newline='', encoding='utf-8') as csv_output:
            writer = csv.writer(csv_output, delimiter=delimitatore2)  
            
            for i, row in enumerate(reader):
                if i < righe or righe == -1:
                    if (row[nome_colonna] != nome_colonna and row[nome_colonna] != ""):
                        writer.writerow([row[nome_colonna]])
                else:
                    break


def estrai_righe(input_file, output_file, delimiter=',', righe=100) -> None:
    """
    Funzione per estrarre righe dal csv

    :param input_file: percorso del file CSV da cui estrarre le righe
    :param output_file: percorso del file CSV dove salvare le righe estratte
    :param delimiter: delimitatore del file CSV da cui estrarre le righe
    :param righe: numero di righe da estrarre
    :return: None
    """
    with open(input_file, mode='r', newline='', encoding='utf-8') as csv_input:
        reader = csv.DictReader(csv_input, delimiter=delimiter)  
        
        with open(output_file, mode='w', newline='', encoding='utf-8') as csv_output:
            writer = csv.writer(csv_output, delimiter=delimiter) 
            writer.writerow(reader.fieldnames)  # type: ignore
            
            for i, row in enumerate(reader):
                if i < righe or righe == -1:
                    writer.writerow(row.values())  
                else:
                    break


def estrai_ingredienti_foodkg(input_file, output_file, min_occurrences=0) -> None:
    """
    Funzione per estrarre e pulire gli ingredienti unici da FoodKg e contarne le occorrenze

    :param input_file: percorso del file CSV da cui estrarre gli ingredienti
    :param output_file: percorso del file CSV dove salvare gli ingredienti estratti
    :param min_occurrences: numero minimo di occorrenze che un ingrediente deve avere per essere salvato
    :return: None
    """

    ingredient_counts = {}  

    for chunk in pd.read_csv(input_file, sep=',', chunksize=10000, usecols=['ingredient_food_kg_names'], on_bad_lines='skip', low_memory=False):
        for ingredient_list_str in chunk['ingredient_food_kg_names'].dropna():

            try:
                ingredient_list = ast.literal_eval(ingredient_list_str)
                for ingredient in ingredient_list:

                    ingredient_counts[ingredient] = ingredient_counts.get(ingredient, 0) + 1
            except (ValueError, SyntaxError):
                continue

    # Salva gli ingredienti unici stampa il numero di ingredienti e occorrenze
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ingredient'])
        for ingredient, count in sorted(ingredient_counts.items()):
            if count >= min_occurrences:
                writer.writerow([ingredient])
        print(f"File {output_file} creato con {len(ingredient_counts)} ingredienti unici e {sum(ingredient_counts.values())} occorrenze totali.")


def estrai_aggettivi_foodkg(input_file, output_file, text_column='text_column_name') -> None:
    """
    Funzione per estrarre aggettivi dagli elementi di foodkg
    
    :param input_file: percorso del file CSV da cui estrarre gli aggettivi
    :param output_file: percorso del file CSV dove salvare gli aggettivi estratti
    :param text_column: nome della colonna che contiene il testo da estrarre gli aggettivi
    :return: None
    """
    unique_adjectives = set()
    data = pd.read_csv(input_file, usecols=[text_column], on_bad_lines='skip')
    
    for text in data[text_column].dropna():
        words = word_tokenize(str(text))  
        # Identifica il POS di ciascuna parola
        tagged_words = pos_tag(words) 
        
        # Filtra e aggiungi solo gli aggettivi
        for word, pos in tagged_words:
            # 'JJ' è il tag per gli aggettivi
            if pos == 'JJ':  
                # Converte in minuscolo per evitare duplicati simili
                unique_adjectives.add(word.lower()) 

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['adjective'])  
        for adjective in sorted(unique_adjectives):
            writer.writerow([adjective])

    print(f"File {output_file} creato con {len(unique_adjectives)} aggettivi unici.")


def estrai_tag_unici(file_csv, file_output) -> None:
    """
    Funzione per estrarre i tag unici da HUMMUS (a scopo di analisi)
    :param file_csv: percorso del file CSV da cui estrarre i tag
    :param file_output: percorso del file CSV dove salvare i tag estratti
    :return: None
    """
    tag_count = {}  

    with open(file_csv, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            tags = row['tags'].strip("[]").replace("'", "").split(', ')
            for tag in tags:
                if tag in tag_count:
                    tag_count[tag] += 1
                else:
                    tag_count[tag] = 1

    sorted_tags = sorted(tag_count.items())

    with open(file_output, mode='w', encoding='utf-8') as txtfile:
        for tag, count in sorted_tags:
            txtfile.write(f'"{tag}", {count}\n')


def crea_food_expert() -> None:
    """
    Funzione per creare un modello per il filtraggio dei brand utilizzabile su ollama
    """    

    #system prompt
    modelfile = '''
    FROM qwen2.5:32b 
    SYSTEM You are a food product classification system that determines whether an input string is the name of a generic food product not tied to a brand or not. \
    You have a single task; you will always be given a single input string. \
    You must return only one word in output. \
    Write "food" if the product is a generic food, like "apple," "pear," "tuna," and therefore not tied to a brand. \
    Write "notfood" if the string is a food brand or any other word, such as an adjective or a place. \
    Do not add any extra comments, translations, or modifications; always respond with the exact word "food" or "notfood." \
    Here are some classification examples: \
    """ \
    apple -> food \
    pear -> food \
    tuna -> food \
    brand -> notfood \
    red -> notfood \
    organic -> notfood \
    banana -> food \
    Barilla -> notfood \
    pasta -> food \
    pain -> food \
    queso -> food \
    Nutella -> notfood \
    Calve -> notfood \
    """
    PARAMETER temperature 0
    PARAMETER top_p 0.8
    PARAMETER top_k 1
    '''

    #crea il modello
    ollama.create(model='food_expert', modelfile=modelfile)


def testa_accuratezza_filtraggio_brand(file1, file) -> None:
    """
    Funzione per testare il modello di filtraggio dei brand
    """
    results = []
    correct_count = 0  
    total_count = 0

    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)  
        for row in reader:
            brand = row['brand_name']
            expected_response = row['expected_response']
            
            # Genera la risposta del modello
            response = ollama.generate(model='food_expert', prompt=brand)
            
            # Controlla se è corretta
            is_correct = response['response'].strip().lower() == expected_response
            results.append({
                'brand_name': brand,
                'model_response': response['response'],
                'expected_response': expected_response,
                'correct': is_correct
            })

            # Aggiorna il conteggio
            total_count += 1
            if is_correct:
                correct_count += 1

    if total_count > 0:
        correct_percentage = (correct_count / total_count) * 100
    else:
        correct_percentage = 0

    for result in results:
        print(f"Brand: {result['brand_name']}, Model Response: {result['model_response']}, "
            f"Expected Response: {result['expected_response']}, Correct: {result['correct']}")
    print(f"\nPercentage of correct responses: {correct_percentage:.2f}%")


def analisi_quantities(input_file, output_file, n) -> None:
    """
    Funzione per analizzare la colonna quantities di off (a scopo di analisi)

    :param input_file: percorso del file CSV di cui fare l'analisi
    :param output_file: percorso del file CSV dove salvare l'analisi
    :param n: numero di occorrenze che una quatità deve avere per essere considerata valida
    :return: None
    """

    brand_counts = dict()  

    for chunk in pd.read_csv(input_file, sep='\t', chunksize=10000, usecols=['quantity'], on_bad_lines='skip', low_memory=False):
        for col in ['quantity']:
            for brand in chunk[col].dropna().unique():
                cleaned_brand = brand
                if cleaned_brand:

                    brand_counts[cleaned_brand] = brand_counts.get(cleaned_brand, 0) + 1
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for brand, count in sorted(brand_counts.items()):
            if count >= n:
                writer.writerow([brand])


def pulitura_brand(input_file, output_file) -> None:
    """
    Funzione per filtrare i brand classificati come cibo dal modello food expert

    :param input_file: percorso del file con i brand da filtrare
    :param output_file: percorso del file dove salvare i brand filtrati
    :return: None
    """

    with open(input_file, newline='') as csvfile, open(output_file, mode='w', newline='') as outfile:
        reader = csv.DictReader(csvfile)
        outfile.write(f"brand_name\n")

        for row in reader:
            brand = row['brand_name']
            response = ollama.generate(model='food_expert', prompt=brand)

            #se `model_response` è "nonfood", scrive il brand nel file di output
            if response['response'] == "notfood":
                outfile.write(f"{brand}\n")
                    
        print(f"\nGenerato il file {output_file}")


def analisi_numero_istanze_per_colonna(input_file, output_file, chunk_size=120000) -> None:
    """
    Funzione per analizzare il numero di istanze per ogni colonna di OFF (a scopo di analisi)

    :param input_file: percorso del file CSV di cui fare l'analisi
    :param output_file: percorso del file CSV dove salvare l'analisi
    :param chunk_size: dimensione del chunk di lettura
    :return: None
    """

    column_counts = {}

    for chunk in pd.read_csv(input_file, sep='\t', chunksize=chunk_size,on_bad_lines='skip',  low_memory=False):
        df = chunk

        for col in df.columns:
            if col not in column_counts:
                column_counts[col] = {'full': 0, 'empty': 0}
            
            column_counts[col]['full'] += df[col].notna().sum()
            column_counts[col]['empty'] += df[col].isna().sum()


    with open(output_file, 'w') as f:
        f.write("Column,Full Cells,Empty Cells\n")
        for col, counts in column_counts.items():
            f.write(f"{col},{counts['full']},{counts['empty']}\n")

    print(f"Analisi completata. Risultati salvati in {output_file}")


def clean_brand_name(brand_name) -> str | None:
    """
    Funzione per pulire i nomi dei brand

    :param brand_name: il nome del brand da pulire
    :return: il nome pulito del brand
    """
    # Rimuove stringhe "&quot" e sostituisci con nulla
    brand_name = brand_name.replace('&quot', '')
    
    # Rimuove punteggiatura e spazi extra
    brand_name = re.sub(r'[^\w\s]', '', brand_name).strip()
   
    if brand_name.isdigit() or len(brand_name) <= 1 or brand_name == " ":
        return None
    return brand_name.lower()


def extract_clean_brands(input_file, output_file, n=1) -> None:
    """
    Funzione per estrarre e pulire i brand unici da off che appaiono piu di n volte

    :param input_file: percorso del file da leggere
    :param output_file: percorso del file di output
    :param n: numero minimo di occorrenze per un brand per essere considerato valido
    """

    brand_counts = dict()  

    for chunk in pd.read_csv(input_file, sep='\t', chunksize=10000, usecols=['brands'], on_bad_lines='skip', low_memory=False):
        for col in ['brands']:
            for brand in chunk[col].dropna().unique():
                cleaned_brand = clean_brand_name(brand)
                if cleaned_brand:

                    brand_counts[cleaned_brand] = brand_counts.get(cleaned_brand, 0) + 1
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['brand_name'])
        for brand, count in sorted(brand_counts.items()):
            if count >= n:
                writer.writerow([brand])


def count_products_by_brand_threshold(csv_file, threshold_range) -> None:
    """
    Funzione per contare quanti prodotti sono legati a brand con meno di n apparizioni

    :param csv_file: percorso del file CSV da leggere
    :param threshold_range: intervallo di numero di apparizioni brand da considerare
    :return: None
    """
    csv.field_size_limit(1_000_000_000)
    brand_counter = Counter()

    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        
        for row_num, row in enumerate(reader, start=1):
            try:
                brand = row.get("brands", "").strip().lower()
                if brand:
                    brand_counter[brand] += 1
            except csv.Error as e:
                print(f"Errore alla riga {row_num}: {e}. Riga saltata.")

    eliminated_counts = {threshold: 0 for threshold in threshold_range}
    remaining_counts = {threshold: 0 for threshold in threshold_range}

    for brand, occurrences in brand_counter.items():
        for threshold in threshold_range:
            if occurrences < threshold:
                eliminated_counts[threshold] += occurrences
            else:
                remaining_counts[threshold] += occurrences

    for threshold in threshold_range:
        print(f"Soglia: {threshold}")
        print(f"Numero di prodotti non considerati: {eliminated_counts[threshold]}")
        print(f"Numero di prodotti considerati: {remaining_counts[threshold]}")


def estrai_descrizione(input_file, output_file, nome_colonna="title", righe=100, delimitatore1=',', delimitatore2=' '):
    """
    Funzione  per estrarre le member description degli utenti di hummus

    :param input_file: percorso del file CSV da leggere
    :param output_file: percorso del file CSV di output
    :param nome_colonna: nome della colonna che contiene il titolo della ricetta
    :param righe: numero di righe da leggere
    :param delimitatore1: delimitatore del file CSV da leggere
    :param delimitatore2: delimitatore del file CSV di output
    :return: None

    """
    with open(input_file, mode='r', newline='', encoding='utf-8') as csv_input:
        reader = csv.DictReader(csv_input, delimiter=delimitatore1) 
        
        with open(output_file, mode='w', newline='', encoding='utf-8') as csv_output:
            writer = csv.writer(csv_output, delimiter=delimitatore2)  
            
            for i, row in enumerate(reader):
                if i < righe or righe == -1:
                    if (row[nome_colonna] != nome_colonna and row[nome_colonna] != ""):
                        writer.writerow([row[nome_colonna]])
                else:
                    break


def crea_estrattore_member_description() -> None:
    """
    Funzione per creare un modello di estrazione di attributi dalle descrizioni degli utenti
    """

    # System prompt
    modelfile = '''
    FROM qwen2.5:32b
    SYSTEM You are a highly skilled attribute extractor model specialized in user profiling for a food ontology. You will receive a description provided by a user, and your task is to extract personal information and make reasonable inferences using a step-by-step, chain of thoughts approach. Specifically, you need to identify and infer the following attributes: \
    \
    - age \
    - weight \
    - height \
    - gender \
    - physical activity category (type of physical activity) \
    - halal (yes/no) \
    - food allergies or intolerances (related to food only) \
    - dietary preferences (e.g., vegan, vegetarian, low CO2 emissions, etc.) \
    \
    ### Guidelines: \
    1. **Chain of Thoughts**: Analyze the input description step by step, considering each detail and making logical inferences based on context. Clearly outline your reasoning process internally before extracting attributes, but do not include any reasoning or explanation in the output. \
    2. **Output Format**: After completing your internal analysis, write "#######" followed directly by the extracted attributes in the format: "attribute: value", separated by commas. \
    - If no attributes can be extracted, return the string "none" after "#######". \
    - Do not include any attribute if there is no information to infer or extract. \
    - Do not provide any comments, explanations, or reasoning after the extracted attributes. \
    \
    ### Examples: \
    - "" -> ####### none \
    - "I like dogs" -> ####### none \
    - "I am a mother and I like Italian cuisine, but I can't eat gluten" -> ####### gender: female, age: 30-50, allergies: gluten \
    - "I love running and I usually avoid dairy products" -> ####### physical activity category: running, allergies: lactose \
    - "I am a grandfather who loves Mediterranean food" -> ####### gender: male, age: 60+ \
    - "I am a software engineer who follows a vegan diet" -> ####### age: 30-50, dietary preference: vegan \
    - "I have two kids and I enjoy hiking on weekends" -> ####### age: 30-50, physical activity category: hiking \
    - "I work as a teacher and can't eat shellfish" -> ####### age: 30-50, allergies: shellfish \
    - "I am a retired army officer who loves spicy food" -> ####### age: 60+, gender: male \
    - "I only eat plant-based food and do yoga every morning" -> ####### dietary preference: vegan, physical activity category: yoga \
    - "I have celiac disease and cannot consume any gluten-containing products" -> ####### allergies: gluten \
    - "As a father of three, I often cook for my family" -> ####### gender: male, age: 30-50 \
    - "I observe Ramadan and avoid eating pork" -> ####### halal: yes \
    \
    PARAMETER temperature 0
    PARAMETER top_p 0.8
    PARAMETER top_k 1
    '''

    # Crea il modello con il system prompt definito e con nome "attribute_extractor"
    ollama.create(model='attribute_extractor', modelfile=modelfile)


def testa_estrazione_attributi_utenti(file) -> None:
    """
    Funzione per testare l'estrazione di attributi dalle descrizioni degli utenti
    prende delle descrizioni degli utenti e estrae gli attributi inferiti

    :param file: percorso del file CSV da leggere
    :return: None
    """
    
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile)  
        for row in reader:            
            response = ollama.generate(model='attribute_extractor', prompt=str(row))
            print("descrizione utente: \n", (str(row)))
            print("attributi estratti: \n", response['response'])
