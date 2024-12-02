"""
File containing scripts and data necessary for the creation of LLM models, analysis of CSV files and ontologies,
and creating files used as input for other scripts.
"""

import ast
import csv
import re
from collections import Counter

import ollama  # type: ignore
import pandas as pd  # type: ignore
from nltk import pos_tag, word_tokenize  # type: ignore


# Script to extract recipes from the CSV
def extract_recipes(
    input_file,
    output_file,
    column_name="title",
    rows=100,
    delimiter1=",",
    delimiter2=" ",
) -> None:
    """
    Function used to extract recipes or products from the CSV

    :param input_file: path to the CSV file from which to extract recipes
    :param output_file: path to the CSV file where extracted recipes will be saved
    :param column_name: name of the column containing the recipe title
    :param rows: number of rows to extract
    :param delimiter1: delimiter of the CSV file from which to extract recipes
    :param delimiter2: delimiter of the CSV file where extracted recipes will be saved
    :return: None
    """

    with open(input_file, mode="r", newline="", encoding="utf-8") as csv_input:
        reader = csv.DictReader(csv_input, delimiter=delimiter1)

        with open(output_file, mode="w", newline="", encoding="utf-8") as csv_output:
            writer = csv.writer(csv_output, delimiter=delimiter2)

            for i, row in enumerate(reader):
                if i < rows or rows == -1:
                    if row[column_name] != column_name and row[column_name] != "":
                        writer.writerow([row[column_name]])
                else:
                    break


def extract_rows(input_file, output_file, delimiter=",", rows=100) -> None:
    """
    Function to extract rows from the CSV

    :param input_file: path to the CSV file from which to extract rows
    :param output_file: path to the CSV file where extracted rows will be saved
    :param delimiter: delimiter of the CSV file from which to extract rows
    :param rows: number of rows to extract
    :return: None
    """
    with open(input_file, mode="r", newline="", encoding="utf-8") as csv_input:
        reader = csv.DictReader(csv_input, delimiter=delimiter)

        with open(output_file, mode="w", newline="", encoding="utf-8") as csv_output:
            writer = csv.writer(csv_output, delimiter=delimiter)
            # Write headers to the output file
            writer.writerow(reader.fieldnames)  # type: ignore

            for i, row in enumerate(reader):
                if i < rows or rows == -1:
                    # Write row values
                    writer.writerow(row.values())
                else:
                    break


def extract_ingredients_foodkg(input_file, output_file, min_occurrences=0) -> None:
    """
    Function to extract and clean unique ingredients from FoodKG and count their occurrences

    :param input_file: path to the CSV file from which to extract ingredients
    :param output_file: path to the CSV file where extracted ingredients will be saved
    :param min_occurrences: minimum number of occurrences an ingredient must have to be saved
    :return: None
    """

    ingredient_counts = {}

    for chunk in pd.read_csv(
        input_file,
        sep=",",
        chunksize=10000,
        usecols=["ingredient_food_kg_names"],
        on_bad_lines="skip",
        low_memory=False,
    ):
        for ingredient_list_str in chunk["ingredient_food_kg_names"].dropna():
            try:
                ingredient_list = ast.literal_eval(ingredient_list_str)
                for ingredient in ingredient_list:
                    ingredient_counts[ingredient] = (
                        ingredient_counts.get(ingredient, 0) + 1
                    )
            except (ValueError, SyntaxError):
                continue

    # Save the unique ingredients and print the number of ingredients and occurrences
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ingredient"])
        for ingredient, count in sorted(ingredient_counts.items()):
            if count >= min_occurrences:
                writer.writerow([ingredient])
        print(
            f"File {output_file} created with {len(ingredient_counts)} unique ingredients and {sum(ingredient_counts.values())} total occurrences."
        )


def extract_adjectives_foodkg(
    input_file, output_file, text_column="text_column_name"
) -> None:
    """
    Function to extract adjectives from FoodKG elements

    :param input_file: path to the CSV file from which to extract adjectives
    :param output_file: path to the CSV file where extracted adjectives will be saved
    :param text_column: name of the column containing the text from which adjectives will be extracted
    :return: None
    """
    unique_adjectives = set()
    data = pd.read_csv(input_file, usecols=[text_column], on_bad_lines="skip")

    for text in data[text_column].dropna():
        words = word_tokenize(str(text))
        # Identify the POS of each word
        tagged_words = pos_tag(words)

        # Filter and add only adjectives
        for word, pos in tagged_words:
            # 'JJ' is the tag for adjectives
            if pos == "JJ":
                # Convert to lowercase to avoid similar duplicates
                unique_adjectives.add(word.lower())

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["adjective"])
        for adjective in sorted(unique_adjectives):
            writer.writerow([adjective])

    print(
        f"File {output_file} created with {len(unique_adjectives)} unique adjectives."
    )


def extract_tags_unique(file_csv, file_output) -> None:
    """
    Function to extract unique tags from HUMMUS (for analysis purposes)

    :param file_csv: path to the CSV file from which to extract the tags
    :param file_output: path to the CSV file where the extracted tags will be saved
    :return: None
    """
    tag_count = {}

    with open(file_csv, mode="r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            tags = row["tags"].strip("[]").replace("'", "").split(", ")
            for tag in tags:
                if tag in tag_count:
                    tag_count[tag] += 1
                else:
                    tag_count[tag] = 1

    sorted_tags = sorted(tag_count.items())

    with open(file_output, mode="w", encoding="utf-8") as txtfile:
        for tag, count in sorted_tags:
            txtfile.write(f'"{tag}", {count}\n')


def create_food_expert() -> None:
    """
    Function to create a model for brand filtering usable on Ollama
    """

    # system prompt
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

    # create the model
    ollama.create(model="food_expert", modelfile=modelfile)


def test_filtering_brand_accuracy(file1, file) -> None:
    """
    Function to test the brand filtering model
    """
    results = []
    correct_count = 0
    total_count = 0

    with open(file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            brand = row["brand_name"]
            expected_response = row["expected_response"]

            # Generate the model's response
            response = ollama.generate(model="food_expert", prompt=brand)

            # Check if the response is correct
            is_correct = response["response"].strip().lower() == expected_response
            results.append(
                {
                    "brand_name": brand,
                    "model_response": response["response"],
                    "expected_response": expected_response,
                    "correct": is_correct,
                }
            )

            # Update counts
            total_count += 1
            if is_correct:
                correct_count += 1

    if total_count > 0:
        correct_percentage = (correct_count / total_count) * 100
    else:
        correct_percentage = 0

    for result in results:
        print(
            f"Brand: {result['brand_name']}, Model Response: {result['model_response']}, "
            f"Expected Response: {result['expected_response']}, Correct: {result['correct']}"
        )
    print(f"\nPercentage of correct responses: {correct_percentage:.2f}%")


def analisi_quantities(input_file, output_file, n) -> None:
    """
    Function to analyze the quantities column of OFF (for analysis purposes)

    :param input_file: path to the CSV file to analyze
    :param output_file: path to the CSV file where the analysis will be saved
    :param n: the minimum number of occurrences for a quantity to be considered valid
    :return: None
    """

    brand_counts = dict()

    for chunk in pd.read_csv(
        input_file,
        sep="\t",
        chunksize=10000,
        usecols=["quantity"],
        on_bad_lines="skip",
        low_memory=False,
    ):
        for col in ["quantity"]:
            for brand in chunk[col].dropna().unique():
                cleaned_brand = brand
                if cleaned_brand:
                    brand_counts[cleaned_brand] = brand_counts.get(cleaned_brand, 0) + 1

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for brand, count in sorted(brand_counts.items()):
            if count >= n:
                writer.writerow([brand])


def brand_filtering(input_file, output_file) -> None:
    """
    Function to filter brands classified as food by the food expert model.

    :param input_file: path to the file with brands to filter
    :param output_file: path to the file where filtered brands will be saved
    :return: None
    """

    with (
        open(input_file, newline="") as csvfile,
        open(output_file, mode="w", newline="") as outfile,
    ):
        reader = csv.DictReader(csvfile)
        outfile.write("brand_name\n")

        for row in reader:
            brand = row["brand_name"]
            response = ollama.generate(model="food_expert", prompt=brand)

            # If the `model_response` is "notfood", write the brand to the output file
            if response["response"] == "notfood":
                outfile.write(f"{brand}\n")

        print(f"\nGenerated the file {output_file}")


def number_of_instance_for_columns(input_file, output_file, chunk_size=120000) -> None:
    """
    Function to analyze the number of instances per column in OFF (for analysis purposes)

    :param input_file: path to the CSV file to analyze
    :param output_file: path to the CSV file where the analysis will be saved
    :param chunk_size: size of the chunk to read
    :return: None
    """

    column_counts = {}

    for chunk in pd.read_csv(
        input_file,
        sep="\t",
        chunksize=chunk_size,
        on_bad_lines="skip",
        low_memory=False,
    ):
        df = chunk

        for col in df.columns:
            if col not in column_counts:
                column_counts[col] = {"full": 0, "empty": 0}

            column_counts[col]["full"] += df[col].notna().sum()
            column_counts[col]["empty"] += df[col].isna().sum()

    with open(output_file, "w") as f:
        f.write("Column,Full Cells,Empty Cells\n")
        for col, counts in column_counts.items():
            f.write(f"{col},{counts['full']},{counts['empty']}\n")

    print(f"Analysis completed. Results saved in {output_file}")


def clean_brand_name(brand_name) -> str | None:
    """
    Function to clean brand names

    :param brand_name: the brand name to clean
    :return: the cleaned brand name
    """
    # Remove "&quot" strings and replace with nothing
    brand_name = brand_name.replace("&quot", "")

    # Remove punctuation and extra spaces
    brand_name = re.sub(r"[^\w\s]", "", brand_name).strip()

    if brand_name.isdigit() or len(brand_name) <= 1 or brand_name == " ":
        return None
    return brand_name.lower()


def extract_clean_brands(input_file, output_file, n=1) -> None:
    """
    Function to extract and clean unique brands from OFF that appear more than n times.

    :param input_file: path to the file to read
    :param output_file: path to the output file
    :param n: minimum number of occurrences for a brand to be considered valid
    """

    brand_counts = dict()

    for chunk in pd.read_csv(
        input_file,
        sep="\t",
        chunksize=10000,
        usecols=["brands"],
        on_bad_lines="skip",
        low_memory=False,
    ):
        for col in ["brands"]:
            for brand in chunk[col].dropna().unique():
                cleaned_brand = clean_brand_name(brand)
                if cleaned_brand:
                    brand_counts[cleaned_brand] = brand_counts.get(cleaned_brand, 0) + 1

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["brand_name"])
        for brand, count in sorted(brand_counts.items()):
            if count >= n:
                writer.writerow([brand])


def count_products_by_brand_threshold(csv_file, threshold_range) -> None:
    """
    Function to count how many products are linked to brands with fewer than n appearances.

    :param csv_file: path to the CSV file to read
    :param threshold_range: range of the number of brand appearances to consider
    :return: None
    """
    csv.field_size_limit(1_000_000_000)
    brand_counter = Counter()

    with open(csv_file, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")

        for row_num, row in enumerate(reader, start=1):
            try:
                brand = row.get("brands", "").strip().lower()
                if brand:
                    brand_counter[brand] += 1
            except csv.Error as e:
                print(f"Error at row {row_num}: {e}. Skipping row.")

    eliminated_counts = {threshold: 0 for threshold in threshold_range}
    remaining_counts = {threshold: 0 for threshold in threshold_range}

    for brand, occurrences in brand_counter.items():
        for threshold in threshold_range:
            if occurrences < threshold:
                eliminated_counts[threshold] += occurrences
            else:
                remaining_counts[threshold] += occurrences

    for threshold in threshold_range:
        print(f"Threshold: {threshold}")
        print(f"Number of products excluded: {eliminated_counts[threshold]}")
        print(f"Number of products considered: {remaining_counts[threshold]}")


def extract_description(
    input_file,
    output_file,
    column_name="title",
    rows=100,
    delimiter1=",",
    delimiter2=" ",
):
    """
    Function to extract the member descriptions of users in HUMMUS.

    :param input_file: path to the CSV file to read
    :param output_file: path to the output CSV file
    :param column_name: name of the column that contains the recipe title
    :param rows: number of rows to read
    :param delimiter1: delimiter for the input CSV file
    :param delimiter2: delimiter for the output CSV file
    :return: None

    """
    with open(input_file, mode="r", newline="", encoding="utf-8") as csv_input:
        reader = csv.DictReader(csv_input, delimiter=delimiter1)

        with open(output_file, mode="w", newline="", encoding="utf-8") as csv_output:
            writer = csv.writer(csv_output, delimiter=delimiter2)

            for i, row in enumerate(reader):
                if i < rows or rows == -1:
                    if row[column_name] != column_name and row[column_name] != "":
                        writer.writerow([row[column_name]])
                else:
                    break


def create_attribute_extractor() -> None:
    """
    Function for create a model of extracting attributes from user descriptions
    """

    # System prompt
    modelfile = """
    FROM qwen2.5:32b
    SYSTEM You are a highly skilled attribute extractor model specialized in user profiling for a food ontology. You will receive a description provided by a user, and your task is to extract personal information and make reasonable inferences using a step-by-step, chain of thoughts approach. Specifically, you need to identify and infer the following attributes: \
    \
    - age \
    - weight \
    - height \
    - gender \
    - physical activity category (type of physical activity) \
    - religious constraint (e.g., halal, kosher, etc.) \
    - food allergies or intolerances (related to food only) \
    - dietary preferences (e.g., vegan, vegetarian, low CO2 emissions, etc.) \
    \
    ### Guidelines: \
    1. **Chain of Thoughts**: Analyze the input description step by step, considering each detail and making logical inferences based on context. Clearly outline your reasoning process internally before extracting attributes, but do not include any reasoning or explanation in the output. \
    2. **Inferred Values**: For any attribute whose value is inferred but not explicitly stated in the input, append "(inferred)" to the value. If a value is explicitly stated, do not include "(inferred)". \
    3. **Output Format**: After completing your internal analysis, write "#######" followed directly by the extracted attributes in the format: "attribute: value", separated by commas. \
    - If no attributes can be extracted, return the string "none" after "#######". \
    - Do not include any attribute if there is no information to infer or extract. \
    - Do not provide any comments, explanations, or reasoning after the extracted attributes. \
    \
    ### Examples: \
    - "" -> ####### none \
    - "I like dogs" -> ####### none \
    - "I am a mother and I like Italian cuisine, but I can't eat gluten" -> ####### gender: female, age: 30-50 (inferred), allergies: gluten \
    - "I love running and I usually avoid dairy products" -> ####### physical activity category: running, allergies: lactose \
    - "I am a grandfather who loves Mediterranean food" -> ####### gender: male, age: 60+ (inferred) \
    - "I am a software engineer who follows a vegan diet" -> ####### age: 30-50 (inferred), dietary preference: vegan \
    - "I have two kids and I enjoy hiking on weekends" -> ####### age: 30-50 (inferred), physical activity category: hiking \
    - "I work as a teacher and can't eat shellfish" -> ####### age: 30-50 (inferred), allergies: shellfish \
    - "I am a retired army officer who loves spicy food" -> ####### age: 60+ (inferred), gender: male \
    - "I only eat plant-based food and do yoga every morning" -> ####### dietary preference: vegan, physical activity category: yoga \
    - "I have celiac disease and cannot consume any gluten-containing products" -> ####### allergies: gluten \
    - "As a father of three, I often cook for my family" -> ####### gender: male, age: 30-50 (inferred) \
    - "I observe Ramadan and avoid eating pork" -> ####### religious constraint: halal \
    \
    PARAMETER temperature 0
    PARAMETER top_p 0.8
    PARAMETER top_k 1
    """

    # Create the model with the defined system prompt and name "attribute_extractor"
    ollama.create(model="attribute_extractor", modelfile=modelfile)


def test_attribute_extraction(file) -> None:
    """
    Function to test the extraction of attributes from user descriptions.
    It takes user descriptions and extracts the inferred attributes.

    :param file: path to the CSV file to read
    :return: None
    """

    with open(file, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            response = ollama.generate(model="attribute_extractor", prompt=str(row))
            print("User description: \n", (str(row)))
            print("Extracted attributes: \n", response["response"])
