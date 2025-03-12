"""
File containing scripts and data necessary for creating files used as input for other scripts.
"""


import ast
import csv
import sys
import re
import random
import ollama
import pandas as pd
from nltk import pos_tag, word_tokenize


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

        with open(
            output_file, mode="w", newline="", encoding="utf-8"
        ) as csv_output:
            writer = csv.writer(csv_output, delimiter=delimiter2)

            for i, row in enumerate(reader):
                if i < rows or rows == -1:
                    if (
                        row[column_name] != column_name
                        and row[column_name] != ""
                    ):
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

    csv.field_size_limit(sys.maxsize)

    with open(input_file, mode="r", newline="", encoding="utf-8") as csv_input:
        reader = csv.DictReader(csv_input, delimiter=delimiter)

        with open(
            output_file, mode="w", newline="", encoding="utf-8"
        ) as csv_output:
            writer = csv.writer(csv_output, delimiter=delimiter)
            # Write headers to the output file
            writer.writerow(reader.fieldnames)  # type: ignore

            for i, row in enumerate(reader):
                if i < rows or rows == -1:
                    # Write row values
                    writer.writerow(row.values())
                else:
                    break


def extract_rows_random(
    input_file, output_file, delimiter=",", rows=100
) -> None:
    """
    Function to extract rows from the CSV randomly

    :param input_file: path to the CSV file from which to extract rows
    :param output_file: path to the CSV file where extracted rows will be saved
    :param delimiter: delimiter of the CSV file from which to extract rows
    :param rows: number of rows to extract
    :return: None
    """

    csv.field_size_limit(1_000_000_000)

    with open(input_file, mode="r", newline="", encoding="utf-8") as csv_input:

        reader = list(csv.DictReader(csv_input, delimiter=delimiter))
        random_rows = random.sample(reader, min(rows, len(reader)))

        with open(
            output_file, mode="w", newline="", encoding="utf-8"
        ) as csv_output:
            writer = csv.writer(csv_output, delimiter=delimiter)
            # Write headers to the output file
            writer.writerow(reader[0].keys())

            for row in random_rows:
                # Write row values
                writer.writerow(row.values())


def extract_ingredients_foodkg(
    input_file, output_file, min_occurrences=0
) -> None:
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


def clean_brand_name(brand_name) -> str | None:
    """
    Function to clean brand names

    :param brand_name: the brand name to clean
    :return: the cleaned brand name or None if the brand name is invalid
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
    :return: None
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
                    brand_counts[cleaned_brand] = (
                        brand_counts.get(cleaned_brand, 0) + 1
                    )

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["brand_name"])
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
    :param column_name: name of the column that contains the member description
    :param rows: number of rows to read
    :param delimiter1: delimiter for the input CSV file
    :param delimiter2: delimiter for the output CSV file
    :return: None

    """
    with open(input_file, mode="r", newline="", encoding="utf-8") as csv_input:
        reader = csv.DictReader(csv_input, delimiter=delimiter1)

        with open(
            output_file, mode="w", newline="", encoding="utf-8"
        ) as csv_output:
            writer = csv.writer(csv_output, delimiter=delimiter2)

            for i, row in enumerate(reader):
                if i < rows or rows == -1:
                    if (
                        row[column_name] != column_name
                        and row[column_name] != ""
                    ):
                        writer.writerow([row[column_name]])
                else:
                    break


def transform_groups_in_tags(
    input_file,
    output_file,
    columns_name,
    delimiter1=",",
    delimiter2=","  
):
    """
    Function to transform specified columns into new tags within the 'tags' column.

    :param input_file: Path to the CSV file to read.
    :param output_file: Path to the output CSV file.
    :param columns_name: List of column names to be transformed.
    :param delimiter1: Delimiter for the input CSV file.
    :param delimiter2: Delimiter for the output CSV file.
    :return: None
    """
    # Load the CSV file
    df = pd.read_csv(input_file, delimiter=delimiter1)

    # Ensure 'tags' column is treated as a string and missing values are handled
    df['tags'] = df['tags'].fillna("[]").astype(str)

    # Process each row
    for col in columns_name:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(" ", "_")
            df['tags'] = df.apply(lambda row: row['tags'].rstrip("]") + ", '" + row[col] + "']" if row['tags'].endswith("]") else row['tags'] + ", '" + row[col] + "']", axis=1)
    
    # Save the updated DataFrame
    df.to_csv(output_file, index=False, sep=delimiter2)
    print(f"File saved successfully to {output_file}")