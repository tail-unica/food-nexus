"""
File containing scripts and data required for the normalization
of ingredient names and the definition of the translation model
"""

import csv
import json
import re
import string
import unicodedata
import os
import sys
import nltk
import ollama
import pint
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Optional
import time

csv.field_size_limit(sys.maxsize)

def add_to_sys_path(folder_name):
    utils_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), folder_name)
    )
    sys.path.append(utils_path)
add_to_sys_path("../ollama_server_file")
from ollama_server import OllamaModel  # type: ignore



# Device on which operations will be performed
device = "cuda" if torch.cuda.is_available() else "cpu"


# Initialize the lemmatizer and the unit registry
lemmatizer = WordNetLemmatizer()
ureg = pint.UnitRegistry()

# Custom definitions of non-standard units
ureg.define("tbsp = 15 * milliliter")
ureg.define("tsp = 5 * milliliter")
ureg.define("cup = 240 * milliliter")
ureg.define("pint = 473.176 * milliliter")
ureg.define("quart = 946.353 * milliliter")
ureg.define("ounce = 28.3495 * gram")
ureg.define("fluid_ounce = 29.5735 * milliliter")


# Function to load json files
def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


# Path of json files
suffixes_and_prefixes_file = "config_file/suffixes_and_prefixes.json"
abbreviations_file = "config_file/abbreviations.json"
unit_conversion_map_file = "config_file/unit_conversion_map.json"
units_to_remove_file = "config_file/units_to_remove.json"

# List of suffixes and prefixes to remove
suffixes_and_prefixes = load_json_file(suffixes_and_prefixes_file)

# Dictionary of acronyms and associated values
abbreviations = load_json_file(abbreviations_file)

# Dictionary of units and their standardized forms
unit_conversion_map = load_json_file(unit_conversion_map_file)

# List of units of measurement to remove
units_to_remove = load_json_file(units_to_remove_file)


def load_adjectives_to_keep_words(
    input_file="./csv_file/aggettivi_FoodKg.csv",
) -> set[str]:
    """
    Function to load unique adjectives into `keep_words`

    :param input_file: path to the CSV file

    :return: a set of strings containing the unique adjectives from FoodKG
    """

    # Initialize the set of strings
    keep_words = set()
    with open(input_file, mode="r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        # Add each adjective found in the file to `keep_words`
        for row in reader:
            adjective = row["adjective"].strip().lower()
            keep_words.add(adjective)
    return keep_words


# List of words to keep in adjectives and adverbs
keep_words = load_adjectives_to_keep_words()


def load_brands_from_csv(
    file_path="./csv_file/brands_filtered.csv",
) -> list[str]:
    """
    Function to load brand names to be removed from the appropriate CSV file

    :param file_path: path to the CSV file

    :return: a list of strings containing brand names to remove
    """
    # Initialize the list of strings
    brands = []
    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Check if the row is not empty
            if row:
                # Add the brand name by removing extra spaces
                brands.append(row[0].strip())

    return brands


"""
I sort them so that if a brand name is included in another, the longer name is removed first
For example, "rio" and "rio mare" are both brands, applying the brand removal to the product
"tonno rio mare" would remove "rio" first and then stop, without removing "mare" if they weren't sorted
"""
# List of brands to remove
brands = sorted(load_brands_from_csv(), key=len, reverse=True)


def remove_text_in_parentheses(text: str) -> str:
    """
    Function to remove text within parentheses

    :param text: the text from which to remove the text within parentheses
    :return: the text with the text within parentheses removed
    """
    return re.sub(r"\(.*?\)", "", text).strip()


def remove_text_after_comma_or_colon(text: str) -> str:
    """
    Removes text after the first comma or colon character.

    :param text: The text from which to remove the part after the first comma or colon.
    :return: The text with the part after the first comma or colon removed.
    """
    return re.split(r"[,:]", text, maxsplit=1)[0].strip()


def remove_or_replace_punctuation(text: str) -> str:
    """
    Removes or replaces punctuation in the text.

    Replaces punctuation between two words with a space and removes any remaining punctuation.

    :param text: The text from which to remove or replace punctuation.
    :return: The text with remaining punctuation removed or replaced.
    """
    # Replace punctuation between two characters with a space
    text = re.sub(r"(?<=\w)[^\w\s](?=\w)", " ", text)

    # Remove any other punctuation that wasn't already replaced
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


def remove_numeric_values(text: str) -> str:
    """
    Removes numeric values from the text, including those with decimal points or commas.

    :param text: The text from which to remove numeric values.
    :return: The text without numeric values.
    """
    # Remove whole numbers and decimal numbers with period or comma
    return re.sub(r"\b\d+[.,]?\d*\b", "", text).strip()
_model_instance = None

def translate_to_english(text: str) -> str:
    global _model_instance
    modelfile = """FROM qwen2.5:32b
SYSTEM You are a highly skilled linguist with a specific task: I will provide you with a single string of input related to food names, ingredients, recipes, or culinary terms. \
Your objective is to determine the language of the input string. If the string is in English, respond with "eng." If the string is not in English, translate it into English. \
Please adhere to the following guidelines: \
- Do not add any comments, explanations, or modifications to your response. \
- Always respond with either "eng" or the English translation of the provided text. \
- Do not remove any punctuation mark, for example (",", ".", ";", ":", "!", "?", "(", ")", "\"") in your response. \
- Do not put ani '"' when you are responding "eng". \
Here are some examples for clarity: \
- "pane al mais" -> "corn bread" \
- "apple" -> "eng" \
- "frutta" -> "fruit" \
- "cibo" -> "food" \
- "birne" -> "pear" \
- "arroz con pollo y verduras frescas" -> "rice with chicken and fresh vegetables" \
- "bánh mì với thịt và rau củ" -> "sandwich with meat and vegetables" \
- "パスタとトマトソース" -> "pasta with tomato sauce" \
- "gnocchi di patate con burro e salvia" -> "potato dumplings with butter and sage" \
- "choucroute garnie avec des saucisses" -> "sauerkraut with sausages" \
- "paella de mariscos y arroz amarillo" -> "seafood paella with yellow rice" \
- "fish and chips" -> "eng" \
- "tonno! ☺️ pizza bio l rustica 1kg (2x500g) - con tonno agli oli evo, una vera bontà" -> "tuna! ☺️ pizza bio l rustic 1kg (2x500g) - with tuna in evo oils, a true delight" \
- "cioccolato 🍫 extra fondente - 85% cacao (con aroma di vaniglia naturale)" -> "extra dark chocolate 85 percent cocoa with natural vanilla aroma" \
- "queso manchego curado 🧀 - ideal para tapas o gratinar" -> "manchego cheese cured ideal for tapas or gratinating" \
- "bánh cuốn nhân thịt 🥢 - món ăn sáng Việt Nam thơm ngon" -> "bánh cuốn with meat a delicious Vietnamese breakfast dish" \
- "寿司 🍣 - 新鮮な魚で作られた最高の日本料理" -> "sushi the finest Japanese dish made with fresh fish" \
- "pomodoro rosso bio 🍅 (1kg) - perfetto per salse fatte in casa" -> "red organic tomato 1kg perfect for homemade sauces" \
Begin processing the input now.
PARAMETER temperature 0
PARAMETER top_p 0.8
PARAMETER top_k 1
    """

    try:
        if _model_instance is None:
            _model_instance = OllamaModel(
                modelfile=modelfile, model_name="translation_expert"
            )
        response = _model_instance.generate(text)
        response = response.strip()

        if response == "eng":
            return text
        else:
            return response
        
    except Exception as e:
        print(f"Translation failed: {e}")
        _model_instance = None
        return translate_to_english(text)
    
    
def replace_abbreviations(text: str) -> str:
    """
    Replaces abbreviations in the text with their full form.

    :param text: The text containing abbreviations to replace.
    :return: The text with abbreviations replaced by their full form.
    """
    for key, value in abbreviations.items():
        text = re.sub(
            r"\b" + re.escape(key) + r"\b", value, text, flags=re.IGNORECASE
        )
    return text


def remove_suffixes_prefixes(text: str) -> str:
    """
    Removes suffixes and prefixes from standalone words in the text.

    :param text: The text from which to remove suffixes and prefixes.
    :return: The text with suffixes and prefixes removed.
    """
    for item in suffixes_and_prefixes:
        text = re.sub(r"\b" + re.escape(item) + r"\b", "", text)

    return re.sub(r"\s+", " ", text).strip()


def remove_brands(text: str) -> str:
    """
    Removes brand names from the text.

    :param text: The text from which to remove brand names.
    :return: The text with brand names removed.
    """
    for brand in brands:
        text = re.sub(
            r"\b" + re.escape(brand) + r"\b", "", text, flags=re.IGNORECASE
        )

    return re.sub(r"\s+", " ", text).strip()


def convert_to_lowercase(text: str) -> str:
    """
    Converts the text to lowercase.

    :param text: The text to convert to lowercase.
    :return: The text converted to lowercase.
    """
    return text.lower()


def normalize_special_characters(text: str) -> str:
    """
    Normalizes special characters by removing accents and non-ASCII characters.

    :param text: The text to normalize.
    :return: The text with special characters normalized.
    """

    # Normalize Unicode characters (NFKD removes accents)
    normalized_text = unicodedata.normalize("NFKD", text)

    # Remove non-ASCII characters
    return "".join(
        c
        for c in normalized_text
        if c in string.ascii_letters + string.digits + string.whitespace
    )


def sort_words_alphabetically(text: str) -> str:
    """
    Sorts the words in the text alphabetically.

    :param text: The text to sort.
    :return: The text with words sorted alphabetically.
    """
    words = text.split()
    sorted_words = sorted(words)
    return " ".join(sorted_words)


def normalize_quantities(text: str) -> str:
    """
    Normalizes quantities in the text, handling units with optional spaces.

    :param text: The text containing quantities to normalize.
    :return: The text with quantities normalized.
    """
    # Use a regular expression to find patterns like "number [optional space] unit"
    pattern = r"(\d+(\.\d+)?)\s*([a-zA-Z_]+)"
    matches = re.finditer(pattern, text)

    for match in matches:
        # Found number
        number = match.group(1)
        # Found unit (in lowercase for uniformity)
        unit = match.group(3).lower()
        # The original match, with optional spaces
        original_quantity = match.group(0)

        # Map the unit to its standardized form (if it exists in the map)
        if unit in unit_conversion_map:
            try:
                standardized_unit = unit_conversion_map[unit]
                # Create the quantity with the standardized unit
                quantity = ureg.Quantity(float(number), unit)

                # Convert to the base unit (gram or milliliter)
                normalized_quantity = quantity.to(standardized_unit)

                # Format the result clearly (rounded to two decimals)
                normalized_text = (
                    f"{normalized_quantity.magnitude:.2f} {standardized_unit}"
                )

                # Replace the original with the normalized quantity
                text = text.replace(original_quantity, normalized_text)
            except (
                # ureg.UndefinedUnitError,
                # ureg.DimensionalityError,
                ValueError
            ):
                # If it's not a valid quantity or there's an incompatibility, continue without modification
                continue

    return text


def remove_stopwords(text: str) -> str:
    """
    Removes stopwords from the text.

    :param text: The text from which to remove stopwords.
    :return: The text with stopwords removed.
    """
    # Load the stopwords for the English language
    stop_words = set(stopwords.words("english"))
    # Tokenize the text
    words = nltk.word_tokenize(text)
    # Remove stopwords
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)


def lemmatize_text(text: str) -> str:
    """
    Lemmatizes the text, reducing words to their base form.

    :param text: The text to lemmatize.
    :return: The lemmatized text.
    """
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized_words)


def remove_units(text: str) -> str:
    """
    Removes units of measure from the text.

    :param text: The text from which to remove units of measure.
    :return: The text without units of measure.
    """
    words = text.split()
    cleaned_words = [word for word in words if word not in units_to_remove]
    return " ".join(cleaned_words)


def remove_unwanted_pos(text: str) -> str:
    """
    Removes verbs, adverbs, and adjectives not included in the allowed list.

    :param text: The text from which to remove unwanted parts of speech.
    :return: The text with unwanted parts of speech removed.
    """
    words = nltk.word_tokenize(text)

    # Removed 'RB' because it removes too much information, to be evaluated if re-adding it
    # 'JJ' = adjectives, 'RB' = adverbs, 'VB' = verbs, 'NNP' = proper nouns
    filtered_words = [
        word
        for word, pos in nltk.pos_tag(words)
        if pos not in ("VB", "JJ", "NNP") or word in keep_words
    ]
    return " ".join(filtered_words)


def remove_duplicate_words(text: str) -> str:
    """
    Removes duplicate words from the text.

    :param text: The text from which to remove duplicate words.
    :return: The text without duplicate words.
    """
    words = nltk.word_tokenize(text)
    filtered_words = list(set(words))
    return " ".join(filtered_words)


def remove_lenght1_words(line):
    """
    Removes all words of length 1 from a string.

    Args:
        line (str): The input string.

    Returns:
        str: The string with words of length 1 removed.
    """
    return " ".join(word for word in line.split() if len(word) > 1)


def pipeline_core(line, show_all=False, show_something=False, translation_nedded=True, only_translation=True) -> str:
    """
    Function to execute the normalization pipeline on a single line

    :param line: the line to normalize
    :param show_all: if True, shows all intermediate normalization steps
    :param show_something: if True, shows some normalization steps
    :return: the normalized line
    """

    line = re.sub(r"&quot;", "", line)

    # Original text
    if show_all or show_something:
        print("original:".ljust(40), line)

    # Convert to lowercase
    line = convert_to_lowercase(line)
    if show_all:
        print("lowercase:".ljust(40), line)

    # Remove brand names
    line = remove_brands(line)
    if show_all:
        print("brand removed:".ljust(40), line)

    # Translate text to English
    if translation_nedded:
        line = translate_to_english(line)
        if show_all:
            print("text translated to English:".ljust(40), line)

    if only_translation == False:
        # Convert to lowercase
        line = convert_to_lowercase(line)
        if show_all:
            print("lowercase:".ljust(40), line)

        # Expand abbreviations
        line = replace_abbreviations(line)
        if show_all:
            print("abbreviations expanded:".ljust(40), line)

        # Remove unwanted adjectives, adverbs, and verbs
        line = remove_unwanted_pos(line)
        if show_all:
            print("remove adjectives:".ljust(40), line)

        # Remove suffixes and prefixes
        line = remove_suffixes_prefixes(line)
        if show_all:
            print("remove suffixes and prefixes:".ljust(40), line)

        # Remove text in parentheses
        line = remove_text_in_parentheses(line)
        if show_all:
            print("remove text in parentheses:".ljust(40), line)

        # Remove text after the first comma or ":"
        line = remove_text_after_comma_or_colon(line)
        if show_all:
            print("remove text after comma:".ljust(40), line)

        # Remove remaining punctuation
        line = remove_or_replace_punctuation(line)
        if show_all:
            print("remove punctuation:".ljust(40), line)

        # Remove stopwords
        line = remove_stopwords(line)
        if show_all:
            print("remove stopwords:".ljust(40), line)

        # Normalize quantities
        line = normalize_quantities(line)
        if show_all:
            print("normalize quantities:".ljust(40), line)

        # Remove units of measurement
        line = remove_units(line)
        if show_all:
            print("remove units of measurement:".ljust(40), line)

        # Remove numeric values
        line = remove_numeric_values(line)
        if show_all:
            print("remove numeric values:".ljust(40), line)

        # Convert text to singular
        line = lemmatize_text(line)
        if show_all:
            print("convert text to singular:".ljust(40), line)

        # Normalize special characters
        line = normalize_special_characters(line)
        if show_all:
            print("normalize special characters:".ljust(40), line)

        # Remove duplicate words
        line = remove_duplicate_words(line)
        if show_all:
            print("remove duplicate words:".ljust(40), line)

        # Remove single-character words
        line = remove_lenght1_words(line)
        if show_all:
            print("remove single-character words:".ljust(40), line)

        # Sort words alphabetically
        line = sort_words_alphabetically(line)
        if show_all:
            print("alphabetical sorting:".ljust(40), line)

    if show_something or show_all:
        print("complete normalization:".ljust(40), line, "\n")

    return line


def pipeline(
    input_file,
    output_file,
    column_name,
    new_column_name,
    delimiter=",",
    show_all=False,
    show_something=False,
    translation_nedded=True,
    only_translation=False
) -> None:
    """
    Function to execute the normalization pipeline on a specific column of a CSV file.

    :param input_file: path of the file to normalize
    :param output_file: path of the output file
    :param column_name: name of the column to normalize
    :param new_column_name: name of the new column for normalized values
    :param delimiter: delimiter used in the input CSV file (default is ',')
    :param show_all: if True, shows all intermediate normalization steps
    :param show_something: if True, shows some normalization steps
    """

    # Check if output file exists and get the number of lines already processed
    start_row = 0
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as outfile:
            existing_lines = sum(1 for _ in outfile) - 1  # Subtract header
            start_row = existing_lines

    # Open input and output files
    with (
        open(input_file, "r", encoding="utf-8") as infile,
        open(output_file, "a" if start_row > 0 else "w", encoding="utf-8", newline="") as outfile,
    ):
        reader = csv.DictReader(infile, delimiter=delimiter)
        fieldnames = reader.fieldnames

        if fieldnames is None:
            raise ValueError("Input file is empty or invalid.")

        if column_name not in fieldnames:
            raise ValueError(
                f"Column '{column_name}' not found in the input file."
            )

        # Add the new column to the output file
        if start_row == 0:  # Write header only if starting from scratch
            fieldnames.append(new_column_name) #type: ignore
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
        else:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames + [new_column_name], delimiter=delimiter) #type: ignore

        # Skip rows that are already processed
        for _ in range(start_row):
            next(reader)

        total_rows = sum(1 for _ in open(input_file, "r", encoding="utf-8")) - 1  # Subtract header
        print(f"Resuming from row {start_row + 1}. Total rows to process: {total_rows}")

        start_time = time.time()

        for i, row in enumerate(reader, start=start_row + 1):
            original_line = row[column_name]
            transformed_line = pipeline_core(
                show_all=show_all,
                show_something=show_something,
                line=original_line,
                translation_nedded=translation_nedded,
                only_translation=only_translation
            )
            # Write the normalized value in the new column
            for i, row in enumerate(reader, start=start_row + 1):
                if None in row:
                    print(f"Row {i} contains None keys: {row}")
                row[new_column_name] = transformed_line
                writer.writerow(row)
            

            # Calculate time remaining
            elapsed_time = time.time() - start_time
            avg_time_per_row = elapsed_time / (i - start_row)
            remaining_time = avg_time_per_row * (total_rows - i)

            print(
                f"Processed {i}/{total_rows} rows. "
                f"Estimated time remaining: {remaining_time:.2f} seconds",
                end="\r"
            )

    print(f"\nNormalization complete. Process resumed from row {start_row + 1}.")
