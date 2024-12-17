"""
File containing scripts and data necessary for analysis of CSV files and ontologies
"""


import csv
from collections import Counter
import ollama
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def analisys_quantities(input_file, output_file, n) -> None:
    """
    Function to analyze the quantities column of OFF (for analysis purposes only)

    :param input_file: path to the CSV file to analyze
    :param output_file: path to the CSV file where the analysis will be saved
    :param n: the minimum number of occurrences for a quantity to be considered valid
    :return: None
    """

    brand_counts = dict()

    for chunk in pd.read_csv(
        input_file,
        sep="\t",
        chunksize=120000,
        usecols=["quantity"],
        on_bad_lines="skip",
        low_memory=False,
    ):
        for col in ["quantity"]:
            for brand in chunk[col].dropna().unique():
                cleaned_brand = brand
                if cleaned_brand:
                    brand_counts[cleaned_brand] = (
                        brand_counts.get(cleaned_brand, 0) + 1
                    )

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for brand, count in sorted(brand_counts.items()):
            if count >= n:
                writer.writerow([brand])


def number_of_instance_for_columns(
    input_file, output_file, chunk_size=120000
) -> None:
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


def translate_to_english_test(text: str) -> str:
    """
    Function that translates a string into English used for testing the model

    :param text: text to be translated
    :return: response from the translation model
    """
    response = ollama.generate(model="translation_expert", prompt=text)
    print(text, response["response"])
    return response["response"]


def test_filtering_brand_accuracy(file) -> None:
    """
    Function to test the brand filtering model

    :param file: path to the CSV file to read
    with brand_name and expected_response with
    :return: None
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
            is_correct = (
                response["response"].strip().lower() == expected_response
            )
            results.append(
                {
                    "brand_name": brand,
                    "model_response": response["response"],
                    "expected_response": expected_response,
                    "correct": is_correct,
                }
            )

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
            response = ollama.generate(
                model="attribute_extractor", prompt=str(row)
            )
            print("User description: \n", (str(row)))
            print("Extracted attributes: \n", response["response"])


def plot_populated_counts(csv_file, output_dir, columns, output_file):
    """
    Creates a bar plot showing how many rows are populated for specified columns in a CSV file.

    :param csv_file (str): Path to the CSV file.
    :param output_dir (str): Directory to save the plot.
    :param columns (list): List of columns to analyze.
    :return: None
    """

    df = pd.read_csv(csv_file, delimiter="\t", on_bad_lines="skip")

    # Ensure the specified columns exist in the DataFrame
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        print(
            f"Warning: The following columns are not in the CSV and will be ignored: {missing_columns}"
        )
        columns = [col for col in columns if col in df.columns]

    # Count the number of non-null values for each specified column
    populated_counts = df[columns].notnull().sum()
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=populated_counts.index, y=populated_counts.values, palette="viridis"
    )
    plt.title("Populated Counts per Specified Column")
    plt.xlabel("Columns")
    plt.ylabel("Number of Populated Rows")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, output_file)
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved at: {plot_path}")
