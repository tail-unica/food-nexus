"""
File thar contains functions for attribute extraction
"""

import csv
import ollama
import os
import sys
import time

csv.field_size_limit(sys.maxsize)


def add_to_sys_path(folder_name):
    utils_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), folder_name)
    )
    sys.path.append(utils_path)
add_to_sys_path("../ollama_server_file")
from ollama_server import OllamaModel  # type: ignore


def count_lines_in_csv(filename, delimiter):
    """
    Count lines in a CSV file, properly handling quoted fields with newlines and delimiters.
    
    :param filename: Path to the CSV file
    :param delimiter: CSV delimiter
    :return: Number of lines (excluding header)
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.reader(
                file, 
                delimiter=delimiter,
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL
            )
            next(reader, None)  # Skip header
            return sum(1 for _ in reader)
    except Exception as e:
        print(f"Error counting lines: {str(e)}")
        return 0


def add_user_attributes(
    input_file,
    output_file,
    column_name,
    new_column_names,
    delimiter=",",
    delimiter2=",",
    show_progress=True,
) -> None:
    """
    Function for infere new attribute about user, given an existing attribute with checkpoint system

    :param input_file: Input file path
    :param output_file: Output file path
    :param column_name: Name of the column to extract attributes from
    :param new_column_names: List of new columns to add
    :param delimiter: Delimiter of the input file
    :param delimiter2: Delimiter of the output file
    :param show_progress: Show progress of the process or not

    :return: None
    """
    
    # Check if output file exists and count its lines
    skip_rows = 0
    if os.path.exists(output_file):
        skip_rows = count_lines_in_csv(output_file, delimiter2)
        print(f"Found existing output file with {skip_rows} processed rows. Continuing from this point.")
        
        # Verify input file total
        input_total = count_lines_in_csv(input_file, delimiter)
        if skip_rows > input_total:
            print(f"Warning: Output file has more rows ({skip_rows}) than input file ({input_total})")
            return
    
    mode = "a" if skip_rows > 0 else "w"
    
    with (
        open(input_file, "r", encoding="utf-8") as infile,
        open(output_file, mode, encoding="utf-8", newline="") as outfile,
    ):
        reader = csv.DictReader(
            infile, 
            delimiter=delimiter,
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL
        )
        fieldnames = reader.fieldnames

        if fieldnames is None:
            raise ValueError("Input file is empty or invalid.")

        if column_name not in fieldnames:
            raise ValueError(
                f"Column '{column_name}' not found in the input file."
            )

        # Add the new columns to the output file
        fieldnames.extend(new_column_names)  # type: ignore

        writer = csv.DictWriter(
            outfile, 
            fieldnames=fieldnames, 
            delimiter=delimiter2,
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL
        )
        
        if mode == "w":
            writer.writeheader()

        infile.seek(0)
        total_lines = sum(1 for _ in infile)
        infile.seek(0)
        next(reader)  # Skip header after seek
        
        for _ in range(skip_rows):
            try:
                next(reader)
            except StopIteration:
                print("Warning: Reached end of file while skipping rows")
                return

        start_time = time.time()
        rows_processed = 0

        for idx, row in enumerate(reader, start=skip_rows+1):
            rows_processed += 1
            extracted_attributes_dictionary = {}

            # Initialize new columns with empty values
            for column in new_column_names:
                row[column] = ""

            if row[column_name] != "":
                original_line = row[column_name]

                extracted_attributes_dictionary = support_add_user_attributes(
                    extracted_attributes_dictionary=extracted_attributes_dictionary,
                    original_line=original_line,
                    show_progress=show_progress
                    )

            # Add extracted attributes to the row
            for column in new_column_names:
                if column in extracted_attributes_dictionary:
                    row[column] = extracted_attributes_dictionary[column]

            writer.writerow(row)

            if rows_processed % 20 == 0:
                elapsed_time = time.time() - start_time
                remaining_rows = total_lines - (skip_rows + rows_processed)
                if rows_processed > 0:
                    time_per_row = elapsed_time / rows_processed
                    estimated_remaining_time = remaining_rows * time_per_row
                    days, remainder = divmod(int(estimated_remaining_time), 86400)
                    hours, remainder = divmod(remainder, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    print(f"Processed {skip_rows + rows_processed} of {total_lines} rows")
                    print(f"Estimated time remaining: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds")

    print("Normalization complete")


def support_add_user_attributes(extracted_attributes_dictionary, original_line, show_progress):
    modelfile = "llm_model_file/attribute_extractor.txt"

    _model_instance = None

    try:

        if _model_instance is None:
            _model_instance = OllamaModel(modelfile=modelfile, model_name="attribute_extractor")

        extracted_attributes_string = _model_instance.generate(original_line).strip()
        extracted_attributes_string = extracted_attributes_string.replace("####### ", "")

        # Parse attributes
        if show_progress:
            print(extracted_attributes_string)

        for attribute in extracted_attributes_string.split(","):
            if ":" in attribute:
                parts = attribute.split(":", 1)  
                if len(parts) == 2 and ":" not in parts[0]: 
                    attribute_name = parts[0].strip()
                    attribute_value = parts[1].strip()
                    if attribute_name.strip() in [
                        "weight",
                        "height",
                        "gender",
                        "age",
                    ]:
                        if (
                            attribute_name
                            not in extracted_attributes_dictionary
                        ):
                            extracted_attributes_dictionary[
                                attribute_name
                            ] = ""
                        if (
                            extracted_attributes_dictionary[attribute_name]
                            == ""
                        ):
                            extracted_attributes_dictionary[
                                attribute_name
                            ] = attribute_value
                        else:
                            extracted_attributes_dictionary[
                                attribute_name
                            ] = (
                                extracted_attributes_dictionary[
                                    attribute_name
                                ]
                                + ";"
                                + attribute_value
                            )
                    elif attribute_name.strip() in [
                        "physical activity category",
                        "religious constraint",
                        "food allergies or intolerances",
                        "dietary preference",
                    ]:
                        if (
                            "user_constraints"
                            not in extracted_attributes_dictionary
                        ):
                            extracted_attributes_dictionary[
                                "user_constraints"
                            ] = ""
                        if (
                            extracted_attributes_dictionary[
                                "user_constraints"
                            ]
                            == ""
                        ):
                            extracted_attributes_dictionary[
                                "user_constraints"
                            ] = (attribute_name + ": " + attribute_value)
                        else:
                            extracted_attributes_dictionary[
                                "user_constraints"
                            ] = (
                                extracted_attributes_dictionary[
                                    "user_constraints"
                                ]
                                + "; "
                                + attribute_name
                                + ": "
                                + attribute_value
                            )
                else:
                    attribute = ""
        return extracted_attributes_dictionary
    except Exception as e:
        print(f"Attribute extraction failed: {e}")
        _model_instance = None
        return support_add_user_attributes(extracted_attributes_dictionary, original_line, show_progress)
