"""
File thar contains functions for attribute extraction
"""

import csv
import ollama
import os
import sys
import time

csv.field_size_limit(new_limit=sys.maxsize)


def add_to_sys_path(folder_name):
    utils_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), folder_name)
    )
    sys.path.append(utils_path)
add_to_sys_path("../ollama_server_file")
from ollama_server import OllamaModel  # type: ignore


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
    Funcion for infere new attribute about user, given an existing attribute

    :param input_file: Input file path
    :param output_file: Output file path
    :param column_name: Name of the column to extract attributes from
    :param new_column_names: List of new columns to add
    :param delimiter: Delimiter of the input file
    :param delimiter2: Delimiter of the output file
    :param show_progress: Show progress of the process or not

    :return: None
    """

    with (
        open(input_file, "r", encoding="utf-8") as infile,
        open(output_file, "w", encoding="utf-8", newline="") as outfile,
    ):
        reader = csv.DictReader(infile, delimiter=delimiter)
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
            outfile, fieldnames=fieldnames, delimiter=delimiter2
        )
        writer.writeheader()

        total_lines = sum(1 for _ in reader)
        infile.seek(0)
        next(reader)

        start_time = time.time()

        for idx, row in enumerate(reader, start=1):
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

            if idx % 5 == 0:
                elapsed_time = time.time() - start_time
                estimated_total_time = (elapsed_time / idx) * total_lines
                remaining_time = estimated_total_time - elapsed_time
                days, remainder = divmod(int(remaining_time), 86400)
                hours, remainder = divmod(remainder, 3600)
                minutes, _ = divmod(remainder, 60)
                print(f"Estimated time remaining: {days} days, {hours} hours, {minutes} minutes")


    print("Normalization complete")


def support_add_user_attributes(extracted_attributes_dictionary, original_line, show_progress):
    modelfile = """FROM qwen2.5:32b
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
    - If no attributes can be extracted do not write "none specified", just write "none". \
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
