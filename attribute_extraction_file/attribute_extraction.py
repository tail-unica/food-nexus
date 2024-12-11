"""
File thar contains functions for attribute extraction
"""


import csv
import ollama


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

        for row in reader:
            extracted_attributes_dictionary = {}

            # Initialize new columns with empty values
            for column in new_column_names:
                row[column] = ""

            if row[column_name] != "":
                original_line = row[column_name]

                # Call the model for attribute extraction
                extracted_attributes_string = ollama.generate(
                    model="attribute_extractor", prompt=original_line
                )

                # Clean the response
                extracted_attributes_string = extracted_attributes_string[
                    "response"
                ].replace("####### ", "")

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
                                    extracted_attributes_dictionary[
                                        attribute_name
                                    ]
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
                                    ] = (
                                        attribute_name + ": " + attribute_value
                                    )
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

            # Add extracted attributes to the row
            for column in new_column_names:
                if column in extracted_attributes_dictionary:
                    row[column] = extracted_attributes_dictionary[column]

            writer.writerow(row)

    print("Normalization complete")
