
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

                # Call the model
                extracted_attributes_string = ollama.generate(
                    model="attribute_extractor", prompt=original_line
                )
                extracted_attributes_string = extracted_attributes_string[
                    "response"
                ].replace("####### ", "")

                # Parse attributes
                if show_progress:
                    print(extracted_attributes_string)

                for attribute in extracted_attributes_string.split(","):
                    if len(attribute.split(":")) > 1:
                        attribute_name, attribute_value = attribute.split(":")
                        attribute_name = attribute_name.strip()
                        attribute_value = attribute_value.strip()
                        # print(attribute_name, attribute_value)
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

            # Add extracted attributes to the row
            for column in new_column_names:
                if column in extracted_attributes_dictionary:
                    row[column] = extracted_attributes_dictionary[column]

            writer.writerow(row)

    print("Normalization complete")
