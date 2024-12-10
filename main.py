"""
BETA of the Script for merging the two ontologies
"""


import os
import sys
from rdflib import RDF, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL
import ollama
import csv


def add_to_sys_path(folder_name):
    utils_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), folder_name)
    )
    sys.path.append(utils_path)


add_to_sys_path("entity_linking_file")
add_to_sys_path("create_rdf_file")
add_to_sys_path("normalization_pipeline_file")

from entity_linking import (  # type: ignore
    read_specified_columns,
    normalize_columns,
    find_k_most_similar_pairs_with_indicators,
)

from create_rdf import sanitize_for_uri  # type: ignore
from pipeline import pipeline_core, pipeline  # type: ignore


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


from collections import Counter
from rdflib import Graph, Literal, RDF
import csv

def analyze_turtle_file(file_path):
    """
    Analyze a Turtle file for ontology statistics, including counts of instances.
    """
    g = Graph()
    g.parse(file_path, format="turtle")
    
    # Insiemi e contatori
    entities = set()
    relations = set()
    attributes = set()
    entity_types = Counter()
    relation_types = Counter()
    
    # Analizza le triple RDF
    for subject, predicate, obj in g:
        # Raccogli entità e relazioni
        entities.add(subject)
        relations.add(predicate)
        if isinstance(obj, Literal):
            attributes.add(obj)
        else:
            entities.add(obj)
        
        # Conta le istanze di tipi di entità
        if predicate == RDF.type:
            entity_types[str(obj)] += 1  # Usa string per contare i tipi come stringhe leggibili
        
        # Conta le relazioni
        relation_types[str(predicate)] += 1
    
    return {
        "num_triples": len(g),
        "num_entities": len(entities),
        "num_relations": len(relations),
        "num_attributes": len(attributes),
        "num_entity_types": len(entity_types),
        "num_relation_types": len(relation_types),
        "entity_types": entity_types,
        "relation_types": relation_types
    }

def ontology_statistics(turtle_files, output_csv):
    """
    Analyze Turtle files and write statistics to a CSV, including instance counts.
    """
    all_entity_types = Counter()
    all_relation_types = Counter()
    
    results = []
    for file_path in turtle_files:
        stats = analyze_turtle_file(file_path)
        results.append({
            "file": file_path.split("/")[-1],
            "num_triples": stats["num_triples"],
            "num_entities": stats["num_entities"],
            "num_relations": stats["num_relations"],
            "num_attributes": stats["num_attributes"],
            "num_entity_types": len(stats["entity_types"]),
            "num_relation_types": len(stats["relation_types"]),
            "entity_types": stats["entity_types"],
            "relation_types": stats["relation_types"]
        })
        all_entity_types.update(stats["entity_types"])
        all_relation_types.update(stats["relation_types"])
    
    # Colonne dinamiche
    entity_type_cols = sorted(all_entity_types.keys())
    relation_type_cols = sorted(all_relation_types.keys())
    
    header = [
        "file", 
        "num_triples",
        "num_entities", 
        "num_relations", 
        "num_attributes", 
        "num_entity_types", 
        "num_relation_types"
    ] + [f"entity_type_{entita.split("/")[-1]}" for entita in entity_type_cols] + [f"relation_type_{relazione.split("/")[-1]}" for relazione in relation_type_cols]
    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        
        for result in results:
            row = {
                "file": result["file"],
                "num_triples": result["num_triples"],
                "num_entities": result["num_entities"],
                "num_relations": result["num_relations"],
                "num_attributes": result["num_attributes"],
                "num_entity_types": result["num_entity_types"],
                "num_relation_types": result["num_relation_types"]
            }
            # Aggiungi i conteggi delle entità
            for entity_type in entity_type_cols:
                row[f"entity_type_{entity_type.split("/")[-1]}"] = result["entity_types"].get(entity_type, 0)
            # Aggiungi i conteggi delle relazioni
            for relation_type in relation_type_cols:
                row[f"relation_type_{relation_type.split("/")[-1]}"] = result["relation_types"].get(relation_type, 0)
            
            writer.writerow(row)



def create_completed_ontology(
    merge: bool = False, threshold_value: float = 0.85, model = "paraphrase-MiniLM-L3-v2"
) -> None:
    """
    Function to create the hummus-off ontology with associate recipe and ingredient
    """

    # Define the output file path
    file_output = "./csv_file/unica_food_ontology.ttl"

    # Initialize the graph and namespaces
    g = Graph()

    # Obtain namespace
    UNICA = Namespace("https://github.com/tail-unica/kgeats/")
    SCHEMA = Namespace("https://schema.org/")

    # Associate namespaces with the graph
    g.bind("unica", UNICA)
    g.bind("schema", SCHEMA)

    # Ontology versioning
    link_ontology = "https://github.com/tail-unica/kgeats/unica_food_ontology"
    ontology_iri = URIRef(f"{link_ontology}")
    version_iri = URIRef(f"{link_ontology}/1.0")
    version_info = Literal("Version 1.0 - Initial release", lang="en")

    g.add((ontology_iri, RDF.type, OWL.Ontology))
    g.add((ontology_iri, OWL.versionIRI, version_iri))
    g.add((ontology_iri, OWL.versionInfo, version_info))

    # Reference to the previous version
    prior_version_iri = URIRef(f"{link_ontology}/0.0")
    g.add((ontology_iri, OWL.priorVersion, prior_version_iri))

    ontology_files = [
        "./csv_file/ontology_hummus.ttl",
        "./csv_file/ontology_off.ttl",
    ]

    # Upload the ontologies
    for file_path in ontology_files:
        temp_graph = Graph()
        temp_graph.namespace_manager.bind("unica", UNICA)
        temp_graph.namespace_manager.bind("schema", SCHEMA)

        # Upload the ontologies data in the new graph
        temp_graph.parse(file_path, format="turtle")
        g += temp_graph

    # Merge the ontologies
    if merge:

        # Columns of the hummus file to be used for the merging
        hummus_file_path = "./csv_file/pp_recipes_normalized_by_pipeline.csv"
        hummus_column = [
            "title",
            "totalFat [g]",
            "totalCarbohydrate [g]",
            "protein [g]",
            "servingSize [g]",
            "title_normalized",
        ]
        list_hummus_recipe = read_specified_columns(
            hummus_file_path, hummus_column, delimiter=","
        )
        # print(list_hummus_recipe[0])

        # Normalize the columns by dividing them by serving size
        list_hummus_recipe = normalize_columns(list_hummus_recipe)

        # print(list_hummus_recipe[0])

        # Columns of the off file to be used for the merging
        off_file_path = "./csv_file/off_normalized_by_pipeline.csv"
        off_column = [
            "product_name",
            "fat_100g",
            "carbohydrates_100g",
            "proteins_100g",
            "product_name_normalized",
        ]
        list_off_recipe = read_specified_columns(
            off_file_path, off_column, delimiter="\t"
        )

        # print(list_off_recipe[0])

        # Columns of the foodkg file to be used for the merging
        food_kg_path = (
            "./csv_file/ingredients_food_kg_normalizzed_by_pipeline.csv"
        )
        foodkg_column = ["ingredient", "ingredient_normalized"]
        list_foodkg = read_specified_columns(
            food_kg_path, foodkg_column, delimiter=","
        )

        # print(list_foodkg[0])

        ### Merge hummus and off ###
        most_similar_pairs = find_k_most_similar_pairs_with_indicators(
            list_hummus_recipe,
            list_off_recipe,
            k=5,
            model=model,
            use_indicator=True,
        )

        for score, original_name1, original_name2 in most_similar_pairs:
            if float(score) > threshold_value:

                hummus_recipes = [
                    recipe
                    for recipe in g.subjects(RDF.type, SCHEMA.Recipe)
                    if (recipe, SCHEMA.name, Literal(original_name1, lang="en"))
                    in g
                    and recipe.startswith(str(UNICA) + "Recipe_hummus")  # type: ignore
                ]

                off_recipes = [
                    recipe
                    for recipe in g.subjects(RDF.type, SCHEMA.Recipe)
                    if (recipe, SCHEMA.name, Literal(original_name2, lang="en"))
                    in g
                    and recipe.startswith(str(UNICA) + "Recipe_off")  # type: ignore
                ]

                for hummus_recipe in hummus_recipes:
                    for off_recipe in off_recipes:
                        g.add((hummus_recipe, SCHEMA.sameAs, off_recipe))
                        g.add((off_recipe, SCHEMA.sameAs, hummus_recipe))

        ### Merge foodkg and off ###
        for i, item in enumerate(list_off_recipe):
            list_off_recipe[i] = [item[0], item[-1]]

        most_similar_pairs = find_k_most_similar_pairs_with_indicators(
            list_off_recipe,
            list_foodkg,
            k=5,
            model=model,
            use_indicator=False,
        )

        for score, original_name1, original_name2 in most_similar_pairs:

            if float(score) > threshold_value:

                off_recipes = [
                    recipe
                    for recipe in g.subjects(RDF.type, SCHEMA.Recipe)
                    if (recipe, SCHEMA.name, Literal(original_name1, lang="en"))
                    in g
                    and recipe.startswith(str(UNICA) + "Recipe_off")  # type: ignore
                ]

                foodkg_recipes = [
                    recipe
                    for recipe in g.subjects(RDF.type, SCHEMA.Recipe)
                    if (
                        recipe,
                        SCHEMA.identifier,
                        Literal(
                            sanitize_for_uri(
                                original_name2.replace(" ", "_").lower()
                            )
                        ),
                    )
                    in g
                    and recipe.startswith(str(UNICA) + "Recipe_Ingredient")  # type: ignore
                ]

                for foodkg_recipe in foodkg_recipes:
                    for off_recipe in off_recipes:
                        g.add((foodkg_recipe, SCHEMA.sameAs, off_recipe))
                        g.add((off_recipe, SCHEMA.sameAs, foodkg_recipe))

    # Save the RDF graph in Turtle format
    g.serialize(destination=file_output, format="turtle")
    print(f"Generated file: {file_output}")
