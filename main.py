"""
Script for merging the two ontologies
"""


import os
import sys
from rdflib import RDF, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL


def add_to_sys_path(folder_name) -> None:
    """
    Function to add a folder to the system path
    :param folder_name: name of the folder to add
    :return: None
    """

    utils_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), folder_name)
    )
    sys.path.append(utils_path)


add_to_sys_path("entity_linking_file")
add_to_sys_path("create_rdf_file")
add_to_sys_path("normalization_pipeline_file")
add_to_sys_path("attribute_extraction_file")


from entity_linking import (  # type: ignore
    read_specified_columns,
    normalize_columns,
    find_k_most_similar_pairs_with_indicators,
)

from create_rdf import sanitize_for_uri  # type: ignore
from pipeline import pipeline_core, pipeline  # type: ignore
from attribute_extraction import add_user_attributes  # type: ignore


def create_completed_ontology(
    merge: bool = False,
    threshold_value: float = 0.85,
    model="paraphrase-MiniLM-L3-v2",
) -> None:
    """
    Function to create the hummus-off merged ontology

    :param merge: if True, merges the two ontologies
    :param threshold_value: threshold value for merging
    :param model: model to use for merging
    :return: None
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

    ### Merge the ontologies ###
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

        # Normalize the columns by dividing them by serving size
        list_hummus_recipe = normalize_columns(list_hummus_recipe)

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

        # Columns of the foodkg file to be used for the merging
        food_kg_path = (
            "./csv_file/ingredients_food_kg_normalizzed_by_pipeline.csv"
        )
        foodkg_column = ["ingredient", "ingredient_normalized"]
        list_foodkg = read_specified_columns(
            food_kg_path, foodkg_column, delimiter=","
        )

        ### Merge hummus and off ###
        most_similar_pairs = find_k_most_similar_pairs_with_indicators(
            list_hummus_recipe,
            list_off_recipe,
            k=1,
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
            k=1,
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
