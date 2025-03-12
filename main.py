"""
Script for merging the two ontologies
"""


from _csv import _writer
import os
import sys
import time
import csv
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
    file_output = "./csv_file/complete_food_ontology.ttl"

    # Initialize the graph and namespaces
    g = Graph()

    # Obtain namespace
    UNICA = Namespace("https://github.com/tail-unica/kgeats/")
    SCHEMA = Namespace("https://schema.org/")

    # Associate namespaces with the graph
    g.bind("unica", UNICA)
    g.bind("schema", SCHEMA)

    # Ontology versioning
    link_ontology = "https://github.com/tail-unica/kgeats/unica_complete_food_ontology"
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

        print(f"added to onthology the subonthology {file_path}")



    ### Merge the ontologies ###
    if merge:

        print(f"starting the merging process")

        ### Merge hummus and off ###
        file_off_hummus = "./csv_file/file_off_hummus.csv"
        off_hummus_columns = ["hummus_id", "off_id"]
        list_off_hummus_recipe = read_specified_columns(
            file_off_hummus, off_hummus_columns, delimiter=","
        )

        for hummus_id, off_id in file_off_hummus:
            hummus_recipes = [
                recipe
                for recipe in g.subjects(RDF.type, SCHEMA.Recipe)
                if (recipe, SCHEMA.identifier, Literal(hummus_id))
                in g
                and recipe.startswith(str(UNICA) + "Recipe_hummus")  # type: ignore
            ]

            off_recipes = [
                recipe
                for recipe in g.subjects(RDF.type, SCHEMA.Recipe)
                if (recipe, SCHEMA.identifier, Literal(off_id))
                in g
                and recipe.startswith(str(UNICA) + "Recipe_off")  # type: ignore
            ]

            for hummus_recipe in hummus_recipes:
                for off_recipe in off_recipes:
                    g.add((hummus_recipe, SCHEMA.sameAs, off_recipe))
                    g.add((off_recipe, SCHEMA.sameAs, hummus_recipe))



        ### Merge foodkg and off ###           
        file_off_foodkg = "./csv_file/file_off_foodkg.csv"
        off_foodkg_columns = ["off_recipe", "foodkg_recipe"]
        list_off_fgk_recipe = read_specified_columns(
            file_off_hummus, off_foodkg_columns, delimiter=","
        )

        for original_name1, original_name2 in file_off_foodkg:

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
