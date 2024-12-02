"""
SO CHE QUESTO FILE è SCRITTO MALE, LO STO USANDO PER FARE PROVE
inoltre si deve ancora definire quale bert usare per il linking
"""

# ruff: noqa: E402

from create_rdf_script import sanitize_for_uri  # type: ignore
from entity_linking_script import (  # type: ignore
    find_k_most_similar_pairs_with_indicators,
    normalize_columns,
    read_specified_columns,
)
from pipeline_script import pipeline_core  # type: ignore
from rdflib import RDF, Graph, Literal, Namespace, URIRef  # type: ignore
from rdflib.namespace import OWL  # type: ignore


def create_completed_ontology(merge: bool = False) -> None:
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
        "./csv_file/ontologia_nostra_hummus.ttl",
        "./csv_file/ontologia_nostra_off.ttl",
    ]

    # Upload the ontologies
    for file_path in ontology_files:
        temp_graph = Graph()
        temp_graph.namespace_manager.bind("unica", UNICA)
        temp_graph.namespace_manager.bind("schema", SCHEMA)

        temp_graph.parse(file_path, format="turtle")
        g += temp_graph

    # Merge the ontologies
    if merge:
        hummus_file_path = "./csv_file/pp_recipes_rows.csv"
        hummus_column = [
            "title",
            "totalFat [g]",
            "totalCarbohydrate [g]",
            "protein [g]",
            "servingSize [g]",
        ]
        list_hummus_recipe = read_specified_columns(
            hummus_file_path, hummus_column, delimiter=","
        )
        list_hummus_recipe = normalize_columns(list_hummus_recipe)

        off_file_path = "./csv_file/off_rows.csv"
        off_column = [
            "product_name",
            "fat_100g",
            "carbohydrates_100g",
            "proteins_100g",
        ]
        list_off_recipe = read_specified_columns(
            off_file_path, off_column, delimiter="\t"
        )

        food_kg_path = "./csv_file/pp_recipes_rows.csv"
        foodkg_column = ["ingredient_food_kg_names"]
        list_foodkg_temp = read_specified_columns(
            food_kg_path, foodkg_column, delimiter=","
        )

        list_foodkg = []
        for elementi in list_foodkg_temp:
            for elementi2 in (
                str(elementi).strip('("[]"),').replace("'", "").split(", ")
            ):
                list_foodkg.append(elementi2)
        list_foodkg = list(set(list_foodkg))

        for lst1 in (list_hummus_recipe, list_off_recipe):
            for i, item in enumerate(lst1):
                if item[0] != "":
                    temp_item = list(item)
                    normalized_name = pipeline_core(
                        line=item[0], show_all=False, show_something=False
                    )
                    if normalized_name != "":
                        temp_item.insert(0, normalized_name)
                        temp_item.append(temp_item.pop(1))
                        lst1[i] = tuple(temp_item)
                    else:
                        lst1.pop(i)

        most_similar_pairs = find_k_most_similar_pairs_with_indicators(
            list_hummus_recipe, list_off_recipe, k=5
        )

        for score, original_name1, original_name2 in most_similar_pairs:
            threshold = 0.05
            if float(score) > threshold:
                hummus_recipes = [
                    recipe
                    for recipe in g.subjects(RDF.type, SCHEMA.Recipe)
                    if (recipe, SCHEMA.name, Literal(original_name1, lang="en")) in g
                    and recipe.startswith(str(UNICA) + "Recipe_hummus")
                ]

                off_recipes = [
                    recipe
                    for recipe in g.subjects(RDF.type, SCHEMA.Recipe)
                    if (recipe, SCHEMA.name, Literal(original_name2, lang="en")) in g
                    and recipe.startswith(str(UNICA) + "Recipe_off")
                ]

                for hummus_recipe in hummus_recipes:
                    for off_recipe in off_recipes:
                        g.add((hummus_recipe, SCHEMA.sameAs, off_recipe))
                        g.add((off_recipe, SCHEMA.sameAs, hummus_recipe))

        for i, item in enumerate(list_foodkg):
            if item != "":
                normalized_name = pipeline_core(
                    line=item, show_all=False, show_something=False
                )
                if normalized_name != "":
                    list_foodkg[i] = [normalized_name, item]
                else:
                    list_foodkg.pop(i)

        for i, item in enumerate(list_off_recipe):
            list_off_recipe[i] = [item[0], item[-1]]

        most_similar_pairs = find_k_most_similar_pairs_with_indicators(
            list_off_recipe, list_foodkg, k=5
        )

        for score, original_name1, original_name2 in most_similar_pairs:
            threshold = 0.05
            if float(score) > threshold:
                off_recipes = [
                    recipe
                    for recipe in g.subjects(RDF.type, SCHEMA.Recipe)
                    if (recipe, SCHEMA.name, Literal(original_name1, lang="en")) in g
                    and recipe.startswith(str(UNICA) + "Recipe_off")
                ]

                foodkg_recipes = [
                    recipe
                    for recipe in g.subjects(RDF.type, SCHEMA.Recipe)
                    if (
                        recipe,
                        SCHEMA.identifier,
                        Literal(
                            sanitize_for_uri(original_name2.replace(" ", "_").lower())
                        ),
                    )
                    in g
                    and recipe.startswith(str(UNICA) + "Recipe_Ingredient")
                ]
                for foodkg_recipe in foodkg_recipes:
                    for off_recipe in off_recipes:
                        g.add((foodkg_recipe, SCHEMA.sameAs, off_recipe))
                        g.add((off_recipe, SCHEMA.sameAs, foodkg_recipe))

    # Save the RDF graph in Turtle format
    g.serialize(destination=file_output, format="turtle")
    print(f"Generated file: {file_output}")
