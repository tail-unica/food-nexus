"""
File with various functions and data necessary for converting HUMMUS and Open Food Facts data into an RDF format.
"""

import re
import pandas as pd
from rdflib import RDF, RDFS, XSD, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL


def sanitize_for_uri(value) -> str:
    """
    Generic sanitization function for URIs

    :param value: value to sanitize

    :return: sanitized value
    """
    return re.sub(r"[^a-zA-Z0-9_]", "", str(value))


def clean_column_name(col_name) -> str:
    """
    Function to clean the column name by removing content within square brackets

    :param col_name: name of the column

    :return: cleaned column name
    """
    return re.sub(r"\s*\[.*?\]", "", col_name).strip()


def create_namespace(namespace_completo=True) -> None:
    """
    Function to create the TTL file with the custom namespace for UNICA
    """

    # Create the graph
    g = Graph()

    # Define the namespaces
    SCHEMA = Namespace("https://schema.org/")
    UNICA = Namespace(
        "https://github.com/tail-unica/kgeats/"
    )  # Custom namespace, the name and address will be decided later
    XSD_NS = Namespace("http://www.w3.org/2001/XMLSchema#")
    RDFS_NS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    RDF_NS = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")

    # Add prefixes to the graph
    g.bind("schema", SCHEMA)
    g.bind("unica", UNICA)
    g.bind("xsd", XSD_NS)
    g.bind("rdfs", RDFS_NS)
    g.bind("rdf", RDF_NS)

    if namespace_completo:
        # Create classes
        g.add((UNICA.FoodProducer, RDF.type, RDFS.Class))
        g.add((UNICA.FoodProducer, RDFS.subClassOf, SCHEMA.Organization))
        g.add(
            (
                UNICA.FoodProducer,
                RDFS.label,
                Literal("Food Producer", lang="en"),
            )
        )
        g.add(
            (
                UNICA.FoodProducer,
                RDFS.comment,
                Literal(
                    "Information about the producer of a food item.", lang="en"
                ),
            )
        )

    g.add((UNICA.UserConstraint, RDF.type, RDFS.Class))
    g.add((UNICA.UserConstraint, RDFS.subClassOf, SCHEMA.Intangible))
    g.add(
        (
            UNICA.UserConstraint,
            RDFS.label,
            Literal("User Constraint", lang="en"),
        )
    )
    g.add(
        (
            UNICA.UserConstraint,
            RDFS.comment,
            Literal(
                "Constraint about what a user can or want to eat.", lang="en"
            ),
        )
    )

    # Propriety for UserConstraint
    g.add((UNICA.constraintName, RDF.type, RDF.Property))
    g.add((UNICA.constraintName, RDFS.domain, UNICA.UserConstraint))
    g.add((UNICA.constraintName, RDFS.range, XSD.string))
    g.add(
        (
            UNICA.constraintName,
            RDFS.label,
            Literal("Constraint Name", lang="en"),
        )
    )
    g.add(
        (
            UNICA.constraintName,
            RDFS.comment,
            Literal("Name of the constraint.", lang="en"),
        )
    )

    g.add((UNICA.constraintDescription, RDF.type, RDF.Property))
    g.add((UNICA.constraintDescription, RDFS.domain, UNICA.UserConstraint))
    g.add((UNICA.constraintDescription, RDFS.range, XSD.string))
    g.add(
        (
            UNICA.constraintDescription,
            RDFS.label,
            Literal("Constraint Description", lang="en"),
        )
    )
    g.add(
        (
            UNICA.constraintDescription,
            RDFS.comment,
            Literal("Description of the constraint.", lang="en"),
        )
    )

    # Create the Indicator class
    g.add((UNICA.Indicator, RDF.type, RDFS.Class))
    g.add((UNICA.Indicator, RDFS.subClassOf, SCHEMA.NutritionInformation))
    g.add((UNICA.Indicator, RDFS.label, Literal("Indicator", lang="en")))
    g.add(
        (
            UNICA.Indicator,
            RDFS.comment,
            Literal(
                "Represents nutritional and sustainability indicators for food items.",
                lang="en",
            ),
        )
    )

    if namespace_completo:
        # Indicator property
        proprieta_indicator = [
            (
                "calciumPer100g",
                "Calcium per 100g (mg)",
                "Calcium content per 100 grams.",
                XSD.float,
            ),
            (
                "ironPer100g",
                "Iron per 100g (mg)",
                "Iron content per 100 grams.",
                XSD.float,
            ),
            (
                "vitaminCPer100g",
                "Vitamin C per 100g (mg)",
                "Vitamin C content per 100 grams.",
                XSD.float,
            ),
            (
                "vitaminAPer100g",
                "Vitamin A per 100g (mcg)",
                "Vitamin A content per 100 grams.",
                XSD.float,
            ),
            (
                "whoScore",
                "WHO Score",
                "A score indicating WHO healthfulness assessment.",
                XSD.float,
            ),
            (
                "fsaScore",
                "FSA Score",
                "A score based on the Food Standards Agency's healthfulness criteria.",
                XSD.float,
            ),
            (
                "nutriScore",
                "Nutri-Score",
                "A score indicating the nutritional value based on the Nutri-Score system.",
                XSD.float,
            ),
            (
                "nutriscoreScore",
                "Nutri-Score Score",
                "A numeric score for Nutri-Score system.",
                XSD.float,
            ),
            (
                "ecoscoreScore",
                "Eco-Score Score",
                "A numeric score for Eco-Score system.",
                XSD.float,
            ),
            (
                "nutritionScoreFrPer100g",
                "Nutrition Score FR per 100g",
                "French nutrition score per 100 grams.",
                XSD.float,
            ),
            (
                "nutriscoreGrade",
                "Nutri-Score Grade",
                "A grade indicating the Nutri-Score nutritional level.",
                XSD.string,
            ),
            (
                "ecoscoreGrade",
                "Eco-Score Grade",
                "A grade indicating the Eco-Score environmental impact level.",
                XSD.string,
            ),
            (
                "novaGroup",
                "NOVA Group",
                "Group classification of processing level based on NOVA system.",
                XSD.string,
            ),
        ]
    else:
        # Indicator Property
        proprieta_indicator = [
            (
                "whoScore",
                "WHO Score",
                "A score indicating WHO healthfulness assessment.",
                XSD.float,
            ),
            (
                "fsaScore",
                "FSA Score",
                "A score based on the Food Standards Agency's healthfulness criteria.",
                XSD.float,
            ),
            (
                "nutriScore",
                "Nutri-Score",
                "A score indicating the nutritional value based on the Nutri-Score system.",
                XSD.float,
            ),
        ]

    for prop, label, comment, range_type in proprieta_indicator:
        g.add((UNICA[prop], RDF.type, RDF.Property))
        g.add((UNICA[prop], RDFS.domain, UNICA.Indicator))
        g.add((UNICA[prop], RDFS.range, range_type))
        g.add((UNICA[prop], RDFS.label, Literal(label, lang="en")))
        g.add((UNICA[prop], RDFS.comment, Literal(comment, lang="en")))

    # Save in Turtle format with the name 'namespace_unica.ttl'
    with open("../csv_file/namespace_unica.ttl", "w") as f:
        f.write(g.serialize(format="turtle"))

    print("namespace created successfully")


def convert_hummus_in_rdf() -> None:
    """
    Function to convert hummus data into RDF format
    """

    # Initialize the graph and namespaces
    g = Graph()

    # Obtain namespace
    UNICA = Namespace("https://github.com/tail-unica/kgeats/")
    SCHEMA = Namespace("https://schema.org/")

    # Associate namespaces with the graph
    g.bind("unica", UNICA)
    g.bind("schema", SCHEMA)

    # Ontology versioning
    link_ontology = "https://github.com/tail-unica/kgeats/hummus"
    ontology_iri = URIRef(f"{link_ontology}")
    version_iri = URIRef(f"{link_ontology}/1.0")
    version_info = Literal("Version 1.0 - Initial release", lang="en")

    g.add((ontology_iri, RDF.type, OWL.Ontology))
    g.add((ontology_iri, OWL.versionIRI, version_iri))
    g.add((ontology_iri, OWL.versionInfo, version_info))

    # Reference to the previous version
    prior_version_iri = URIRef(f"{link_ontology}/0.0")
    g.add((ontology_iri, OWL.priorVersion, prior_version_iri))

    # input file
    # file_ricette = "../csv_file/pp_recipes.csv"
    # file_review = "../csv_file/pp_reviews.csv"
    # file_utenti = "../csv_file/pp_members_normalized.csv"

    file_ricette = "../csv_file/pp_recipes_rows.csv"
    file_review = "../csv_file/pp_reviews_rows.csv"
    file_utenti = "../csv_file/pp_members_normalized.csv"

    file_output = "../csv_file/ontology_hummus.ttl"

    # Upload the CSV
    df_ricette = pd.read_csv(file_ricette, on_bad_lines="skip")
    df_review = pd.read_csv(file_review, on_bad_lines="skip")
    df_utenti = pd.read_csv(file_utenti, on_bad_lines="skip")

    tag_count = {}
    ingredient_count = {}
    constraint_count = {}

    # Indicator measurement mapping
    indicator_fields = {
        "servingSize": "grams",
        "calories": "cal",
        "caloriesFromFat": "cal",
        "totalFat": "grams",
        "saturatedFat": "grams",
        "cholesterol": "mg",
        "sodium": "mg",
        "totalCarbohydrate": "grams",
        "dietaryFiber": "grams",
        "sugars": "grams",
        "protein": "grams",
        "who_score": None,
        "fsa_score": None,
        "nutri_score": None,
    }

    qualitatives_indicators_hummus = [
        "servingSize",
        "who_score",
        "fsa_score",
        "nutri_score",
    ]

    # Create the entity UserGroup
    for idx, row in df_utenti.iterrows():
        if pd.notna(row["member_id"]):
            group_id = URIRef(
                UNICA[f"UserGroup_{sanitize_for_uri(row['member_id'])}"]
            )
            g.add((group_id, RDF.type, SCHEMA.Organization))
            if pd.notna(row["member_name"]):
                g.add(
                    (
                        group_id,
                        SCHEMA.name,
                        Literal(row["member_name"], lang="en"),
                    )
                )
            if pd.notna(row["age"]):
                if isinstance(row["age"], int):
                    g.add(
                        (
                            group_id,
                            SCHEMA.age,
                            Literal(row["age"], datatype=XSD.integer),
                        )
                    )
                else:
                    g.add(
                        (
                            group_id,
                            SCHEMA.typicalAgeRange,
                            Literal(row["age"], datatype=XSD.string),
                        )
                    )
            if pd.notna(row["gender"]):
                g.add(
                    (
                        group_id,
                        SCHEMA.gender,
                        Literal(row["gender"], datatype=XSD.string),
                    )
                )
            if pd.notna(row["weight"]):
                g.add(
                    (
                        group_id,
                        SCHEMA.weight,
                        Literal(row["weight"], datatype=XSD.float),
                    )
                )
            if pd.notna(row["height"]):
                g.add(
                    (
                        group_id,
                        SCHEMA.height,
                        Literal(row["height"], datatype=XSD.float),
                    )
                )

            if pd.notna(row["user_constraints"]):
                contraint_list = row["user_constraints"].split(";")
                for contraint in contraint_list:
                    constraint_name, constraint_value = contraint.split(":")
                    tag_id = URIRef(
                        UNICA[
                            f"Constraint_{sanitize_for_uri(constraint_name).strip()}_{sanitize_for_uri(constraint_value).strip()}"
                        ]
                    )
                    if tag_id not in constraint_count:
                        constraint_count[tag_id] = 1
                        g.add((tag_id, RDF.type, UNICA.UserConstraint))
                        g.add(
                            (
                                tag_id,
                                SCHEMA.constraintName,
                                Literal(constraint_name, lang="en"),
                            )
                        )
                        g.add(
                            (
                                tag_id,
                                SCHEMA.constraintDescription,
                                Literal(
                                    f"is a user constraint about {constraint_name}",
                                    lang="en",
                                ),
                            )
                        )
                    g.add((group_id, SCHEMA.hasConstraint, tag_id))

    # Create the entity Recipe
    for idx, row in df_ricette.iterrows():
        if pd.notna(row["recipe_id"]):
            recipe_id = URIRef(
                UNICA[f"Recipe_hummus{sanitize_for_uri(row['recipe_id'])}"]
            )
            g.add((recipe_id, RDF.type, SCHEMA.Recipe))
            if pd.notna(row["title"]):
                g.add(
                    (recipe_id, SCHEMA.name, Literal(row["title"], lang="en"))
                )

            g.add((recipe_id, SCHEMA.identifier, Literal(row["recipe_id"])))
            if pd.notna(row["directions"]):
                g.add(
                    (
                        recipe_id,
                        SCHEMA.RecipeInstructions,
                        Literal(row["directions"], lang="en"),
                    )
                )

            if pd.notna(row["author_id"]):
                author_id = URIRef(
                    UNICA[f"UserGroup_{sanitize_for_uri(row['author_id'])}"]
                )
                g.add((author_id, SCHEMA.publishesRecipe, recipe_id))

            # UserConstraint
            if pd.notna(row["tags"]):
                tags = (
                    str(row["tags"])
                    .strip("[]")
                    .replace("'", "")
                    .split(sep=", ")
                )
                for tag in tags:
                    tag1 = tag
                    tag = sanitize_for_uri(tag.replace("-", "_").lower())
                    tag_id = URIRef(UNICA[f"Tag_{tag}"])
                    if tag not in tag_count:
                        tag_count[tag] = 1
                        g.add((tag_id, RDF.type, UNICA.UserConstraint))
                        g.add(
                            (
                                tag_id,
                                SCHEMA.constraintName,
                                Literal(tag, lang="en"),
                            )
                        )
                        g.add(
                            (
                                tag_id,
                                SCHEMA.constraintDescription,
                                Literal(
                                    f"is a user constraint about {tag1}",
                                    lang="en",
                                ),
                            )
                        )

                    g.add((tag_id, SCHEMA.suitableForDiet, recipe_id))

            # Indicator
            for col, unit in indicator_fields.items():
                # Find the column in the CSV corresponding to the indicator, after removing the square brackets
                csv_column = next(
                    (
                        csv_col
                        for csv_col in row.index
                        if clean_column_name(csv_col) == col
                    ),
                    None,
                )
                # Continue only if we found a match and the value is not NaN
                if csv_column and pd.notna(row[csv_column]):
                    indicator_id = URIRef(
                        UNICA[
                            f"Indicator_{sanitize_for_uri(col.replace(' ', '_').lower())}_{sanitize_for_uri(row['recipe_id'])}"
                        ]
                    )
                    g.add((indicator_id, RDF.type, UNICA.Indicator))
                    g.add((indicator_id, SCHEMA.type, Literal(col)))
                    if unit:
                        g.add((indicator_id, SCHEMA.unitText, Literal(unit)))

                    stringa = str(row[csv_column])
                    if col in qualitatives_indicators_hummus:
                        quantità = (
                            float(stringa) / float(row["servingSize [g]"]) * 100
                        )
                    else:
                        quantità = float(stringa)

                    g.add(
                        (
                            indicator_id,
                            SCHEMA.quantity,
                            Literal(quantità, datatype=XSD.float),
                        )
                    )
                    g.add(
                        (recipe_id, SCHEMA.NutritionInformation, indicator_id)
                    )

            # Ingredients
            if pd.notna(row["ingredient_food_kg_names"]):
                ingredients = row["ingredient_food_kg_names"].split(", ")
                for ing in ingredients:
                    ingredient_id = URIRef(
                        UNICA[
                            f"Recipe_Ingredient_{sanitize_for_uri(ing.replace(' ', '_').lower())}"
                        ]
                    )
                    if ing not in ingredient_count:
                        ingredient_count[ing] = 1
                        g.add((ingredient_id, RDF.type, SCHEMA.Recipe))

                        g.add(
                            (
                                ingredient_id,
                                SCHEMA.identifier,
                                Literal(
                                    sanitize_for_uri(
                                        ing.replace(" ", "_").lower()
                                    )
                                ),
                            )
                        )

                    ingredient_id_for_recipe = (
                        ingredient_id + "_" + str(object=idx)
                    )
                    g.add(
                        (
                            ingredient_id_for_recipe,
                            RDF.type,
                            SCHEMA.QuantitativeValue,
                        )
                    )
                    g.add(
                        (
                            ingredient_id_for_recipe,
                            SCHEMA.isRelatedTo,
                            ingredient_id,
                        )
                    )

                    g.add(
                        (
                            recipe_id,
                            SCHEMA.hasPart,
                            ingredient_id_for_recipe,
                        )
                    )
                    # g.add((ingredient_id_for_recipe, SCHEMA.isPartOf, recipe_id))

    # Create UserReview entities and relationships
    for idx, row in df_review.iterrows():
        if pd.notna(row["rating"]) and isinstance(row["rating"], (int, float)):
            review_id = URIRef(UNICA[f"Review_{sanitize_for_uri(idx)}"])
            if pd.notna(row["rating"]):
                g.add(
                    (
                        review_id,
                        SCHEMA.reviewRating,
                        Literal(row["rating"], datatype=XSD.float),
                    )
                )

            g.add((review_id, RDF.type, SCHEMA.UserReview))
            if pd.notna(row["text"]):
                g.add(
                    (
                        review_id,
                        SCHEMA.reviewBody,
                        Literal(row["text"], lang="en"),
                    )
                )

            if pd.notna(row["recipe_id"]):
                g.add(
                    (
                        review_id,
                        SCHEMA.itemReviewed,
                        URIRef(
                            UNICA[
                                f"Recipe_hummus{sanitize_for_uri(row['recipe_id'])}"
                            ]
                        ),
                    )
                )
            if pd.notna(row["member_id"]):
                g.add(
                    (
                        URIRef(
                            UNICA[
                                f"UserGroup_{sanitize_for_uri(row['member_id'])}"
                            ]
                        ),
                        SCHEMA.publishesReview,
                        review_id,
                    )
                )

    # Save the RDF graph in Turtle format
    g.serialize(destination=file_output, format="turtle")
    print(f"Generated file: {file_output}")


def convert_off_in_rdf() -> None:
    """
    Function to convert off data into RDF format
    """

    # Initialize the graph and namespaces
    g = Graph()

    # Define namespaces
    UNICA = Namespace("https://github.com/tail-unica/kgeats/")
    SCHEMA = Namespace("https://schema.org/")

    # Associate namespaces with the graph
    g.bind("unica", UNICA)
    g.bind("schema", SCHEMA)

    # Ontology versioning
    link_ontology = "https://github.com/tail-unica/kgeats/off"
    ontology_iri = URIRef(f"{link_ontology}")
    version_iri = URIRef(f"{link_ontology}/1.0")
    version_info = Literal("Version 1.0 - Initial release", lang="en")

    g.add((ontology_iri, RDF.type, OWL.Ontology))
    g.add((ontology_iri, OWL.versionIRI, version_iri))
    g.add((ontology_iri, OWL.versionInfo, version_info))

    # Reference to the previous version
    prior_version_iri = URIRef(f"{link_ontology}/0.0")
    g.add((ontology_iri, OWL.priorVersion, prior_version_iri))

    # File of input and output
    # off_file = "../csv_file/en.openfoodfacts.org.products.csv"
    off_file = "../csv_file/off_rows.csv"
    file_output = "../csv_file/ontology_off.ttl"

    # Upload the CSV
    df_off = pd.read_csv(off_file, sep="\t", on_bad_lines="skip")

    qualitatives_indicators = [
        "nutriscore_grade",
        "nova_group",
        "ecoscore_grade",
        "ecoscore_score",
    ]

    traces_and_allergies = {}

    # Iterate through each row to create Recipe, UserConstraint and Indicator entities
    for idx, row in df_off.iterrows():
        if pd.notna(row["product_name"]) and row["product_name"].strip() != "":
            # Create the recipe
            recipe_id = URIRef(
                UNICA[f"Recipe_off_{idx}"]
            )  # I put _off to differentiate them from the hummus ids
            g.add((recipe_id, RDF.type, SCHEMA.Recipe))
            g.add(
                (
                    recipe_id,
                    SCHEMA.name,
                    Literal(row["product_name"], lang="en"),
                )
            )
            g.add(
                (
                    recipe_id,
                    SCHEMA.identifier,
                    Literal(idx, datatype=XSD.integer),
                )
            )

            # UserConstraint
            if pd.notna(row["allergens"]):
                for allergen in row["allergens"].split(","):
                    allergen1 = allergen.replace("en:", "")
                    allergen = sanitize_for_uri(
                        allergen.replace("en:", "").replace("-", "_").lower()
                    )
                    allergen = URIRef(
                        UNICA[f"allergen_{sanitize_for_uri(allergen)}"]
                    )
                    if allergen not in traces_and_allergies:
                        traces_and_allergies[allergen] = 1
                        g.add((allergen, RDF.type, UNICA.UserConstraint))
                        g.add(
                            (
                                allergen,
                                SCHEMA.constraintName,
                                Literal(allergen1, lang="en"),
                            )
                        )
                        g.add(
                            (
                                allergen,
                                SCHEMA.constraintDescription,
                                Literal(
                                    f"is a user constraint about having an allergy to {allergen1}",
                                    lang="en",
                                ),
                            )
                        )

                    g.add((allergen, SCHEMA.suitableForDiet, recipe_id))

            if pd.notna(row["traces_en"]):
                for trace in row["traces_en"].split(","):
                    trace1 = trace
                    trace = sanitize_for_uri(trace.replace("-", "_").lower())
                    trace = URIRef(UNICA[f"trace_{sanitize_for_uri(trace)}"])
                    if trace not in traces_and_allergies:
                        traces_and_allergies[trace] = 1
                        g.add((trace, RDF.type, UNICA.UserConstraint))
                        g.add(
                            (
                                trace,
                                SCHEMA.constraintName,
                                Literal(trace1, lang="en"),
                            )
                        )
                        g.add(
                            (
                                trace,
                                SCHEMA.constraintDescription,
                                Literal(
                                    f"is a user constraint about having a trace of {trace1} in the product",
                                    lang="en",
                                ),
                            )
                        )

                    g.add((trace, SCHEMA.suitableForDiet, recipe_id))

            # Indicator
            for column in df_off.columns:
                if "100g" in column or column in qualitatives_indicators:
                    indicator_value = row[column]

                    if (
                        pd.notna(indicator_value)
                        and indicator_value != "unknown"
                    ):
                        # Create the indicator
                        column = column.replace("-", "_")
                        indicator_id = URIRef(
                            UNICA[
                                f"{sanitize_for_uri(re.sub('_100g', '', column))}_{idx}"
                            ]
                        )
                        g.add((indicator_id, RDF.type, UNICA.Indicator))
                        g.add((indicator_id, SCHEMA.type, Literal(column)))

                        if column not in qualitatives_indicators:
                            g.add(
                                (
                                    indicator_id,
                                    SCHEMA.unitText,
                                    Literal("grams"),
                                )
                            )
                            g.add(
                                (
                                    indicator_id,
                                    SCHEMA.quantity,
                                    Literal(
                                        indicator_value, datatype=XSD.float
                                    ),
                                )
                            )
                        else:
                            g.add(
                                (
                                    indicator_id,
                                    SCHEMA.quantity,
                                    Literal(
                                        indicator_value, datatype=XSD.string
                                    ),
                                )
                            )

                        # Add the relationship between the recipe and the indicator
                        g.add(
                            (
                                recipe_id,
                                SCHEMA.NutritionInformation,
                                indicator_id,
                            )
                        )

    # Relationship of alternative between recipes, is ingredient, has ingredient
    for idx1, row1 in df_off.iterrows():
        for idx2, row2 in df_off.iterrows():
            if (
                idx1 != idx2
                and pd.notna(row1["product_name"])
                and row1["product_name"].strip() != ""
                and pd.notna(row2["product_name"])
                and row2["product_name"].strip() != ""
            ):
                # Relationdhip Alternative Recipe
                if (
                    pd.notna(row1["generic_name"])
                    and pd.notna(row2["generic_name"])
                    and row1["generic_name"] == row2["generic_name"]
                ):
                    g.add(
                        (
                            URIRef(UNICA[f"recipe_{idx1}"]),
                            SCHEMA.isSimilarTo,
                            URIRef(UNICA[f"recipe_{idx2}"]),
                        )
                    )

                # Relationship of has ingredient / is ingredient
                if pd.notna(row2["ingredients_text"]) and row1[
                    "generic_name"
                ] in str(row2["ingredients_text"]).split(", "):
                    g.add(
                        (
                            URIRef(UNICA[f"recipe_{idx1}"]),
                            SCHEMA.isPartOf,
                            URIRef(UNICA[f"recipe_{idx2}"]),
                        )
                    )
                    g.add(
                        (
                            URIRef(UNICA[f"recipe_{idx2}"]),
                            SCHEMA.hasPart,
                            URIRef(UNICA[f"recipe_{idx1}"]),
                        )
                    )

    # Save the RDF graph in Turtle format
    g.serialize(destination=file_output, format="turtle")
    print(f"Generated file: {file_output}")
