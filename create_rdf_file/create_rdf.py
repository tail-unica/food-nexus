"""
File with various functions and data necessary for converting HUMMUS and
Open Food Facts data into an RDF format and for create our custom namespace.
"""

import re
import pandas as pd
from rdflib import RDF, RDFS, XSD, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL
from sentence_transformers import SentenceTransformer, util
import ast
import gc 
import os
import time


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


#def create_namespace(namespace_completo=True) -> None:
#    """
#    Function to create the TTL file with the custom namespace for UNICA
#    """
#
#    # Create the graph
#    g = Graph()
#
#    # Define the namespaces
#    SCHEMA = Namespace("https://schema.org/")
#    UNICA = Namespace(
#        "https://github.com/tail-unica/kgeats/"
#    )
#    XSD_NS = Namespace("http://www.w3.org/2001/XMLSchema#")
#    RDFS_NS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
#    RDF_NS = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
#
#    # Add prefixes to the graph
#    g.bind("schema", SCHEMA)
#    g.bind("unica", UNICA)
#    g.bind("xsd", XSD_NS)
#    g.bind("rdfs", RDFS_NS)
#    g.bind("rdf", RDF_NS)
#
#    if namespace_completo:
#        # Create classes
#        g.add((UNICA.FoodProducer, RDF.type, RDFS.Class))
#        g.add((UNICA.FoodProducer, RDFS.subClassOf, SCHEMA.Organization))
#        g.add(
#            (
#                UNICA.FoodProducer,
#                RDFS.label,
#                Literal("Food Producer", lang="en"),
#            )
#        )
#        g.add(
#            (
#                UNICA.FoodProducer,
#                RDFS.comment,
#                Literal(
#                    "Information about the producer of a food item.", lang="en"
#                ),
#            )
#        )
#
#    g.add((UNICA.Tag, RDF.type, RDFS.Class))
#    g.add((UNICA.Tag, RDFS.subClassOf, SCHEMA.Intangible))
#    g.add(
#        (
#            UNICA.Tag,
#            RDFS.label,
#            Literal("User Constraint", lang="en"),
#        )
#    )
#    g.add(
#        (
#            UNICA.Tag,
#            RDFS.comment,
#            Literal(
#                "Constraint about what a user can or want to eat.", lang="en"
#            ),
#        )
#    )
#
#    # Propriety for Tag
#    g.add((UNICA.constraintName, RDF.type, RDF.Property))
#    g.add((UNICA.constraintName, RDFS.domain, UNICA.Tag))
#    g.add((UNICA.constraintName, RDFS.range, XSD.string))
#    g.add(
#        (
#            UNICA.constraintName,
#            RDFS.label,
#            Literal("Constraint Name", lang="en"),
#        )
#    )
#    g.add(
#        (
#            UNICA.constraintName,
#            RDFS.comment,
#            Literal("Name of the constraint.", lang="en"),
#        )
#    )
#
#    g.add((UNICA.description, RDF.type, RDF.Property))
#    g.add((UNICA.description, RDFS.domain, UNICA.Tag))
#    g.add((UNICA.description, RDFS.range, XSD.string))
#    g.add(
#        (
#            UNICA.description,
#            RDFS.label,
#            Literal("Constraint Description", lang="en"),
#        )
#    )
#    g.add(
#        (
#            UNICA.description,
#            RDFS.comment,
#            Literal("Description of the constraint.", lang="en"),
#        )
#    )
#
#    # Create the Indicator class
#    g.add((UNICA.Indicator, RDF.type, RDFS.Class))
#    g.add((UNICA.Indicator, RDFS.subClassOf, SCHEMA.NutritionInformation))
#    g.add((UNICA.Indicator, RDFS.label, Literal("Indicator", lang="en")))
#    g.add(
#        (
#            UNICA.Indicator,
#            RDFS.comment,
#            Literal(
#                "Represents nutritional and sustainability indicators for food items.",
#                lang="en",
#            ),
#        )
#    )
#
#    if namespace_completo:
#        # Indicator property
#        proprieta_indicator = [
#            (
#                "calcium",
#                "Calcium for 100g",
#                "Calcium content for 100 grams.",
#                XSD.float,
#            ),
#            (
#                "iron",
#                "Iron for 100g",
#                "Iron content per 100 grams.",
#                XSD.float,
#            ),
#            (
#                "vitaminC",
#                "Vitamin C for 100g",
#                "Vitamin C content per 100 grams.",
#                XSD.float,
#            ),
#            (
#                "vitaminA",
#                "Vitamin A for 100g",
#                "Vitamin A content per 100 grams.",
#                XSD.float,
#            ),
#            (
#                "whoScore",
#                "WHO Score",
#                "A score indicating WHO healthfulness assessment.",
#                XSD.float,
#            ),
#            (
#                "fsaScore",
#                "FSA Score",
#                "A score based on the Food Standards Agency's healthfulness criteria.",
#                XSD.float,
#            ),
#            (
#                "nutriScore",
#                "Nutri-Score",
#                "A score indicating the nutritional value based on the Nutri-Score system.",
#                XSD.float,
#            ),
#            (
#                "nutriscoreScore",
#                "Nutri-Score Score",
#                "A numeric score for Nutri-Score system.",
#                XSD.float,
#            ),
#            (
#                "ecoscoreScore",
#                "Eco-Score Score",
#                "A numeric score for Eco-Score system.",
#                XSD.float,
#            ),
#            (
#                "nutritionScoreFr",
#                "Nutrition Score FR for 100g",
#                "French nutrition score per 100 grams.",
#                XSD.float,
#            ),
#            (
#                "nutriscoreGrade",
#                "Nutri-Score Grade",
#                "A grade indicating the Nutri-Score nutritional level.",
#                XSD.string,
#            ),
#            (
#                "ecoscoreGrade",
#                "Eco-Score Grade",
#                "A grade indicating the Eco-Score environmental impact level.",
#                XSD.string,
#            ),
#            (
#                "novaGroup",
#                "NOVA Group",
#                "Group classification of processing level based on NOVA system.",
#                XSD.string,
#            ),
#        ]
#    else:
#        # Indicator Property
#        proprieta_indicator = [
#            (
#                "whoScore",
#                "WHO Score",
#                "A score indicating WHO healthfulness assessment.",
#                XSD.float,
#            ),
#            (
#                "fsaScore",
#                "FSA Score",
#                "A score based on the Food Standards Agency's healthfulness criteria.",
#                XSD.float,
#            ),
#            (
#                "nutriScore",
#                "Nutri-Score",
#                "A score indicating the nutritional value based on the Nutri-Score system.",
#                XSD.float,
#            ),
#        ]
#
#    for prop, label, comment, range_type in proprieta_indicator:
#        g.add((UNICA[prop], RDF.type, RDF.Property))
#        g.add((UNICA[prop], RDFS.domain, UNICA.Indicator))
#        g.add((UNICA[prop], RDFS.range, range_type))
#        g.add((UNICA[prop], RDFS.label, Literal(label, lang="en")))
#        g.add((UNICA[prop], RDFS.comment, Literal(comment, lang="en")))
#
#    # Save in Turtle format with the name 'namespace_unica.ttl'
#    with open("../csv_file/namespace_unica.ttl", "w") as f:
#        f.write(g.serialize(format="turtle"))
#
#    print("namespace created successfully")


def convert_hummus_in_rdf(
    use_infered_attributes_description=False,
    use_infered_attributes_review=False,
    use_row=False,
) -> None:
    """
    Function to convert hummus data into RDF format
    """

    model = SentenceTransformer('all-MiniLM-L6-v2')

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

    print("added the versioning info")

    # input file
    if use_row:
        file_recipes = "../csv_file/pp_recipes_rows.csv"
        file_users = "../csv_file/pp_members_rows.csv"
        file_review = "../csv_file/pp_reviews_rows.csv"
    else:
        file_recipes = "../csv_file/pp_recipes_normalized_by_pipeline.csv"
        file_users = "../csv_file/pp_members_with_attributes.csv"
        file_review = "../csv_file/pp_reviews_with_attributes.csv"

    if use_infered_attributes_review or use_infered_attributes_description:
        file_output_ttl = "../csv_file/ontology_hummus_infered.ttl"
        file_output_nt = "../csv_file/ontology_hummus_infered.nt"
    else:
        file_output_ttl = "../csv_file/ontology_hummus.ttl"
        file_output_nt = "../csv_file/ontology_hummus.nt"

    # Upload the CSV
    df_ricette = pd.read_csv(filepath_or_buffer=file_recipes, on_bad_lines="skip", sep=";",  low_memory=False)
    df_review = pd.read_csv(file_review, on_bad_lines="skip",  sep=",",  low_memory=False)
    df_utenti = pd.read_csv(file_users, on_bad_lines="skip", sep=",", low_memory=False)

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

    qualitatives_indicators_hummus = {
        "who_score": "whoScore",
        "fsa_score": "fsaScore",
        "nutri_score": "nutriScore",
    }

    print("starting user creation")

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
            if use_infered_attributes_description:
                if pd.notna(row["age"]):
                    if isinstance(row["age"], int):
                        g.add(
                            (
                                group_id,
                                SCHEMA.birthDate,
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
                            Literal(row["weight"], datatype=XSD.string),
                        )
                    )
                if pd.notna(row["height"]):
                    g.add(
                        (
                            group_id,
                            SCHEMA.height,
                            Literal(row["height"], datatype=XSD.string),
                        )
                    )
                if pd.notna(row["user_constraints"]):
                    contraint_list = row["user_constraints"].split(";")
                    for contraint in contraint_list:
                        #constraint_name, constraint_value = contraint.split(":")
                        try: 
                            constraint_name, constraint_value = contraint.split(":", 1)
                        except ValueError:
                            #print(f"Bad format: {contraint}, idx: {idx}")
                            continue

                        if constraint_name == "physical activity category":
                            g.add(
                                (
                                    group_id,
                                    SCHEMA.PhysicalActivityCategory,
                                    Literal(constraint_value, lang="en"),
                                )
                            )
                        else:
                            tag_id = URIRef(
                                UNICA[
                                    f"Tag_{sanitize_for_uri(constraint_name).strip()}_{sanitize_for_uri(constraint_value).strip()}"
                                ]
                            )
                            if tag_id not in tag_count:
                                tag_count[tag_id] = 1
                                g.add((tag_id, RDF.type, UNICA.Tag))
                                g.add(
                                    (
                                        tag_id,
                                        SCHEMA.name,
                                        Literal(constraint_name, lang="en"),
                                    )
                                )
                                g.add(
                                    (
                                        tag_id,
                                        SCHEMA.description,
                                        Literal(
                                            f"is a tag about {constraint_name}",
                                            lang="en",
                                        ),
                                    )
                                )
                            g.add((group_id, UNICA.hasConstraint, tag_id))


    print("starting recipe creation")

    # Create the entity Recipe
    conta = 0
    for idx, row in df_ricette.iterrows():
        conta +=1
        if conta % 10000 == 0:
            print(f"row: {conta}")


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
                g.add((author_id, UNICA.publishesRecipe, recipe_id))

            # Tag
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
                    if tag_id != "Tag_":
                        if tag not in tag_count:
                            tag_count[tag_id] = 1
                            g.add((tag_id, RDF.type, UNICA.Tag))
                            g.add(
                                (
                                    tag_id,
                                    SCHEMA.name,
                                    Literal(tag, lang="en"),
                                )
                            )
                            g.add(
                                (
                                    tag_id,
                                    SCHEMA.description,
                                    Literal(
                                        f"is a tag about {tag1.replace('-', ' ')}",
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
                if csv_column and pd.notna(row[csv_column]): #type: ignore
                    if col in qualitatives_indicators_hummus.keys():
                        col = qualitatives_indicators_hummus[col]
                    indicator_id = URIRef(
                        UNICA[
                            f"Indicator_{sanitize_for_uri(value=col.replace(' ', '_').lower())}_{sanitize_for_uri(row['recipe_id'])}"
                        ]
                    )
                    g.add((indicator_id, RDF.type, UNICA.Indicator))
                    g.add((indicator_id, SCHEMA.type, Literal(lexical_or_value=col)))
                    if unit:
                        g.add((indicator_id, SCHEMA.unitText, Literal(unit)))

                    stringa = str(row[csv_column])
                    if (
                        col in qualitatives_indicators_hummus.values()
                        and row["servingSize [g]"] != 0
                        and row["servingSize [g]"] != ""
                    ):
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
                        (recipe_id, UNICA.hasIndicator, indicator_id)
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

                    if pd.notna(row["ingredients"]):
                        list1 = [ing]
                        try:
                            list2 = ast.literal_eval(row["ingredients"])
                            if not isinstance(list2, list):
                                list2 = []
                        except (ValueError, SyntaxError):
                            list2 = []

                        list2_strings = [item[0] for item in list2 if isinstance(item, list) and len(item) > 0]

                        if not list2_strings:
                            continue  

                        for l1 in list1:
                            match = next((l2 for l2 in list2 if l1 in l2[0]), None)
                            if match:
                                # Direct Match
                                measure_value = match[1].strip()
                                measure, unit = measure_value.split('time(s)')
                                measure = measure.strip()
                                unit = unit.strip()
                            else:
                                # Similarity Match with Sentence Transformer
                                embeddings1 = model.encode(l1, convert_to_tensor=True).to("cpu")
                                embeddings2 = model.encode(list2_strings, convert_to_tensor=True).to("cpu")
                                
                                if embeddings2.shape[0] == 0:
                                    continue
                                
                                cosine_scores = util.cos_sim(embeddings1, embeddings2)
                                best_match_idx = cosine_scores.argmax().item()
                                best_match = list2[best_match_idx] #type: ignore

                                measure_value = best_match[1].strip()
                                measure, unit = measure_value.split('time(s)')
                                measure = measure.strip()
                                unit = unit.strip()

                            if unit != "":
                                g.add(
                                    (ingredient_id_for_recipe,
                                    SCHEMA.quantity,
                                    Literal(unit, datatype=XSD.string))
                                )
                            if measure != "":
                                g.add(
                                    (ingredient_id_for_recipe,
                                    SCHEMA.unitText,
                                    Literal(measure, datatype=XSD.string))
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

    print("starting review creation")

    conta = 0
    # Create UserReview entities and relationships
    for idx, row in df_review.iterrows():
        conta += 1
        if conta % 10000 == 0:
            print(f"row: {conta}")
    
        if pd.notna(row["rating"]) and isinstance(row["rating"], (int, float)):
            review_id = URIRef(UNICA[f"Review_{sanitize_for_uri(idx)}"])
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
                group_id = URIRef(
                    UNICA[f"UserGroup_{sanitize_for_uri(row['member_id'])}"]
                )
                g.add(
                    (
                        group_id,
                        UNICA.publishesReview,
                        review_id,
                    )
                )

                if use_infered_attributes_review:
                    if pd.notna(row["age"]):
                        if isinstance(row["age"], int):
                            if not g.value(
                                subject=group_id, predicate=SCHEMA.birthDate
                            ):
                                g.add(
                                    (
                                        group_id,
                                        SCHEMA.birthDate,
                                        Literal(
                                            row["age"], datatype=XSD.integer
                                        ),
                                    )
                                )
                        else:
                            if not g.value(
                                subject=group_id,
                                predicate=SCHEMA.typicalAgeRange,
                            ) and not g.value(
                                subject=group_id, predicate=SCHEMA.birthDate
                            ):
                                g.add(
                                    (
                                        group_id,
                                        SCHEMA.typicalAgeRange,
                                        Literal(
                                            row["age"], datatype=XSD.string
                                        ),
                                    )
                                )
                    if pd.notna(row["gender"]):
                        if not g.value(
                            subject=group_id, predicate=SCHEMA.gender
                        ):
                            g.add(
                                (
                                    group_id,
                                    SCHEMA.gender,
                                    Literal(row["gender"], datatype=XSD.string),
                                )
                            )
                    if pd.notna(row["weight"]):
                        if not g.value(
                            subject=group_id, predicate=SCHEMA.weight
                        ):
                            g.add(
                                (
                                    group_id,
                                    SCHEMA.weight,
                                    Literal(row["weight"], datatype=XSD.string),
                                )
                            )
                    if pd.notna(row["height"]):
                        if not g.value(
                            subject=group_id, predicate=SCHEMA.height
                        ):
                            g.add(
                                (
                                    group_id,
                                    SCHEMA.height,
                                    Literal(row["height"], datatype=XSD.string),
                                )
                            )

                    if pd.notna(row["user_constraints"]):
                        contraint_list = row["user_constraints"].split(";")
                        for contraint in contraint_list:
                            try: 
                                constraint_name, constraint_value = contraint.split(":", 1)
                            except ValueError:
                                print(f"Bad format: {contraint}")
                                continue
                            
                            tag_id = URIRef(
                                UNICA[
                                    f"Tag_{sanitize_for_uri(constraint_name).strip()}_{sanitize_for_uri(constraint_value).strip()}"
                                ]
                            )
                            if tag_id not in tag_count:
                                tag_count[tag_id] = 1
                                g.add((tag_id, RDF.type, UNICA.Tag))
                                g.add(
                                    (
                                        tag_id,
                                        SCHEMA.name,
                                        Literal(constraint_name, lang="en"),
                                    )
                                )
                                g.add(
                                    (
                                        tag_id,
                                        SCHEMA.description,
                                        Literal(
                                            f"is a tag about {constraint_name}",
                                            lang="en",
                                        ),
                                    )
                                )
                            g.add((group_id, UNICA.hasConstraint, tag_id))




    # Add tag similarity
    forbidden_words = ['no-', 'non-', 'free-', 'high', 'low', 'avoid', 'free', 'less', 'nothing', 'without']
    model = SentenceTransformer('BAAI/bge-en-icl')
    tag_list = list(tag_count.keys())


    def preprocess_tag(tag_uriref): 
        tag_full_uri_string = str(tag_uriref)

        prefix_to_remove = str(UNICA["Tag_"])
        if tag_full_uri_string.startswith(prefix_to_remove):
            processed = tag_full_uri_string[len(prefix_to_remove):]
        else:

            local_name = tag_full_uri_string.split('/')[-1]
            if local_name.startswith("Tag_"):
                 processed = local_name[len("Tag_"):]
            else:
                 processed = local_name

        processed = processed.replace("_", " ")
        return processed.strip()


    original_tags_valid = []
    processed_tags_for_embedding = []

    for original_tag in tag_list:
        processed = preprocess_tag(original_tag)
        if processed: 
            original_tags_valid.append(original_tag)
            processed_tags_for_embedding.append(processed)

    if processed_tags_for_embedding:

        embeddings = model.encode(processed_tags_for_embedding, convert_to_tensor=True, show_progress_bar=True)
        cosine_scores = util.cos_sim(embeddings, embeddings)
        num_tags = len(original_tags_valid)

        for i in range(num_tags):
            for j in range(i + 1, num_tags):

                original_tag1 = original_tags_valid[i]
                original_tag2 = original_tags_valid[j]
                
                processed_tag1_text = processed_tags_for_embedding[i]
                processed_tag2_text = processed_tags_for_embedding[j]
                
                similarity = cosine_scores[i][j].item() 

                if similarity > 0.85:

                    tag1_text_lower = processed_tag1_text.lower()
                    tag2_text_lower = processed_tag2_text.lower()

                    negative_keywords = {'no-', 'non-', 'not', 'free-', 'free', 'without', 'less', 'low', 'reduced', 'zero', 'nothing', 'anti-', 'avoid', 'exempt', 'un-', 'de-'}
                    positive_keywords = {'high', 'rich', 'extra', 'added', 'with', 'contains', 'plus', 'pro-', 'more', 'full', 'enriched'}

                    tag1_has_negative = any(kw in tag1_text_lower for kw in negative_keywords)
                    tag1_has_positive = any(kw in tag1_text_lower for kw in positive_keywords)

                    tag2_has_negative = any(kw in tag2_text_lower for kw in negative_keywords)
                    tag2_has_positive = any(kw in tag2_text_lower for kw in positive_keywords)

                    associate = True

                    if (tag1_has_negative and tag2_has_positive) or \
                    (tag1_has_positive and tag2_has_negative):
                        associate = False
                
                    if associate:
                        g.add((original_tag1, SCHEMA.sameAs, original_tag2))




    #Sustainability information
    #df_sustainability = pd.read_csv(filepath_or_buffer=file_recipes, on_bad_lines="skip", low_memory=False)
    #print("starting sustainability creation")
#
    #conta = 0
    ## Create UserReview entities and relationships
    #for idx, row in df_sustainability.iterrows():
    #    conta += 1
    #    if conta % 1000 == 0:
    #        print(f"row: {conta}")
    #    if pd.notna(row["recipe_id"]) and isinstance(row["recipe_id"], (int, float)):
    #        
    #        for colonna in ["recipe_CF_kg", "recipe_WF_kg"]:
    #            tipo = colonna.replace('recipe_', '')
    #            indicator_id = URIRef(
    #                    UNICA[
    #                        f"Indicator_{sanitize_for_uri(value=tipo.lower())}_{sanitize_for_uri(row['recipe_id'])}"
    #                    ]
    #                )   
#
    #            g.add((indicator_id, RDF.type, UNICA.Indicator))
    #            g.add((indicator_id, SCHEMA.type, Literal(lexical_or_value=tipo)))
    #            g.add((indicator_id, SCHEMA.unitText, Literal(lexical_or_value="kg")))
    #            g.add(
    #            (
    #                indicator_id,
    #                SCHEMA.quantity,
    #                Literal(row[colonna], datatype=XSD.float),
    #            )
    #            )
    #            g.add(
    #                (row["recipe_id"], SCHEMA.NutritionInformation, indicator_id)
    #            )


    # Save the RDF graph in Turtle format
    g.serialize(destination=file_output_ttl, format="turtle")
    print(f"Generated file: {file_output_ttl}")

    g.serialize(destination=file_output_nt, format="nt", encoding="utf-8")
    print(f"Generated file: {file_output_nt}")


def convert_off_in_rdf(use_row=False) -> None:
    """
    Function to convert off data into RDF format
    """

    # File paths
    if use_row:
        off_file = "../csv_file/off_rows.csv"
    else:
        off_file = "../csv_file/off_normalized_final.csv"
    
    # Create output directory if it doesn't exist
    output_dir = "../csv_file/chunks/"
    os.makedirs(output_dir, exist_ok=True)
    
    final_output_ttl = "../csv_file/ontology_off.ttl"
    final_output_nt = "../csv_file/ontology_off.nt"
    
    qualitatives_indicators: dict[str, str] = {
        "nutriscore_grade": "nutriscoreGrade",
        "nova_group": "novaGroup",
        "ecoscore_grade": "ecoscoreGrade",
        "nutrition-score-fr_100g": "nutritionScoreFr",
        "nutriscore_score": "nutriscoreScore",
        "ecoscore_score": "ecoscoreScore",
    }

    off_indicators = {
        "calcium_100g": "calcium",
        "iron_100g": "iron",
        "vitamin-c_100g": "vitaminC",
        "vitamin-a_100g": "vitaminA",
        "nutriscore_score": "nutriscoreScore",
        "nutrition-score-fr_100g": "nutritionScoreFr",
        "nutriscore_grade": "nutriscoreGrade",
        "nova_group": "novaGroup",
        "ecoscore_grade": "ecoscoreGrade",
        "ecoscore_score": "ecoscoreScore",
        "energy_100g": "calories",
        "energy-from-fat_100g": "caloriesFromFat",
        "fat_100g": "totalFat",
        "saturated-fat_100g": "saturatedFat",
        "cholesterol_100g": "cholesterol",
        "sodium_100g": "sodium",
        "carbohydrates_100g": "totalCarbohydrate",
        "fiber_100g": "dietaryFiber",
        "sugars_100g": "sugars",
        "proteins_100g": "protein",
        #less important indicator
        "caprylic-acid_100g": "caprylicAcid",
        "capric-acid_100g": "capricAcid",
        "lauric-acid_100g": "lauricAcid",
        "myristic-acid_100g": "myristicAcid",
        "palmitic-acid_100g": "palmiticAcid",
        "stearic-acid_100g": "stearicAcid",
        "arachidic-acid_100g": "arachidicAcid",
        "behenic-acid_100g": "behenicAcid",
        "lignoceric-acid_100g": "lignocericAcid",
        "cerotic-acid_100g": "ceroticAcid",
        "montanic-acid_100g": "montanicAcid",
        "melissic-acid_100g": "melissicAcid",
        "unsaturated-fat_100g": "unsaturatedFat",
        "monounsaturated-fat_100g": "monounsaturatedFat",
        "omega-9-fat_100g": "omega9Fat",
        "polyunsaturated-fat_100g": "polyunsaturatedFat",
        "omega-3-fat_100g": "omega3Fat",
        "omega-6-fat_100g": "omega6Fat",
        "alpha-linolenic-acid_100g": "alphaLinolenicAcid",
        "eicosapentaenoic-acid_100g": "eicosapentaenoicAcid",
        "docosahexaenoic-acid_100g": "docosahexaenoicAcid",
        "linoleic-acid_100g": "linoleicAcid",
        "arachidonic-acid_100g": "arachidonicAcid",
        "gamma-linolenic-acid_100g": "gammaLinolenicAcid",
        "dihomo-gamma-linolenic-acid_100g": "dihomoGammaLinolenicAcid",
        "oleic-acid_100g": "oleicAcid",
        "elaidic-acid_100g": "elaidicAcid",
        "gondoic-acid_100g": "gondoicAcid",
        "mead-acid_100g": "meadAcid",
        "erucic-acid_100g": "erucicAcid",
        "nervonic-acid_100g": "nervonicAcid",
        "trans-fat_100g": "transFat",
        "added-sugars_100g": "addedSugars",
        "sucrose_100g": "sucrose",
        "glucose_100g": "glucose",
        "fructose_100g": "fructose",
        "lactose_100g": "lactose",
        "maltose_100g": "maltose",
        "maltodextrins_100g": "maltodextrins",
        "starch_100g": "starch",
        "polyols_100g": "polyols",
        "erythritol_100g": "erythritol",
        "soluble-fiber_100g": "solubleFiber",
        "insoluble-fiber_100g": "insolubleFiber",
        "casein_100g": "casein",
        "serum-proteins_100g": "serumProteins",
        "nucleotides_100g": "nucleotides",
        "salt_100g": "salt",
        "added-salt_100g": "addedSalt",
        "alcohol_100g": "alcohol",
        "beta-carotene_100g": "betaCarotene",
        "vitamin-d_100g": "vitaminD",
        "vitamin-e_100g": "vitaminE",
        "vitamin-k_100g": "vitaminK",
        "vitamin-b1_100g": "vitaminB1",
        "vitamin-b2_100g": "vitaminB2",
        "vitamin-pp_100g": "vitaminPP",
        "vitamin-b6_100g": "vitaminB6",
        "vitamin-b9_100g": "vitaminB9",
        "folates_100g": "folates",
        "vitamin-b12_100g": "vitaminB12",
        "biotin_100g": "biotin",
        "pantothenic-acid_100g": "pantothenicAcid",
        "silica_100g": "silica",
        "bicarbonate_100g": "bicarbonate",
        "potassium_100g": "potassium",
        "chloride_100g": "chloride",
        "phosphorus_100g": "phosphorus",
        "magnesium_100g": "magnesium",
        "zinc_100g": "zinc",
        "copper_100g": "copper",
        "manganese_100g": "manganese",
        "fluoride_100g": "fluoride",
        "selenium_100g": "selenium",
        "chromium_100g": "chromium",
        "molybdenum_100g": "molybdenum",
        "iodine_100g": "iodine",
        "caffeine_100g": "caffeine",
        "taurine_100g": "taurine",
        "ph_100g": "ph",
        "fruits-vegetables-nuts_100g": "fruitsVegetablesNuts",
        "fruits-vegetables-nuts-dried_100g": "fruitsVegetablesNutsDried",
        "fruits-vegetables-nuts-estimate_100g": "fruitsVegetablesNutsEstimate",
        "fruits-vegetables-nuts-estimate-from-ingredients_100g": "fruitsVegetablesNutsEstimateFromIngredients",
        "collagen-meat-protein-ratio_100g": "collagenMeatProteinRatio",
        "cocoa_100g": "cocoa",
        "chlorophyl_100g": "chlorophyl",
        "carbon-footprint_100g": "carbonFootprint",
        "carbon-footprint-from-meat-or-fish_100g": "carbonFootprintFromMeatOrFish",
        "nutrition-score-uk_100g": "nutritionScoreUk",
        "glycemic-index_100g": "glycemicIndex",
        "water-hardness_100g": "waterHardness",
        "choline_100g": "choline",
        "phylloquinone_100g": "phylloquinone",
        "energy-kj_100g": "energyKj",
        "energy-kcal_100g": "energyKcal",
        "acidity_100g": "acidity",
        "nitrate_100g": "nitrate",
        "sulphate_100g": "sulphate",
        "carnitine_100g": "carnitine",
        "inositol_100g": "inositol",
        "beta-glucan_100g": "betaGlucan",
        "butyric-acid_100g": "butyricAcid",
        "caproic-acid_100g": "caproicAcid"
    }
    

    city_columns: list[str] = ["cities_tags"]#, "cities"]

    tags_columns = [
        "categories", "categories_en", "categories_tags",
        "main_category_en", "nutrient_levels_tags", 
        "food_groups_en", "food_groups_tags", "ingredients_analysis_tags", "ingredients_tags", #ingredient a parte?
        "labels_en", "labels_tags", "labels", "allergens", "traces_en", "packaging_en", "serving_size", "serving_quantity",
        "additives_n", "quantity", "product_quantity", "pnns_groups_2", "pnns_groups_1", "additives_en", "generic_name"
        ]
    
    countries_column = ["purchase_places", "origins_tags", "origins", "origins_en", "countries", "countries_tags", "countries_en"]

    foodproducer_columns =["brands", "brands_tags", "brand_owner"]


    # Initialize a set to track already processed allergens and traces
    processed_constraints = set()
    processed_food_producer = set()
    processed_store = set()
    processed_cities = set()
    processed_countries = set()
    
    # Define namespaces (will be used in each chunk)
    UNICA = Namespace("https://github.com/tail-unica/kgeats/")
    SCHEMA = Namespace("https://schema.org/")
    
    # Process in smaller chunks for better memory management
    chunksize = 50000  #10000 if the code run out of memory
    chunk_files_ttl = []
    chunk_files_nt = []

    # First, create a graph with just the ontology information
    ontology_graph = Graph()
    ontology_graph.bind("unica", UNICA)
    ontology_graph.bind("schema", SCHEMA)
    
    # Ontology versioning
    link_ontology = "https://github.com/tail-unica/kgeats/off"
    ontology_iri = URIRef(f"{link_ontology}")
    version_iri = URIRef(f"{link_ontology}/1.0")
    version_info = Literal("Version 1.0 - Initial release", lang="en")

    ontology_graph.add((ontology_iri, RDF.type, OWL.Ontology))
    ontology_graph.add((ontology_iri, OWL.versionIRI, version_iri))
    ontology_graph.add((ontology_iri, OWL.versionInfo, version_info))

    # Reference to the previous version
    prior_version_iri = URIRef(f"{link_ontology}/0.0")
    ontology_graph.add((ontology_iri, OWL.priorVersion, prior_version_iri))
    
    # Save the ontology information
    ontology_file = f"{output_dir}ontology_header.ttl"
    ontology_graph.serialize(destination=ontology_file, format="turtle")
    chunk_files_ttl.append(ontology_file)
    
        # Save the ontology information
    ontology_file = f"{output_dir}ontology_header.nt"
    ontology_graph.serialize(destination=ontology_file, format="nt", encoding="utf-8")
    chunk_files_nt.append(ontology_file)

    # Process the CSV in chunks
    cont_chunk = 0
    for df_off_chunk in pd.read_csv(off_file, sep="\t", on_bad_lines="skip", chunksize=chunksize, low_memory=False):
        print(f"Processing rows from {chunksize * cont_chunk} to {chunksize * (cont_chunk+1)}")
        
        # Create a new graph for each chunk
        chunk_graph = Graph()
        chunk_graph.bind("unica", UNICA)
        chunk_graph.bind("schema", SCHEMA)
        
        # Track constraints processed in this chunk
        chunk_constraints = set()
        
        for idx, row in df_off_chunk.iterrows():
            if pd.notna(row["product_name"]) and row["product_name"].strip() != "":
                # Create the recipe
                recipe_id = URIRef(UNICA[f"Recipe_off_{row["code"]}"]) # I put _off to differentiate them from the hummus ids
                chunk_graph.add((recipe_id, RDF.type, SCHEMA.Product))
                chunk_graph.add((recipe_id, SCHEMA.name, Literal(row["product_name"], lang="en")))
                chunk_graph.add((recipe_id, SCHEMA.identifier, Literal(idx, datatype=XSD.integer)))
                chunk_graph.add((recipe_id, SCHEMA.image, Literal(row["image_small_url"])))

                recipe_tags = set()

                #Add entity Tag
                for contraint in tags_columns:
                    if pd.notna(row[contraint]):
                        for tag in str(row[contraint]).split(","):
                            tag = tag.split(":", 1)[1] if ":" in tag else tag
                            tag1 = tag
                            tag_uri = sanitize_for_uri(tag.replace("en:", "").replace("fr:", "").replace("-", "_").lower())
                            tag_ref = URIRef(UNICA[f"Contraint_{contraint}_{tag_uri}"])
                            recipe_tags.add(tag_ref)

                            # Check if we've already processed this allergen
                            if tag_ref not in processed_constraints:
                                processed_constraints.add(tag_ref)
                                chunk_constraints.add(tag_ref)
                                
                                chunk_graph.add((tag_ref, RDF.type, UNICA.Tag))
                                chunk_graph.add((tag_ref, SCHEMA.name, Literal(tag1, lang="en")))
                                chunk_graph.add((
                                    tag_ref, 
                                    SCHEMA.description, 
                                    Literal(f"is a tag about {tag1}", lang="en")
                                ))
                            
                for tags in recipe_tags:
                    chunk_graph.add(triple=(tags, SCHEMA.suitableForDiet, recipe_id))

                # Process indicators
                for column in off_indicators.keys():
                    if column in df_off_chunk.columns:
                        indicator_value = row[column]
                        if pd.notna(indicator_value) and indicator_value != "unknown":
                            column_name = off_indicators[column]
                            indicator_id = URIRef(UNICA[f"{column_name}_{idx}"])
                            
                            chunk_graph.add((indicator_id, RDF.type, UNICA.Indicator))
                            chunk_graph.add((indicator_id, SCHEMA.type, Literal(column_name)))

                            is_qualitative = column_name in qualitatives_indicators.values()
                            
                            is_numeric = False
                            if not is_qualitative:
                                try:
                                    float(indicator_value)
                                    is_numeric = True
                                except (ValueError, TypeError):
                                    is_numeric = False
                            
                            if is_numeric:
                                chunk_graph.add((indicator_id, SCHEMA.unitText, Literal("grams")))
                                chunk_graph.add((
                                    indicator_id, 
                                    SCHEMA.quantity, 
                                    Literal(float(indicator_value), datatype=XSD.float)
                                ))
                            else:
                                chunk_graph.add((
                                    indicator_id, 
                                    SCHEMA.quantity, 
                                    Literal(indicator_value, datatype=XSD.string)
                                ))

                            # Add the relationship between the recipe and the indicator
                            chunk_graph.add((recipe_id, UNICA.hasIndicator, indicator_id))
            


                

                # Add foodproducer entity
                recipe_foodproducer = set()
                for column_foodproducer in foodproducer_columns:
                    if pd.notna(row[column_foodproducer]):
                        for food_producer in row[column_foodproducer].split(","):
                            food_producer1 = food_producer
                            food_producer_uri = sanitize_for_uri(food_producer.replace("-", "_").lower())
                            food_producer_ref = URIRef(UNICA[f"food_producer_{food_producer_uri}"])
                            recipe_foodproducer.add(food_producer_ref)
                            
                            if food_producer_ref not in processed_food_producer:
                                processed_constraints.add(food_producer_ref)
                                chunk_constraints.add(food_producer_ref)
                            
                                chunk_graph.add((food_producer_ref, RDF.type, UNICA.FoodProducer))
                                chunk_graph.add((food_producer_ref, SCHEMA.name, Literal(food_producer1, lang="en")))
                
                for foodproducerentity in recipe_foodproducer:
                    chunk_graph.add((foodproducerentity, SCHEMA.produces, recipe_id))




                #Add country entity (Produced)
                recipe_countries = set()

                for column_countries in ["manufacturing_places_tags", "manufacturing_places"]:
                    if pd.notna(row[column_countries]):
                        for countries in row[column_countries].split(","):
                            countries1 = countries
                            countries_uri = sanitize_for_uri(countries.replace("-", "_").lower())
                            countries_ref = URIRef(UNICA[f"food_producer_{countries_uri}"])
                            recipe_countries.add(countries_ref)

                            if countries_ref not in processed_countries:
                                processed_constraints.add(countries_ref)
                                chunk_constraints.add(countries_ref)
                            
                                chunk_graph.add((countries_ref, RDF.type, SCHEMA.Country))
                                chunk_graph.add((countries_ref, SCHEMA.name, Literal(countries1, lang="en")))
                            
                for country in recipe_countries:
                    chunk_graph.add(triple=(recipe_id, SCHEMA.countryOfAssembly, country))


                #Add country entity (Located)
                recipe_countries = set()

                for column_countries in countries_column:
                    if pd.notna(row[column_countries]):
                        for countries in row[column_countries].split(","):
                            countries = countries.split(":", 1)[1] if ":" in countries else countries
                            countries1 = countries
                            countries_uri = sanitize_for_uri(countries.replace("-", "_").lower())
                            countries_ref = URIRef(UNICA[f"food_producer_{countries_uri}"])
                            recipe_countries.add(countries_ref)

                            if countries_ref not in processed_countries:
                                processed_constraints.add(countries_ref)
                                chunk_constraints.add(countries_ref)
                            
                                chunk_graph.add((countries_ref, RDF.type, SCHEMA.Country))
                                chunk_graph.add((countries_ref, SCHEMA.name, Literal(countries1, lang="en")))


                #Add city entity
                recipe_cities = set()

                for column_city in city_columns:
                    if pd.notna(row[column_city]):
                        for city in row[column_city].split(","):
                            city1 = city
                            city_uri = sanitize_for_uri(city.replace("-", "_").lower())
                            city_ref = URIRef(UNICA[f"food_producer_{city_uri}"])
                            recipe_cities.add(city_ref)

                            if city_ref not in processed_cities:
                                processed_cities.add(city_ref)
                                chunk_constraints.add(city_ref)
                            
                                chunk_graph.add((city_ref, RDF.type, SCHEMA.City))
                                chunk_graph.add((city_ref, SCHEMA.name, Literal(city1, lang="en")))

                            for country in recipe_countries:
                                chunk_graph.add(triple=(city_ref, UNICA.isPlaceIn, country))


                #Add store entity
                if pd.notna(row["stores"]):
                    for stores in row["stores"].split(","):
                        stores1 = stores
                        stores_uri = sanitize_for_uri(stores.replace("-", "_").lower())
                        stores_ref = URIRef(UNICA[f"food_producer_{stores_uri}"])

                        if stores_ref not in processed_store:
                            processed_store.add(stores_ref)
                            chunk_constraints.add(stores_ref)
                        
                            chunk_graph.add((stores_ref, RDF.type, SCHEMA.Store))
                            chunk_graph.add((stores_ref, SCHEMA.name, Literal(stores1, lang="en")))

                        chunk_graph.add(triple=(stores_ref, SCHEMA.offers, recipe_id))

                        for city in recipe_cities:
                            chunk_graph.add(triple=(stores_ref, UNICA.isPlaceIn, city))

        # Save this chunk
        chunk_file = f"{output_dir}chunk_{cont_chunk}.ttl"
        chunk_graph.serialize(destination=chunk_file, format="turtle")
        chunk_files_ttl.append(chunk_file)
        
        # Save this chunk
        chunk_file = f"{output_dir}chunk_{cont_chunk}.nt"
        chunk_graph.serialize(destination=chunk_file, format="nt", encoding="utf-8")
        chunk_files_nt.append(chunk_file)
        
        cont_chunk += 1
        
        # Force garbage collection after each chunk
        del chunk_graph
        gc.collect()
    
    print(f"Generated {len(chunk_files_ttl)} chunk files")
    print("Combining all files into the final output...")
    
    # Combine all the chunks into a single file
    with open(final_output_ttl, 'w') as outfile:
        for chunk_file in chunk_files_ttl:
            with open(chunk_file, 'r') as infile:
                outfile.write(infile.read())
                outfile.write("\n")
    
    print(f"Generated final file: {final_output_ttl}")
    
    # Combine all the chunks into a single file
    with open(final_output_nt, 'w') as outfile:
        for chunk_file in chunk_files_nt:
            with open(chunk_file, 'r') as infile:
                outfile.write(infile.read())
                outfile.write("\n")
    
    print(f"Generated final file: {final_output_nt}")


    # Optional: clean up temporary chunk files
    for chunk_file in chunk_files_ttl:
        try:
            os.remove(chunk_file)
        except:
            print(f"Could not remove {chunk_file}")
    print("Temporary files deleted")

    for chunk_file in chunk_files_nt:
        try:
            os.remove(chunk_file)
        except:
            print(f"Could not remove {chunk_file}")
    print("Temporary files deleted")







def create_merge_ontology():

    UNICA = Namespace("https://github.com/tail-unica/kgeats/")

    dizionario_hum = {}
    dizionario_off = {}

    hum_file = "../csv_file/pp_recipes_normalized_by_pipeline.csv"
    off_file = "../csv_file/off_normalized_final.csv"
    hum_off_file = "../csv_file/file_off_hummus_filtered_975.csv"
    file_output_nt =  "../csv_file/ontology_merge.nt"

    chunksize = 100000
    cont_chunk = 0

    for df_off_chunk in pd.read_csv(off_file, sep="\t", on_bad_lines="skip", chunksize=chunksize, low_memory=False, usecols=["product_name_normalized", "code"]):
        print(f"Processing rows off from {chunksize * cont_chunk} to {chunksize * (cont_chunk+1)}")
        
        for idx, row in df_off_chunk.iterrows():
            if(row["product_name_normalized"] != None and row["product_name_normalized"] != ""):
                id = URIRef(value=UNICA[f"Recipe_off_{row["code"]}"])
                if id != None:
                    if row["product_name_normalized"] not in dizionario_off:
                        dizionario_off[row["product_name_normalized"]] = [id]
                    else: 
                        dizionario_off[row["product_name_normalized"]].append(id)
        cont_chunk += 1

    cont_chunk = 0
    for df_hum_chunk in pd.read_csv(hum_file, sep=";", on_bad_lines="skip", chunksize=chunksize, low_memory=False, usecols=["title_normalized", "recipe_id"]):
        print(f"Processing rows hummus from {chunksize * cont_chunk} to {chunksize * (cont_chunk+1)}")
        
        for idx, row in df_hum_chunk.iterrows():
            if(row["title_normalized"] != None and row["title_normalized"] != ""):
                id = URIRef(UNICA[f"Recipe_hummus{sanitize_for_uri(row['recipe_id'])}"])
                if id != None:
                    if row["title_normalized"] not in dizionario_hum:
                        dizionario_hum[row["title_normalized"]] = [id]
                    else: 
                        dizionario_hum[row["title_normalized"]].append(id)
        cont_chunk += 1


    numchunk = 0
    chunksize = 1000

    hum_keys = set(dizionario_hum.keys())
    off_keys = set(dizionario_off.keys())

    hum_off_df = pd.read_csv(hum_off_file, sep=",", low_memory=False, on_bad_lines="skip")
    total_lines  = len(hum_off_df)

    total_chunks = (total_lines // chunksize) + 1
    start_total = time.time()

    with open(file_output_nt, "w", encoding="utf-8") as f_out:

        for df_merge_chunk in pd.read_csv(hum_off_file, sep=",", on_bad_lines="skip", chunksize=chunksize, low_memory=False, usecols=["title_normalized", "product_name_normalized"]):
            chunk_start = time.time()
            print(f"\nProcessing chunk {numchunk+1}/{total_chunks}")

            for row in df_merge_chunk.itertuples(index=False):
                title = row.title_normalized #need to be modified in the csv fle
                product = row.product_name_normalized

                if title in hum_keys and product in off_keys:
                    for hum_ricetta in dizionario_hum[title]:
                        for off_ricetta in dizionario_off[product]: 
                            triple_str = f"<{off_ricetta}> <https://schema.org/sameAs> <{hum_ricetta}> .\n"
                            f_out.write(triple_str)
 

            del df_merge_chunk
            gc.collect() 

            chunk_time = time.time() - chunk_start
            avg_time_per_chunk = (time.time() - start_total) / (numchunk + 1)
            remaining_chunks = total_chunks - (numchunk + 1)
            est_remaining = avg_time_per_chunk * remaining_chunks
            print(f"Chunk time: {chunk_time:.2f}s — Estimated remaining: {est_remaining/60:.1f} min")
            numchunk += 1

        total_time = time.time() - start_total
        print(f"\nTotal processing time: {total_time/60:.2f} minutes")





def create_merge_ontology2():

    UNICA = Namespace("https://github.com/tail-unica/kgeats/")

    dizionario_hum = {}
    dizionario_off = {}

    hum_file = "../csv_file/ingredients_food_kg_normalizzed_by_pipeline.csv"
    off_file = "../csv_file/off_normalized_final.csv"
    hum_off_file = "../csv_file/file_off_foodkg_filtered_975.csv"
    file_output_nt =  "../csv_file/ontology_merge_2.nt"

    chunksize = 100000
    cont_chunk = 0

    for df_off_chunk in pd.read_csv(off_file, sep="\t", on_bad_lines="skip", chunksize=chunksize, low_memory=False, usecols=["product_name_normalized", "code"]):
        print(f"Processing rows off from {chunksize * cont_chunk} to {chunksize * (cont_chunk+1)}")
        
        for idx, row in df_off_chunk.iterrows():
            if(row["product_name_normalized"] != None and row["product_name_normalized"] != ""):
                id = URIRef(value=UNICA[f"Recipe_off_{row["code"]}"])
                if id != None:
                    if row["product_name_normalized"] not in dizionario_off:
                        dizionario_off[row["product_name_normalized"]] = [id]
                    else: 
                        dizionario_off[row["product_name_normalized"]].append(id)
        cont_chunk += 1

    cont_chunk = 0


    for df_hum_chunk in pd.read_csv(hum_file, sep=",", on_bad_lines="skip", chunksize=chunksize, low_memory=False, usecols=["ingredient", "ingredient_normalized"]):
        print(f"Processing rows hummus from {chunksize * cont_chunk} to {chunksize * (cont_chunk+1)}")
        
        for idx, row in df_hum_chunk.iterrows():
            if(row["ingredient_normalized"] != None and row["ingredient_normalized"] != ""):
                id = URIRef(
                UNICA[
                    f"Recipe_Ingredient_{sanitize_for_uri(row["ingredient"].replace(' ', '_').lower())}"
                            ]
                        )
                if id != None:
                    if row["ingredient_normalized"] not in dizionario_hum:
                        dizionario_hum[row["ingredient_normalized"]] = [id]
                    else: 
                        dizionario_hum[row["ingredient_normalized"]].append(id)
        cont_chunk += 1


    numchunk = 0
    chunksize = 1000

    hum_keys = set(dizionario_hum.keys())
    off_keys = set(dizionario_off.keys())

    hum_off_df = pd.read_csv(hum_off_file, sep=",", low_memory=False, on_bad_lines="skip")
    total_lines  = len(hum_off_df)

    total_chunks = (total_lines // chunksize) + 1
    start_total = time.time()

    with open(file_output_nt, "w", encoding="utf-8") as f_out:

        for df_merge_chunk in pd.read_csv(hum_off_file, sep=",", on_bad_lines="skip", chunksize=chunksize, low_memory=False, usecols=["ingredient_normalized", "product_name_normalized"]):
            chunk_start = time.time()
            print(f"\nProcessing chunk {numchunk+1}/{total_chunks}")

            for row in df_merge_chunk.itertuples(index=False):
                title = row.ingredient_normalized #need to be modified in the csv fle
                product = row.product_name_normalized

                if title in hum_keys and product in off_keys:
                    for hum_ricetta in dizionario_hum[title]:
                        for off_ricetta in dizionario_off[product]: 
                            triple_str = f"<{off_ricetta}> <https://schema.org/sameAs> <{hum_ricetta}> .\n"
                            f_out.write(triple_str)

            del df_merge_chunk
            gc.collect() 

            chunk_time = time.time() - chunk_start
            avg_time_per_chunk = (time.time() - start_total) / (numchunk + 1)
            remaining_chunks = total_chunks - (numchunk + 1)
            est_remaining = avg_time_per_chunk * remaining_chunks
            print(f"Chunk time: {chunk_time:.2f}s — Estimated remaining: {est_remaining/60:.1f} min")
            numchunk += 1

        total_time = time.time() - start_total
        print(f"\nTotal processing time: {total_time/60:.2f} minutes")



def create_merge_ontology3():

    UNICA = Namespace("https://github.com/tail-unica/kgeats/")

    dizionario_hum = {}
    dizionario_off = {}

    hum_file = "../csv_file/pp_recipes_normalized_by_pipeline.csv"
    off_file = "../csv_file/pp_recipes_normalized_by_pipeline.csv"
    hum_off_file = "../csv_file/file_hummus_hummus_filtered_975.csv"
    file_output_nt =  "../csv_file/ontology_merge3.nt"

    chunksize = 100000
    cont_chunk = 0

    for df_off_chunk in pd.read_csv(off_file, sep=";", on_bad_lines="skip", chunksize=chunksize, low_memory=False, usecols=["title_normalized", "recipe_id"]):
        print(f"Processing rows hum from {chunksize * cont_chunk} to {chunksize * (cont_chunk+1)}")
        
        for idx, row in df_off_chunk.iterrows():
            if(row["title_normalized"] != None and row["title_normalized"] != ""):
                id = URIRef(UNICA[f"Recipe_hummus{sanitize_for_uri(row['recipe_id'])}"])
                if id != None:
                    if row["title_normalized"] not in dizionario_off:
                        dizionario_off[row["title_normalized"]] = [id]
                    else: 
                        dizionario_off[row["title_normalized"]].append(id)
        cont_chunk += 1

    cont_chunk = 0
    for df_hum_chunk in pd.read_csv(hum_file, sep=";", on_bad_lines="skip", chunksize=chunksize, low_memory=False, usecols=["title_normalized", "recipe_id"]):
        print(f"Processing rows hummus from {chunksize * cont_chunk} to {chunksize * (cont_chunk+1)}")
        
        for idx, row in df_hum_chunk.iterrows():
            if(row["title_normalized"] != None and row["title_normalized"] != ""):
                id = URIRef(UNICA[f"Recipe_hummus{sanitize_for_uri(row['recipe_id'])}"])
                if id != None:
                    if row["title_normalized"] not in dizionario_hum:
                        dizionario_hum[row["title_normalized"]] = [id]
                    else: 
                        dizionario_hum[row["title_normalized"]].append(id)
        cont_chunk += 1


    numchunk = 0
    chunksize = 1000

    hum_keys = set(dizionario_hum.keys())
    off_keys = set(dizionario_off.keys())

    hum_off_df = pd.read_csv(hum_off_file, sep=",", low_memory=False, on_bad_lines="skip")
    total_lines  = len(hum_off_df)

    total_chunks = (total_lines // chunksize) + 1
    start_total = time.time()

    with open(file_output_nt, "w", encoding="utf-8") as f_out:

        for df_merge_chunk in pd.read_csv(hum_off_file, sep=",", on_bad_lines="skip", chunksize=chunksize, low_memory=False, usecols=["name_file1", "name_file2"]):

            chunk_start = time.time()
            print(f"\nProcessing chunk {numchunk+1}/{total_chunks}")

            for row in df_merge_chunk.itertuples(index=False):
                title = row.name_file1
                product = row.name_file2

                if title in hum_keys and product in off_keys:
                    for hum_ricetta in dizionario_hum[title]:
                        for off_ricetta in dizionario_off[product]: 
                            if off_ricetta != hum_ricetta:
                                triple_str = f"<{off_ricetta}> <https://schema.org/sameAs> <{hum_ricetta}> .\n"
                                f_out.write(triple_str)
 

            del df_merge_chunk
            gc.collect() 

            chunk_time = time.time() - chunk_start
            avg_time_per_chunk = (time.time() - start_total) / (numchunk + 1)
            remaining_chunks = total_chunks - (numchunk + 1)
            est_remaining = avg_time_per_chunk * remaining_chunks
            print(f"Chunk time: {chunk_time:.2f}s — Estimated remaining: {est_remaining/60:.1f} min")
            numchunk += 1

        total_time = time.time() - start_total
        print(f"\nTotal processing time: {total_time/60:.2f} minutes")


