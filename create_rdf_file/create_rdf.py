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
    )
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
                "calcium",
                "Calcium for 100g",
                "Calcium content for 100 grams.",
                XSD.float,
            ),
            (
                "iron",
                "Iron for 100g",
                "Iron content per 100 grams.",
                XSD.float,
            ),
            (
                "vitaminC",
                "Vitamin C for 100g",
                "Vitamin C content per 100 grams.",
                XSD.float,
            ),
            (
                "vitaminA",
                "Vitamin A for 100g",
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
                "nutritionScoreFr",
                "Nutrition Score FR for 100g",
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
        file_recipes = "../csv_file/pp_recipes.csv"
        if use_infered_attributes_description:
            file_users = "../csv_file/pp_members_with_attributes.csv"
        else:
            file_users = "../csv_file/pp_members.csv"

        if use_infered_attributes_review:
            file_review = "../csv_file/pp_reviews_with_attributes.csv"
        else:
            file_review = "../csv_file/pp_reviews.csv"

    if use_infered_attributes_review or use_infered_attributes_description:
        file_output = "../csv_file/ontology_hummus.ttl"
    else:
        file_output = "../csv_file/ontology_hummus_not_infered.ttl"

    # Upload the CSV
    df_ricette = pd.read_csv(filepath_or_buffer=file_recipes, on_bad_lines="skip", low_memory=False)
    df_review = pd.read_csv(file_review, on_bad_lines="skip", low_memory=False)
    df_utenti = pd.read_csv(file_users, on_bad_lines="skip", low_memory=False)

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
                    if tag_id != "Tag_":
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
                                        f"is a user constraint about {tag1.replace('-', ' ')}",
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
                        SCHEMA.publishesReview,
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


    df_sustainability = pd.read_csv(filepath_or_buffer=file_recipes, on_bad_lines="skip", low_memory=False)
    print("starting sustainability creation")

    conta = 0
    # Create UserReview entities and relationships
    for idx, row in df_sustainability.iterrows():
        conta += 1
        if conta % 1000 == 0:
            print(f"row: {conta}")
        if pd.notna(row["recipe_id"]) and isinstance(row["recipe_id"], (int, float)):
            
            for colonna in ["recipe_CF_kg", "recipe_WF_kg"]:
                tipo = colonna.replace('recipe_', '')
                indicator_id = URIRef(
                        UNICA[
                            f"Indicator_{sanitize_for_uri(value=tipo.lower())}_{sanitize_for_uri(row['recipe_id'])}"
                        ]
                    )   

                g.add((indicator_id, RDF.type, UNICA.Indicator))
                g.add((indicator_id, SCHEMA.type, Literal(lexical_or_value=tipo)))
                g.add((indicator_id, SCHEMA.unitText, Literal(lexical_or_value="kg")))
                g.add(
                (
                    indicator_id,
                    SCHEMA.quantity,
                    Literal(row[colonna], datatype=XSD.float),
                )
                )
                g.add(
                    (row["recipe_id"], SCHEMA.NutritionInformation, indicator_id)
                )

    # Save the RDF graph in Turtle format
    g.serialize(destination=file_output, format="turtle")
    print(f"Generated file: {file_output}")


def convert_off_in_rdf(use_row=False) -> None:
    """
    Function to convert off data into RDF format
    """

    # File paths
    if use_row:
        off_file = "../csv_file/off_rows.csv"
    else:
        off_file = "../csv_file/off_english.csv"
    
    # Create output directory if it doesn't exist
    output_dir = "../csv_file/chunks/"
    os.makedirs(output_dir, exist_ok=True)
    
    final_output = "../csv_file/ontology_off.ttl"
    
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
    }
    
    # Initialize a set to track already processed allergens and traces
    processed_constraints = set()
    
    # Define namespaces (will be used in each chunk)
    UNICA = Namespace("https://github.com/tail-unica/kgeats/")
    SCHEMA = Namespace("https://schema.org/")
    
    # Process in smaller chunks for better memory management
    chunksize = 50000  #10000 if the code run out of memory
    chunk_files = []
    
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
    chunk_files.append(ontology_file)
    
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
                recipe_id = URIRef(UNICA[f"Recipe_off_{idx}"]) # I put _off to differentiate them from the hummus ids
                chunk_graph.add((recipe_id, RDF.type, SCHEMA.Recipe))
                chunk_graph.add((recipe_id, SCHEMA.name, Literal(row["product_name"], lang="en")))
                chunk_graph.add((recipe_id, SCHEMA.identifier, Literal(idx, datatype=XSD.integer)))

                # Process allergens
                if pd.notna(row["allergens"]):
                    for allergen in row["allergens"].split(","):
                        allergen1 = re.sub(r"..:", "", str(allergen))
                        allergen_uri = sanitize_for_uri(allergen.replace("en:", "").replace("-", "_").lower())
                        allergen_ref = URIRef(UNICA[f"allergen_{allergen_uri}"])
                        
                        # Check if we've already processed this allergen
                        if allergen_ref not in processed_constraints:
                            processed_constraints.add(allergen_ref)
                            chunk_constraints.add(allergen_ref)
                            
                            chunk_graph.add((allergen_ref, RDF.type, UNICA.UserConstraint))
                            chunk_graph.add((allergen_ref, SCHEMA.constraintName, Literal(allergen1, lang="en")))
                            chunk_graph.add((
                                allergen_ref, 
                                SCHEMA.constraintDescription, 
                                Literal(f"is a user constraint about having an allergy to {allergen1}", lang="en")
                            ))
                        
                        chunk_graph.add((allergen_ref, SCHEMA.suitableForDiet, recipe_id))

                # Process traces
                if pd.notna(row["traces_en"]):
                    for trace in row["traces_en"].split(","):
                        trace1 = trace
                        trace_uri = sanitize_for_uri(trace.replace("-", "_").lower())
                        trace_ref = URIRef(UNICA[f"trace_{trace_uri}"])
                        
                        if trace_ref not in processed_constraints:
                            processed_constraints.add(trace_ref)
                            chunk_constraints.add(trace_ref)
                            
                            chunk_graph.add((trace_ref, RDF.type, UNICA.UserConstraint))
                            chunk_graph.add((trace_ref, SCHEMA.constraintName, Literal(trace1, lang="en")))
                            chunk_graph.add((
                                trace_ref, 
                                SCHEMA.constraintDescription, 
                                Literal(f"is a user constraint about having a trace of {trace1} in the product", lang="en")
                            ))
                        
                        chunk_graph.add((trace_ref, SCHEMA.suitableForDiet, recipe_id))

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
                            chunk_graph.add((recipe_id, SCHEMA.NutritionInformation, indicator_id))
        
        # Save this chunk
        chunk_file = f"{output_dir}chunk_{cont_chunk}.ttl"
        chunk_graph.serialize(destination=chunk_file, format="turtle")
        chunk_files.append(chunk_file)
        
        cont_chunk += 1
        
        # Force garbage collection after each chunk
        del chunk_graph
        gc.collect()
    
    print(f"Generated {len(chunk_files)} chunk files")
    print("Combining all files into the final output...")
    
    # Combine all the chunks into a single file
    with open(final_output, 'w') as outfile:
        for chunk_file in chunk_files:
            with open(chunk_file, 'r') as infile:
                outfile.write(infile.read())
                outfile.write("\n")
    
    print(f"Generated final file: {final_output}")
    
    # Optional: clean up temporary chunk files
    for chunk_file in chunk_files:
        try:
            os.remove(chunk_file)
        except:
            print(f"Could not remove {chunk_file}")
    print("Temporary files deleted")

